"""libfranka Gripper Server v3 backed by MuJoCo finger physics."""

import errno
import logging
import select
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from remu.protocol.gripper_protocol import (
    GRIPPER_COMMAND_PORT,
    GRIPPER_HEADER_SIZE,
    GRIPPER_VERSION,
    GripperCommand,
    GripperConnectStatus,
    GripperHeader,
    GripperStatus,
    connect_response,
    pack_state,
    response,
)
from remu.sim.mujoco_sim import FRANKA_HAND_MAX_FORCE, FRANKA_HAND_MAX_WIDTH, MujocoSim

logger = logging.getLogger(__name__)

_BROADCAST_HZ = 60.0
_POLL_PERIOD = 0.01
_SETTLE_VELOCITY = 1e-3
_SETTLE_SAMPLES = 3
_TARGET_TOLERANCE = 1e-3
_MOTION_TIMEOUT = 5.0


@dataclass
class _PendingMotion:
    command: GripperCommand
    command_id: int
    target_width: float
    initial_width: float
    deadline: float
    epsilon_inner: float = 0.0
    epsilon_outer: float = 0.0
    moved: bool = False
    settled_samples: int = 0


class FrankaGripperServer:
    """TCP command server plus UDP state stream on the standard gripper port."""

    def __init__(
        self,
        sim: MujocoSim,
        host: str = "0.0.0.0",
        port: int = GRIPPER_COMMAND_PORT,
        on_command: Optional[Callable[[dict], None]] = None,
        on_raw_packet: Optional[Callable[..., None]] = None,
    ):
        if not sim.enable_gripper:
            raise ValueError("cannot start a gripper server for a gripper-disabled simulation")
        self.sim = sim
        self.host = host
        self.port = port
        self.on_command = on_command
        self.on_raw_packet = on_raw_packet
        self.server_socket: Optional[socket.socket] = None
        self.client_socket: Optional[socket.socket] = None
        self.udp_socket: Optional[socket.socket] = None
        self.running = False
        self.connection_running = False
        self.client_address: Optional[str] = None
        self.client_udp_port: Optional[int] = None
        self._broadcast_thread: Optional[threading.Thread] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._pending: Optional[_PendingMotion] = None
        self._is_grasped = False

    @staticmethod
    def _receive_exact(sock: socket.socket, size: int) -> Optional[bytes]:
        data = bytearray()
        while len(data) < size:
            try:
                chunk = sock.recv(size - len(data))
            except OSError:
                return None
            if not chunk:
                return None
            data.extend(chunk)
        return bytes(data)

    def _receive_message(self, sock: socket.socket):
        header_data = self._receive_exact(sock, GRIPPER_HEADER_SIZE)
        if header_data is None:
            return None, None
        header = GripperHeader.from_bytes(header_data)
        payload_size = header.size - GRIPPER_HEADER_SIZE
        if payload_size < 0 or payload_size > 4096:
            raise ValueError(f"invalid gripper message size {header.size}")
        payload = self._receive_exact(sock, payload_size) if payload_size else b""
        if payload is None:
            return None, None
        if self.on_raw_packet is not None:
            self.on_raw_packet(
                transport="tcp", direction="client_to_remu",
                endpoint="gripper", data=header_data + payload,
            )
        return header, payload

    def _send_status(self, sock, command, command_id, status):
        packet = response(command, command_id, status)
        sock.sendall(packet)
        if self.on_raw_packet is not None:
            self.on_raw_packet(
                transport="tcp", direction="remu_to_client",
                endpoint="gripper", data=packet,
            )

    def _emit_command(self, command, command_id, **values) -> None:
        if self.on_command is not None:
            self.on_command({
                "command": command.name,
                "command_id": command_id,
                **values,
            })

    def _start_motion(
        self,
        sock,
        command: GripperCommand,
        command_id: int,
        width: float,
        speed: float,
        force: float,
        epsilon_inner: float = 0.0,
        epsilon_outer: float = 0.0,
        evaluation_width: Optional[float] = None,
    ) -> None:
        if self._pending is not None:
            self._send_status(sock, command, command_id, GripperStatus.kFail)
            return
        try:
            self.sim.set_gripper_target(width, speed, force)
        except (ValueError, RuntimeError) as exc:
            logger.info("Rejecting gripper command: %s", exc)
            self._send_status(sock, command, command_id, GripperStatus.kFail)
            return
        state = self.sim.get_finger_state()
        self._is_grasped = False
        self._pending = _PendingMotion(
            command=command,
            command_id=command_id,
            target_width=width if evaluation_width is None else evaluation_width,
            initial_width=float(state["width"]),
            deadline=time.monotonic() + _MOTION_TIMEOUT,
            epsilon_inner=epsilon_inner,
            epsilon_outer=epsilon_outer,
        )

    def _abort_pending(self, sock: Optional[socket.socket]) -> None:
        pending, self._pending = self._pending, None
        if pending is not None and sock is not None:
            try:
                self._send_status(
                    sock, pending.command, pending.command_id, GripperStatus.kAborted
                )
            except OSError:
                pass

    def _poll_motion(self, sock: socket.socket) -> None:
        pending = self._pending
        if pending is None:
            if self._is_grasped and not self.sim.get_finger_state()["contact_body_ids"]:
                self._is_grasped = False
            return

        state = self.sim.get_finger_state()
        width = float(state["width"])
        speed = float(np.max(np.abs(state["dq"])))
        pending.moved = pending.moved or abs(width - pending.initial_width) > 1e-4
        began_at_target = abs(pending.initial_width - pending.target_width) <= _TARGET_TOLERANCE
        if speed < _SETTLE_VELOCITY and (pending.moved or began_at_target):
            pending.settled_samples += 1
        else:
            pending.settled_samples = 0

        timed_out = time.monotonic() >= pending.deadline
        if pending.settled_samples < _SETTLE_SAMPLES and not timed_out:
            return

        if pending.command == GripperCommand.kGrasp:
            in_band = (
                pending.target_width - pending.epsilon_inner
                < width
                < pending.target_width + pending.epsilon_outer
            )
            successful = in_band and bool(state["contact_body_ids"]) and not timed_out
            self._is_grasped = successful
        else:
            successful = abs(width - pending.target_width) <= _TARGET_TOLERANCE and not timed_out

        status = GripperStatus.kSuccess if successful else GripperStatus.kUnsuccessful
        self._pending = None
        if not successful:
            self.sim.stop_gripper()
        self._send_status(sock, pending.command, pending.command_id, status)

    def _handle_connect(self, sock, header, payload) -> None:
        if len(payload) != 4:
            self._send_status(sock, header.command, header.command_id, GripperStatus.kFail)
            return
        version, udp_port = struct.unpack("<HH", payload)
        status = (
            GripperConnectStatus.kSuccess
            if version == GRIPPER_VERSION
            else GripperConnectStatus.kIncompatibleLibraryVersion
        )
        sock.sendall(connect_response(header.command_id, status))
        if status != GripperConnectStatus.kSuccess:
            return
        self.client_address = sock.getpeername()[0]
        self.client_udp_port = udp_port
        if self.udp_socket is None:
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._broadcast_thread = threading.Thread(target=self._broadcast_state, daemon=True)
            self._broadcast_thread.start()

    def _dispatch(self, sock, header, payload) -> None:
        try:
            command = GripperCommand(header.command)
        except ValueError:
            self._send_status(sock, header.command, header.command_id, GripperStatus.kFail)
            return

        if command == GripperCommand.kConnect:
            self._emit_command(command, header.command_id)
            self._handle_connect(sock, header, payload)
        elif command == GripperCommand.kHoming:
            if payload:
                self._send_status(sock, command, header.command_id, GripperStatus.kFail)
            else:
                self._emit_command(
                    command, header.command_id, width=FRANKA_HAND_MAX_WIDTH,
                    speed=0.05, force=FRANKA_HAND_MAX_FORCE,
                )
                self._start_motion(
                    sock, command, header.command_id, FRANKA_HAND_MAX_WIDTH, 0.05,
                    FRANKA_HAND_MAX_FORCE,
                )
        elif command == GripperCommand.kMove:
            if len(payload) != 16:
                self._send_status(sock, command, header.command_id, GripperStatus.kFail)
            else:
                width, speed = struct.unpack("<dd", payload)
                self._emit_command(
                    command, header.command_id, width=width, speed=speed,
                    force=FRANKA_HAND_MAX_FORCE,
                )
                self._start_motion(
                    sock, command, header.command_id, width, speed, FRANKA_HAND_MAX_FORCE
                )
        elif command == GripperCommand.kGrasp:
            if len(payload) != 40:
                self._send_status(sock, command, header.command_id, GripperStatus.kFail)
            else:
                width, eps_inner, eps_outer, speed, force = struct.unpack("<ddddd", payload)
                if (
                    not np.all(np.isfinite([width, eps_inner, eps_outer, speed, force]))
                    or not 0.0 <= width <= FRANKA_HAND_MAX_WIDTH
                    or min(eps_inner, eps_outer) < 0.0
                ):
                    self._send_status(sock, command, header.command_id, GripperStatus.kFail)
                else:
                    self._emit_command(
                        command, header.command_id, width=width, speed=speed, force=force,
                        epsilon_inner=eps_inner, epsilon_outer=eps_outer,
                    )
                    # ``width`` describes the expected held object, not a
                    # no-load position target. Close fully; contact stops the
                    # fingers and the requested force caps the resulting load.
                    physical_width = 0.0
                    self._start_motion(
                        sock, command, header.command_id, physical_width, speed, force,
                        eps_inner, eps_outer, evaluation_width=width,
                    )
        elif command == GripperCommand.kStop:
            self._emit_command(command, header.command_id)
            self.sim.stop_gripper()
            self._is_grasped = False
            self._abort_pending(sock)
            self._send_status(sock, command, header.command_id, GripperStatus.kSuccess)

    def _broadcast_state(self) -> None:
        period = 1.0 / _BROADCAST_HZ
        next_deadline = time.perf_counter()
        message_id = 0
        while self.running and self.connection_running:
            address, port, udp = self.client_address, self.client_udp_port, self.udp_socket
            if address is not None and port is not None and udp is not None:
                state = self.sim.get_finger_state()
                is_grasped = self._is_grasped and bool(state["contact_body_ids"])
                try:
                    packet = pack_state(
                        message_id, float(state["width"]), FRANKA_HAND_MAX_WIDTH,
                        is_grasped, 30,
                    )
                    udp.sendto(packet, (address, port))
                    if self.on_raw_packet is not None:
                        self.on_raw_packet(
                            transport="udp", direction="remu_to_client",
                            endpoint="gripper", data=packet,
                        )
                    message_id = (message_id + 1) & 0xFFFFFFFF
                except OSError:
                    pass
            next_deadline += period
            remaining = next_deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            else:
                next_deadline = time.perf_counter()

    def _handle_client(self, sock: socket.socket) -> None:
        self.client_socket = sock
        self.connection_running = True
        try:
            while self.running and self.connection_running:
                readable, _, _ = select.select([sock], [], [], _POLL_PERIOD)
                if readable:
                    header, payload = self._receive_message(sock)
                    if header is None:
                        break
                    self._dispatch(sock, header, payload)
                self._poll_motion(sock)
        except (ConnectionError, OSError, ValueError):
            logger.debug("gripper client disconnected", exc_info=True)
        finally:
            self.connection_running = False
            self.sim.stop_gripper()
            self._abort_pending(None)
            if self._broadcast_thread is not None:
                self._broadcast_thread.join(timeout=1.0)
                self._broadcast_thread = None
            if self.udp_socket is not None:
                self.udp_socket.close()
                self.udp_socket = None
            self.client_address = None
            self.client_udp_port = None
            self.client_socket = None
            sock.close()

    def _accept_loop(self) -> None:
        while self.running:
            try:
                client, address = self.server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            logger.info("New gripper connection from %s:%d", *address)
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._handle_client(client)

    def start(self, background: bool = True):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(1.0)
        try:
            sock.bind((self.host, self.port))
        except OSError as exc:
            sock.close()
            if exc.errno == errno.EADDRINUSE:
                logger.error("Gripper port %s:%s is already in use", self.host, self.port)
            raise
        sock.listen(1)
        self.server_socket = sock
        self.port = sock.getsockname()[1]
        self.running = True
        logger.info("Gripper server listening on %s:%d", self.host, self.port)
        if not background:
            self._accept_loop()
            return None
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()
        return self._accept_thread

    def stop(self) -> None:
        self.running = False
        self.connection_running = False
        self.sim.stop_gripper()
        for sock in (self.client_socket, self.server_socket, self.udp_socket):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        self.server_socket = None
        if self._accept_thread is not None and self._accept_thread is not threading.current_thread():
            self._accept_thread.join(timeout=2.0)
        self._accept_thread = None
