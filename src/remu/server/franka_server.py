"""FCI (Franka Control Interface) network server.

Implements the libfranka research_interface wire protocol (v10, robot server
10) on top of a :class:`~remu.sim.mujoco_sim.MujocoSim` physics backend, so a
real ``franka::Robot`` client (or a libfranka-based controller) can connect to
``127.0.0.1`` and drive the simulated arm exactly as it would drive hardware.

Server structure (TCP command channel + UDP 1kHz state channel) is adapted
from the ``libfranka-sim`` reference implementation vendored under
``remu/references/libfranka-sim`` (Apache-2.0).
"""

import errno
import logging
import select
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Optional

from remu.protocol.franka_protocol import (
    COMMAND_PORT,
    Command,
    ConnectStatus,
    ControllerMode,
    LibfrankaControllerMode,
    LibfrankaMotionGeneratorMode,
    MessageHeader,
    MotionGeneratorMode,
    MoveCommand,
    MoveStatus,
    PROTOCOL_VERSION,
    RobotMode,
    SetCartesianImpedanceCommand,
    SetCollisionBehaviorCommand,
    SetJointImpedanceCommand,
    convert_to_libfranka_controller_mode,
    convert_to_libfranka_motion_mode,
)
from remu.protocol.robot_state import RobotState
from remu.sim.mujoco_sim import ControlMode, MujocoSim

logger = logging.getLogger(__name__)

DEFAULT_URDF = Path(__file__).resolve().parent.parent / "models" / "fr3.urdf"

# RobotCommand packet size (matches libfranka's RobotCommand struct):
# message_id(8) + MotionGeneratorCommand(q_c 56 + dq_c 56 + O_T_EE_c 128
# + O_dP_EE_c 48 + elbow_c 16 + valid_elbow 1 + motion_finished 1)
# + ControllerCommand(tau_J_d 56 + torque_finished 1)
_ROBOT_COMMAND_SIZE = 8 + (56 + 56 + 128 + 48 + 16 + 1 + 1) + (56 + 1)


class FrankaFciServer:
    """TCP command + UDP state server implementing the FCI protocol over MuJoCo."""

    def __init__(
        self,
        sim: MujocoSim,
        host: str = "0.0.0.0",
        port: int = COMMAND_PORT,
        urdf_path: Optional[Path] = None,
    ):
        self.sim = sim
        self.host = host
        self.port = port
        self.library_version = PROTOCOL_VERSION

        self.server_socket: Optional[socket.socket] = None
        self.client_socket: Optional[socket.socket] = None
        self.udp_socket: Optional[socket.socket] = None
        self.client_address: Optional[str] = None
        self.client_udp_port: Optional[int] = None

        self.running = False
        self.connection_running = False
        self.transmitting_state = False
        self.current_motion_id = 0
        self.control_mode = ControlMode.NONE

        self.robot_state = RobotState()
        self.urdf_string = Path(urdf_path or DEFAULT_URDF).read_text(encoding="utf-8")

        self._threads = []

    # -- connection bookkeeping -----------------------------------------
    def reset_state(self):
        self.transmitting_state = False
        self.current_motion_id = 0
        self.client_socket = None
        self.udp_socket = None
        self.client_address = None
        self.client_udp_port = None
        self.control_mode = ControlMode.NONE
        self.connection_running = False
        self.robot_state = RobotState()

    # -- framing ----------------------------------------------------------
    def _receive_exact(self, sock: socket.socket, size: int) -> Optional[bytes]:
        data = bytearray()
        remaining = size
        while remaining > 0:
            try:
                chunk = sock.recv(remaining)
                if not chunk:
                    return None
                data.extend(chunk)
                remaining -= len(chunk)
            except socket.error:
                return None
        return bytes(data)

    def _receive_message(self, client_socket):
        header_data = self._receive_exact(client_socket, 12)
        if not header_data:
            raise ConnectionError("Failed to receive message header")
        header = MessageHeader.from_bytes(header_data)
        payload_size = header.size - 12
        payload = None
        if payload_size > 0:
            payload = self._receive_exact(client_socket, payload_size)
            if not payload:
                raise ConnectionError("Failed to receive message payload")
        return header, payload

    # -- TCP command handlers ---------------------------------------------
    def _send_connect_response(self, client_socket, command_id, status):
        total_size = 12 + 3
        header = MessageHeader(Command.kConnect, command_id, total_size)
        response = struct.pack("<BH", status.value, self.library_version)
        client_socket.sendall(header.to_bytes() + response)

    def _handle_get_robot_model(self, client_socket, header):
        urdf_bytes = self.urdf_string.encode("utf-8")
        payload = struct.pack("<B", 0) + urdf_bytes
        response_header = MessageHeader(Command.kGetRobotModel, header.command_id, 12 + len(payload))
        client_socket.sendall(response_header.to_bytes() + payload)

    def _send_status_response(self, client_socket, command, command_id, status_value=0):
        total_size = 12 + 4
        response_header = MessageHeader(command, command_id, total_size)
        client_socket.sendall(response_header.to_bytes() + struct.pack("<B3x", status_value))

    def _send_move_response(self, client_socket, command_id, status: MoveStatus):
        total_size = 12 + 4
        header = MessageHeader(Command.kMove, command_id, total_size)
        client_socket.sendall(header.to_bytes() + struct.pack("<B3x", status.value))

    def _handle_move_command(self, client_socket, header, payload):
        try:
            move_cmd = MoveCommand.from_bytes(payload)
        except ValueError:
            self._send_move_response(client_socket, header.command_id, MoveStatus.kInvalidArgumentRejected)
            return

        self.robot_state.set_motion_generator_mode(
            convert_to_libfranka_motion_mode(move_cmd.motion_generator_mode)
        )
        self.robot_state.set_controller_mode(
            convert_to_libfranka_controller_mode(move_cmd.controller_mode)
        )
        self.robot_state.state["robot_mode"] = RobotMode.kMove
        self.current_motion_id = header.command_id

        if (
            move_cmd.controller_mode == ControllerMode.kJointImpedance
            and move_cmd.motion_generator_mode == MotionGeneratorMode.kJointPosition
        ):
            self.sim.set_control_mode(ControlMode.POSITION)
            self.control_mode = ControlMode.POSITION
        elif (
            move_cmd.controller_mode == ControllerMode.kJointImpedance
            and move_cmd.motion_generator_mode == MotionGeneratorMode.kJointVelocity
        ):
            self.sim.set_control_mode(ControlMode.VELOCITY)
            self.control_mode = ControlMode.VELOCITY
        elif move_cmd.controller_mode == ControllerMode.kExternalController:
            self.sim.set_control_mode(ControlMode.TORQUE)
            self.control_mode = ControlMode.TORQUE

        self._send_move_response(client_socket, header.command_id, MoveStatus.kMotionStarted)

    def _handle_stop_move_command(self, client_socket, header):
        self._send_status_response(client_socket, Command.kStopMove, header.command_id)

        if self.control_mode != ControlMode.POSITION:
            current_q = self.sim.get_robot_state()["q"]
            self.sim.update_joint_positions(current_q)
            self.sim.set_control_mode(ControlMode.POSITION)
            self.control_mode = ControlMode.POSITION

        if self.udp_socket:
            self.robot_state.state["motion_generator_mode"] = 0
            self.robot_state.state["controller_mode"] = 3
            self.robot_state.state["robot_mode"] = RobotMode.kIdle
            self.robot_state.update()
            self.udp_socket.sendto(
                self.robot_state.pack_state(), (self.client_address, self.client_udp_port)
            )

        self.transmitting_state = False

        if self.current_motion_id:
            move_header = MessageHeader(Command.kMove, self.current_motion_id, 16)
            client_socket.sendall(move_header.to_bytes() + struct.pack("<B3x", MoveStatus.kSuccess.value))
            self.current_motion_id = 0

        self.connection_running = False

    def _handle_automatic_error_recovery(self, client_socket, header):
        self.robot_state.state["robot_mode"] = RobotMode.kIdle
        self._send_status_response(client_socket, Command.kAutomaticErrorRecovery, header.command_id)

    def _handle_tcp_messages(self, client_socket):
        while self.running:
            try:
                try:
                    client_socket.getpeername()
                except socket.error:
                    self.transmitting_state = False
                    self.connection_running = False
                    break

                readable, _, _ = select.select([client_socket], [], [], 0.1)
                if not readable:
                    continue

                header, payload = self._receive_message(client_socket)

                if header.command == Command.kMove:
                    self._handle_move_command(client_socket, header, payload)
                elif header.command == Command.kStopMove:
                    self._handle_stop_move_command(client_socket, header)
                elif header.command == Command.kSetCollisionBehavior:
                    SetCollisionBehaviorCommand.from_bytes(payload)
                    self._send_status_response(client_socket, header.command, header.command_id)
                elif header.command == Command.kSetJointImpedance:
                    SetJointImpedanceCommand.from_bytes(payload)
                    self._send_status_response(client_socket, header.command, header.command_id)
                elif header.command == Command.kSetCartesianImpedance:
                    SetCartesianImpedanceCommand.from_bytes(payload)
                    self._send_status_response(client_socket, header.command, header.command_id)
                elif header.command == Command.kGetRobotModel:
                    self._handle_get_robot_model(client_socket, header)
                elif header.command == Command.kAutomaticErrorRecovery:
                    self._handle_automatic_error_recovery(client_socket, header)
                else:
                    logger.warning("Unhandled command: %s", header.command)
            except ConnectionError:
                self.transmitting_state = False
                self.connection_running = False
                break
            except Exception:
                logger.exception("Error in TCP thread")
                if not self.running:
                    break
                self.transmitting_state = False
                self.connection_running = False
                break

    # -- UDP command channel (motion generator / controller commands) ----
    def _handle_udp_commands(self):
        poller = select.poll()
        poller.register(self.udp_socket.fileno(), select.POLLIN)

        while self.running and self.connection_running:
            udp_socket = self.udp_socket
            if udp_socket is None:
                break
            if not poller.poll(1):
                continue

            try:
                data, _ = udp_socket.recvfrom(_ROBOT_COMMAND_SIZE)
            except (BlockingIOError, OSError):
                continue
            if len(data) != _ROBOT_COMMAND_SIZE:
                continue

            offset = 0
            message_id = struct.unpack("<Q", data[offset : offset + 8])[0]
            offset += 8
            q_c = struct.unpack("<7d", data[offset : offset + 56])
            offset += 56
            dq_c = struct.unpack("<7d", data[offset : offset + 56])
            offset += 56
            offset += 128 + 48 + 16  # O_T_EE_c, O_dP_EE_c, elbow_c (unused: no cartesian control yet)
            offset += 1  # valid_elbow
            motion_generation_finished = bool(data[offset])
            offset += 1
            tau_J_d = struct.unpack("<7d", data[offset : offset + 56])
            offset += 56
            torque_command_finished = bool(data[offset])

            if message_id == 0:
                continue

            if motion_generation_finished or torque_command_finished:
                if self.control_mode != ControlMode.POSITION:
                    current_q = self.sim.get_robot_state()["q"]
                    self.sim.update_joint_positions(current_q)
                    self.sim.set_control_mode(ControlMode.POSITION)
                    self.control_mode = ControlMode.POSITION

                self.robot_state.state["motion_generator_mode"] = 0
                self.robot_state.state["controller_mode"] = 3
                self.robot_state.state["robot_mode"] = RobotMode.kIdle
                self.robot_state.update()
                if self.udp_socket is not None:
                    self.udp_socket.sendto(
                        self.robot_state.pack_state(), (self.client_address, self.client_udp_port)
                    )

                if self.current_motion_id and self.client_socket is not None:
                    move_header = MessageHeader(Command.kMove, self.current_motion_id, 16)
                    self.client_socket.sendall(
                        move_header.to_bytes() + struct.pack("<B3x", MoveStatus.kSuccess.value)
                    )
                    self.current_motion_id = 0
                continue

            controller_mode = self.robot_state.state["controller_mode"]
            motion_mode = self.robot_state.state["motion_generator_mode"]

            if (
                controller_mode == LibfrankaControllerMode.kJointImpedance
                and motion_mode == LibfrankaMotionGeneratorMode.kJointPosition
            ):
                self.robot_state.state["q_d"] = list(q_c)
                self.sim.update_joint_positions(q_c)
            elif (
                controller_mode == LibfrankaControllerMode.kJointImpedance
                and motion_mode == LibfrankaMotionGeneratorMode.kJointVelocity
            ):
                self.robot_state.state["dq_d"] = list(dq_c)
                self.sim.update_joint_velocities(dq_c)
            elif controller_mode == LibfrankaControllerMode.kExternalController:
                self.robot_state.state["tau_J_d"] = list(tau_J_d)
                self.sim.update_torques(tau_J_d)

    # -- UDP state channel (1kHz robot state broadcast) -------------------
    def _run_state_transmission(self, client_address: str, client_udp_port: int):
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        command_thread = threading.Thread(target=self._handle_udp_commands, daemon=True)
        command_thread.start()
        self._threads.append(command_thread)

        self.client_address = client_address
        self.client_udp_port = client_udp_port
        self.transmitting_state = True
        first_state_sent = False

        period = 0.001
        next_deadline = time.perf_counter()

        while self.running and self.connection_running and self.transmitting_state:
            sim_state = self.sim.get_robot_state()
            if not first_state_sent:
                self.robot_state.state["q_d"] = list(sim_state["q"])

            self.robot_state.state["q"] = list(sim_state["q"])
            self.robot_state.state["dq"] = list(sim_state["dq"])
            self.robot_state.state["tau_J"] = list(sim_state["tau_J"])
            self.robot_state.state["O_T_EE"] = list(sim_state["O_T_EE"])

            if self.udp_socket and not self.udp_socket._closed:
                self.udp_socket.sendto(
                    self.robot_state.pack_state(), (client_address, client_udp_port)
                )

            if not first_state_sent and self.current_motion_id:
                self._send_move_response(self.client_socket, self.current_motion_id, MoveStatus.kSuccess)
                first_state_sent = True

            self.robot_state.update()

            next_deadline += period
            remaining = next_deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            elif remaining < -period:
                next_deadline = time.perf_counter()

        self.transmitting_state = False
        if self.udp_socket:
            self.udp_socket.close()
            self.udp_socket = None

    # -- client connection lifecycle ---------------------------------------
    def _handle_client(self, client_socket):
        try:
            self.reset_state()
            self.client_socket = client_socket
            self.connection_running = True

            header, payload = self._receive_message(client_socket)
            if header.command != Command.kConnect:
                logger.error("Expected Connect, got %s", header.command)
                return
            if not payload or len(payload) < 4:
                logger.error("Invalid Connect payload")
                return

            _version, network_udp_port = struct.unpack("<HH", payload[:4])
            self._send_connect_response(client_socket, header.command_id, ConnectStatus.kSuccess)

            tcp_thread = threading.Thread(target=self._handle_tcp_messages, args=(client_socket,), daemon=True)
            tcp_thread.start()
            self._threads.append(tcp_thread)

            client_address = client_socket.getpeername()[0]
            self._run_state_transmission(client_address, network_udp_port)

            while self.connection_running and self.running:
                time.sleep(0.1)

            tcp_thread.join(timeout=1.0)
        except Exception:
            logger.exception("Error handling client")
        finally:
            client_socket.close()
            self.reset_state()

    def _accept_loop(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.settimeout(1.0)
        try:
            self.server_socket.bind((self.host, self.port))
        except OSError as e:
            if e.errno == errno.EADDRINUSE:
                logger.warning("Port %d in use, retrying bind", self.port)
                time.sleep(1)
                self.server_socket.bind((self.host, self.port))
            else:
                raise
        self.server_socket.listen(1)
        logger.info("FCI server listening on %s:%d", self.host, self.port)

        while self.running:
            try:
                self.reset_state()
                client_socket, address = self.server_socket.accept()
                logger.info("New FCI connection from %s:%d", *address)
                client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._handle_client(client_socket)
            except socket.timeout:
                continue
            except Exception:
                logger.exception("Connection handling error")
                continue

    # -- lifecycle ----------------------------------------------------------
    def start(self, background: bool = True):
        """Start the accept loop. Runs in a daemon thread unless background=False."""
        self.running = True
        if background:
            thread = threading.Thread(target=self._accept_loop, daemon=True)
            thread.start()
            self._threads.append(thread)
            return thread
        self._accept_loop()
        return None

    def stop(self):
        self.running = False
        self.connection_running = False
        self.transmitting_state = False
        for sock in (self.client_socket, self.server_socket, self.udp_socket):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        self.reset_state()
