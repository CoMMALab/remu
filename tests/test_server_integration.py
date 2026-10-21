import socket
import struct
import threading
import time

import numpy as np
import pytest

from remu.protocol.franka_protocol import (
    Command,
    ConnectStatus,
    LibfrankaMotionGeneratorMode,
    MessageHeader,
    MoveStatus,
    RobotMode,
)
from remu.server.franka_server import FrankaFciServer, _ROBOT_COMMAND_SIZE
from remu.sim.mujoco_sim import MujocoSim


@pytest.fixture
def running_server(scene_path):
    sim = MujocoSim(scene_path, realtime=True)
    sim.build()
    sim_thread = threading.Thread(target=sim.run, daemon=True)
    sim_thread.start()

    server = FrankaFciServer(sim, host="127.0.0.1", port=0)
    # Bind to an ephemeral port for test isolation, then discover it.
    server.port = _free_port()
    server.start(background=True)
    time.sleep(0.2)

    yield server, sim

    server.stop()
    sim.stop()


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Client:
    def __init__(self, tcp: socket.socket, udp: socket.socket):
        self.tcp = tcp
        self.udp = udp

    def sendall(self, data):
        self.tcp.sendall(data)

    def recv(self, size):
        return self.tcp.recv(size)

    def close(self):
        self.tcp.close()
        self.udp.close()


def _connect(server) -> _Client:
    for _ in range(50):
        try:
            sock = socket.create_connection(("127.0.0.1", server.port), timeout=2.0)
            break
        except ConnectionRefusedError:
            time.sleep(0.1)
    else:
        raise RuntimeError("server never came up")

    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.bind(("127.0.0.1", 0))
    udp_port = udp_sock.getsockname()[1]

    payload = struct.pack("<HH", 9, udp_port)
    header = MessageHeader(Command.kConnect, 1, 12 + len(payload))
    sock.sendall(header.to_bytes() + payload)

    resp_header = MessageHeader.from_bytes(sock.recv(12))
    status, version = struct.unpack("<BH", sock.recv(3))
    assert resp_header.command == Command.kConnect
    assert status == ConnectStatus.kSuccess
    assert version == 9

    return _Client(sock, udp_sock)


def test_connect_handshake(running_server):
    server, _sim = running_server
    sock = _connect(server)
    sock.close()


def test_incompatible_protocol_version_is_rejected(running_server):
    server, _sim = running_server
    sock = socket.create_connection(("127.0.0.1", server.port), timeout=2.0)
    payload = struct.pack("<HH", 10, 12345)
    header = MessageHeader(Command.kConnect, 1, 12 + len(payload))
    sock.sendall(header.to_bytes() + payload)

    response_header = MessageHeader.from_bytes(sock.recv(12))
    status, version = struct.unpack("<BH", sock.recv(3))
    assert response_header.command == Command.kConnect
    assert status == ConnectStatus.kIncompatibleLibraryVersion
    assert version == 9
    sock.close()


def test_get_robot_model_returns_urdf(running_server):
    server, _sim = running_server
    sock = _connect(server)

    header = MessageHeader(Command.kGetRobotModel, 2, 12)
    sock.sendall(header.to_bytes())

    resp_header_bytes = sock.recv(12)
    resp_header = MessageHeader.from_bytes(resp_header_bytes)
    payload = b""
    while len(payload) < resp_header.size - 12:
        payload += sock.recv(resp_header.size - 12 - len(payload))

    assert resp_header.command == Command.kGetRobotModel
    assert payload[0] == 0
    assert b"<robot" in payload[1:]
    sock.close()


def test_load_model_library_returns_unpadded_binary(running_server, tmp_path):
    server, _sim = running_server
    expected = b"\x7fELFremu-test-model"
    library_path = tmp_path / "libfcimodels.so"
    library_path.write_bytes(expected)
    server.model_library_path = library_path
    sock = _connect(server)

    request = struct.pack("<BB", 3, 0)  # ARM64, Linux
    header = MessageHeader(Command.kLoadModelLibrary, 3, 12 + len(request))
    sock.sendall(header.to_bytes() + request)
    response_header = MessageHeader.from_bytes(sock.recv(12))
    payload = b""
    while len(payload) < response_header.size - 12:
        payload += sock.recv(response_header.size - 12 - len(payload))

    assert response_header.command == Command.kLoadModelLibrary
    assert response_header.size == 13 + len(expected)
    assert payload == b"\x00" + expected
    sock.close()


def test_udp_state_broadcast_after_connect(running_server):
    server, _sim = running_server
    sock = _connect(server)
    sock.udp.settimeout(2.0)

    data, _addr = sock.udp.recvfrom(4096)
    assert len(data) == 2373
    message_id = struct.unpack("<Q", data[:8])[0]
    assert message_id >= 0
    sock.close()


def test_move_and_position_command_moves_robot(running_server):
    server, sim = running_server
    sock = _connect(server)
    sock.udp.settimeout(2.0)

    controller_mode = 0  # kJointImpedance
    motion_mode = 0  # kJointPosition
    move_payload = struct.pack("<II", controller_mode, motion_mode)
    move_payload += struct.pack("<ddd", 0, 0, 0)
    move_payload += struct.pack("<ddd", 0, 0, 0)
    header = MessageHeader(Command.kMove, 3, 12 + len(move_payload))
    sock.sendall(header.to_bytes() + move_payload)

    resp_header = MessageHeader.from_bytes(sock.recv(12))
    (status,) = struct.unpack("<B3x", sock.recv(4))
    assert resp_header.command == Command.kMove
    assert status == MoveStatus.kMotionStarted

    # Discover the server's UDP command port from an incoming state packet's
    # sender address, then stream RobotCommand packets driving q toward a target.
    data, server_udp_addr = sock.udp.recvfrom(4096)

    target = sim.home_q + np.array([0.2, 0, 0, 0, 0, 0, 0])
    for i in range(1, 3000):
        packet = _build_robot_command(message_id=i, q_c=target)
        sock.udp.sendto(packet, server_udp_addr)
        try:
            sock.udp.recvfrom(4096)
        except socket.timeout:
            pass
        if i % 200 == 0:
            q = sim.get_robot_state()["q"]
            if abs(q[0] - target[0]) < 0.02:
                break
        time.sleep(0.001)

    q = sim.get_robot_state()["q"]
    assert abs(q[0] - target[0]) < 0.05
    sock.close()


def _build_robot_command(message_id, q_c, dq_c=None, tau_J_d=None, motion_finished=False):
    dq_c = dq_c if dq_c is not None else [0.0] * 7
    tau_J_d = tau_J_d if tau_J_d is not None else [0.0] * 7
    packet = struct.pack("<Q", message_id)
    packet += struct.pack("<7d", *q_c)
    packet += struct.pack("<7d", *dq_c)
    packet += struct.pack("<16d", *([1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0]))
    packet += struct.pack("<6d", *([0.0] * 6))
    packet += struct.pack("<2d", *([0.0] * 2))
    packet += struct.pack("<B", 0)  # valid_elbow
    packet += struct.pack("<B", int(motion_finished))  # motion_generation_finished
    packet += struct.pack("<7d", *tau_J_d)
    packet += struct.pack("<B", 0)  # torque_command_finished
    assert len(packet) == _ROBOT_COMMAND_SIZE
    return packet



# The last fields of a RobotState packet: motion_generator_mode, controller_mode (B, B),
# errors, reflex_reason (41B each), robot_mode (B), control_command_success_rate (d).
_MODES_TAIL = struct.calcsize("<BB41B41BBd")


def _start_velocity_motion(sock, command_id=3):
    move_payload = struct.pack("<II", 0, 1)   # kJointImpedance, kJointVelocity
    move_payload += struct.pack("<ddd", 0, 0, 0) + struct.pack("<ddd", 0, 0, 0)
    sock.sendall(MessageHeader(Command.kMove, command_id, 12 + len(move_payload)).to_bytes()
                 + move_payload)


def _move_reply(sock):
    header = MessageHeader.from_bytes(sock.recv(12))
    (status,) = struct.unpack("<B3x", sock.recv(4))
    assert header.command == Command.kMove
    return header.command_id, status


def test_motion_started_follows_a_state_in_the_new_mode(running_server):
    """libfranka takes a state during a motion that isn't in the motion's mode as the motion
    having failed, and then waits, with no timeout, for a reply that never comes. So when
    kMotionStarted goes out, the newest state sent must already be in the new mode."""
    server, _sim = running_server
    packed, at_reply = {}, []
    real_pack, real_reply = server.robot_state.pack_state, server._send_move_response

    def pack_state():
        st = server.robot_state.state
        packed[st["message_id"]] = (st["motion_generator_mode"], st["robot_mode"])
        return real_pack()

    def send_move_response(client_socket, command_id, status):
        if status == MoveStatus.kMotionStarted:
            last_sent = server.robot_state.message_id - 1   # sent, then message_id advanced
            at_reply.append(packed.get(last_sent))
        real_reply(client_socket, command_id, status)

    for trial in range(5):
        sock = _connect(server)          # each connection makes a fresh RobotState
        server.robot_state.pack_state = pack_state
        server._send_move_response = send_move_response
        sock.udp.settimeout(2.0)
        _, server_udp_addr = sock.udp.recvfrom(4096)
        _start_velocity_motion(sock)
        assert _move_reply(sock) == (3, MoveStatus.kMotionStarted)
        assert at_reply[-1] == (LibfrankaMotionGeneratorMode.kJointVelocity, RobotMode.kMove), \
            f"trial {trial}: kMotionStarted went out after a state in mode {at_reply[-1]}"
        sock.udp.sendto(_build_robot_command(1, q_c=[0.0] * 7, motion_finished=True),
                        server_udp_addr)
        sock.close()
        time.sleep(0.1)


def test_move_succeeds_only_when_the_motion_finishes(running_server):
    """kSuccess answers a Move when the motion is over -- not while it is still running."""
    server, sim = running_server
    sock = _connect(server)
    sock.udp.settimeout(2.0)
    _, server_udp_addr = sock.udp.recvfrom(4096)
    _start_velocity_motion(sock)
    assert _move_reply(sock) == (3, MoveStatus.kMotionStarted)

    sock.tcp.settimeout(0.0)
    for i in range(1, 200):                # 0.2 s of zero-velocity commands
        sock.udp.sendto(_build_robot_command(i, q_c=sim.home_q), server_udp_addr)
        time.sleep(0.001)
        try:
            early = sock.tcp.recv(16)
        except BlockingIOError:
            continue
        pytest.fail(f"a Move reply while the motion runs: {early!r}")
    sock.tcp.settimeout(2.0)
    sock.udp.sendto(_build_robot_command(200, q_c=sim.home_q, motion_finished=True),
                    server_udp_addr)
    assert _move_reply(sock) == (3, MoveStatus.kSuccess)
    sock.close()
