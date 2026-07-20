import socket
import struct
import threading
import time

import numpy as np
import pytest

from remu.protocol.franka_protocol import Command, ConnectStatus, MessageHeader, MoveStatus
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

    payload = struct.pack("<HH", 10, udp_port)
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


def _build_robot_command(message_id, q_c, dq_c=None, tau_J_d=None):
    dq_c = dq_c if dq_c is not None else [0.0] * 7
    tau_J_d = tau_J_d if tau_J_d is not None else [0.0] * 7
    packet = struct.pack("<Q", message_id)
    packet += struct.pack("<7d", *q_c)
    packet += struct.pack("<7d", *dq_c)
    packet += struct.pack("<16d", *([1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0]))
    packet += struct.pack("<6d", *([0.0] * 6))
    packet += struct.pack("<2d", *([0.0] * 2))
    packet += struct.pack("<B", 0)  # valid_elbow
    packet += struct.pack("<B", 0)  # motion_generation_finished
    packet += struct.pack("<7d", *tau_J_d)
    packet += struct.pack("<B", 0)  # torque_command_finished
    assert len(packet) == _ROBOT_COMMAND_SIZE
    return packet
