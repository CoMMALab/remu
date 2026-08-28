import socket
import struct
import threading
import time

import pytest

from remu.protocol.gripper_protocol import (
    GRIPPER_HEADER_SIZE,
    GRIPPER_STATE_SIZE,
    GripperCommand,
    GripperHeader,
    GripperStatus,
)
from remu.server.gripper_server import FrankaGripperServer
from remu.sim.mujoco_sim import MujocoSim
from remu.sim.scene import build_scene_xml


def _message(command, command_id, payload=b""):
    return GripperHeader(command, command_id, GRIPPER_HEADER_SIZE + len(payload)).to_bytes() + payload


def _recv(sock):
    header_data = b""
    while len(header_data) < GRIPPER_HEADER_SIZE:
        header_data += sock.recv(GRIPPER_HEADER_SIZE - len(header_data))
    header = GripperHeader.from_bytes(header_data)
    payload = b""
    while len(payload) < header.size - GRIPPER_HEADER_SIZE:
        payload += sock.recv(header.size - GRIPPER_HEADER_SIZE - len(payload))
    return header, payload


@pytest.fixture
def gripper_stack(scene_path):
    sim = MujocoSim(scene_path, realtime=True)
    sim.build()
    sim_thread = threading.Thread(target=sim.run, daemon=True)
    sim_thread.start()
    server = FrankaGripperServer(sim, host="127.0.0.1", port=0)
    server.start()
    yield sim, server
    server.stop()
    sim.stop()
    sim_thread.join(timeout=2)


def _connect(server):
    tcp = socket.create_connection(("127.0.0.1", server.port), timeout=2)
    tcp.settimeout(5)
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(("127.0.0.1", 0))
    udp.settimeout(2)
    payload = struct.pack("<HH", 3, udp.getsockname()[1])
    tcp.sendall(_message(GripperCommand.kConnect, 1, payload))
    header, response = _recv(tcp)
    assert header.command == GripperCommand.kConnect
    assert struct.unpack("<HH", response) == (0, 3)
    return tcp, udp


def test_connect_and_broadcast_state(gripper_stack):
    _sim, server = gripper_stack
    tcp, udp = _connect(server)
    try:
        state = udp.recv(64)
        assert len(state) == GRIPPER_STATE_SIZE
        _message_id, width, max_width, grasped, temperature = struct.unpack("<Idd?H", state)
        assert width == pytest.approx(0.08)
        assert max_width == pytest.approx(0.08)
        assert grasped is False
        assert temperature == 30
    finally:
        tcp.close()
        udp.close()


def test_homing_and_move_are_blocking_commands(gripper_stack):
    sim, server = gripper_stack
    tcp, udp = _connect(server)
    try:
        tcp.sendall(_message(GripperCommand.kMove, 2, struct.pack("<dd", 0.02, 0.05)))
        header, payload = _recv(tcp)
        assert header.command == GripperCommand.kMove
        assert struct.unpack("<H", payload) == (GripperStatus.kSuccess,)
        assert sim.get_finger_state()["width"] == pytest.approx(0.02, abs=1e-3)

        tcp.sendall(_message(GripperCommand.kHoming, 3))
        header, payload = _recv(tcp)
        assert header.command == GripperCommand.kHoming
        assert struct.unpack("<H", payload) == (GripperStatus.kSuccess,)
        assert sim.get_finger_state()["width"] == pytest.approx(0.08, abs=1e-3)
    finally:
        tcp.close()
        udp.close()


def test_stop_aborts_an_in_progress_move(gripper_stack):
    sim, server = gripper_stack
    tcp, udp = _connect(server)
    try:
        tcp.sendall(_message(GripperCommand.kMove, 10, struct.pack("<dd", 0.0, 0.001)))
        time.sleep(0.05)
        tcp.sendall(_message(GripperCommand.kStop, 11))

        first_header, first_payload = _recv(tcp)
        second_header, second_payload = _recv(tcp)
        assert (first_header.command, struct.unpack("<H", first_payload)[0]) == (
            GripperCommand.kMove, GripperStatus.kAborted
        )
        assert (second_header.command, struct.unpack("<H", second_payload)[0]) == (
            GripperCommand.kStop, GripperStatus.kSuccess
        )
        time.sleep(0.02)
        assert sim._gripper_goal[3] is False
    finally:
        tcp.close()
        udp.close()


def test_grasp_succeeds_only_with_bilateral_physics_contact():
    box = (
        '<body name="grasp_box" pos="0.5545 0 0.522">'
        '<geom type="box" size="0.015 0.015 0.02"/></body>'
    )
    scene = build_scene_xml(extra_body_xml=[box])
    sim = MujocoSim(scene, realtime=True)
    sim.build()
    sim_thread = threading.Thread(target=sim.run, daemon=True)
    sim_thread.start()
    server = FrankaGripperServer(sim, host="127.0.0.1", port=0)
    server.start()
    tcp, udp = _connect(server)
    try:
        payload = struct.pack("<ddddd", 0.03, 0.002, 0.002, 0.05, 40.0)
        tcp.sendall(_message(GripperCommand.kGrasp, 20, payload))
        header, response = _recv(tcp)
        assert header.command == GripperCommand.kGrasp
        assert struct.unpack("<H", response) == (GripperStatus.kSuccess,), (
            sim.get_finger_state()
        )
        assert sim.get_finger_state()["contact_body_ids"]

        grasped = False
        # Drain packets queued while the blocking grasp command was running.
        for _ in range(200):
            state = udp.recv(64)
            grasped = struct.unpack("<Idd?H", state)[3]
            if grasped:
                break
        assert grasped is True
    finally:
        tcp.close()
        udp.close()
        server.stop()
        sim.stop()
        sim_thread.join(timeout=2)
        scene.unlink(missing_ok=True)
