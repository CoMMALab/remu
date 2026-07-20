import struct

from remu.protocol.franka_protocol import (
    Command,
    ControllerMode,
    MessageHeader,
    MotionGeneratorMode,
    MoveCommand,
    SetCartesianImpedanceCommand,
    SetCollisionBehaviorCommand,
    SetJointImpedanceCommand,
    convert_to_libfranka_controller_mode,
    convert_to_libfranka_motion_mode,
)


def test_message_header_roundtrip():
    header = MessageHeader(Command.kMove, command_id=42, size=100)
    data = header.to_bytes()
    assert len(data) == 12
    parsed = MessageHeader.from_bytes(data)
    assert parsed == header


def test_move_command_from_bytes():
    payload = struct.pack("<II", ControllerMode.kJointImpedance, MotionGeneratorMode.kJointPosition)
    payload += struct.pack("<ddd", 1.0, 2.0, 3.0)
    payload += struct.pack("<ddd", 4.0, 5.0, 6.0)
    cmd = MoveCommand.from_bytes(payload)
    assert cmd.controller_mode == ControllerMode.kJointImpedance
    assert cmd.motion_generator_mode == MotionGeneratorMode.kJointPosition
    assert cmd.maximum_path_deviation == (1.0, 2.0, 3.0)
    assert cmd.maximum_goal_pose_deviation == (4.0, 5.0, 6.0)


def test_move_command_invalid_mode_raises():
    payload = struct.pack("<II", 99, 99) + b"\x00" * 48
    try:
        MoveCommand.from_bytes(payload)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_set_collision_behavior_roundtrip():
    values = list(range(7)) * 4 + list(range(6)) * 4
    payload = struct.pack("<" + "7d" * 4 + "6d" * 4, *values)
    cmd = SetCollisionBehaviorCommand.from_bytes(payload)
    assert cmd.lower_torque_thresholds_acceleration == [float(i) for i in range(7)]
    assert cmd.upper_force_thresholds_nominal == [float(i) for i in range(6)]


def test_set_joint_impedance_roundtrip():
    payload = struct.pack("<7d", *range(7))
    cmd = SetJointImpedanceCommand.from_bytes(payload)
    assert cmd.K_theta == [float(i) for i in range(7)]


def test_set_cartesian_impedance_roundtrip():
    payload = struct.pack("<6d", *range(6))
    cmd = SetCartesianImpedanceCommand.from_bytes(payload)
    assert cmd.K_x == [float(i) for i in range(6)]


def test_mode_conversions():
    assert convert_to_libfranka_controller_mode(ControllerMode.kExternalController).name == "kExternalController"
    assert convert_to_libfranka_motion_mode(MotionGeneratorMode.kJointVelocity).name == "kJointVelocity"
