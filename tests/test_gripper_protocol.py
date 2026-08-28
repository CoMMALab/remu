import struct

from remu.protocol.gripper_protocol import (
    GRIPPER_HEADER_SIZE,
    GRIPPER_STATE_SIZE,
    GRIPPER_VERSION,
    GripperCommand,
    GripperConnectStatus,
    GripperHeader,
    GripperStatus,
    connect_response,
    pack_state,
    response,
)


def test_gripper_protocol_constants_and_header():
    assert GRIPPER_VERSION == 3
    assert GRIPPER_HEADER_SIZE == 10
    header = GripperHeader(GripperCommand.kMove, 17, 26)
    assert GripperHeader.from_bytes(header.to_bytes()) == header


def test_gripper_responses_are_packed():
    message = connect_response(2, GripperConnectStatus.kSuccess)
    assert len(message) == 14
    assert struct.unpack("<HH", message[10:]) == (0, 3)

    message = response(GripperCommand.kHoming, 3, GripperStatus.kAborted)
    assert len(message) == 12
    assert struct.unpack("<H", message[10:]) == (GripperStatus.kAborted,)


def test_gripper_state_is_exact_wire_size():
    message = pack_state(42, 0.03, 0.08, True, 30)
    assert len(message) == GRIPPER_STATE_SIZE == 23
    assert struct.unpack("<Idd?H", message) == (42, 0.03, 0.08, True, 30)
