"""Packed libfranka Gripper Server v3 wire types."""

import enum
import struct
from dataclasses import dataclass


GRIPPER_COMMAND_PORT = 1338
GRIPPER_VERSION = 3
_HEADER = struct.Struct("<HII")
GRIPPER_HEADER_SIZE = _HEADER.size
_STATE = struct.Struct("<Idd?H")
GRIPPER_STATE_SIZE = _STATE.size


class GripperCommand(enum.IntEnum):
    kConnect = 0
    kHoming = 1
    kGrasp = 2
    kMove = 3
    kStop = 4


class GripperConnectStatus(enum.IntEnum):
    kSuccess = 0
    kIncompatibleLibraryVersion = 1


class GripperStatus(enum.IntEnum):
    kSuccess = 0
    kFail = 1
    kUnsuccessful = 2
    kAborted = 3


@dataclass(frozen=True)
class GripperHeader:
    command: int
    command_id: int
    size: int

    @classmethod
    def from_bytes(cls, data: bytes) -> "GripperHeader":
        return cls(*_HEADER.unpack(data[:GRIPPER_HEADER_SIZE]))

    def to_bytes(self) -> bytes:
        return _HEADER.pack(int(self.command), self.command_id, self.size)


def response(command: int, command_id: int, status: enum.IntEnum) -> bytes:
    payload = struct.pack("<H", int(status))
    return GripperHeader(command, command_id, GRIPPER_HEADER_SIZE + len(payload)).to_bytes() + payload


def connect_response(
    command_id: int, status: GripperConnectStatus, version: int = GRIPPER_VERSION
) -> bytes:
    payload = struct.pack("<HH", int(status), version)
    return GripperHeader(
        GripperCommand.kConnect, command_id, GRIPPER_HEADER_SIZE + len(payload)
    ).to_bytes() + payload


def pack_state(
    message_id: int, width: float, max_width: float, is_grasped: bool, temperature: int
) -> bytes:
    return _STATE.pack(message_id, width, max_width, is_grasped, temperature)
