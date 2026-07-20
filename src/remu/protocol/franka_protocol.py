"""libfranka FCI wire protocol (research_interface, robot server version 9).

Targets libfranka 0.15.x -- verified against the 0.15.3 headers vendored at
``frankabridge/libfranka`` (``kVersion = 9``). Values must match
``research_interface::robot::Command`` exactly.

Note this is *not* the v10 protocol used by libfranka >= 0.18: v9 is
double-based on the wire (see :mod:`remu.protocol.robot_state`) and its
``Command`` enum includes ``kLoadModelLibrary``, which shifts
``kGetRobotModel`` from 11 to 12.
"""

import enum
import struct
from dataclasses import dataclass

COMMAND_PORT = 1337

# research_interface::robot::kVersion for libfranka 0.15.x.
PROTOCOL_VERSION = 9


class Command(enum.IntEnum):
    """Commands supported by the Franka robot interface protocol (v9)."""

    kConnect = 0
    kMove = 1
    kStopMove = 2
    kSetCollisionBehavior = 3
    kSetJointImpedance = 4
    kSetCartesianImpedance = 5
    kSetGuidingMode = 6
    kSetEEToK = 7
    kSetNEToEE = 8
    kSetLoad = 9
    kAutomaticErrorRecovery = 10
    kLoadModelLibrary = 11
    kGetRobotModel = 12


class ConnectStatus(enum.IntEnum):
    kSuccess = 0
    kIncompatibleLibraryVersion = 1


class MoveStatus(enum.IntEnum):
    kSuccess = 0
    kMotionStarted = 1
    kPreempted = 2
    kPreemptedDueToActivatedSafetyFunctions = 3
    kCommandRejectedDueToActivatedSafetyFunctions = 4
    kCommandNotPossibleRejected = 5
    kStartAtSingularPoseRejected = 6
    kInvalidArgumentRejected = 7
    kReflexAborted = 8
    kEmergencyAborted = 9
    kInputErrorAborted = 10
    kAborted = 11


class ControllerMode(enum.IntEnum):
    kJointImpedance = 0
    kCartesianImpedance = 1
    kExternalController = 2


class MotionGeneratorMode(enum.IntEnum):
    kJointPosition = 0
    kJointVelocity = 1
    kCartesianPosition = 2
    kCartesianVelocity = 3
    kNone = 4


class LibfrankaControllerMode(enum.IntEnum):
    kJointImpedance = 0
    kCartesianImpedance = 1
    kExternalController = 2
    kOther = 3


class LibfrankaMotionGeneratorMode(enum.IntEnum):
    kIdle = 0
    kJointPosition = 1
    kJointVelocity = 2
    kCartesianPosition = 3
    kCartesianVelocity = 4
    kNone = 5


def convert_to_libfranka_motion_mode(mode: MotionGeneratorMode) -> LibfrankaMotionGeneratorMode:
    """Convert a Move-command motion mode to the RobotState wire enum."""
    conversion_map = {
        MotionGeneratorMode.kJointPosition: LibfrankaMotionGeneratorMode.kJointPosition,
        MotionGeneratorMode.kJointVelocity: LibfrankaMotionGeneratorMode.kJointVelocity,
        MotionGeneratorMode.kCartesianPosition: LibfrankaMotionGeneratorMode.kCartesianPosition,
        MotionGeneratorMode.kCartesianVelocity: LibfrankaMotionGeneratorMode.kCartesianVelocity,
        MotionGeneratorMode.kNone: LibfrankaMotionGeneratorMode.kNone,
    }
    return conversion_map[mode]


def convert_to_libfranka_controller_mode(mode: ControllerMode) -> LibfrankaControllerMode:
    """Convert a Move-command controller mode to the RobotState wire enum."""
    conversion_map = {
        ControllerMode.kJointImpedance: LibfrankaControllerMode.kJointImpedance,
        ControllerMode.kCartesianImpedance: LibfrankaControllerMode.kCartesianImpedance,
        ControllerMode.kExternalController: LibfrankaControllerMode.kExternalController,
    }
    return conversion_map[mode]


class RobotMode(enum.IntEnum):
    kOther = 0
    kIdle = 1
    kMove = 2
    kGuiding = 3
    kReflex = 4
    kUserStopped = 5
    kAutomaticErrorRecovery = 6


@dataclass
class MessageHeader:
    """Every libfranka message begins with this 12-byte header."""

    command: Command
    command_id: int
    size: int

    @classmethod
    def from_bytes(cls, data: bytes) -> "MessageHeader":
        command, command_id, size = struct.unpack("<III", data)
        return cls(Command(command), command_id, size)

    def to_bytes(self) -> bytes:
        return struct.pack("<III", self.command.value, self.command_id, self.size)


@dataclass
class MoveCommand:
    controller_mode: ControllerMode
    motion_generator_mode: MotionGeneratorMode
    maximum_path_deviation: tuple
    maximum_goal_pose_deviation: tuple

    @classmethod
    def from_bytes(cls, data: bytes) -> "MoveCommand":
        controller_mode, motion_generator_mode = struct.unpack("<II", data[:8])
        try:
            controller_mode = ControllerMode(controller_mode)
            motion_generator_mode = MotionGeneratorMode(motion_generator_mode)
        except ValueError as e:
            raise ValueError(f"Invalid controller mode or motion generator mode: {e}") from e

        path_dev = struct.unpack("<ddd", data[8:32])
        goal_dev = struct.unpack("<ddd", data[32:56])

        return cls(controller_mode, motion_generator_mode, path_dev, goal_dev)


@dataclass
class SetCollisionBehaviorCommand:
    lower_torque_thresholds_acceleration: list
    upper_torque_thresholds_acceleration: list
    lower_torque_thresholds_nominal: list
    upper_torque_thresholds_nominal: list
    lower_force_thresholds_acceleration: list
    upper_force_thresholds_acceleration: list
    lower_force_thresholds_nominal: list
    upper_force_thresholds_nominal: list

    @classmethod
    def from_bytes(cls, data: bytes) -> "SetCollisionBehaviorCommand":
        offset = 0
        lower_torque_acc = list(struct.unpack("<7d", data[offset : offset + 56]))
        offset += 56
        upper_torque_acc = list(struct.unpack("<7d", data[offset : offset + 56]))
        offset += 56
        lower_torque_nom = list(struct.unpack("<7d", data[offset : offset + 56]))
        offset += 56
        upper_torque_nom = list(struct.unpack("<7d", data[offset : offset + 56]))
        offset += 56

        lower_force_acc = list(struct.unpack("<6d", data[offset : offset + 48]))
        offset += 48
        upper_force_acc = list(struct.unpack("<6d", data[offset : offset + 48]))
        offset += 48
        lower_force_nom = list(struct.unpack("<6d", data[offset : offset + 48]))
        offset += 48
        upper_force_nom = list(struct.unpack("<6d", data[offset : offset + 48]))

        return cls(
            lower_torque_acc,
            upper_torque_acc,
            lower_torque_nom,
            upper_torque_nom,
            lower_force_acc,
            upper_force_acc,
            lower_force_nom,
            upper_force_nom,
        )


@dataclass
class SetJointImpedanceCommand:
    K_theta: list

    @classmethod
    def from_bytes(cls, data: bytes) -> "SetJointImpedanceCommand":
        return cls(list(struct.unpack("<7d", data[:56])))


@dataclass
class SetCartesianImpedanceCommand:
    K_x: list

    @classmethod
    def from_bytes(cls, data: bytes) -> "SetCartesianImpedanceCommand":
        return cls(list(struct.unpack("<6d", data[:48])))
