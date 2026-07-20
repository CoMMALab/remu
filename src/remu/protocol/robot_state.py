"""libfranka v9 RobotState wire format (double-based, 2373 bytes, packed).

Must match ``research_interface::robot::RobotState`` (``#pragma pack(push, 1)``)
byte for byte, or libfranka throws ``ProtocolException("libfranka: incorrect
object size")`` from ``network.h`` on the first UDP state packet.

Verified against the libfranka 0.15.3 headers vendored at
``frankabridge/libfranka`` (``kVersion = 9``, ``sizeof(RobotState) == 2373``).
Note v9 differs from the newer v10 format in two ways: every float array is a
``double`` (not ``float``), and v9 has no ``accelerometer_top`` /
``accelerometer_bottom`` fields at all.
"""

import struct
from typing import Any, Dict

from remu.protocol.franka_protocol import RobotMode

_IDENTITY_4X4 = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
]  # fmt: skip

_ROBOT_STATE_FORMAT = (
    "<"
    "Q"  # message_id (uint64)
    + "16d" * 6  # O_T_EE, O_T_EE_d, F_T_EE, EE_T_K, F_T_NE, NE_T_EE
    + "d9d3d"  # m_ee, I_ee, F_x_Cee
    + "d9d3d"  # m_load, I_load, F_x_Cload
    + "2d2d"  # elbow, elbow_d
    + "7d" * 8  # tau_J, tau_J_d, dtau_J, q, q_d, dq, dq_d, ddq_d
    + "7d6d7d6d"  # joint_contact, cartesian_contact, joint_collision, cartesian_collision
    + "7d6d6d6d"  # tau_ext_hat_filtered, O_F_ext_hat_K, K_F_ext_hat_K, O_dP_EE_d
    + "3d2d2d2d"  # O_ddP_O, elbow_c, delbow_c, ddelbow_c
    + "16d6d6d"  # O_T_EE_c, O_dP_EE_c, O_ddP_EE_c
    + "7d7d"  # theta, dtheta
    + "BB"  # motion_generator_mode, controller_mode
    + "41B41B"  # errors, reflex_reason
    + "Bd"  # robot_mode, control_command_success_rate
)
_ROBOT_STATE_PACKER = struct.Struct(_ROBOT_STATE_FORMAT)

# sizeof(research_interface::robot::RobotState) for protocol v9.
ROBOT_STATE_SIZE = 2373
assert _ROBOT_STATE_PACKER.size == ROBOT_STATE_SIZE, (
    f"RobotState layout is {_ROBOT_STATE_PACKER.size} bytes, expected {ROBOT_STATE_SIZE}"
)


class RobotState:
    """Holds robot state as a dict and serializes it to the v10 wire format."""

    def __init__(self):
        self.state = self._initialize_state()
        self._message_id = 0

    def _initialize_state(self) -> Dict[str, Any]:
        return {
            "message_id": 0,
            "q": [0.0] * 7,
            "q_d": [0.0] * 7,
            "dq": [0.0] * 7,
            "dq_d": [0.0] * 7,
            "ddq_d": [0.0] * 7,
            "tau_J": [0.0] * 7,
            "dtau_J": [0.0] * 7,
            "tau_J_d": [0.0] * 7,
            "theta": [0.0] * 7,
            "dtheta": [0.0] * 7,
            "robot_mode": RobotMode.kIdle.value,
            "control_command_success_rate": 1.0,
            "O_T_EE": list(_IDENTITY_4X4),
            "O_T_EE_d": list(_IDENTITY_4X4),
            "F_T_EE": list(_IDENTITY_4X4),
            "EE_T_K": list(_IDENTITY_4X4),
            "F_T_NE": list(_IDENTITY_4X4),
            "NE_T_EE": list(_IDENTITY_4X4),
            "tau_ext_hat_filtered": [0.0] * 7,
            "F_x_Cee": [0.0] * 3,
            "I_ee": [0.0] * 9,
            "m_ee": 0.0,
            "K_F_ext_hat_K": [0.0] * 6,
            "elbow": [0.0] * 2,
            "elbow_d": [0.0] * 2,
            "joint_contact": [0.0] * 7,
            "cartesian_contact": [0.0] * 6,
            "joint_collision": [0.0] * 7,
            "cartesian_collision": [0.0] * 6,
            "errors": [False] * 41,
            "m_load": 0.0,
            "I_load": [0.0] * 9,
            "F_x_Cload": [0.0] * 3,
            "O_F_ext_hat_K": [0.0] * 6,
            "O_dP_EE_d": [0.0] * 6,
            "O_ddP_O": [0.0] * 3,
            "elbow_c": [0.0] * 2,
            "delbow_c": [0.0] * 2,
            "ddelbow_c": [0.0] * 2,
            "O_T_EE_c": list(_IDENTITY_4X4),
            "O_dP_EE_c": [0.0] * 6,
            "O_ddP_EE_c": [0.0] * 6,
            "motion_generator_mode": 0,
            "controller_mode": 0,
            "reflex_reason": [False] * 41,
        }

    def pack_state(self) -> bytes:
        s = self.state
        return _ROBOT_STATE_PACKER.pack(
            s["message_id"],
            *s["O_T_EE"],
            *s["O_T_EE_d"],
            *s["F_T_EE"],
            *s["EE_T_K"],
            *s["F_T_NE"],
            *s["NE_T_EE"],
            s["m_ee"],
            *s["I_ee"],
            *s["F_x_Cee"][:3],
            s["m_load"],
            *s["I_load"],
            *s["F_x_Cload"][:3],
            *s["elbow"],
            *s["elbow_d"],
            *s["tau_J"],
            *s["tau_J_d"],
            *s["dtau_J"],
            *s["q"],
            *s["q_d"],
            *s["dq"],
            *s["dq_d"],
            *s["ddq_d"],
            *s["joint_contact"],
            *s["cartesian_contact"],
            *s["joint_collision"],
            *s["cartesian_collision"],
            *s["tau_ext_hat_filtered"],
            *s["O_F_ext_hat_K"],
            *s["K_F_ext_hat_K"],
            *s["O_dP_EE_d"],
            *s["O_ddP_O"][:3],
            *s["elbow_c"],
            *s["delbow_c"],
            *s["ddelbow_c"],
            *s["O_T_EE_c"],
            *s["O_dP_EE_c"],
            *s["O_ddP_EE_c"],
            *s["theta"],
            *s["dtheta"],
            s["motion_generator_mode"],
            s["controller_mode"],
            *s["errors"],
            *s["reflex_reason"],
            s["robot_mode"],
            s["control_command_success_rate"],
        )

    def update(self):
        """Advance to the next state frame with a monotonic message_id."""
        self._message_id += 1
        self.state["message_id"] = self._message_id

    def set_motion_generator_mode(self, mode: int):
        self.state["motion_generator_mode"] = mode

    def set_controller_mode(self, mode: int):
        self.state["controller_mode"] = mode
