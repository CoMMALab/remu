import os


def _configure_mujoco_gl():
    """Use a headless-safe GL backend unless the caller selected one."""
    os.environ.setdefault("MUJOCO_GL", "egl")


_configure_mujoco_gl()

from remu.server.franka_server import FrankaFciServer
from remu.server.gripper_server import FrankaGripperServer
from remu.sim.mujoco_sim import ControlMode, MujocoSim
from remu.sim.scene import build_scene_xml, default_fr3_mjcf, default_hand_mjcf

__all__ = [
    "FrankaFciServer",
    "FrankaGripperServer",
    "MujocoSim",
    "ControlMode",
    "build_scene_xml",
    "default_fr3_mjcf",
    "default_hand_mjcf",
]
