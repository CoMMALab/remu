import pytest

from remu.sim.mujoco_sim import MujocoSim
from remu.sim.scene import build_scene_xml


@pytest.fixture(scope="session")
def scene_path():
    return build_scene_xml()


@pytest.fixture
def sim(scene_path):
    s = MujocoSim(scene_path, realtime=False)
    s.build()
    yield s
    s.stop()
