import numpy as np
import pytest

from remu.sim.mujoco_sim import FRANKA_HAND_MAX_WIDTH, MujocoSim
from remu.sim.scene import build_scene_xml


def test_default_hand_starts_open(sim):
    state = sim.get_finger_state()
    assert state["width"] == pytest.approx(FRANKA_HAND_MAX_WIDTH)
    assert state["q"] == pytest.approx([0.04, 0.04])


def test_gripper_target_moves_symmetric_fingers(sim):
    sim.set_gripper_target(0.02, 0.05, 30.0)
    for _ in range(1500):
        sim.step()
    state = sim.get_finger_state()
    assert state["width"] == pytest.approx(0.02, abs=1e-4)
    assert state["q"][0] == pytest.approx(state["q"][1], abs=1e-6)


def test_gripper_speed_is_limited(sim):
    initial = sim.get_finger_state()["width"]
    sim.set_gripper_target(0.0, 0.02, 30.0)
    for _ in range(100):
        sim.step()
    assert initial - sim.get_finger_state()["width"] <= 0.0021


def test_stop_removes_finger_effort(sim):
    sim.set_gripper_target(0.0, 0.05, 30.0)
    sim.step()
    sim.stop_gripper()
    sim.step()
    assert np.array_equal(sim.data.qfrc_applied[sim._finger_dof_adr], np.zeros(2))


def test_bilateral_contacts_identify_a_physical_grasp():
    # Fixed 30 mm box centred between the fingertips in the default home pose.
    box = (
        '<body name="grasp_box" pos="0.5545 0 0.522">'
        '<geom type="box" size="0.015 0.015 0.02"/></body>'
    )
    scene = build_scene_xml(extra_body_xml=[box])
    grasp_sim = MujocoSim(scene, realtime=False)
    try:
        grasp_sim.build()
        grasp_sim.set_gripper_target(0.0, 0.05, 40.0)
        for _ in range(2000):
            grasp_sim.step()
        state = grasp_sim.get_finger_state()
        assert state["width"] == pytest.approx(0.03, abs=1e-3)
        assert state["contact_body_ids"], "both fingers should contact the same box"
    finally:
        grasp_sim.stop()
        scene.unlink(missing_ok=True)


@pytest.mark.parametrize(
    "args", [(-0.01, 0.05, 10), (0.09, 0.05, 10), (0.02, 0, 10), (0.02, 0.05, 71)]
)
def test_rejects_invalid_gripper_targets(sim, args):
    with pytest.raises(ValueError):
        sim.set_gripper_target(*args)
