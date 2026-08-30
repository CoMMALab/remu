import mujoco

import remu.sim.scene as scene_module
from remu.sim.scene import build_scene_xml, default_fr3_mjcf


def test_default_fr3_mjcf_resolves_to_existing_file():
    path = default_fr3_mjcf()
    assert path.exists()
    assert path.suffix == ".xml"


def test_build_scene_xml_loads_in_mujoco():
    scene_path = build_scene_xml()
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    assert model.nq == 9
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)
    ]
    assert joint_names == [f"fr3_joint{i}" for i in range(1, 8)] + [
        "fr3_finger_joint1", "fr3_finger_joint2"
    ]


def test_build_scene_xml_can_disable_gripper_without_resolving_hand(monkeypatch):
    def fail_if_called():
        raise AssertionError("the hand asset should not be resolved")

    monkeypatch.setattr(scene_module, "default_hand_mjcf", fail_if_called)
    scene_path = build_scene_xml(add_gripper=False)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    assert model.nq == 7


def test_scene_with_existing_hand_does_not_resolve_another_hand(monkeypatch):
    scene_path = build_scene_xml()

    def fail_if_called():
        raise AssertionError("an existing hand should be reused")

    monkeypatch.setattr(scene_module, "default_hand_mjcf", fail_if_called)
    composed_path = build_scene_xml(robot_mjcf=scene_path)
    model = mujoco.MjModel.from_xml_path(str(composed_path))
    assert model.nq == 9


def test_build_scene_xml_adds_ground_plane():
    scene_path = build_scene_xml(add_ground=True)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    geom_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)
    ]
    assert "remu_floor" in geom_names


def test_build_scene_xml_without_ground():
    scene_path = build_scene_xml(add_ground=False)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    geom_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)
    ]
    assert "remu_floor" not in geom_names
