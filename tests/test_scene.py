import mujoco

from remu.sim.scene import build_scene_xml, default_fr3_mjcf


def test_default_fr3_mjcf_resolves_to_existing_file():
    path = default_fr3_mjcf()
    assert path.exists()
    assert path.suffix == ".xml"


def test_build_scene_xml_loads_in_mujoco():
    scene_path = build_scene_xml()
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    assert model.nq == 7
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)
    ]
    assert joint_names == [f"fr3_joint{i}" for i in range(1, 8)]


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
