import copy

import numpy as np
import pytest
import mujoco

from remu.camera import parse_camera_config
from remu.sim.scene import build_scene_xml


def _config():
    return {
        "version": 1,
        "robot_base_body": "fr3_link0",
        "cameras": [{
            "vendor": "realsense", "model": "d435i", "serial": "rs1",
            "base_from_optical": np.eye(4).tolist(),
            "pipeline": {
                "fps": 30,
                "color": {"width": 640, "height": 480, "format": "rgb8"},
                "depth": {"width": 640, "height": 480, "format": "z16"},
            },
        }, {
            "vendor": "orbbec", "model": "femto_mega", "serial": "fm1",
            "base_from_optical": np.eye(4).tolist(),
            "pipeline": {
                "fps": 15,
                "color": {"width": 1280, "height": 720, "format": "rgb8"},
                "depth": {"width": 512, "height": 512, "format": "z16"},
            },
        }],
    }


def test_mixed_camera_config_builds_model_specific_devices():
    rig = parse_camera_config(_config())
    assert [camera.vendor for camera in rig.cameras] == ["realsense", "orbbec"]
    assert rig.cameras[0].device_name == "Intel RealSense D435I"
    assert rig.cameras[1].device_name == "Orbbec Femto Mega"
    assert rig.cameras[1].depth_profile.fovy_deg == 120.0
    assert rig.cameras[1].parent_body == "fr3_link0"

@pytest.mark.parametrize("model, name", [
    ("gemini_2", "Orbbec Gemini 2"),
    ("gemini_2_l", "Orbbec Gemini 2 L"),
    ("gemini_330", "Orbbec Gemini 330"),
    ("gemini_330l", "Orbbec Gemini 330L"),
    ("gemini_335", "Orbbec Gemini 335"),
    ("gemini_335l", "Orbbec Gemini 335L"),
    ("gemini_336", "Orbbec Gemini 336"),
    ("gemini_336l", "Orbbec Gemini 336L"),
])
def test_orbbec_gemini_models_are_supported(model, name):
    value = _config()
    value["cameras"][1]["model"] = model

    camera = parse_camera_config(value).cameras[1]

    assert camera.model == model
    assert camera.device_name == name


def test_each_camera_can_override_the_rig_parent_body():
    value = _config()
    value["cameras"][0]["parent_body"] = "fr3_hand"
    value["cameras"][1]["parent_body"] = "fr3_link7"

    rig = parse_camera_config(value)

    assert [camera.parent_body for camera in rig.cameras] == ["fr3_hand", "fr3_link7"]



@pytest.mark.parametrize("mutation, match", [
    (lambda value: value.update(version=2), "version"),
    (lambda value: value["cameras"][0].update(model="unknown"), "unsupported"),
    (lambda value: value["cameras"][0]["pipeline"].update(fps=0), "positive"),
    (lambda value: value["cameras"][0].update(base_from_optical=[[1, 0], [0, 1]]), "4x4"),
])
def test_invalid_camera_config_is_rejected(mutation, match):
    value = _config()
    mutation(value)
    with pytest.raises(ValueError, match=match):
        parse_camera_config(value)


def test_duplicate_serials_are_rejected_across_vendors():
    value = _config()
    value["cameras"][1]["serial"] = value["cameras"][0]["serial"]
    with pytest.raises(ValueError, match="duplicate camera serial"):
        parse_camera_config(value)


def test_pose_must_be_a_rigid_right_handed_transform():
    value = copy.deepcopy(_config())
    value["cameras"][0]["base_from_optical"][0][0] = 2.0
    with pytest.raises(ValueError, match="orthonormal"):
        parse_camera_config(value)


def test_configured_camera_is_attached_to_named_robot_body(tmp_path):
    robot = tmp_path / "robot.xml"
    robot.write_text(
        '<mujoco><worldbody><body name="fr3_link0" pos="0.2 0.3 0.4"/></worldbody></mujoco>'
    )
    camera = parse_camera_config(_config()).cameras[0]
    scene = build_scene_xml(
        robot_mjcf=robot, cameras=[camera], add_ground=False, add_gripper=False
    )
    try:
        model = mujoco.MjModel.from_xml_path(str(scene))
        camera_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{camera.mjcf_camera_name}_body"
        )
        base_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fr3_link0")
        assert model.body_parentid[camera_body] == base_body
    finally:
        scene.unlink(missing_ok=True)


def test_missing_configured_base_body_is_reported(tmp_path):
    robot = tmp_path / "robot.xml"
    robot.write_text('<mujoco><worldbody/></mujoco>')
    camera = parse_camera_config(_config()).cameras[0]
    with pytest.raises(ValueError, match="missing robot base body"):
        build_scene_xml(
            robot_mjcf=robot, cameras=[camera], add_ground=False, add_gripper=False
        )
