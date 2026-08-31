"""Build remu's MuJoCo scene: an FR3, Franka Hand, and optional objects."""

import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Optional

import mujoco


_GROUND_SNIPPET = (
    '<geom name="remu_floor" size="0 0 0.05" type="plane" '
    'material="remu_groundplane"/>'
)

DEFAULT_FINGER_JOINT_NAMES = ("fr3_finger_joint1", "fr3_finger_joint2")
FLANGE_TO_HAND_QUAT = (0.3826834, 0.0, 0.0, 0.9238795)
FLANGE_OFFSET_Z = 0.107


def default_fr3_mjcf() -> Path:
    """Resolve the MuJoCo Menagerie FR3 model."""
    override = os.environ.get("FR3_MJCF")
    if override:
        return Path(override)
    try:
        from robot_descriptions import fr3_mj_description

        return Path(fr3_mj_description.MJCF_PATH)
    except Exception as exc:
        raise RuntimeError(
            f"Could not obtain the FR3 model via robot_descriptions ({type(exc).__name__}: "
            f"{exc}). Set $FR3_MJCF to a local MJCF path for offline use."
        ) from exc


def default_hand_mjcf() -> Path:
    """Resolve the standalone MuJoCo Menagerie Franka Hand model."""
    override = os.environ.get("FRANKA_HAND_MJCF")
    if override:
        return Path(override)
    try:
        from robot_descriptions import panda_mj_description

        return Path(panda_mj_description.PACKAGE_PATH) / "hand.xml"
    except Exception as exc:
        raise RuntimeError(
            f"Could not obtain the Franka Hand model via robot_descriptions "
            f"({type(exc).__name__}: {exc}). Set $FRANKA_HAND_MJCF for offline use."
        ) from exc


def _asset_paths(spec: mujoco.MjSpec, source: Path, kind: str) -> list[Path]:
    """Resolve file-backed assets before attachment discards source dirs."""
    model_dir = Path(spec.modelfiledir or source.parent)
    asset_dir_value = getattr(spec, f"{kind}dir")
    asset_dir = Path(asset_dir_value) if asset_dir_value else Path()
    paths = []
    collection = "meshes" if kind == "mesh" else "textures"
    for item in getattr(spec, collection):
        if not item.file:
            continue
        path = Path(item.file)
        if not path.is_absolute():
            path = model_dir / asset_dir / path
        paths.append(path.resolve())
    return paths


def _scene_root(
    robot_path: Path, add_gripper: bool, hand_path: Optional[Path]
) -> ET.Element:
    """Load the robot, optionally graft the hand, and return portable XML."""
    spec = mujoco.MjSpec.from_file(str(robot_path))
    mesh_paths = _asset_paths(spec, robot_path, "mesh")
    texture_paths = _asset_paths(spec, robot_path, "texture")

    joint_names = {joint.name for joint in spec.joints}
    has_hand = all(name in joint_names for name in DEFAULT_FINGER_JOINT_NAMES)
    hand_spec = None
    if add_gripper and not has_hand:
        if hand_path is None:
            hand_path = default_hand_mjcf()
        if not hand_path.exists():
            raise FileNotFoundError(f"Franka Hand MJCF not found: {hand_path}")
        hand_spec = mujoco.MjSpec.from_file(str(hand_path))
        hand_meshes = _asset_paths(hand_spec, hand_path, "mesh")
        hand_textures = _asset_paths(hand_spec, hand_path, "texture")

        flange = None
        offset = (0.0, 0.0, 0.0)
        for name in ("fr3_link8", "fr3_link7"):
            try:
                flange = spec.body(name)
            except KeyError:
                flange = None
            if flange is not None:
                if name.endswith("link7"):
                    offset = (0.0, 0.0, FLANGE_OFFSET_Z)
                break
        if flange is None:
            raise ValueError(
                "Gripper enabled, but the MJCF has neither 'fr3_link8' nor "
                "'fr3_link7'; use a conventionally named FR3 scene or --no-gripper"
            )

        frame = flange.add_frame(pos=offset, quat=FLANGE_TO_HAND_QUAT)
        frame.attach_body(hand_spec.worldbody.first_body(), "fr3_", "")
        mesh_paths.extend(hand_meshes)
        texture_paths.extend(hand_textures)

    root = ET.fromstring(spec.to_xml())
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
        compiler.attrib.pop("texturedir", None)

    meshes = root.findall("./asset/mesh")
    if len(meshes) != len(mesh_paths):
        raise RuntimeError(
            f"MJCF asset count changed while composing hand: {len(meshes)} != {len(mesh_paths)}"
        )
    for mesh, path in zip(meshes, mesh_paths):
        mesh.set("file", str(path))

    textures = [node for node in root.findall("./asset/texture") if node.get("file")]
    if len(textures) != len(texture_paths):
        raise RuntimeError(
            "MJCF texture count changed while composing hand: "
            f"{len(textures)} != {len(texture_paths)}"
        )
    for texture, path in zip(textures, texture_paths):
        texture.set("file", str(path))

    del hand_spec
    return root


def _upsert(parent: ET.Element, tag: str, **attrs) -> ET.Element:
    element = parent.find(tag)
    if element is None:
        element = ET.SubElement(parent, tag)
    element.attrib.update({key: str(value) for key, value in attrs.items()})
    return element


def build_scene_xml(
    robot_mjcf: Optional[Path] = None,
    extra_object_mjcfs: Iterable[Path] = (),
    extra_body_xml: Iterable[str] = (),
    add_ground: bool = True,
    cameras: Iterable = (),
    add_gripper: bool = True,
    hand_mjcf: Optional[Path] = None,
) -> Path:
    """Compose a scene and return a temporary, self-contained MJCF path.

    The Franka Hand is enabled by default. A custom model may either already
    contain ``fr3_finger_joint1/2`` or expose ``fr3_link8``/``fr3_link7`` for
    attachment. Pass ``add_gripper=False`` for an arm-only custom model.
    """
    robot_path = Path(robot_mjcf) if robot_mjcf is not None else default_fr3_mjcf()
    if not robot_path.exists():
        raise FileNotFoundError(f"Robot MJCF not found: {robot_path}")
    hand_path = Path(hand_mjcf) if hand_mjcf is not None else None
    cameras = tuple(cameras)

    root = _scene_root(robot_path, add_gripper, hand_path)

    for object_path in extra_object_mjcfs:
        ET.SubElement(root, "include", file=str(Path(object_path).resolve()))

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    _upsert(
        visual, "headlight", diffuse="0.6 0.6 0.6", ambient="0.3 0.3 0.3",
        specular="0 0 0",
    )
    _upsert(visual, "rgba", haze="0.15 0.25 0.35 1")
    global_visual = _upsert(visual, "global", azimuth="120", elevation="-20")
    if cameras:
        profiles = [
            profile
            for camera in cameras
            for profile in (camera.color_profile, camera.depth_profile)
        ]
        offwidth = max(
            int(global_visual.get("offwidth", "0")), *(p.width for p in profiles)
        )
        offheight = max(
            int(global_visual.get("offheight", "0")), *(p.height for p in profiles)
        )
        global_visual.set("offwidth", str(offwidth))
        global_visual.set("offheight", str(offheight))

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    if asset.find("./texture[@type='skybox']") is None:
        ET.SubElement(
            asset, "texture", type="skybox", builtin="gradient", rgb1="0.3 0.5 0.7",
            rgb2="0 0 0", width="512", height="3072",
        )
    if asset.find("./texture[@name='remu_groundplane']") is None:
        ET.SubElement(
            asset, "texture", type="2d", name="remu_groundplane", builtin="checker",
            mark="edge", rgb1="0.2 0.3 0.4", rgb2="0.1 0.2 0.3",
            markrgb="0.8 0.8 0.8", width="300", height="300",
        )
    if asset.find("./material[@name='remu_groundplane']") is None:
        ET.SubElement(
            asset, "material", name="remu_groundplane", texture="remu_groundplane",
            texuniform="true", texrepeat="5 5", reflectance="0.2",
        )

    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "light", pos="0 0 1.5", dir="0 0 -1", directional="true")
    if add_ground and worldbody.find(".//geom[@name='remu_floor']") is None:
        worldbody.append(ET.fromstring(_GROUND_SNIPPET))
    for snippet in extra_body_xml:
        worldbody.append(ET.fromstring(snippet))
    for camera in cameras:
        parent = worldbody
        parent_body = getattr(camera, "parent_body", None)
        if parent_body:
            parent = worldbody.find(f".//body[@name='{parent_body}']")
            if parent is None:
                raise ValueError(
                    f"camera {camera.serial!r} references missing robot base body {parent_body!r}"
                )
        parent.append(ET.fromstring(camera.mjcf_body()))

    fd, path = tempfile.mkstemp(prefix="remu_scene_", suffix=".xml")
    with os.fdopen(fd, "wb") as file:
        ET.ElementTree(root).write(file, encoding="utf-8", xml_declaration=True)
    return Path(path)
