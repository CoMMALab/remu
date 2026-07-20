"""Build the MJCF scene the emulator serves: an FR3 robot plus optional
user-supplied objects, wrapped with a ground plane and lighting via MuJoCo's
native ``<include>`` mechanism.
"""

import os
import tempfile
from pathlib import Path
from typing import Iterable, Optional

_SCENE_TEMPLATE = """<mujoco model="remu_scene">
  <include file="{robot_path}"/>
{object_includes}
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="remu_groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
      rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="remu_groundplane" texture="remu_groundplane" texuniform="true" texrepeat="5 5"
      reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
{ground}
{extra_bodies}
  </worldbody>
</mujoco>
"""

_GROUND_SNIPPET = '    <geom name="remu_floor" size="0 0 0.05" type="plane" material="remu_groundplane"/>'


def default_fr3_mjcf() -> Path:
    """Resolve the FR3 MJCF (MuJoCo Menagerie ``franka_fr3/fr3.xml``).

    Resolution order: the ``$FR3_MJCF`` env override, otherwise
    ``robot_descriptions``, which downloads and caches the model on first use.
    """
    override = os.environ.get("FR3_MJCF")
    if override:
        return Path(override)
    try:
        from robot_descriptions import fr3_mj_description

        return Path(fr3_mj_description.MJCF_PATH)
    except Exception as exc:  # offline / proxy / fetch failure on first use
        raise RuntimeError(
            f"Could not obtain the FR3 model via robot_descriptions ({type(exc).__name__}: "
            f"{exc}). It is downloaded and cached on first use, so the first run needs "
            "network access; set $FR3_MJCF to a local MJCF path to run fully offline."
        ) from exc


def build_scene_xml(
    robot_mjcf: Optional[Path] = None,
    extra_object_mjcfs: Iterable[Path] = (),
    extra_body_xml: Iterable[str] = (),
    add_ground: bool = True,
    cameras: Iterable = (),
) -> Path:
    """Compose a scene MJCF and write it to a temp file; returns the file path.

    Args:
        robot_mjcf: Path to the robot MJCF (defaults to the bundled FR3).
        extra_object_mjcfs: Paths to standalone MJCF files to ``<include>``
            (each should define its own top-level body/bodies).
        extra_body_xml: Raw ``<body>...</body>`` XML snippets to place
            directly in the scene worldbody (e.g. a table, a prop).
        add_ground: Whether to add a checkered ground plane + directional light.
        cameras: :class:`~remu.camera.d435i.EmulatedD435i` instances to place
            in the scene. Each contributes a ``<camera>`` the camera server
            renders from; a camera not included here cannot be bound.
    """
    robot_path = Path(robot_mjcf) if robot_mjcf is not None else default_fr3_mjcf()
    if not robot_path.exists():
        raise FileNotFoundError(f"Robot MJCF not found: {robot_path}")

    object_includes = "\n".join(
        f'  <include file="{Path(p).resolve()}"/>' for p in extra_object_mjcfs
    )
    ground = _GROUND_SNIPPET if add_ground else ""
    extra_bodies = "\n".join(list(extra_body_xml) + [c.mjcf_body() for c in cameras])

    xml = _SCENE_TEMPLATE.format(
        robot_path=robot_path.resolve(),
        object_includes=object_includes,
        ground=ground,
        extra_bodies=extra_bodies,
    )

    # MuJoCo resolves a relative <compiler meshdir="..."/> against the
    # *top-level* file's directory, not the <include>d file's directory. So
    # the generated scene must live next to the robot MJCF for its meshdir to
    # still resolve, unless the robot's meshdir happens to be absolute.
    try:
        fd, path = tempfile.mkstemp(
            prefix="remu_scene_", suffix=".xml", dir=str(robot_path.parent)
        )
    except OSError:
        fd, path = tempfile.mkstemp(prefix="remu_scene_", suffix=".xml")
    with os.fdopen(fd, "w") as f:
        f.write(xml)
    return Path(path)
