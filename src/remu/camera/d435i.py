"""Emulated Intel RealSense D435i, rendered from a MuJoCo scene camera.

One :class:`EmulatedD435i` owns a ``<camera>`` in the scene MJCF and an
offscreen ``mujoco.Renderer``, and turns each render into the pair of buffers
a real D435i delivers over USB: an RGB8 color image and a Z16 depth image
already aligned to it.

Alignment is free here. On real hardware depth and color come from different
physical sensors and ``rs.align`` reprojects one into the other; in the sim
both images are rendered from the same MuJoCo camera at the same resolution,
so they are aligned by construction and the shim's ``align.process`` is an
identity pass.

Frame conventions
-----------------
MuJoCo cameras look down **-z** with +x right and +y up. RealSense optical
frames look down **+z** with +x right and +y **down**. The two differ by a
180-degree rotation about x, which :meth:`EmulatedD435i.optical_pose` applies
-- so the transform it returns is directly comparable to what the AprilTag
calibration in pointcloud_perception solves for.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# D435i color stream defaults. The vertical FOV is the spec'd 42.5 deg for the
# 640x480 RGB mode; fx/fy/ppx/ppy are *derived* from it in Intrinsics.from_fovy
# rather than hardcoded, so the numbers handed to clients can never drift out
# of sync with the MuJoCo camera actually being rendered.
D435I_WIDTH = 640
D435I_HEIGHT = 480
D435I_FPS = 30
D435I_COLOR_FOVY_DEG = 42.5

# Usable depth range of a D435i, in metres. Outside it real hardware returns 0
# (= "no reading"), which is what the emulator writes too, so the perception
# code's zero-vertex rejection exercises the same path it does on hardware.
D435I_MIN_DEPTH_M = 0.105
D435I_MAX_DEPTH_M = 10.0


@dataclass
class Intrinsics:
    """Pinhole intrinsics, matching ``rs.intrinsics`` field-for-field."""

    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    # The emulated lens is a perfect pinhole, so distortion is identically
    # zero. Real D435i color streams report the "inverse Brown-Conrady" model
    # with near-zero coefficients, so this is a fair stand-in.
    coeffs: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_fovy(cls, width: int, height: int, fovy_deg: float) -> "Intrinsics":
        """Derive intrinsics from a MuJoCo camera's vertical field of view."""
        fy = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
        # Square pixels: MuJoCo's projection is symmetric, so fx == fy and the
        # horizontal FOV falls out of the aspect ratio.
        return cls(width=width, height=height, fx=fy, fy=fy, ppx=width / 2.0, ppy=height / 2.0)

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "ppx": self.ppx,
            "ppy": self.ppy,
            "coeffs": list(self.coeffs),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Intrinsics":
        return cls(
            width=d["width"], height=d["height"],
            fx=d["fx"], fy=d["fy"], ppx=d["ppx"], ppy=d["ppy"],
            coeffs=tuple(d.get("coeffs", (0.0,) * 5)),
        )


def look_at_axes(eye: Sequence[float], target: Sequence[float],
                 up: Sequence[float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    """MuJoCo camera ``xyaxes`` (right, up) for a camera at ``eye`` aimed at ``target``."""
    eye = np.asarray(eye, dtype=float)
    forward = np.asarray(target, dtype=float) - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-9:
        raise ValueError("camera eye and target coincide -- no viewing direction")
    forward /= norm

    right = np.cross(forward, np.asarray(up, dtype=float))
    if np.linalg.norm(right) < 1e-9:
        raise ValueError(
            "camera view direction is parallel to `up`; pass a different up vector"
        )
    right /= np.linalg.norm(right)
    # MuJoCo's camera z axis points backwards (-forward), so up = z x right.
    cam_up = np.cross(-forward, right)
    return np.stack([right, cam_up])


@dataclass
class EmulatedD435i:
    """A simulated D435i bound to a named MuJoCo camera.

    Args:
        serial: Device serial reported to clients. Any string works; using a
            plausible 12-digit serial keeps ``stream_output.yaml``-style
            pinning meaningful.
        eye: Camera position in world coordinates.
        target: Point the camera is aimed at.
        up: World up vector used to resolve roll about the view axis.
    """

    serial: str
    eye: Sequence[float]
    target: Sequence[float] = (0.0, 0.0, 0.3)
    up: Sequence[float] = (0.0, 0.0, 1.0)
    width: int = D435I_WIDTH
    height: int = D435I_HEIGHT
    fps: int = D435I_FPS
    fovy_deg: float = D435I_COLOR_FOVY_DEG
    min_depth_m: float = D435I_MIN_DEPTH_M
    max_depth_m: float = D435I_MAX_DEPTH_M
    mjcf_camera_name: str = field(init=False)

    def __post_init__(self):
        self.mjcf_camera_name = f"remu_cam_{self.serial}"
        self.eye = np.asarray(self.eye, dtype=float)
        self.target = np.asarray(self.target, dtype=float)
        self._renderer = None
        self._camera_id = None

    @property
    def name(self) -> str:
        """The key pointcloud_perception uses for this camera in calibration.json."""
        return f"realsense/{self.serial}"

    @property
    def intrinsics(self) -> Intrinsics:
        return Intrinsics.from_fovy(self.width, self.height, self.fovy_deg)

    def mjcf_body(self) -> str:
        """A ``<camera>`` element to splice into the scene worldbody.

        Rendered as a small visual marker plus the camera itself, so the
        camera is visible in the viewer instead of being an invisible frame.
        """
        axes = look_at_axes(self.eye, self.target, self.up)
        xyaxes = " ".join(f"{v:.6f}" for v in axes.reshape(-1))
        pos = " ".join(f"{v:.6f}" for v in self.eye)
        return (
            # A body with no joint is static, which is what a tripod-mounted
            # camera is; the marker geom is non-colliding so it can't perturb
            # the arm if the camera is placed inside its workspace.
            f'    <body name="{self.mjcf_camera_name}_body" pos="{pos}">\n'
            f'      <geom name="{self.mjcf_camera_name}_marker" type="box"'
            f' size="0.045 0.0125 0.0125" rgba="0.1 0.1 0.1 1" contype="0" conaffinity="0"/>\n'
            f'      <camera name="{self.mjcf_camera_name}" pos="0 0 0" xyaxes="{xyaxes}"'
            f' fovy="{self.fovy_deg}"/>\n'
            f"    </body>"
        )

    def pose(self) -> np.ndarray:
        """4x4 camera->world transform in MuJoCo's camera convention (-z forward)."""
        axes = look_at_axes(self.eye, self.target, self.up)
        right, cam_up = axes[0], axes[1]
        back = np.cross(right, cam_up)  # +z, pointing behind the camera
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2] = right, cam_up, back
        T[:3, 3] = self.eye
        return T

    def optical_pose(self) -> np.ndarray:
        """4x4 optical-frame->world transform, RealSense convention (+z forward, +y down).

        This is the ground-truth extrinsic for this camera: the same quantity
        pointcloud_perception's AprilTag calibration estimates and stores in
        calibration.json under :attr:`name`.
        """
        flip = np.diag([1.0, -1.0, -1.0, 1.0])  # 180 deg about x
        return self.pose() @ flip

    # -- rendering ---------------------------------------------------------

    def bind(self, model) -> None:
        """Resolve the MJCF camera id and create the offscreen renderer.

        Must be called from the thread that will call :meth:`render`: the
        renderer owns an OpenGL context, and contexts are not shareable
        across threads.
        """
        import mujoco

        self._camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, self.mjcf_camera_name
        )
        if self._camera_id < 0:
            raise ValueError(
                f"camera '{self.mjcf_camera_name}' is not in the scene -- pass this "
                "EmulatedD435i to build_scene_xml(cameras=...) so its MJCF is included"
            )
        self._renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        logger.info("camera %s bound to MJCF camera '%s'", self.serial, self.mjcf_camera_name)

    def render(self, model, data) -> Tuple[np.ndarray, np.ndarray]:
        """Render one aligned (color RGB8, depth Z16-in-millimetres) pair."""
        if self._renderer is None:
            self.bind(model)

        self._renderer.disable_depth_rendering()
        self._renderer.update_scene(data, camera=self._camera_id)
        color = self._renderer.render().copy()

        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(data, camera=self._camera_id)
        depth_m = self._renderer.render()
        self._renderer.disable_depth_rendering()

        return color, self.quantize_depth(depth_m)

    def quantize_depth(self, depth_m: np.ndarray) -> np.ndarray:
        """Convert metric depth to Z16 millimetres, zeroing out-of-range pixels.

        Zero is RealSense's "no reading" sentinel, so anything nearer than the
        minimum range or beyond the maximum -- including the MuJoCo far-plane
        background, which comes back as a large finite depth rather than a
        miss -- has to become 0 rather than a bogus measurement.
        """
        valid = (depth_m >= self.min_depth_m) & (depth_m <= self.max_depth_m)
        return np.where(valid, depth_m * 1000.0, 0.0).astype(np.uint16)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def d435i_in_front_of_robot(
    serial: str = "934222071887",
    distance_m: float = 1.0,
    height_m: float = 0.6,
    look_at_height_m: float = 0.35,
) -> EmulatedD435i:
    """A D435i standing in front of the arm, tilted down at its workspace.

    ``distance_m`` is measured along +x from the robot base, which is at the
    world origin in the remu scene; the camera looks back along -x.
    """
    return EmulatedD435i(
        serial=serial,
        eye=(distance_m, 0.0, height_m),
        target=(0.0, 0.0, look_at_height_m),
    )


def optical_pose_to_calibration(cameras) -> dict:
    """Ground-truth ``calibration.json`` contents for ``cameras``.

    pointcloud_perception recovers these extrinsics from an AprilTag; in the
    emulator they are known exactly, so this skips calibration entirely.
    Keys and the row-major 4x4 layout match what ``load_calibration`` reads.
    """
    return {cam.name: cam.optical_pose().tolist() for cam in cameras}
