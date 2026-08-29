"""Vendor-neutral MuJoCo-backed RGB-D camera implementation."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .wire import DEPTH_SCALE

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics shared by the simulated SDK adapters."""

    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    coeffs: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_fovy(cls, width: int, height: int, fovy_deg: float) -> "Intrinsics":
        fy = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
        return cls(width, height, fy, fy, width / 2.0, height / 2.0)

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
    def from_dict(cls, value: dict) -> "Intrinsics":
        return cls(
            width=value["width"], height=value["height"],
            fx=value["fx"], fy=value["fy"],
            ppx=value["ppx"], ppy=value["ppy"],
            coeffs=tuple(value.get("coeffs", (0.0,) * 5)),
        )


@dataclass(frozen=True)
class StreamProfile:
    width: int
    height: int
    format: str
    fovy_deg: float

    @property
    def intrinsics(self) -> Intrinsics:
        return Intrinsics.from_fovy(self.width, self.height, self.fovy_deg)

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "format": self.format,
            "intrinsics": self.intrinsics.to_dict(),
        }


def look_at_axes(
    eye: Sequence[float], target: Sequence[float], up: Sequence[float] = (0.0, 0.0, 1.0)
) -> np.ndarray:
    """Return MuJoCo camera right/up axes for a world-space look-at pose."""
    eye_array = np.asarray(eye, dtype=float)
    forward = np.asarray(target, dtype=float) - eye_array
    norm = np.linalg.norm(forward)
    if norm < 1e-9:
        raise ValueError("camera eye and target coincide -- no viewing direction")
    forward /= norm
    right = np.cross(forward, np.asarray(up, dtype=float))
    if np.linalg.norm(right) < 1e-9:
        raise ValueError("camera view direction is parallel to `up`; pass a different up vector")
    right /= np.linalg.norm(right)
    camera_up = np.cross(-forward, right)
    return np.stack([right, camera_up])


def optical_pose_from_look_at(
    eye: Sequence[float], target: Sequence[float], up: Sequence[float] = (0.0, 0.0, 1.0)
) -> np.ndarray:
    axes = look_at_axes(eye, target, up)
    right, camera_up = axes
    back = np.cross(right, camera_up)
    mujoco_pose = np.eye(4)
    mujoco_pose[:3, 0], mujoco_pose[:3, 1], mujoco_pose[:3, 2] = right, camera_up, back
    mujoco_pose[:3, 3] = np.asarray(eye, dtype=float)
    return mujoco_pose @ np.diag([1.0, -1.0, -1.0, 1.0])


def _matrix_to_quaternion(rotation: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to a normalized MuJoCo wxyz quaternion."""
    m = rotation
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = (0.25 * s, (m[2, 1] - m[1, 2]) / s,
                (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    else:
        index = int(np.argmax(np.diag(m)))
        if index == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            quat = ((m[2, 1] - m[1, 2]) / s, 0.25 * s,
                    (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
        elif index == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            quat = ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                    0.25 * s, (m[1, 2] + m[2, 1]) / s)
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            quat = ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                    (m[1, 2] + m[2, 1]) / s, 0.25 * s)
    values = np.asarray(quat, dtype=float)
    values /= np.linalg.norm(values)
    return tuple(float(value) for value in values)


class EmulatedRgbdCamera:
    """A configured RGB-D device rendered from two co-located MuJoCo cameras."""

    def __init__(
        self,
        *,
        vendor: str,
        model: str,
        serial: str,
        device_name: str,
        base_from_optical: Sequence[Sequence[float]],
        color: StreamProfile,
        depth: StreamProfile,
        fps: int,
        parent_body: Optional[str] = None,
        min_depth_m: float = 0.105,
        max_depth_m: float = 10.0,
        depth_scale: float = DEPTH_SCALE,
    ):
        self.vendor = vendor
        self.model = model
        self.serial = str(serial)
        self.device_name = device_name
        self.base_from_optical = np.asarray(base_from_optical, dtype=float)
        self.color_profile = color
        self.depth_profile = depth
        self.fps = int(fps)
        self.parent_body = parent_body
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.depth_scale = float(depth_scale)

        safe_serial = re.sub(r"[^A-Za-z0-9_]", "_", self.serial)
        self.mjcf_camera_name = f"remu_cam_{safe_serial}"
        self._color_camera_id = None
        self._depth_camera_id = None
        self._color_renderer = None
        self._depth_renderer = None

    @property
    def width(self) -> int:
        return self.color_profile.width

    @property
    def height(self) -> int:
        return self.color_profile.height

    @property
    def fovy_deg(self) -> float:
        return self.color_profile.fovy_deg

    @property
    def intrinsics(self) -> Intrinsics:
        return self.color_profile.intrinsics

    @property
    def depth_intrinsics(self) -> Intrinsics:
        return self.depth_profile.intrinsics

    @property
    def name(self) -> str:
        return f"{self.vendor}/{self.serial}"

    def optical_pose(self) -> np.ndarray:
        return self.base_from_optical.copy()

    def pose(self) -> np.ndarray:
        return self.base_from_optical @ np.diag([1.0, -1.0, -1.0, 1.0])

    def mjcf_body(self) -> str:
        mujoco_pose = self.pose()
        pos = " ".join(f"{value:.9g}" for value in mujoco_pose[:3, 3])
        quat = " ".join(f"{value:.9g}" for value in _matrix_to_quaternion(mujoco_pose[:3, :3]))
        return (
            f'<body name="{self.mjcf_camera_name}_body" pos="{pos}" quat="{quat}">\n'
            f'  <geom name="{self.mjcf_camera_name}_marker" type="box" '
            'size="0.045 0.0125 0.0125" rgba="0.1 0.1 0.1 1" '
            'contype="0" conaffinity="0"/>\n'
            f'  <camera name="{self.mjcf_camera_name}" fovy="{self.color_profile.fovy_deg}"/>\n'
            f'  <camera name="{self.mjcf_camera_name}_depth" fovy="{self.depth_profile.fovy_deg}"/>\n'
            '</body>'
        )

    def bind(self, model) -> None:
        import mujoco

        self._color_camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, self.mjcf_camera_name
        )
        self._depth_camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, f"{self.mjcf_camera_name}_depth"
        )
        if self._color_camera_id < 0 or self._depth_camera_id < 0:
            raise ValueError(
                f"camera '{self.mjcf_camera_name}' is not in the scene; pass it to build_scene_xml"
            )
        self._color_renderer = mujoco.Renderer(
            model, height=self.color_profile.height, width=self.color_profile.width
        )
        self._depth_renderer = mujoco.Renderer(
            model, height=self.depth_profile.height, width=self.depth_profile.width
        )
        logger.info("camera %s bound to MuJoCo", self.serial)

    def render(self, model, data) -> Tuple[np.ndarray, np.ndarray]:
        if self._color_renderer is None:
            self.bind(model)
        self._color_renderer.disable_depth_rendering()
        self._color_renderer.update_scene(data, camera=self._color_camera_id)
        color = self._color_renderer.render().copy()

        self._depth_renderer.enable_depth_rendering()
        self._depth_renderer.update_scene(data, camera=self._depth_camera_id)
        depth_m = self._depth_renderer.render().copy()
        self._depth_renderer.disable_depth_rendering()
        return color, self.quantize_depth(depth_m)

    def quantize_depth(self, depth_m: np.ndarray) -> np.ndarray:
        valid = (depth_m >= self.min_depth_m) & (depth_m <= self.max_depth_m)
        return np.rint(np.where(valid, depth_m / self.depth_scale, 0.0)).astype(np.uint16)

    def device_dict(self) -> dict:
        return {
            "vendor": self.vendor,
            "model": self.model,
            "serial": self.serial,
            "name": self.device_name,
            "fps": self.fps,
            "depth_scale": self.depth_scale,
            "color": self.color_profile.to_dict(),
            "depth": self.depth_profile.to_dict(),
        }

    def close(self) -> None:
        for renderer_name in ("_color_renderer", "_depth_renderer"):
            renderer = getattr(self, renderer_name)
            if renderer is not None:
                renderer.close()
                setattr(self, renderer_name, None)
