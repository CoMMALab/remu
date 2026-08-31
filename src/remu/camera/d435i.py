"""Intel RealSense D435i compatibility wrapper."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .rgbd import (
    EmulatedRgbdCamera,
    Intrinsics,
    StreamProfile,
    look_at_axes,
    optical_pose_from_look_at,
)

D435I_WIDTH = 640
D435I_HEIGHT = 480
D435I_FPS = 30
D435I_COLOR_FOVY_DEG = 42.5
D435I_DEPTH_FOVY_DEG = 58.0
D435I_MIN_DEPTH_M = 0.105
D435I_MAX_DEPTH_M = 10.0


class EmulatedD435i(EmulatedRgbdCamera):
    """Backwards-compatible D435i constructor using an eye/target pose."""

    def __init__(
        self,
        serial: str,
        eye: Sequence[float],
        target: Sequence[float] = (0.0, 0.0, 0.3),
        up: Sequence[float] = (0.0, 0.0, 1.0),
        width: int = D435I_WIDTH,
        height: int = D435I_HEIGHT,
        fps: int = D435I_FPS,
        fovy_deg: float = D435I_COLOR_FOVY_DEG,
        min_depth_m: float = D435I_MIN_DEPTH_M,
        max_depth_m: float = D435I_MAX_DEPTH_M,
    ):
        self.eye = np.asarray(eye, dtype=float)
        self.target = np.asarray(target, dtype=float)
        self.up = np.asarray(up, dtype=float)
        super().__init__(
            vendor="realsense",
            model="d435i",
            serial=serial,
            device_name="Intel RealSense D435I",
            link_eye_tf=optical_pose_from_look_at(self.eye, self.target, self.up),
            color=StreamProfile(width, height, "rgb8", fovy_deg),
            depth=StreamProfile(width, height, "z16", fovy_deg),
            fps=fps,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
        )


def d435i_in_front_of_robot(
    serial: str = "934222071887",
    distance_m: float = 1.0,
    height_m: float = 0.6,
    look_at_height_m: float = 0.35,
) -> EmulatedD435i:
    return EmulatedD435i(
        serial=serial,
        eye=(distance_m, 0.0, height_m),
        target=(0.0, 0.0, look_at_height_m),
    )


def optical_pose_to_calibration(cameras) -> dict:
    return {camera.name: camera.optical_pose().tolist() for camera in cameras}


__all__ = [
    "D435I_WIDTH", "D435I_HEIGHT", "D435I_FPS", "D435I_COLOR_FOVY_DEG",
    "D435I_MIN_DEPTH_M", "D435I_MAX_DEPTH_M", "EmulatedD435i", "Intrinsics",
    "d435i_in_front_of_robot", "look_at_axes", "optical_pose_to_calibration",
]
