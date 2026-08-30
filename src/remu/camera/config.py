"""YAML configuration for simulated RGB-D camera rigs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

from .rgbd import EmulatedRgbdCamera, StreamProfile


@dataclass(frozen=True)
class CameraRigConfig:
    robot_base_body: str | None
    cameras: tuple[EmulatedRgbdCamera, ...]


_MODELS = {
    ("realsense", "d435i"): {
        "name": "Intel RealSense D435I",
        "color_fovy": 42.5,
        "depth_fovy": 58.0,
        "min_depth": 0.105,
        "max_depth": 10.0,
    },
    ("orbbec", "femto_mega"): {
        "name": "Orbbec Femto Mega",
        "color_fovy": 51.0,
        "depth_fovy": 65.0,
        "min_depth": 0.25,
        "max_depth": 5.46,
    },
}


def _mapping(value, where: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a mapping")
    return value


def _positive_int(value, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{where} must be a positive integer")
    return value


def _pose(value, where: str) -> np.ndarray:
    try:
        transform = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where} must contain numeric values") from exc
    if transform.shape != (4, 4):
        raise ValueError(f"{where} must be a 4x4 matrix")
    if not np.all(np.isfinite(transform)):
        raise ValueError(f"{where} must contain only finite values")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{where} must have homogeneous final row [0, 0, 0, 1]")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{where} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{where} rotation must be right-handed")
    return transform


def _stream(value, where: str, expected_format: str, default_fovy: float) -> StreamProfile:
    stream = _mapping(value, where)
    fmt = str(stream.get("format", expected_format)).lower()
    if fmt != expected_format:
        raise ValueError(f"{where}.format must be {expected_format!r}")
    fovy = float(stream.get("fovy_deg", default_fovy))
    if not 0.0 < fovy < 180.0:
        raise ValueError(f"{where}.fovy_deg must be between 0 and 180")
    return StreamProfile(
        width=_positive_int(stream.get("width"), f"{where}.width"),
        height=_positive_int(stream.get("height"), f"{where}.height"),
        format=fmt,
        fovy_deg=fovy,
    )


def parse_camera_config(value: dict) -> CameraRigConfig:
    root = _mapping(value, "camera config")
    if root.get("version") != 1:
        raise ValueError("camera config version must be 1")
    base_body = root.get("robot_base_body")
    if base_body is not None and (not isinstance(base_body, str) or not base_body):
        raise ValueError("robot_base_body must be a non-empty MuJoCo body name when provided")
    entries = root.get("cameras")
    if not isinstance(entries, list) or not entries:
        raise ValueError("cameras must be a non-empty list")

    cameras = []
    identities = set()
    for index, raw in enumerate(entries):
        where = f"cameras[{index}]"
        entry = _mapping(raw, where)
        vendor = str(entry.get("vendor", "")).lower()
        model = str(entry.get("model", "")).lower()
        descriptor = _MODELS.get((vendor, model))
        if descriptor is None:
            supported = ", ".join(f"{v}/{m}" for v, m in _MODELS)
            raise ValueError(f"{where} has unsupported camera {vendor}/{model}; expected {supported}")
        serial = entry.get("serial")
        if not isinstance(serial, (str, int)) or not str(serial):
            raise ValueError(f"{where}.serial must be a non-empty string")
        serial = str(serial)
        identity = serial
        if identity in identities:
            raise ValueError(f"duplicate camera serial {serial}")
        identities.add(identity)
        parent_body = entry.get("parent_body", base_body)
        if not isinstance(parent_body, str) or not parent_body:
            raise ValueError(
                f"{where}.parent_body is required when robot_base_body is not configured"
            )

        pipeline = _mapping(entry.get("pipeline"), f"{where}.pipeline")
        fps = _positive_int(pipeline.get("fps"), f"{where}.pipeline.fps")
        color = _stream(
            pipeline.get("color"), f"{where}.pipeline.color", "rgb8", descriptor["color_fovy"]
        )
        depth_value = _mapping(pipeline.get("depth"), f"{where}.pipeline.depth")
        depth_fovy = descriptor["depth_fovy"]
        if (vendor, model) == ("orbbec", "femto_mega") and (
            depth_value.get("width"), depth_value.get("height")
        ) in ((512, 512), (1024, 1024)):
            depth_fovy = 120.0
        depth = _stream(
            depth_value, f"{where}.pipeline.depth", "z16", depth_fovy
        )
        cameras.append(EmulatedRgbdCamera(
            vendor=vendor,
            model=model,
            serial=serial,
            device_name=descriptor["name"],
            base_from_optical=_pose(entry.get("base_from_optical"), f"{where}.base_from_optical"),
            color=color,
            depth=depth,
            fps=fps,
            parent_body=parent_body,
            min_depth_m=descriptor["min_depth"],
            max_depth_m=descriptor["max_depth"],
        ))
    return CameraRigConfig(base_body, tuple(cameras))


def load_camera_config(path: Path | str) -> CameraRigConfig:
    config_path = Path(path)
    try:
        value = yaml.safe_load(config_path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid camera YAML in {config_path}: {exc}") from exc
    return parse_camera_config(value)


def cameras_to_calibration(cameras: Sequence[EmulatedRgbdCamera]) -> dict:
    return {camera.name: camera.optical_pose().tolist() for camera in cameras}
