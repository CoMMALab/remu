"""Versioned top-level configuration for remu runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LoadedRunConfig:
    """Argument defaults plus an optional inline camera-rig document."""

    defaults: dict[str, Any]
    camera_rig: dict[str, Any] | None
    source: dict[str, Any]


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a mapping")
    return value


def _resolve_path(base: Path, value: Any) -> str | None:
    if value is None:
        return None
    path = Path(value)
    return str(path if path.is_absolute() else (base / path).resolve())


def load_run_config(path: str | Path) -> LoadedRunConfig:
    """Load version-1 unified YAML and flatten it into CLI-compatible defaults."""
    config_path = Path(path).resolve()
    try:
        source = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid run YAML in {config_path}: {exc}") from exc
    root = _mapping(source, "run config")
    if root.get("version") != 1:
        raise ValueError("run config version must be 1")

    robot = _mapping(root.get("robot"), "robot")
    simulation = _mapping(root.get("simulation"), "simulation")
    network = _mapping(root.get("network"), "network")
    viewer = _mapping(root.get("viewer"), "viewer")
    ephemeral = _mapping(root.get("ephemeral"), "ephemeral")
    recording = _mapping(root.get("recording"), "recording")
    base = config_path.parent

    defaults: dict[str, Any] = {}
    scalar_values = {
        "mode": root.get("mode"),
        "robot_mjcf": _resolve_path(base, robot.get("mjcf")),
        "scene_mjcf": _resolve_path(base, robot.get("scene_mjcf")),
        "urdf": _resolve_path(base, robot.get("urdf")),
        "model_library": _resolve_path(base, robot.get("model_library")),
        "joint_names": robot.get("joint_names"),
        "initial_q": robot.get("initial_q"),
        "dt": simulation.get("dt"),
        "host": network.get("host"),
        "port": network.get("fci_port"),
        "gripper_port": network.get("gripper_port"),
        "camera_port": network.get("camera_port"),
        "viewer": viewer.get("backend"),
        "viser_port": viewer.get("viser_port"),
        "camera_calibration_out": _resolve_path(base, root.get("camera_calibration_out")),
        "output": _resolve_path(base, ephemeral.get("output")),
        "render_workers": ephemeral.get("render_workers"),
        "overwrite": ephemeral.get("overwrite"),
        "record": _resolve_path(base, recording.get("path")),
        "record_level": recording.get("level"),
        "record_chunk_mib": recording.get("chunk_mib"),
        "record_rotate_seconds": recording.get("rotate_seconds"),
        "record_rotate_mib": recording.get("rotate_mib"),
    }
    defaults.update({key: value for key, value in scalar_values.items() if value is not None})

    objects = robot.get("objects")
    if objects is not None:
        if not isinstance(objects, list):
            raise ValueError("robot.objects must be a list")
        defaults["objects"] = [_resolve_path(base, value) for value in objects]
    if "gripper" in robot:
        if not isinstance(robot["gripper"], bool):
            raise ValueError("robot.gripper must be a boolean")
        defaults["no_gripper"] = not robot["gripper"]

    camera_rig = root.get("camera_rig")
    if camera_rig is not None:
        camera_rig = dict(_mapping(camera_rig, "camera_rig"))
        camera_rig.setdefault("version", 1)

    return LoadedRunConfig(defaults=defaults, camera_rig=camera_rig, source=root)
