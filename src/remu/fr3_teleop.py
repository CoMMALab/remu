"""Export a Remu HDF5 run as a fr3-teleop staged episode."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image



def _link_eye_tf(group):
    """The camera-to-link transform recorded beside the frames, or ``None``.

    Two compatibilities at once. The dataset was called ``base_from_optical``
    in files written before the rename, and it is written only for cameras that
    had a calibration -- so an unconditional lookup raised KeyError on any
    uncalibrated camera, which is what this used to do.
    """
    for key in ("link_eye_tf", "base_from_optical"):
        if key in group:
            return group[key][:].tolist()
    return None


def _camera_groups(capture: h5py.File):
    cameras = capture.get("cameras")
    if cameras is None:
        return
    for vendor, vendor_group in cameras.items():
        for serial, camera_group in vendor_group.items():
            yield f"{vendor}/{serial}", vendor, serial, camera_group


def export_episode(source: str | Path, destination: str | Path, *, overwrite: bool = False) -> Path:
    """Convert one ``remu-ephemeral`` HDF5 file to fr3-teleop staging layout."""
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"destination already exists: {destination}")
    partial = destination.with_name(f"{destination.name}.partial")
    if partial.exists() and not overwrite:
        raise FileExistsError(f"partial destination already exists: {partial}")
    if overwrite:
        for path in (destination, partial):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
    partial.mkdir(parents=True)

    try:
        with h5py.File(source, "r") as capture:
            if capture.attrs.get("format") != "remu-ephemeral":
                raise ValueError(f"not a remu ephemeral dataset: {source}")
            trajectory = capture["trajectory"]
            time_s = trajectory["time_s"][:]
            qpos = trajectory["qpos"]
            qvel = trajectory["qvel"]
            if not len(time_s) or qpos.shape[1] < 7 or qvel.shape[1] < 7:
                raise ValueError("trajectory must contain timestamps and at least seven arm joints")
            start_s = float(time_s[0])

            state_path = partial / "state.jsonl"
            with state_path.open("w", encoding="utf-8") as state_file:
                for index, stamp_s in enumerate(time_s):
                    q = np.asarray(qpos[index, :7], dtype=float)
                    dq = np.asarray(qvel[index, :7], dtype=float)
                    width = 0.0
                    if qpos.shape[1] >= 9:
                        width = float(np.clip(np.sum(qpos[index, 7:9]), 0.0, 0.08))
                    row = {
                        "stamp_ns": int(round(float(stamp_s) * 1e9)),
                        "seq": index,
                        "q": q.tolist(),
                        "dq": dq.tolist(),
                        "tau_J": [0.0] * 7,
                        # Remu intentionally records physical state only. An
                        # overlapping target ghost is the least misleading
                        # representation available to the existing replay UI.
                        "q_d": q.tolist(),
                        "gripper_width": width,
                        "gripper_target": width,
                        "mode": 0,
                        "fault": 0,
                    }
                    state_file.write(json.dumps(row, separators=(",", ":")) + "\n")

            camera_meta = {}
            for name, vendor, serial, group in _camera_groups(capture):
                directory = partial / name
                directory.mkdir(parents=True)
                stamps = group["time_s"][:]
                scheduled = group["scheduled_time_s"][:]
                colors = group["color"]
                frame_index = directory / "frames.jsonl"
                with frame_index.open("w", encoding="utf-8") as index_file:
                    for index, stamp_s in enumerate(stamps):
                        filename = f"{index:06d}.jpg"
                        path = directory / filename
                        Image.fromarray(colors[index], mode="RGB").save(
                            path, format="JPEG", quality=90, optimize=True
                        )
                        row = {
                            "seq": index,
                            "stamp_ns": int(round(float(stamp_s) * 1e9)),
                            "device_stamp_us": int(round(float(scheduled[index]) * 1e6)),
                            "file": filename,
                            "bytes": path.stat().st_size,
                            "width": int(colors.shape[2]),
                            "height": int(colors.shape[1]),
                            "encoding": 0,
                        }
                        index_file.write(json.dumps(row, separators=(",", ":")) + "\n")

                duration = float(stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
                measured_fps = (len(stamps) - 1) / duration if duration > 0 else 0.0
                device = json.loads(group.attrs.get("device", "{}"))
                camera_meta[name] = {
                    "frames": len(stamps),
                    "dropped": 0,
                    "duration_s": round(duration, 4),
                    "measured_fps": round(measured_fps, 2),
                    "first_stamp_ns": int(round(float(stamps[0]) * 1e9)) if len(stamps) else None,
                    "last_stamp_ns": int(round(float(stamps[-1]) * 1e9)) if len(stamps) else None,
                    "vendor": vendor,
                    "model": device.get("model", "unknown"),
                    "serial": serial,
                    "rotate_deg": 0,
                    "depth_enabled": False,
                    "source_depth_in_hdf5": "depth" in group,
                    "link_eye_tf": _link_eye_tf(group),
                }

            duration = float(time_s[-1] - start_s)
            fps_values = [entry["measured_fps"] for entry in camera_meta.values()]
            meta = {
                "episode_id": destination.name,
                "task": "Remu ephemeral replay",
                "duration_s": round(duration, 4),
                "fps_target": round(fps_values[0]) if fps_values else 30,
                "state_rows": len(time_s),
                "robot": {"backend": "remu", "ip": "offline", "libfranka": "0.15.x", "limits": "FR3"},
                "cameras": camera_meta,
                "source_hdf5": str(source),
            }
            (partial / "episode.json").write_text(
                json.dumps(meta, indent=2) + "\n", encoding="utf-8"
            )
        os.replace(partial, destination)
    except BaseException:
        # Keep the partial directory for diagnosis/recovery.
        raise
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Export Remu HDF5 as a fr3-teleop Rerun staged episode"
    )
    parser.add_argument("source", help="remu-ephemeral HDF5 dataset")
    parser.add_argument("destination", help="staged episode output directory")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    output = export_episode(args.source, args.destination, overwrite=args.overwrite)
    print(output)
    return 0
