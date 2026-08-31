"""Export canonical Remu MCAP recordings to dense HDF5 arrays."""

from __future__ import annotations

import argparse
import io
import json
import os
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from remu.recording import TOPIC_SIM_STATE, iter_messages, read_attachments, read_metadata


def _decode_frame(message) -> np.ndarray:
    if message.encoding == "png":
        return np.asarray(Image.open(io.BytesIO(message.data))).copy()
    if message.encoding == "raw_rgb8":
        return np.frombuffer(message.data, dtype=np.uint8).reshape(
            message.height, message.width, 3
        ).copy()
    if message.encoding == "raw_z16_le":
        return np.frombuffer(message.data, dtype="<u2").reshape(
            message.height, message.width
        ).copy()
    raise ValueError(f"unsupported camera encoding {message.encoding!r}")


def export_hdf5(
    source: str | Path,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    source = Path(source)
    destination = Path(destination)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.stem}.partial{destination.suffix or '.h5'}")
    partial.unlink(missing_ok=True)

    states = []
    cameras = defaultdict(lambda: {"color": [], "depth": []})
    for record in iter_messages(source):
        if record.topic == TOPIC_SIM_STATE:
            states.append(record.proto_msg)
        elif record.topic.startswith("/remu/camera/"):
            kind = record.topic.rsplit("/", 1)[-1]
            if kind in ("color", "depth"):
                cameras[record.proto_msg.camera_name][kind].append(record.proto_msg)
    if not states:
        raise ValueError(f"{source} has no {TOPIC_SIM_STATE} messages")

    attachments = read_attachments(source)
    metadata = read_metadata(source)
    calibration = {}
    if "camera_calibration.json" in attachments:
        calibration = json.loads(attachments["camera_calibration.json"].data)

    ticks = np.asarray([row.tick_index for row in states], dtype=np.int64)
    tick_to_index = {int(tick): index for index, tick in enumerate(ticks)}
    try:
        with h5py.File(partial, "w") as output:
            output.attrs["format"] = "remu-hdf5-export"
            output.attrs["format_version"] = 1
            output.attrs["source"] = str(source)
            output.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
            if "scene.xml" in attachments:
                output.attrs["scene_xml"] = attachments["scene.xml"].data.decode()

            trajectory = output.create_group("trajectory")
            trajectory.create_dataset(
                "time_s", data=np.asarray([row.sim_time_ns for row in states]) / 1e9
            )
            trajectory.create_dataset(
                "qpos", data=np.asarray([row.qpos for row in states], dtype=np.float64)
            )
            trajectory.create_dataset(
                "qvel", data=np.asarray([row.qvel for row in states], dtype=np.float64)
            )

            for camera_name, streams in cameras.items():
                vendor, serial = camera_name.split("/", 1)
                group = output.require_group(f"cameras/{vendor}/{serial}")
                color_rows = sorted(streams["color"], key=lambda row: row.frame_index)
                depth_rows = sorted(streams["depth"], key=lambda row: row.frame_index)
                if color_rows:
                    group.create_dataset(
                        "color",
                        data=np.stack([_decode_frame(row) for row in color_rows]),
                        compression="lzf",
                    )
                    rows = color_rows
                else:
                    rows = depth_rows
                if depth_rows:
                    group.create_dataset(
                        "depth",
                        data=np.stack([_decode_frame(row) for row in depth_rows]),
                        compression="lzf",
                    )
                    group.attrs["depth_scale"] = depth_rows[0].depth_scale
                group.create_dataset(
                    "time_s",
                    data=np.asarray([row.actual_sim_time_ns for row in rows]) / 1e9,
                )
                group.create_dataset(
                    "scheduled_time_s",
                    data=np.asarray([row.scheduled_sim_time_ns for row in rows]) / 1e9,
                )
                group.create_dataset(
                    "trajectory_index",
                    data=np.asarray([
                        tick_to_index[int(row.trajectory_tick_index)] for row in rows
                    ], dtype=np.int64),
                )
                if camera_name in calibration:
                    group.create_dataset(
                        "link_eye_tf", data=np.asarray(calibration[camera_name])
                    )
                group.attrs["device"] = json.dumps({"name": camera_name}, sort_keys=True)
        os.replace(partial, destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="canonical Remu MCAP recording")
    parser.add_argument("destination", help="dense HDF5 output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    print(export_hdf5(args.source, args.destination, overwrite=args.overwrite))
    return 0
