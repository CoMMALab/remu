"""One-shot trajectory capture and parallel offline RGB-D rendering."""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import queue
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import mujoco
import numpy as np

from remu.camera.rgbd import EmulatedRgbdCamera

logger = logging.getLogger(__name__)


class TrajectoryRecorder:
    """Record every physics tick while moving HDF5 I/O off the physics thread."""

    def __init__(self, output_path: str | Path, *, nq: int, nv: int, block_size: int = 4096):
        self.output_path = Path(output_path)
        self.nq = int(nq)
        self.nv = int(nv)
        self.block_size = int(block_size)
        self.active = False
        self.sample_count = 0
        self._lock = threading.Lock()
        self._blocks: queue.Queue = queue.Queue()
        self._writer: threading.Thread | None = None
        self._writer_error: BaseException | None = None
        self._time = self._qpos = self._qvel = None
        self._used = 0

    def _new_block(self) -> None:
        self._time = np.empty(self.block_size, dtype=np.float64)
        self._qpos = np.empty((self.block_size, self.nq), dtype=np.float64)
        self._qvel = np.empty((self.block_size, self.nv), dtype=np.float64)
        self._used = 0

    def start(self) -> None:
        with self._lock:
            if self.active or self._writer is not None:
                raise RuntimeError("trajectory recorder can only be started once")
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._new_block()
            self.active = True
            self._writer = threading.Thread(target=self._write_blocks, daemon=True)
            self._writer.start()

    def capture(self, _model, data) -> None:
        with self._lock:
            if not self.active:
                return
            index = self._used
            self._time[index] = data.time
            self._qpos[index] = data.qpos
            self._qvel[index] = data.qvel
            self._used += 1
            self.sample_count += 1
            if self._used == self.block_size:
                self._blocks.put((self._time, self._qpos, self._qvel, self._used))
                self._new_block()

    def stop(self) -> int:
        with self._lock:
            if self.active:
                self.active = False
                if self._used:
                    self._blocks.put((self._time, self._qpos, self._qvel, self._used))
                self._blocks.put(None)
        if self._writer is not None:
            self._writer.join()
        if self._writer_error is not None:
            raise RuntimeError("trajectory HDF5 writer failed") from self._writer_error
        return self.sample_count

    def _write_blocks(self) -> None:
        try:
            import h5py

            with h5py.File(self.output_path, "w") as output:
                output.attrs["format"] = "remu-ephemeral"
                output.attrs["format_version"] = 1
                output.attrs["wall_start_unix_ns"] = time.time_ns()
                group = output.create_group("trajectory")
                times = group.create_dataset(
                    "time_s", shape=(0,), maxshape=(None,), chunks=(self.block_size,), dtype="f8"
                )
                qpos = group.create_dataset(
                    "qpos", shape=(0, self.nq), maxshape=(None, self.nq),
                    chunks=(self.block_size, self.nq), dtype="f8",
                )
                qvel = group.create_dataset(
                    "qvel", shape=(0, self.nv), maxshape=(None, self.nv),
                    chunks=(self.block_size, self.nv), dtype="f8",
                )
                offset = 0
                while True:
                    block = self._blocks.get()
                    if block is None:
                        break
                    block_time, block_qpos, block_qvel, used = block
                    end = offset + used
                    times.resize((end,))
                    qpos.resize((end, self.nq))
                    qvel.resize((end, self.nv))
                    times[offset:end] = block_time[:used]
                    qpos[offset:end] = block_qpos[:used]
                    qvel[offset:end] = block_qvel[:used]
                    offset = end
                output.attrs["trajectory_samples"] = offset
        except BaseException as exc:
            self._writer_error = exc


@dataclass(frozen=True)
class CameraSchedule:
    trajectory_index: np.ndarray
    scheduled_time_s: np.ndarray


def schedule_camera_frames(
    time_s: np.ndarray, cameras: Sequence[EmulatedRgbdCamera]
) -> dict[str, CameraSchedule]:
    """Select the first physics tick at or after every camera boundary."""
    times = np.asarray(time_s, dtype=np.float64)
    if times.ndim != 1 or not len(times):
        raise ValueError("trajectory must contain at least one timestamp")
    if np.any(np.diff(times) <= 0):
        raise ValueError("trajectory timestamps must be strictly increasing")

    schedules: dict[str, CameraSchedule] = {}
    duration = max(0.0, float(times[-1] - times[0]))
    for camera in cameras:
        count = int(np.floor(duration * camera.fps + 1e-12)) + 1
        boundaries = times[0] + np.arange(count, dtype=np.float64) / camera.fps
        indices = np.searchsorted(times, boundaries, side="left")
        valid = indices < len(times)
        indices = indices[valid]
        boundaries = boundaries[valid]
        if len(indices):
            keep = np.r_[True, np.diff(indices) != 0]
            indices = indices[keep]
            boundaries = boundaries[keep]
        schedules[camera.name] = CameraSchedule(indices.astype(np.int64), boundaries)
    return schedules


def _render_chunk(args: tuple) -> dict[str, Any]:
    scene_path, capture_path, shard_path, cameras, schedules, chunk = args
    import h5py

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    chunk = np.asarray(chunk, dtype=np.int64)
    due: dict[int, list[tuple[EmulatedRgbdCamera, int, float]]] = {}
    counts: dict[str, int] = {}
    chunk_set = set(int(value) for value in chunk)
    for camera in cameras:
        schedule = schedules[camera.name]
        selected = [i for i, value in enumerate(schedule.trajectory_index) if int(value) in chunk_set]
        counts[camera.name] = len(selected)
        for frame_index in selected:
            trajectory_index = int(schedule.trajectory_index[frame_index])
            due.setdefault(trajectory_index, []).append(
                (camera, frame_index, float(schedule.scheduled_time_s[frame_index]))
            )

    started = time.perf_counter()
    try:
        with h5py.File(capture_path, "r") as capture, h5py.File(shard_path, "w") as shard:
            outputs = {}
            for camera in cameras:
                count = counts[camera.name]
                if count == 0:
                    continue
                group = shard.require_group(f"cameras/{camera.vendor}/{camera.serial}")
                outputs[camera.name] = {
                    "used": 0,
                    "frame_index": group.create_dataset("frame_index", (count,), dtype="i8"),
                    "trajectory_index": group.create_dataset("trajectory_index", (count,), dtype="i8"),
                    "scheduled_time_s": group.create_dataset("scheduled_time_s", (count,), dtype="f8"),
                    "time_s": group.create_dataset("time_s", (count,), dtype="f8"),
                    "color": group.create_dataset(
                        "color", (count, camera.color_profile.height, camera.color_profile.width, 3),
                        dtype="u1", chunks=(1, camera.color_profile.height, camera.color_profile.width, 3),
                        compression="lzf",
                    ),
                    "depth": group.create_dataset(
                        "depth", (count, camera.depth_profile.height, camera.depth_profile.width),
                        dtype="u2", chunks=(1, camera.depth_profile.height, camera.depth_profile.width),
                        compression="lzf",
                    ),
                }

            trajectory = capture["trajectory"]
            for trajectory_index in chunk:
                index = int(trajectory_index)
                data.time = float(trajectory["time_s"][index])
                data.qpos[:] = trajectory["qpos"][index]
                data.qvel[:] = trajectory["qvel"][index]
                mujoco.mj_forward(model, data)
                for camera, frame_index, scheduled_time in due.get(index, ()):
                    color, depth = camera.render(model, data)
                    target = outputs[camera.name]
                    row = target["used"]
                    target["frame_index"][row] = frame_index
                    target["trajectory_index"][row] = index
                    target["scheduled_time_s"][row] = scheduled_time
                    target["time_s"][row] = data.time
                    target["color"][row] = color
                    target["depth"][row] = depth
                    target["used"] += 1
    finally:
        for camera in cameras:
            camera.close()
    return {
        "shard": str(shard_path),
        "frames": sum(counts.values()),
        "elapsed_s": time.perf_counter() - started,
    }


def _camera_group(file, camera: EmulatedRgbdCamera):
    return file.require_group(f"cameras/{camera.vendor}/{camera.serial}")


def _merge_shards(
    capture_path: Path,
    shards: Sequence[Path],
    cameras: Sequence[EmulatedRgbdCamera],
    schedules: dict[str, CameraSchedule],
) -> None:
    import h5py

    with h5py.File(capture_path, "r+") as output:
        for camera in cameras:
            schedule = schedules[camera.name]
            count = len(schedule.trajectory_index)
            group = _camera_group(output, camera)
            group.attrs["device"] = json.dumps(camera.device_dict(), sort_keys=True)
            group.create_dataset("base_from_optical", data=camera.optical_pose())
            datasets = {
                "trajectory_index": group.create_dataset("trajectory_index", (count,), dtype="i8"),
                "scheduled_time_s": group.create_dataset("scheduled_time_s", (count,), dtype="f8"),
                "time_s": group.create_dataset("time_s", (count,), dtype="f8"),
                "color": group.create_dataset(
                    "color", (count, camera.color_profile.height, camera.color_profile.width, 3),
                    dtype="u1", chunks=(1, camera.color_profile.height, camera.color_profile.width, 3),
                    compression="lzf",
                ),
                "depth": group.create_dataset(
                    "depth", (count, camera.depth_profile.height, camera.depth_profile.width),
                    dtype="u2", chunks=(1, camera.depth_profile.height, camera.depth_profile.width),
                    compression="lzf",
                ),
            }
            covered = np.zeros(count, dtype=bool)
            for shard_path in shards:
                with h5py.File(shard_path, "r") as shard:
                    path = f"cameras/{camera.vendor}/{camera.serial}"
                    if path not in shard:
                        continue
                    source = shard[path]
                    frame_indices = source["frame_index"][:]
                    if np.any(frame_indices < 0) or np.any(frame_indices >= count):
                        raise ValueError(f"invalid frame index in shard {shard_path}")
                    if np.any(covered[frame_indices]):
                        raise ValueError(f"duplicate frames in shard {shard_path}")
                    for name, dataset in datasets.items():
                        dataset[frame_indices] = source[name][:]
                    covered[frame_indices] = True
            if not np.all(covered):
                missing = np.flatnonzero(~covered)
                raise ValueError(f"camera {camera.name} is missing frames {missing[:10].tolist()}")
            if not np.array_equal(datasets["trajectory_index"][:], schedule.trajectory_index):
                raise ValueError(f"camera {camera.name} frame order changed during merge")
            if np.any(np.diff(datasets["time_s"][:]) <= 0):
                raise ValueError(f"camera {camera.name} timestamps are not strictly increasing")


def render_offline(
    *,
    scene_path: str | Path,
    capture_path: str | Path,
    output_path: str | Path,
    cameras: Sequence[EmulatedRgbdCamera],
    workers: int = 1,
    overwrite: bool = False,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render a completed capture into an atomic, ordered HDF5 dataset."""
    import h5py

    scene_path = Path(scene_path)
    capture_path = Path(capture_path)
    output_path = Path(output_path)
    if workers < 1:
        raise ValueError("render_workers must be at least 1")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}")

    with h5py.File(capture_path, "r") as capture:
        time_s = capture["trajectory/time_s"][:]
    if not len(time_s):
        raise ValueError("FCI session ended before any physics ticks were captured")
    schedules = schedule_camera_frames(time_s, cameras)
    union = np.unique(np.concatenate([
        schedule.trajectory_index for schedule in schedules.values()
    ])) if schedules else np.empty(0, dtype=np.int64)

    shard_dir = Path(tempfile.mkdtemp(prefix="remu_render_", dir=output_path.parent))
    shards: list[Path] = []
    render_results: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        if len(union):
            worker_count = min(workers, len(union))
            chunks = [chunk for chunk in np.array_split(union, worker_count) if len(chunk)]
            tasks = []
            for index, chunk in enumerate(chunks):
                shard_path = shard_dir / f"shard_{index:04d}.h5"
                shards.append(shard_path)
                tasks.append((scene_path, capture_path, shard_path, list(cameras), schedules, chunk))
            context = multiprocessing.get_context("spawn")
            if worker_count == 1:
                render_results = [_render_chunk(tasks[0])]
            else:
                with context.Pool(worker_count) as pool:
                    render_results = pool.map(_render_chunk, tasks)
        else:
            worker_count = 0

        _merge_shards(capture_path, shards, cameras, schedules)
        with h5py.File(capture_path, "r+") as output:
            output.attrs["physics_dt_s"] = float(np.median(np.diff(time_s))) if len(time_s) > 1 else 0.0
            output.attrs["render_workers"] = worker_count
            output.attrs["render_elapsed_s"] = time.perf_counter() - started
            output.attrs["rendered_frames"] = sum(len(value.trajectory_index) for value in schedules.values())
            output.attrs["scene_xml"] = scene_path.read_text(encoding="utf-8")
            output.attrs["metadata_json"] = json.dumps(metadata or {}, sort_keys=True)
        os.replace(capture_path, output_path)
    except BaseException:
        logger.error("offline render failed; partial capture retained at %s", capture_path)
        raise
    else:
        shutil.rmtree(shard_dir)

    elapsed = time.perf_counter() - started
    frame_count = sum(result["frames"] for result in render_results)
    return {
        "output": str(output_path),
        "trajectory_samples": len(time_s),
        "rendered_frames": frame_count,
        "render_workers": worker_count,
        "elapsed_s": elapsed,
        "frames_per_second": frame_count / elapsed if elapsed else 0.0,
    }
