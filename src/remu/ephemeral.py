"""One-shot MCAP trajectory capture and parallel offline RGB-D rendering."""

from __future__ import annotations

import io
import json
import logging
import multiprocessing
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import mujoco
import numpy as np
from PIL import Image

from remu.camera.rgbd import EmulatedRgbdCamera
from remu.recording import (
    AsyncRecorder,
    RecordingWriter,
    TOPIC_SIM_STATE,
    color_topic,
    depth_topic,
    iter_messages,
    merge_recordings,
    read_attachments,
    read_metadata,
    recording_pb2,
)

logger = logging.getLogger(__name__)


class TrajectoryRecorder:
    """Compatibility wrapper that captures authoritative physics snapshots to MCAP."""

    def __init__(self, output_path: str | Path, *, nq: int, nv: int, block_size: int = 4096):
        self.output_path = Path(output_path)
        self.nq = int(nq)
        self.nv = int(nv)
        self.block_size = int(block_size)
        self.sample_count = 0
        self.active = False
        self._recorder = AsyncRecorder(
            self.output_path,
            metadata={"format": "remu-mcap", "format_version": "1", "phase": "capture"},
            queue_size=max(2 * self.block_size, 1024),
        )

    def start(self) -> None:
        self._recorder.start()
        self.active = True

    def capture(self, _model, data) -> None:
        if not self.active:
            return
        self.sample_count += 1
        sim_time_ns = int(round(float(data.time) * 1e9))
        self._recorder.record(
            TOPIC_SIM_STATE,
            recording_pb2.SimState(
                tick_index=self.sample_count,
                sim_time_ns=sim_time_ns,
                qpos=data.qpos,
                qvel=data.qvel,
            ),
            log_time_ns=sim_time_ns,
            sequence=self.sample_count,
        )

    def stop(self) -> int:
        if self.active:
            self.active = False
            self._recorder.stop()
        return self.sample_count


@dataclass(frozen=True)
class CameraSchedule:
    trajectory_index: np.ndarray
    scheduled_time_s: np.ndarray


@dataclass(frozen=True)
class Trajectory:
    tick_index: np.ndarray
    time_s: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray


def load_trajectory(path: str | Path) -> Trajectory:
    rows = [record.proto_msg for record in iter_messages(path, TOPIC_SIM_STATE)]
    if not rows:
        raise ValueError("FCI session ended before any physics ticks were captured")
    return Trajectory(
        tick_index=np.asarray([row.tick_index for row in rows], dtype=np.int64),
        time_s=np.asarray([row.sim_time_ns for row in rows], dtype=np.float64) / 1e9,
        qpos=np.asarray([row.qpos for row in rows], dtype=np.float64),
        qvel=np.asarray([row.qvel for row in rows], dtype=np.float64),
    )


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


def _png_color(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(output, format="PNG", compress_level=1)
    return output.getvalue()


def _png_depth(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(array.astype("<u2", copy=False)).save(
        output, format="PNG", compress_level=1
    )
    return output.getvalue()


def _render_chunk(args: tuple) -> dict[str, Any]:
    scene_path, capture_path, shard_path, cameras, schedules, chunk = args
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    trajectory = load_trajectory(capture_path)
    chunk = np.asarray(chunk, dtype=np.int64)
    chunk_set = set(map(int, chunk))
    due: dict[int, list[tuple[EmulatedRgbdCamera, int, float]]] = {}
    for camera in cameras:
        schedule = schedules[camera.name]
        for frame_index, trajectory_index in enumerate(schedule.trajectory_index):
            if int(trajectory_index) in chunk_set:
                due.setdefault(int(trajectory_index), []).append(
                    (camera, frame_index, float(schedule.scheduled_time_s[frame_index]))
                )

    started = time.perf_counter()
    frame_count = 0
    try:
        with RecordingWriter(
            shard_path,
            metadata={"format": "remu-mcap", "phase": "render-shard"},
        ) as writer:
            for trajectory_index in chunk:
                index = int(trajectory_index)
                data.time = float(trajectory.time_s[index])
                data.qpos[:] = trajectory.qpos[index]
                data.qvel[:] = trajectory.qvel[index]
                mujoco.mj_forward(model, data)
                actual_ns = int(round(data.time * 1e9))
                tick_index = int(trajectory.tick_index[index])
                for camera, frame_index, scheduled_time in due.get(index, ()):
                    color, depth = camera.render(model, data)
                    common = {
                        "camera_name": camera.name,
                        "frame_index": frame_index,
                        "trajectory_tick_index": tick_index,
                        "scheduled_sim_time_ns": int(round(scheduled_time * 1e9)),
                        "actual_sim_time_ns": actual_ns,
                    }
                    writer.write(
                        color_topic(camera.name),
                        recording_pb2.CameraFrame(
                            **common,
                            encoding="png",
                            width=camera.color_profile.width,
                            height=camera.color_profile.height,
                            data=_png_color(color),
                        ),
                        log_time_ns=actual_ns,
                        sequence=frame_index,
                    )
                    writer.write(
                        depth_topic(camera.name),
                        recording_pb2.CameraFrame(
                            **common,
                            encoding="png",
                            width=camera.depth_profile.width,
                            height=camera.depth_profile.height,
                            depth_scale=camera.depth_scale,
                            data=_png_depth(depth),
                        ),
                        log_time_ns=actual_ns,
                        sequence=frame_index,
                    )
                    frame_count += 1
    finally:
        for camera in cameras:
            camera.close()
    return {
        "shard": str(shard_path),
        "frames": frame_count,
        "elapsed_s": time.perf_counter() - started,
    }


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
    """Render a completed MCAP capture, merge shards, and atomically publish it."""
    scene_path = Path(scene_path)
    capture_path = Path(capture_path)
    output_path = Path(output_path)
    if workers < 1:
        raise ValueError("render_workers must be at least 1")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}")

    trajectory = load_trajectory(capture_path)
    schedules = schedule_camera_frames(trajectory.time_s, cameras)
    union = np.unique(np.concatenate([
        schedule.trajectory_index for schedule in schedules.values()
    ])) if schedules else np.empty(0, dtype=np.int64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shard_dir = Path(tempfile.mkdtemp(prefix="remu_render_", dir=output_path.parent))
    merge_path = output_path.with_name(f"{output_path.stem}.merging{output_path.suffix}")
    shards: list[Path] = []
    render_results: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        if len(union):
            worker_count = min(workers, len(union))
            chunks = [chunk for chunk in np.array_split(union, worker_count) if len(chunk)]
            tasks = []
            for index, chunk in enumerate(chunks):
                shard_path = shard_dir / f"worker-{index:03d}.mcap"
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

        final_metadata = read_metadata(capture_path)
        final_metadata.update({
            "phase": "final",
            "render_workers": str(worker_count),
            "rendered_frames": str(sum(len(value.trajectory_index) for value in schedules.values())),
            "metadata_json": json.dumps(metadata or {}, sort_keys=True),
        })
        merge_recordings(
            [capture_path, *shards],
            merge_path,
            metadata=final_metadata,
            attachments=read_attachments(capture_path),
        )
        os.replace(merge_path, output_path)
    except BaseException:
        logger.error("offline render failed; phase-one capture retained at %s", capture_path)
        merge_path.unlink(missing_ok=True)
        raise
    else:
        capture_path.unlink()
        shutil.rmtree(shard_dir)

    elapsed = time.perf_counter() - started
    frame_count = sum(result["frames"] for result in render_results)
    return {
        "output": str(output_path),
        "trajectory_samples": len(trajectory.time_s),
        "rendered_frames": frame_count,
        "render_workers": worker_count,
        "elapsed_s": elapsed,
        "frames_per_second": frame_count / elapsed if elapsed else 0.0,
    }
