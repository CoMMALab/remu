from types import SimpleNamespace

import h5py
import numpy as np

from remu.cli import _parse_args
from remu.ephemeral import TrajectoryRecorder, schedule_camera_frames


def test_scheduler_uses_first_tick_at_or_after_each_boundary():
    camera = SimpleNamespace(name="realsense/test", fps=30)
    times = np.arange(0.001, 0.102, 0.001)

    schedule = schedule_camera_frames(times, [camera])[camera.name]

    assert schedule.trajectory_index.tolist() == [0, 34, 67, 100]
    assert np.all(times[schedule.trajectory_index] >= schedule.scheduled_time_s)
    assert np.all(times[schedule.trajectory_index] - schedule.scheduled_time_s < 0.001 + 1e-12)


def test_recorder_writes_complete_qpos_and_qvel_for_every_tick(tmp_path):
    path = tmp_path / "capture.h5"
    recorder = TrajectoryRecorder(path, nq=4, nv=3, block_size=2)
    recorder.start()
    for tick in range(5):
        recorder.capture(None, SimpleNamespace(
            time=(tick + 1) * 0.001,
            qpos=np.arange(4, dtype=float) + tick,
            qvel=np.arange(3, dtype=float) - tick,
        ))
    assert recorder.stop() == 5

    with h5py.File(path) as capture:
        assert capture["trajectory/time_s"].shape == (5,)
        assert capture["trajectory/qpos"].shape == (5, 4)
        assert capture["trajectory/qvel"].shape == (5, 3)
        assert np.array_equal(capture["trajectory/qpos"][4], np.arange(4) + 4)
        assert np.array_equal(capture["trajectory/qvel"][4], np.arange(3) - 4)


def test_unified_config_paths_and_cli_overrides(tmp_path):
    config = tmp_path / "run.yaml"
    config.write_text("""
version: 1
mode: ephemeral
robot:
  scene_mjcf: scene.xml
  initial_q: [0, -0.2, 0, -2.4, 0, 2.2, 0.7]
simulation: {dt: 0.001}
viewer: {backend: none}
ephemeral:
  output: data/run.h5
  render_workers: 2
""")

    args, loaded = _parse_args([
        "--config", str(config), "--render-workers", "4",
        "--initial-q", "0.1", "-0.2", "0", "-2.4", "0", "2.2", "0.7",
    ])

    assert loaded is not None
    assert args.mode == "ephemeral"
    assert args.viewer == "none"
    assert args.scene_mjcf == str(tmp_path / "scene.xml")
    assert args.output == str(tmp_path / "data/run.h5")
    assert args.render_workers == 4
    assert args.initial_q[0] == 0.1


def test_parallel_chunk_boundaries_match_single_worker(tmp_path):
    import shutil

    import pytest

    from remu.camera import EmulatedRgbdCamera, StreamProfile
    from remu.ephemeral import render_offline
    from remu.sim.mujoco_sim import MujocoSim
    from remu.sim.scene import build_scene_xml

    camera = EmulatedRgbdCamera(
        vendor="realsense", model="d435i", serial="parallel-test", device_name="test",
        base_from_optical=[[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0.6], [0, 0, 0, 1]],
        color=StreamProfile(32, 24, "rgb8", 42.5),
        depth=StreamProfile(32, 24, "z16", 58.0),
        fps=30, parent_body="fr3_link0",
    )
    scene = build_scene_xml(cameras=[camera])
    try:
        sim = MujocoSim(scene, realtime=False)
        sim.build()
        capture_one = tmp_path / "one.partial.h5"
        recorder = TrajectoryRecorder(capture_one, nq=sim.model.nq, nv=sim.model.nv, block_size=16)
        sim.on_step_callbacks.append(recorder.capture)
        recorder.start()
        for _ in range(70):
            sim.step()
        recorder.stop()
        capture_two = tmp_path / "two.partial.h5"
        shutil.copyfile(capture_one, capture_two)

        try:
            output_one = tmp_path / "one.h5"
            output_two = tmp_path / "two.h5"
            render_offline(
                scene_path=scene, capture_path=capture_one, output_path=output_one,
                cameras=[camera], workers=1,
            )
            render_offline(
                scene_path=scene, capture_path=capture_two, output_path=output_two,
                cameras=[camera], workers=2,
            )
        except Exception as exc:
            pytest.skip(f"offscreen multiprocessing render unavailable: {exc}")

        with h5py.File(output_one) as first, h5py.File(output_two) as second:
            path = "cameras/realsense/parallel-test"
            for name in ("trajectory_index", "scheduled_time_s", "time_s", "color", "depth"):
                assert np.array_equal(first[f"{path}/{name}"][:], second[f"{path}/{name}"][:])
    finally:
        scene.unlink(missing_ok=True)


def test_export_fr3_teleop_staging_layout(tmp_path):
    import json

    from PIL import Image

    from remu.fr3_teleop import export_episode

    source = tmp_path / "run.h5"
    with h5py.File(source, "w") as capture:
        capture.attrs["format"] = "remu-ephemeral"
        trajectory = capture.create_group("trajectory")
        trajectory.create_dataset("time_s", data=[2.0, 2.001, 2.002])
        trajectory.create_dataset("qpos", data=np.tile(np.arange(9), (3, 1)) / 100.0)
        trajectory.create_dataset("qvel", data=np.zeros((3, 9)))
        camera = capture.create_group("cameras/realsense/test")
        camera.create_dataset("time_s", data=[2.0, 2.034])
        camera.create_dataset("scheduled_time_s", data=[2.0, 2.0 + 1 / 30])
        camera.create_dataset("trajectory_index", data=[0, 2])
        camera.create_dataset("color", data=np.full((2, 3, 4, 3), 127, dtype=np.uint8))
        camera.create_dataset("depth", data=np.full((2, 3, 4), 500, dtype=np.uint16))
        camera.create_dataset("base_from_optical", data=np.eye(4))
        camera.attrs["device"] = json.dumps({"model": "d435i"})

    output = export_episode(source, tmp_path / "episode")

    meta = json.loads((output / "episode.json").read_text())
    states = [json.loads(line) for line in (output / "state.jsonl").read_text().splitlines()]
    frames = [json.loads(line) for line in (
        output / "realsense/test/frames.jsonl"
    ).read_text().splitlines()]
    assert meta["state_rows"] == 3
    assert meta["cameras"]["realsense/test"]["frames"] == 2
    assert states[0]["q"] == states[0]["q_d"]
    assert states[0]["gripper_width"] == 0.08
    assert frames[1]["encoding"] == 0
    assert Image.open(output / "realsense/test/000001.jpg").size == (4, 3)
