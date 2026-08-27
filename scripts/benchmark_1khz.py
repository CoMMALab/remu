#!/usr/bin/env python3
"""Measure FR3 simulation throughput and 1 kHz wall-clock pacing quality."""

import argparse
import json
import threading
import time

import numpy as np

from remu.sim.mujoco_sim import ControlMode, MujocoSim
from remu.sim.scene import build_scene_xml


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile))


def _configure_sim(scene, *, realtime: bool) -> MujocoSim:
    sim = MujocoSim(scene, realtime=realtime)
    sim.build()
    target = sim.home_q + np.array([0.3, 0.2, -0.2, 0.2, 0.2, 0.2, -0.2])
    sim.update_joint_positions(target)
    sim.set_control_mode(ControlMode.POSITION)
    return sim


def run_benchmark(duration: float, raw_steps: int) -> dict:
    scene = build_scene_xml()
    try:
        raw_sim = _configure_sim(scene, realtime=False)
        for _ in range(1000):
            raw_sim.step()

        raw_start = time.perf_counter()
        for _ in range(raw_steps):
            raw_sim.step()
        raw_elapsed = time.perf_counter() - raw_start
        raw_sim.stop()

        paced_sim = _configure_sim(scene, realtime=True)
        timestamps = []
        paced_sim.on_step_callbacks.append(
            lambda _model, _data: timestamps.append(time.perf_counter())
        )

        thread = threading.Thread(target=paced_sim.run, daemon=True)
        wall_start = time.perf_counter()
        thread.start()
        time.sleep(duration)
        paced_sim.stop()
        thread.join(timeout=2.0)
        wall_elapsed = time.perf_counter() - wall_start

        if len(timestamps) < 2:
            raise RuntimeError("Simulator produced fewer than two timed steps")

        intervals = np.diff(np.asarray(timestamps))
        state_finite = all(
            np.all(np.isfinite(value)) for value in paced_sim.get_robot_state().values()
        )
        measured_elapsed = timestamps[-1] - timestamps[0]
        expected_steps = wall_elapsed * 1000.0

        return {
            "raw": {
                "steps": raw_steps,
                "elapsed_s": raw_elapsed,
                "steps_per_s": raw_steps / raw_elapsed,
                "realtime_headroom": raw_steps / raw_elapsed / 1000.0,
            },
            "paced": {
                "requested_duration_s": duration,
                "wall_elapsed_s": wall_elapsed,
                "steps": len(timestamps),
                "effective_hz": (len(timestamps) - 1) / measured_elapsed,
                "step_count_error_pct": (
                    (len(timestamps) - expected_steps) / expected_steps * 100.0
                ),
                "interval_ms_p50": _percentile(intervals, 50) * 1000.0,
                "interval_ms_p95": _percentile(intervals, 95) * 1000.0,
                "interval_ms_p99": _percentile(intervals, 99) * 1000.0,
                "interval_ms_max": float(np.max(intervals)) * 1000.0,
                "intervals_over_1_5ms": int(np.count_nonzero(intervals > 0.0015)),
                "intervals_over_2ms": int(np.count_nonzero(intervals > 0.002)),
                "final_state_finite": state_finite,
                "thread_stopped": not thread.is_alive(),
            },
        }
    finally:
        scene.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration", type=float, default=30.0, help="paced test duration in seconds"
    )
    parser.add_argument(
        "--raw-steps", type=int, default=20_000, help="unpaced throughput sample size"
    )
    args = parser.parse_args()
    if args.duration <= 0 or args.raw_steps <= 0:
        parser.error("--duration and --raw-steps must be positive")

    print(json.dumps(run_benchmark(args.duration, args.raw_steps), indent=2))


if __name__ == "__main__":
    main()
