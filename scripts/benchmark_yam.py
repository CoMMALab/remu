#!/usr/bin/env python3
"""Measure shared-world YAM physics headroom and best-effort 1 kHz pacing."""
import argparse
import json
import time

import numpy as np
from remu.yam.world import YamWorld


def benchmark(duration=5., raw_steps=5000):
    if not np.isfinite(duration) or duration <= 0 or raw_steps < 2:
        raise ValueError('duration must be positive and finite; raw_steps must be at least two')
    world = YamWorld(watchdog=0)
    for arm in world.arms:
        target = arm.get_joint_pos()
        target[0] += .2
        arm.command_joint_pos(target)
    for _ in range(500):
        world.step()
    started = time.perf_counter()
    for _ in range(raw_steps):
        world.step()
    raw_elapsed = time.perf_counter() - started
    timestamps = []
    deadline = time.perf_counter()
    end = deadline + duration
    while time.perf_counter() < end:
        world.step()
        timestamps.append(time.perf_counter())
        deadline += world.model.opt.timestep
        delay = deadline - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        elif delay < -.01:
            deadline = time.perf_counter()
    if len(timestamps) < 2:
        raise RuntimeError('fewer than two paced samples')
    intervals = np.diff(timestamps) * 1000
    return dict(raw_steps_per_s=raw_steps / raw_elapsed,
        raw_realtime_headroom=raw_steps / raw_elapsed / 1000,
        paced_steps=len(timestamps), effective_hz=1000 / float(np.mean(intervals)),
        interval_ms_p50=float(np.percentile(intervals, 50)),
        interval_ms_p95=float(np.percentile(intervals, 95)),
        interval_ms_p99=float(np.percentile(intervals, 99)), interval_ms_max=float(intervals.max()),
        intervals_over_1_5ms=int(np.sum(intervals > 1.5)),
        state_finite=bool(np.all(np.isfinite(world.data.qpos)) and np.all(np.isfinite(world.data.qvel))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=5.)
    parser.add_argument('--raw-steps', type=int, default=5000)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.duration, args.raw_steps), indent=2))


if __name__ == '__main__':
    main()
