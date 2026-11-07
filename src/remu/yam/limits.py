"""Configurable YAM simulation limits, in radians and normalized gripper units.

Defaults are simulation tuning, not manufacturer-certified hardware limits.
Position ranges come from the loaded model; URDF velocity/effort are placeholders.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


def positive_vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (7,) or not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ValueError(f'{name} must contain seven finite positive values')
    return result.copy()


@dataclass
class YamLimits:
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray
    effort: np.ndarray
    effort_rate: np.ndarray

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            setattr(self, name, positive_vector(getattr(self, name), name))

    @classmethod
    def defaults(cls, finger_span=.0939):
        return cls(np.r_[np.full(6, 1.5), 2.], np.r_[np.full(6, 10.), 20.],
                   np.r_[np.full(6, 100.), 200.], np.r_[np.full(6, 10.), 20 * finger_span],
                   np.r_[np.full(6, 100.), 200 * finger_span])

    @classmethod
    def load(cls, path, finger_span=.0939):
        values = cls.defaults(finger_span).asdict()
        supplied = json.loads(Path(path).read_text())
        if not isinstance(supplied, dict) or set(supplied) - set(values):
            raise ValueError('limits file must contain only velocity, acceleration, jerk, effort, effort_rate')
        values.update(supplied)
        return cls(**values)

    def asdict(self):
        return {name: getattr(self, name).tolist() for name in self.__dataclass_fields__}


class JointTrajectory:
    def __init__(self, position, ranges, limits):
        self.ranges = np.asarray(ranges).copy()
        self.limits = limits
        self.reset(position)

    def reset(self, position):
        self.position = np.clip(position, self.ranges[:, 0], self.ranges[:, 1]).copy()
        self.velocity = np.zeros(7)
        self.acceleration = np.zeros(7)

    def velocity_bounds(self, position):
        # Braking envelope depends on remaining travel, rather than allowing
        # full velocity right up to a position stop.
        lo = -np.minimum(self.limits.velocity,
            np.sqrt(np.maximum(0., 2 * self.limits.acceleration * (position - self.ranges[:, 0]))))
        hi = np.minimum(self.limits.velocity,
            np.sqrt(np.maximum(0., 2 * self.limits.acceleration * (self.ranges[:, 1] - position))))
        # A conservative linear approach envelope starts braking before the
        # square-root boundary, leaving time to unwind acceleration with jerk.
        lo = np.maximum(lo, -2. * np.maximum(0., position - self.ranges[:, 0]))
        hi = np.minimum(hi, 2. * np.maximum(0., self.ranges[:, 1] - position))
        return lo, hi

    def advance(self, mode, target, velocity, dt, gripper_speed=None):
        lo, hi = self.velocity_bounds(self.position)
        if mode in ('position', 'impedance', 'hold'):
            error = target - self.position
            # Smooth braking also accounts for the time needed to unwind
            # acceleration under the jerk bound; PD tracking avoids snapping
            # position/derivatives at a target crossing.
            desired = 4. * error
        else:
            desired = velocity
        if gripper_speed is not None:
            desired[6] = np.clip(desired[6], -gripper_speed, gripper_speed)
        desired = np.clip(desired, lo, hi)
        desired_acceleration = np.clip((desired - self.velocity) * 10.,
            -self.limits.acceleration, self.limits.acceleration)
        self.acceleration += np.clip(desired_acceleration - self.acceleration,
            -self.limits.jerk * dt, self.limits.jerk * dt)
        self.velocity = np.clip(self.velocity + self.acceleration * dt, -self.limits.velocity, self.limits.velocity)
        self.position = np.clip(self.position + self.velocity * dt,
            self.ranges[:, 0], self.ranges[:, 1])
        stopped = ((self.position <= self.ranges[:, 0]) & (self.velocity < 0)) | (
            (self.position >= self.ranges[:, 1]) & (self.velocity > 0))
        self.velocity[stopped] = 0
        self.acceleration[stopped] = 0
