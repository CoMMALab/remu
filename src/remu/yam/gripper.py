"""Asynchronous simulated gripper operations with contact-based grasp results."""
import numpy as np


class GripperMotion:
    def __init__(self, sim):
        self.sim = sim
        self.motion_id = 0
        self.status = 'idle'
        self.operation = 'stop'
        self.started = 0.
        self.timeout = 5.
        self.opening = 1.
        self.epsilon_inner = .02
        self.epsilon_outer = .02
        self.settled = 0
        self.speed = 2.
        self.force = 20.
        self.active = False

    def start(self, operation, opening=1., speed=1., force=20., epsilon_inner=.02,
              epsilon_outer=.02, timeout=5.):
        values = np.array([opening, speed, force, epsilon_inner, epsilon_outer, timeout])
        if (not np.all(np.isfinite(values)) or not 0 <= opening <= 1 or speed <= 0
                or force <= 0 or force > 20 or epsilon_inner < 0 or epsilon_outer < 0 or timeout <= 0):
            raise ValueError('invalid gripper opening, speed, force, tolerance, or timeout')
        if speed > self.sim.limits.velocity[6]:
            raise ValueError('gripper speed exceeds configured velocity limit')
        self.motion_id += 1
        self.operation = operation
        self.opening = float(opening)
        self.speed, self.force, self.timeout = float(speed), float(force), float(timeout)
        self.epsilon_inner, self.epsilon_outer = float(epsilon_inner), float(epsilon_outer)
        self.started = self.sim.data.time
        self.status = 'moving'
        self.active = True
        self.settled = 0
        if self.sim._mode == 'torque':
            self.sim._target = self.sim._positions()
            self.sim._trajectory.reset(self.sim._target)
            self.sim._mode = 'hold'
            self.sim._torque.fill(0.)
        self.sim._target[6] = 0. if operation == 'grasp' else opening
        self.sim._trajectory.position[6] = self.sim._positions()[6]
        self.sim._trajectory.velocity[6] = self.sim._trajectory.acceleration[6] = 0.
        self.sim._touch_command()
        return self.state()

    def stop(self, status='stopped'):
        self.active = False
        self.status = status
        self.sim._target[6] = self.sim._positions()[6]
        self.sim._trajectory.position[6] = self.sim._target[6]
        self.sim._trajectory.velocity[6] = self.sim._trajectory.acceleration[6] = 0.
        return self.state()

    def contacts(self):
        sim = self.sim
        left, right = sim._finger_bodies
        touching = [set(), set()]
        for contact in sim.data.contact[:sim.data.ncon]:
            a, b = (int(sim.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
            for i, finger in enumerate((left, right)):
                other = b if a == finger else a if b == finger else -1
                if other > 0 and other not in sim._robot_bodies:
                    touching[i].add(other)
        return touching[0] & touching[1]

    def update(self):
        if self.status == 'grasped' and not self.contacts():
            self.status = 'lost'
        if not self.active:
            return
        opening = self.sim._positions()[6]
        speed = abs(self.sim._logical_velocity()[6])
        self.settled = self.settled + 1 if speed < .02 else 0
        if self.sim.data.time - self.started > self.timeout:
            self.stop('timeout')
        elif self.operation == 'grasp' and self.settled >= 20:
            band = self.opening - self.epsilon_inner <= opening <= self.opening + self.epsilon_outer
            if self.contacts() and band:
                self.status, self.active = 'grasped', False
            elif opening < .005:
                self.stop('failed')
        elif self.operation != 'grasp' and self.settled >= 20 and abs(opening - self.opening) < .01:
            self.status, self.active = 'succeeded', False

    def state(self):
        opening = float(self.sim._positions()[6])
        return dict(motion_id=self.motion_id, operation=self.operation, status=self.status,
                    opening=opening, width=opening * self.sim._finger_span.sum(),
                    max_width=float(self.sim._finger_span.sum()),
                    is_grasped=self.status == 'grasped', contacts=sorted(self.contacts()))
