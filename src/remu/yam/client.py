"""Bounded Portal robot adapter; a failed connection must be replaced."""
import math
import threading
import time


class YamClient:
    """No automatic reconnect/replay: create a new client after any failed RPC."""

    def __init__(self, address, timeout=1):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be finite and positive')
        import portal
        self._client = portal.Client(address, autoconn=False, maxinflight=1)
        self.timeout = timeout
        self._lock = threading.Lock()
        self._connected = False
        self._closed = False

    def _call(self, method, *args):
        deadline = time.monotonic() + self.timeout
        if not self._lock.acquire(timeout=self.timeout):
            raise TimeoutError('another RPC is still in flight')
        try:
            if self._closed:
                raise ConnectionError('client failed or closed; create a new connection')
            if not self._connected:
                if not self._client.connect(timeout=max(0., deadline - time.monotonic())):
                    raise TimeoutError('Portal connection deadline exceeded')
                self._connected = True
            future = getattr(self._client, method)(*args)
            return future.result(timeout=max(0., deadline - time.monotonic()))
        except Exception:
            # Do not leave timed-out futures queued or replay commands on reconnect.
            self.close()
            raise
        finally:
            self._lock.release()

    def num_dofs(self):
        return self._call('num_dofs')

    def get_joint_pos(self):
        return self._call('get_joint_pos')

    def command_joint_pos(self, joint_pos):
        return self._call('command_joint_pos', joint_pos)

    def get_observations(self):
        return self._call('get_observations')

    def command_joint_state(self, state):
        return self._call('command_joint_state', state)

    def command_joint_vel(self, velocity):
        return self._call('command_joint_vel', velocity)

    def command_joint_torque(self, effort):
        return self._call('command_joint_torque', effort)

    def get_robot_info(self):
        return self._call('get_robot_info')

    def get_robot_model(self):
        return self._call('get_robot_model')

    def get_dynamics(self):
        return self._call('get_dynamics')

    def get_end_effector_pose(self):
        return self._call('get_end_effector_pose')

    def gripper_move(self, opening, speed=1., force=20., timeout=5.):
        return self._call('gripper_move', opening, speed, force, timeout)

    def gripper_grasp(self, opening, speed=1., force=20., epsilon_inner=.02, epsilon_outer=.02, timeout=5.):
        return self._call('gripper_grasp', opening, speed, force, epsilon_inner, epsilon_outer, timeout)

    def gripper_home(self):
        return self._call('gripper_home')

    def gripper_stop(self):
        return self._call('gripper_stop')

    def get_gripper_state(self):
        return self._call('get_gripper_state')

    def keepalive(self):
        return self._call('keepalive')

    def stop(self):
        return self._call('stop')

    def close(self):
        if not self._closed:
            self._closed = True
            self._client.close(timeout=self.timeout)
