"""Native MuJoCo passive-viewer window, synced from the sim's physics loop."""

import logging

import mujoco.viewer

logger = logging.getLogger(__name__)


class MujocoPassiveViewer:
    """Wraps ``mujoco.viewer.launch_passive`` and syncs it on every sim step.

    Usage: construct after ``sim.build()``, then register ``viewer.sync`` as
    a step callback on the sim (``sim.on_step_callbacks.append(viewer.sync)``),
    or call :meth:`attach` to do both.
    """

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self._handle = mujoco.viewer.launch_passive(model, data)

    def sync(self, model, data):
        if self._handle.is_running():
            self._handle.sync()

    def attach(self, sim):
        sim.on_step_callbacks.append(self.sync)
        return self

    def is_running(self) -> bool:
        return self._handle.is_running()

    def close(self):
        self._handle.close()
