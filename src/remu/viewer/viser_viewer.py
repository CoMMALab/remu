"""Browser-based viewer using ``mjviser`` (viser + MuJoCo).

Unlike ``mjviser.Viewer`` (which owns its own step loop), remu already steps
physics in :class:`~remu.sim.mujoco_sim.MujocoSim`'s realtime loop -- so this
wraps ``ViserMujocoScene`` directly and pushes state to it as a per-step
callback, the same pattern as :class:`~remu.viewer.mujoco_viewer.MujocoPassiveViewer`.
"""

import logging

import mjviser
import viser

logger = logging.getLogger(__name__)


class ViserViewer:
    """Streams the sim's MuJoCo state to a browser-based viser scene."""

    def __init__(self, model, data, host: str = "0.0.0.0", port: int = 8080):
        self.model = model
        self.data = data
        self.server = viser.ViserServer(host=host, port=port)
        self.scene = mjviser.ViserMujocoScene(self.server, model, num_envs=1)
        self.scene.update_from_mjdata(data)
        logger.info("mjviser scene serving at http://localhost:%d", port)

    def sync(self, model, data):
        self.scene.update_from_mjdata(data)

    def attach(self, sim):
        sim.on_step_callbacks.append(self.sync)
        return self

    def close(self):
        self.server.stop()
