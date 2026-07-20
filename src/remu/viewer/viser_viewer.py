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

_IDENTITY_WXYZ = (1.0, 0.0, 0.0, 0.0)


def _mat3_to_wxyz(R):
    """Rotation matrix -> viser's (w, x, y, z) quaternion.

    Done by hand rather than via scipy: the viewer's only other dependencies
    are mujoco/viser, and pulling scipy in for one conversion isn't worth it.
    Uses the branch on the largest diagonal term, which stays numerically
    stable for rotations near 180 degrees where the naive trace formula loses
    precision.
    """
    import numpy as np

    R = np.asarray(R, dtype=float)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return np.array([w, x, y, z])


class ViserViewer:
    """Streams the sim's MuJoCo state to a browser-based viser scene."""

    def __init__(self, model, data, host: str = "0.0.0.0", port: int = 8080):
        self.model = model
        self.data = data
        self.server = viser.ViserServer(host=host, port=port)
        self.scene = mjviser.ViserMujocoScene(self.server, model, num_envs=1)
        self.scene.update_from_mjdata(data)
        logger.info("mjviser scene serving at http://localhost:%d", port)

    def add_camera(self, camera):
        """Draw an :class:`~remu.camera.d435i.EmulatedD435i` as a labelled frustum.

        Uses the camera's *optical* pose (+z forward, +y down), which is
        viser's own frustum convention -- so the drawn frustum shows exactly
        the volume the emulated sensor sees, and the axes triad shows the
        frame its point cloud is expressed in before calibration is applied.
        """
        import numpy as np

        T = camera.optical_pose()
        wxyz = _mat3_to_wxyz(T[:3, :3])
        position = T[:3, 3]
        intrin = camera.intrinsics
        # add_camera_frustum takes the *vertical* FOV in radians.
        self.server.scene.add_frame(
            f"/cameras/{camera.serial}", wxyz=wxyz, position=position,
            axes_length=0.08, axes_radius=0.004,
        )
        self.server.scene.add_camera_frustum(
            f"/cameras/{camera.serial}/frustum",
            fov=2.0 * np.arctan2(intrin.height / 2.0, intrin.fy),
            aspect=intrin.width / intrin.height,
            scale=0.12,
            color=(255, 130, 40),
            wxyz=_IDENTITY_WXYZ,
            position=(0.0, 0.0, 0.0),
        )
        logger.info(
            "camera %s drawn at %s (optical frame)",
            camera.serial, np.round(position, 3).tolist(),
        )

    def sync(self, model, data):
        self.scene.update_from_mjdata(data)

    def attach(self, sim):
        sim.on_step_callbacks.append(self.sync)
        return self

    def close(self):
        self.server.stop()
