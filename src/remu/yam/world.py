"""Compose both YAMs, movable objects and cameras into one MuJoCo world."""
from pathlib import Path
import threading
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .sim import DEFAULT_URDF, YamSim, load_yam_spec


def _pose(value):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError('mount position must contain three finite coordinates')
    return result


def _quaternion(value):
    result = np.asarray(value, dtype=float)
    if result.shape != (4,) or not np.all(np.isfinite(result)) or np.linalg.norm(result) < 1e-10:
        raise ValueError('mount quaternion must contain four finite values with nonzero norm')
    return result / np.linalg.norm(result)


def _external_spec(path):
    path = Path(path).resolve()
    spec = mujoco.MjSpec.from_file(str(path))
    for items, directory in ((spec.meshes, spec.meshdir), (spec.textures, spec.texturedir)):
        for item in items:
            if item.file:
                value = Path(item.file)
                if not value.is_absolute():
                    value = path.parent / directory / value
                item.file = str(value.resolve())
    spec.meshdir = spec.texturedir = ''
    return spec


class YamWorld:
    def __init__(self, urdf=DEFAULT_URDF, *, dt=.001, left_position=(0., -.35, 0.),
                 right_position=(0., .35, 0.), left_quaternion=(1., 0., 0., 0.),
                 right_quaternion=(1., 0., 0., 0.), scene=None, objects=(), cameras=(),
                 limits=None, watchdog=.5, bimanual=True):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be finite and positive')
        spec = mujoco.MjSpec()
        spec.option.timestep = dt
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.visual.global_.offwidth = max([640, *[c.width for c in cameras]])
        spec.visual.global_.offheight = max([480, *[c.height for c in cameras]])
        if scene is None:
            spec.worldbody.add_geom(name='remu_floor', type=mujoco.mjtGeom.mjGEOM_PLANE,
                size=[0., 0., .05], rgba=[.22, .25, .28, 1.], contype=1, conaffinity=6)
        spec.worldbody.add_light(pos=[0., -1., 2.], dir=[0., .5, -1.])
        mounts = (
                ('left', left_position, left_quaternion, 2, 4),
                ('right', right_position, right_quaternion, 4, 2))
        if not bimanual:
            mounts = (mounts[0],)
        for name, pos, quat, bit, other in mounts:
            arm = load_yam_spec(urdf, dt, add_floor=False)
            for geom in arm.geoms:
                if geom.contype or geom.conaffinity:
                    geom.contype, geom.conaffinity = bit, 1 | other
            frame = spec.worldbody.add_frame(name=name + '_mount', pos=_pose(pos), quat=_quaternion(quat))
            spec.attach(arm, prefix=name + '_', frame=frame)
        environment = ([scene] if scene else []) + list(objects)
        for i, path in enumerate(environment):
            obj = _external_spec(path)
            for geom in obj.geoms:
                if geom.contype or geom.conaffinity:
                    geom.contype, geom.conaffinity = 1, 7
            spec.attach(obj, prefix=f'object{i}_', frame=spec.worldbody.add_frame())
        for camera in cameras:
            # Reuse the exact camera convention, intrinsics, and markers used
            # by the Franka scene, including the RealSense optical transform.
            root = ET.fromstring(camera.mjcf_body())
            body = spec.worldbody.add_body(name=root.get('name'), pos=camera.eye)
            body.add_geom(name=camera.mjcf_camera_name + '_marker', type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[.045, .0125, .0125], rgba=[.1, .1, .1, 1.], contype=0, conaffinity=0)
            from remu.camera.d435i import look_at_axes
            body.add_camera(name=camera.mjcf_camera_name, fovy=camera.fovy_deg,
                xyaxes=look_at_axes(camera.eye, camera.target, camera.up).reshape(-1))
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.lock = threading.RLock()
        self.left = YamSim(urdf, model=self.model, data=self.data, namespace='left_', lock=self.lock,
                           limits=limits, watchdog=watchdog)
        self.right = (YamSim(urdf, model=self.model, data=self.data, namespace='right_', lock=self.lock,
                            limits=limits, watchdog=watchdog) if bimanual else None)
        self.arms = (self.left, self.right) if bimanual else (self.left,)
        self.on_step_callbacks = []
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        with self.lock:
            for arm in self.arms:
                arm._prepare_step()
            mujoco.mj_step(self.model, self.data)
            for arm in self.arms:
                arm._after_step()
            for callback in self.on_step_callbacks:
                callback(self.model, self.data)
