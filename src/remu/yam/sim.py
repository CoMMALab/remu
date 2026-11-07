"""Six-axis YAM + parallel gripper, loaded from the supplied URDF.

The public command vector is six joint angles followed by normalized gripper
opening (0 closed, 1 open), as in i2rt's MotorChainRobot. The two physical
finger slides are driven together and exposed as one logical DOF.
"""

from copy import deepcopy
from pathlib import Path
import threading
import time

from .limits import YamLimits, JointTrajectory
from .gripper import GripperMotion
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


DEFAULT_URDF = Path(__file__).resolve().parents[1] / "models" / "yam" / "yam.urdf"
# i2rt/robots/config/yam_v1.yml (arm only).
ARM_KP = np.array([80., 80., 80., 10., 10., 10.])
ARM_KD = np.array([5., 5., 5., 1.5, 1.5, 1.5])


def _resolve_mesh(filename, urdf_path):
    if filename.startswith("package://"):
        package, separator, relative = filename[len("package://"):].partition("/")
        if not separator:
            raise ValueError(f"Invalid URDF mesh URI: {filename}")
        candidates = [urdf_path.parent / relative,
                      urdf_path.parent / package / relative,
                      urdf_path.parent.parent / package / relative]
        # The user supplied XML, but no mesh package. Reuse the bundled i2rt
        # YAM meshes with explicit aliases for the original export's capitals.
        if urdf_path == DEFAULT_URDF.resolve() and package == "urdf_top_assembly":
            aliases = {"Base.stl": "base.stl", "Gripper.stl": "gripper.stl"}
            name = Path(relative).name
            if relative == f"meshes/{name}":
                candidates.append(urdf_path.parent / "assets" / aliases.get(name, name))
    else:
        candidates = [urdf_path.parent / filename]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot resolve URDF mesh {filename!r} relative to {urdf_path}")


def load_yam_spec(urdf_path=DEFAULT_URDF, dt=0.001, *, add_floor=True):
    """Import a YAM URDF with its meshes, inertias, limits, and finger joints.

    The supplied URDF contains visual meshes only. Copy those into collision geometry;
    MuJoCo uses convex mesh hulls, so contact geometry is an approximation.
    Nonadjacent self-collision is disabled to avoid convex-hull intersections.
    """
    path = Path(urdf_path).resolve()
    root = ET.parse(path).getroot()
    # A zero-range revolute joint is an Onshape mounting transform, not an
    # actuated DOF. MuJoCo requires limited hinges to have nonzero travel.
    mount = root.find("joint[@name='dof_joint0']")
    if mount is not None and mount.get("type") == "revolute":
        limit = mount.find("limit")
        if limit is not None and float(limit.get("lower")) == float(limit.get("upper")) == 0:
            mount.set("type", "fixed")
            for tag in ("axis", "limit"):
                element = mount.find(tag)
                if element is not None:
                    mount.remove(element)
    extension = root.find("mujoco")
    if extension is None:
        extension = ET.SubElement(root, "mujoco")
    compiler = extension.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(extension, "compiler")
    compiler.attrib.update(discardvisual="false", fusestatic="false", strippath="false")
    for link in root.findall("link"):
        if link.find("collision") is None:
            for visual in link.findall("visual"):
                collision = deepcopy(visual)
                collision.tag = "collision"
                material = collision.find("material")
                if material is not None:
                    collision.remove(material)
                link.append(collision)
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename", "")
        mesh.set("filename", str(_resolve_mesh(filename, path)))
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    spec.option.timestep = dt
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:
            # Robot geoms collide with the floor/objects, but not each other.
            geom.contype = 2
            geom.conaffinity = 1
    if add_floor:
        spec.worldbody.add_geom(
            name="remu_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0, 0, .05], rgba=[.22, .25, .28, 1], contype=1, conaffinity=2,
        )
        spec.worldbody.add_light(pos=[0, -1, 2], dir=[0, .5, -1], diffuse=[.8, .8, .8])
    # Affine joint actuators expose PD velocity derivatives to IMPLICITFAST.
    # Applying -kd*qvel as an external force is explicit and destabilizes the
    # low-inertia wrist when the fingers are partially open.
    for joint in spec.joints:
        if joint.name in {f"{prefix}{i}" for prefix in ("dof_joint", "joint") for i in range(1, 9)}:
            spec.add_actuator(
                name=f"remu_{joint.name}", target=joint.name,
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
                gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                gainprm=[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                forcelimited=True, forcerange=[-20., 20.],
            )
    return spec


def load_yam_model(urdf_path=DEFAULT_URDF, dt=.001):
    return load_yam_spec(urdf_path, dt).compile()


class YamSim:
    """Thread-safe i2rt robot API backed by a manually stepped MuJoCo model."""

    def __init__(self, urdf_path=DEFAULT_URDF, dt=0.001, *, model=None, data=None,
                 namespace='', lock=None, limits=None, watchdog=0.):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        self.model = model if model is not None else load_yam_model(urdf_path, dt)
        self.data = data if data is not None else mujoco.MjData(self.model)
        self.urdf_path = Path(urdf_path).resolve()
        self.namespace = namespace
        self.dt = float(self.model.opt.timestep)
        self.on_step_callbacks = []
        self._owns_physics = data is None
        names = {self.model.joint(i).name for i in range(self.model.njnt)}
        for prefix in ("dof_joint", "joint"):
            joint_names = tuple(f"{namespace}{prefix}{i}" for i in range(1, 9))
            if set(joint_names) <= names:
                break
        else:
            raise ValueError("YAM URDF must expose dof_joint1–8 or joint1–8")
        self.joint_names = joint_names
        joints = [self.model.joint(name) for name in joint_names]
        self._qadr = np.array([j.qposadr[0] for j in joints])
        self._vadr = np.array([j.dofadr[0] for j in joints])
        self._ranges = np.array([j.range for j in joints])
        if (not np.all(np.isfinite(self._ranges))
                or np.any(self._ranges[:, 1] <= self._ranges[:, 0])):
            raise ValueError("YAM joints must have finite nonzero position ranges")
        self._lock = lock if lock is not None else threading.RLock()
        self._kp = np.r_[ARM_KP, 30.]
        self._kd = np.r_[ARM_KD, 1.]
        # Simulation caps, matching the upstream YAM MJCF arm effort range.
        # URDF effort=1 is an export placeholder, not a calibrated motor limit.
        self._force_limits = np.r_[np.full(6, 10.), 20., 20.]
        self._actadr = np.array([self.model.actuator(f"{namespace}remu_{name[len(namespace):]}").id for name in joint_names])
        self.model.actuator_forcerange[self._actadr] = np.column_stack(
            (-self._force_limits, self._force_limits))
        # The export's effort=1 placeholders otherwise impose an additional
        # joint-level actuator clamp that defeats gravity compensation.
        self.model.jnt_actfrcrange[[joint.id for joint in joints]] = np.column_stack(
            (-self._force_limits, self._force_limits))
        self._target = np.r_[np.clip([0, 1.0, 1.5, 0, 0, 0],
                                    self._ranges[:6, 0], self._ranges[:6, 1]), 1.]
        self._velocity = np.zeros(7)
        self._effort = np.zeros(8)
        self.data.qpos[self._qadr] = self._physical_pos(self._target)
        mujoco.mj_forward(self.model, self.data)
        self._finger_span = np.diff(self._ranges[6:], axis=1)[:, 0]
        self.limits = (YamLimits.load(limits, self._finger_span.sum()) if isinstance(limits, (str, Path))
                       else limits or YamLimits.defaults(self._finger_span.sum()))
        ranges = np.vstack([self._ranges[:6], [0., 1.]])
        self._trajectory = JointTrajectory(self._target, ranges, self.limits)
        self._mode = 'position'
        self._torque = np.zeros(7)
        self._last_command = time.monotonic()
        if not np.isfinite(watchdog) or watchdog < 0:
            raise ValueError('watchdog must be finite and nonnegative')
        self.watchdog = watchdog
        self._watchdog_expired = False
        self._ee_body = self.model.body(namespace + 'gripper').id
        self._finger_bodies = [self.model.body(namespace + name).id for name in ('tip_left', 'tip_right')]
        self._robot_bodies = set()
        for i in range(1, self.model.nbody):
            parent = i
            while parent:
                if parent == self._ee_body or self.model.body(parent).name == namespace + 'root':
                    self._robot_bodies.add(i)
                    break
                parent = int(self.model.body_parentid[parent])
        self.gripper = GripperMotion(self)
        self._force_limits = np.r_[self.limits.effort[:6], np.full(2, self.limits.effort[6] / self._finger_span.sum())]
        self.model.jnt_actfrcrange[[joint.id for joint in joints]] = np.column_stack((-self._force_limits, self._force_limits))

    def num_dofs(self):
        return 7

    @staticmethod
    def _vector(value, name):
        array = np.asarray(value, dtype=float)
        if array.shape != (7,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain seven finite values (six arm joints + gripper)")
        return array.copy()

    def _physical_pos(self, command):
        fingers = self._ranges[6:, 0] + command[6] * np.diff(self._ranges[6:], axis=1)[:, 0]
        return np.r_[command[:6], fingers]

    def _touch_command(self):
        self._last_command = time.monotonic()
        self._watchdog_expired = False

    def keepalive(self):
        with self._lock:
            self._touch_command()

    def command_joint_pos(self, joint_pos):
        self.command_joint_state({"pos": joint_pos})

    def command_joint_vel(self, velocity):
        self.command_joint_state({'mode': 'velocity', 'vel': velocity})

    def command_joint_torque(self, effort):
        self.command_joint_state({'mode': 'torque', 'eff': effort})

    def command_joint_state(self, joint_state):
        mode = joint_state.get('mode', 'position')
        if mode not in ('position', 'velocity', 'torque', 'impedance'):
            raise ValueError('mode must be position, velocity, torque, or impedance')
        key = 'position' if 'position' in joint_state else 'pos'
        if mode in ('position', 'impedance') and key not in joint_state:
            raise ValueError('position mode requires pos')
        if mode == 'velocity' and 'vel' not in joint_state:
            raise ValueError('velocity mode requires vel')
        if mode == 'torque' and 'eff' not in joint_state:
            raise ValueError('torque mode requires eff')
        pos = self._vector(joint_state.get(key, self._target), 'pos')
        vel = self._vector(joint_state.get('vel', np.zeros(7)), 'vel')
        effort = self._vector(joint_state.get('eff', np.zeros(7)), 'eff')
        kp = self._vector(joint_state['kp'], 'kp') if 'kp' in joint_state else None
        kd = self._vector(joint_state['kd'], 'kd') if 'kd' in joint_state else None
        if any(g is not None and np.any(g < 0) for g in (kp, kd)):
            raise ValueError('kp and kd must be nonnegative')
        pos = np.clip(pos, self._trajectory.ranges[:, 0], self._trajectory.ranges[:, 1])
        with self._lock:
            if mode != self._mode:
                self._trajectory.reset(self._positions())
            self._mode = mode
            self._target, self._velocity = pos, np.clip(vel, -self.limits.velocity, self.limits.velocity)
            self._torque = np.clip(effort, -self.limits.effort, self.limits.effort)
            if kp is not None:
                self._kp = kp
            if kd is not None:
                self._kd = kd
            self.gripper.active = False
            self.gripper.status = 'idle'
            self._touch_command()

    def stop(self):
        with self._lock:
            self._target = self._positions()
            self._trajectory.reset(self._target)
            self._velocity.fill(0.)
            self._torque.fill(0.)
            self._mode = 'hold'
            self.gripper.stop()
            self._touch_command()

    def gripper_move(self, opening, speed=1., force=20., timeout=5.):
        with self._lock:
            return self.gripper.start('move', opening, speed, force, timeout=timeout)

    def gripper_grasp(self, opening, speed=1., force=20., epsilon_inner=.02, epsilon_outer=.02, timeout=5.):
        with self._lock:
            return self.gripper.start('grasp', opening, speed, force, epsilon_inner, epsilon_outer, timeout)

    def gripper_home(self):
        with self._lock:
            return self.gripper.start('home', 1.)

    def gripper_stop(self):
        with self._lock:
            self._touch_command()
            return self.gripper.stop()

    def get_gripper_state(self):
        with self._lock:
            return self.gripper.state()

    def _logical_velocity(self):
        velocity = self.data.qvel[self._vadr]
        return np.r_[velocity[:6], np.mean(velocity[6:] / self._finger_span)]

    def _prepare_step(self):
        if self.watchdog and time.monotonic() - self._last_command > self.watchdog:
            if not self._watchdog_expired:
                self.stop()
                self._watchdog_expired = True
        # stop() touches the clock; keep expiry latched until a new command.
        if self._mode != 'torque':
            self._trajectory.advance(self._mode, self._target, self._velocity, self.dt,
                gripper_speed=self.gripper.speed if self.gripper.active else None)
        target = self._physical_pos(self._trajectory.position)
        logical_velocity = self._trajectory.velocity.copy()
        if self._mode in ('position', 'impedance'):
            logical_velocity = np.clip(logical_velocity + self._velocity, -self.limits.velocity, self.limits.velocity)
        velocity = np.r_[logical_velocity[:6], logical_velocity[6] * self._finger_span]
        kp = np.r_[self._kp[:6], np.full(2, self._kp[6] * 100.)]
        kd = np.r_[self._kd[:6], np.full(2, self._kd[6])]
        if self._mode == 'velocity':
            kp[:6] = 0.
        elif self._mode == 'torque':
            kp.fill(0.)
            kd.fill(0.)
        effort = np.r_[self._torque[:6], self._torque[6] * self._finger_span / (self._finger_span @ self._finger_span)]
        control = kp * target + kd * velocity + effort
        if self._mode != 'torque':
            control += self.data.qfrc_bias[self._vadr]
        self.model.actuator_biasprm[self._actadr, 1] = -kp
        self.model.actuator_biasprm[self._actadr, 2] = -kd
        self.data.ctrl[self._actadr] = control
        finger_force = self.limits.effort[6] / self._finger_span.sum()
        if self.gripper.active or self.gripper.status == 'grasped':
            finger_force = min(finger_force, self.gripper.force)
        caps = np.r_[self.limits.effort[:6], np.full(2, finger_force)]
        rates = np.r_[self.limits.effort_rate[:6], np.full(2, self.limits.effort_rate[6] / self._finger_span.sum())]
        low = np.maximum(-caps, self._effort - rates * self.dt)
        high = np.minimum(caps, self._effort + rates * self.dt)
        # New force caps take precedence if an operation reduced its limit.
        low = np.minimum(low, high)
        self.model.actuator_forcerange[self._actadr] = np.column_stack((low, high))

    def _after_step(self):
        self._effort = self.data.actuator_force[self._actadr].copy()
        self.gripper.update()

    def step(self):
        if not self._owns_physics:
            raise RuntimeError('step the shared YamWorld, not an individual arm')
        with self._lock:
            self._prepare_step()
            mujoco.mj_step(self.model, self.data)
            self._after_step()
            for callback in self.on_step_callbacks:
                callback(self.model, self.data)

    def _positions(self):
        q = self.data.qpos[self._qadr]
        opening = (q[6:] - self._ranges[6:, 0]) / np.diff(self._ranges[6:], axis=1)[:, 0]
        return np.r_[q[:6], np.clip(np.mean(opening), 0, 1)]

    def get_joint_pos(self):
        with self._lock:
            return self._positions()

    def get_observations(self):
        with self._lock:
            pos = self._positions()
            vel = self.data.qvel[self._vadr].copy()
            span = np.diff(self._ranges[6:], axis=1)[:, 0]
            # Generalized effort conjugate to the normalized opening coordinate.
            return {
                "joint_pos": pos[:6], "joint_vel": vel[:6],
                "joint_eff": self._effort[:6].copy(),
                "gripper_pos": pos[6:],
                "gripper_vel": np.array([np.mean(vel[6:] / span)]),
                "gripper_eff": np.array([self._effort[6:] @ span]),
                "desired_pos": self._trajectory.position.copy(),
                "desired_vel": self._trajectory.velocity.copy(),
                "desired_acc": self._trajectory.acceleration.copy(),
                "desired_eff": np.r_[self._effort[:6], self._effort[6:] @ span],
                "requested_eff": self._torque.copy(),
                "ee_pose": self.get_end_effector_pose(),
                "control_mode": self._mode,
                "watchdog_expired": self._watchdog_expired,
                "contact": self._has_external_contact(),
                "gripper_state": self.gripper.state(),
                "simulation_time": float(self.data.time),
            }

    def get_robot_model(self):
        return self.urdf_path.read_text()

    def get_dynamics(self):
        """Logical seven-DOF mass, bias and world-frame geometric Jacobian."""
        with self._lock:
            full = np.zeros((self.model.nv, self.model.nv))
            mujoco.mj_fullM(self.model, full, self.data.qM)
            mapping = np.zeros((8, 7))
            mapping[:6, :6] = np.eye(6)
            mapping[6:, 6] = self._finger_span
            jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jp, jr, self._ee_body)
            return dict(mass=mapping.T @ full[np.ix_(self._vadr, self._vadr)] @ mapping,
                bias=mapping.T @ self.data.qfrc_bias[self._vadr],
                jacobian=np.vstack([jp[:, self._vadr], jr[:, self._vadr]]) @ mapping,
                pose=self.get_end_effector_pose(), coordinates='urdf', frame='world')

    def get_end_effector_pose(self):
        with self._lock:
            pose = np.eye(4)
            pose[:3, :3] = self.data.xmat[self._ee_body].reshape(3, 3)
            pose[:3, 3] = self.data.xpos[self._ee_body]
            return pose

    def _has_external_contact(self):
        for contact in self.data.contact[:self.data.ncon]:
            a, b = (int(self.model.geom_bodyid[g]) in self._robot_bodies for g in (contact.geom1, contact.geom2))
            if a != b:
                return True
        return False

    def get_robot_info(self):
        with self._lock:
            return {
                "arm_type": "yam", "gripper_type": "parallel_urdf",
                "kp": self._kp.copy(), "kd": self._kd.copy(),
                "joint_limits": self._ranges[:6].copy(),
                "gripper_limits": np.array([0., 1.]), "gripper_index": 6,
                "joint_names": self.joint_names,
                "joint_coordinates": "urdf",
                "simulated": True,
                "control_modes": ['position', 'velocity', 'torque', 'impedance'],
                "gripper_operations": ['move', 'stop', 'home', 'grasp'],
                "trajectory_limits": self.limits.asdict(),
                "watchdog": self.watchdog,
                "ee_frame": self.namespace + 'gripper',
                "pose_frame": 'world',
                "model_queries": True,
                "limit_provenance": 'simulation tuning; not certified hardware limits',
            }
