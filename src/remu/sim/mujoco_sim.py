"""MuJoCo physics backend driving the FCI emulator.

Mirrors the interface the ``libfranka-sim`` reference implementation exposes
around Genesis (``update_torques`` / ``update_joint_positions`` /
``update_joint_velocities`` / ``get_robot_state``), so the FCI server layer
is physics-engine-agnostic. Any native MJCF actuators on the arm joints are
disabled at load time: control is applied uniformly as a joint torque
(``qfrc_applied``) computed here, so position/velocity/torque control modes
behave the same way regardless of what actuators the source MJCF ships with.
"""

import logging
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import mujoco
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
# Franka "ready" pose (matches the FR3 MJCF's "home" keyframe).
DEFAULT_HOME_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

# Joint impedance gains used for position/velocity control, tuned to be stiff
# but stable at the physics dt below. Approximate the real robot's default
# joint impedance controller; override via MujocoSim(kp=..., kd=...).
DEFAULT_KP = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0])
DEFAULT_KD = np.array([50.0, 50.0, 50.0, 50.0, 15.0, 10.0, 5.0])
DEFAULT_KD_VELOCITY = np.array([20.0, 20.0, 20.0, 20.0, 10.0, 8.0, 5.0])

# FR3 actuator torque limits (J1-4: +/-87 Nm, J5-7: +/-12 Nm).
DEFAULT_TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])


class ControlMode(Enum):
    NONE = "none"
    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"


class MujocoSim:
    """Steps a MuJoCo scene in real time and exposes an FCI-shaped state/command API."""

    def __init__(
        self,
        scene_xml_path,
        joint_names: Optional[Sequence[str]] = None,
        dt: float = 0.001,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        kd_velocity: Optional[np.ndarray] = None,
        torque_limits: Optional[np.ndarray] = None,
        home_q: Optional[np.ndarray] = None,
        realtime: bool = True,
    ):
        self.scene_xml_path = Path(scene_xml_path)
        self.joint_names = list(joint_names) if joint_names else list(DEFAULT_JOINT_NAMES)
        self.dt = dt
        self.kp = np.asarray(kp if kp is not None else DEFAULT_KP, dtype=float)
        self.kd = np.asarray(kd if kd is not None else DEFAULT_KD, dtype=float)
        self.kd_velocity = np.asarray(
            kd_velocity if kd_velocity is not None else DEFAULT_KD_VELOCITY, dtype=float
        )
        self.torque_limits = np.asarray(
            torque_limits if torque_limits is not None else DEFAULT_TORQUE_LIMITS, dtype=float
        )
        self.home_q = np.asarray(home_q if home_q is not None else DEFAULT_HOME_Q, dtype=float)
        self.realtime = realtime

        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None
        self._dof_adr: Optional[np.ndarray] = None
        self._qpos_adr: Optional[np.ndarray] = None
        self._ee_body_id: Optional[int] = None

        self.control_mode = ControlMode.NONE
        self.running = False
        self._lock = threading.Lock()

        # Latest commands from the network thread, published as atomic
        # reference swaps (see update_torques et al.) and read once per
        # physics step -- no lock needed, single writer / single reader.
        self.latest_torques = np.zeros(7)
        self.latest_joint_positions = self.home_q.copy()
        self.latest_joint_velocities = np.zeros(7)

        # State snapshot published each physics step for the network thread.
        self._state_snapshot: Dict[str, np.ndarray] = {
            "q": self.home_q.copy(),
            "dq": np.zeros(7),
            "ddq": np.zeros(7),
            "q_d": self.home_q.copy(),
            "dq_d": np.zeros(7),
            "ddq_d": np.zeros(7),
            "tau_J": np.zeros(7),
            "O_T_EE": np.eye(4).T.flatten(),
        }
        self._prev_dq = np.zeros(7)
        self._ddq_filtered = np.zeros(7)
        self._alpha_acc = 0.95

        # Optional callback invoked once per step (viewer sync hooks).
        self.on_step_callbacks: List = []

    # -- setup ---------------------------------------------------------
    def build(self):
        """Load the model, resolve joint handles, and disable native actuation."""
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_xml_path))
        self.model.opt.timestep = self.dt
        self.data = mujoco.MjData(self.model)

        dof_adr = []
        qpos_adr = []
        for name in self.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint '{name}' not found in scene MJCF")
            dof_adr.append(self.model.jnt_dofadr[jid])
            qpos_adr.append(self.model.jnt_qposadr[jid])
        self._dof_adr = np.array(dof_adr)
        self._qpos_adr = np.array(qpos_adr)

        # Disable any native actuators on the arm so remu's own torque law is
        # the sole source of control (gain/bias zeroed => zero force regardless
        # of ctrl), independent of what actuator types the source MJCF ships.
        self.model.actuator_gainprm[:, :] = 0.0
        self.model.actuator_biasprm[:, :] = 0.0

        flange_name = self.joint_names[-1].replace("joint7", "link7")
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, flange_name)
        self._ee_body_id = body_id if body_id >= 0 else self.model.nbody - 1

        # Try the "home" keyframe if present, else fall back to home_q.
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            self.data.qpos[self._qpos_adr] = self.model.key_qpos[key_id][self._qpos_adr]
            self.home_q = self.data.qpos[self._qpos_adr].copy()
            self.latest_joint_positions = self.home_q.copy()
        else:
            self.data.qpos[self._qpos_adr] = self.home_q

        mujoco.mj_forward(self.model, self.data)
        self._publish_state()
        logger.info("MuJoCo scene built: %s (%d joints)", self.scene_xml_path, len(self.joint_names))

    # -- command interface (mirrors the Genesis backend) ---------------
    def set_control_mode(self, mode: ControlMode):
        if not isinstance(mode, ControlMode):
            raise ValueError(f"Mode must be a ControlMode, got {type(mode)}")
        logger.info("Switching control mode to: %s", mode.value)
        self.control_mode = mode

    def update_torques(self, torques):
        self.latest_torques = np.array(torques, dtype=float)

    def update_joint_positions(self, positions):
        self.latest_joint_positions = np.array(positions, dtype=float)

    def update_joint_velocities(self, velocities):
        self.latest_joint_velocities = np.array(velocities, dtype=float)

    def get_robot_state(self) -> Dict[str, np.ndarray]:
        """Return the latest published state snapshot (lock-free read)."""
        return self._state_snapshot

    # -- physics ---------------------------------------------------------
    def _compute_control_torque(self) -> np.ndarray:
        q = self.data.qpos[self._qpos_adr]
        dq = self.data.qvel[self._dof_adr]
        # Gravity + Coriolis/centrifugal bias for the whole model; take our
        # joints' rows so position/velocity impedance control doesn't sag
        # under gravity, matching the real robot's internal compensation.
        bias = self.data.qfrc_bias[self._dof_adr]

        mode = self.control_mode
        if mode == ControlMode.POSITION:
            tau = self.kp * (self.latest_joint_positions - q) - self.kd * dq + bias
        elif mode == ControlMode.VELOCITY:
            tau = self.kd_velocity * (self.latest_joint_velocities - dq) + bias
        elif mode == ControlMode.TORQUE:
            # Real FCI external-torque control applies the commanded torque
            # directly; the caller is responsible for any compensation.
            tau = self.latest_torques.copy()
        else:  # NONE: hold the last commanded/measured position
            tau = self.kp * (self.latest_joint_positions - q) - self.kd * dq + bias

        return np.clip(tau, -self.torque_limits, self.torque_limits)

    def _publish_state(self):
        q = self.data.qpos[self._qpos_adr].copy()
        dq = self.data.qvel[self._dof_adr].copy()

        ddq_raw = (dq - self._prev_dq) / self.dt
        self._ddq_filtered = self._alpha_acc * self._ddq_filtered + (1 - self._alpha_acc) * ddq_raw
        self._prev_dq = dq.copy()

        ee_pos = self.data.xpos[self._ee_body_id].copy()
        ee_mat = self.data.xmat[self._ee_body_id].reshape(3, 3).copy()
        O_T_EE = np.eye(4)
        O_T_EE[:3, :3] = ee_mat
        O_T_EE[:3, 3] = ee_pos

        applied_tau = self.data.qfrc_applied[self._dof_adr].copy()

        self._state_snapshot = {
            "q": q,
            "dq": dq,
            "ddq": self._ddq_filtered.copy(),
            "q_d": self.latest_joint_positions.copy(),
            "dq_d": dq.copy(),
            "ddq_d": self._ddq_filtered.copy(),
            "tau_J": applied_tau,
            "O_T_EE": O_T_EE.T.flatten(),  # column-major, per libfranka convention
        }

    def step(self):
        """Advance physics by one dt: compute control torque, step, publish state."""
        tau = self._compute_control_torque()
        self.data.qfrc_applied[self._dof_adr] = tau
        mujoco.mj_step(self.model, self.data)
        self._publish_state()
        for cb in self.on_step_callbacks:
            cb(self.model, self.data)

    def run(self):
        """Run the physics loop, paced to wall-clock realtime, until stop()."""
        self.running = True
        next_step = time.perf_counter()
        while self.running:
            self.step()
            if self.realtime:
                next_step += self.dt
                slack = next_step - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                elif slack < -self.dt:
                    next_step = time.perf_counter()

    def stop(self):
        self.running = False
