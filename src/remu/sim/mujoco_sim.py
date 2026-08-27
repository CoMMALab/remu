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

# Franka Research 3 command limits.  These are kept here (rather than read
# from the MJCF) because acceleration, jerk, torque-rate and the asymmetric
# position-dependent velocity envelope are FCI properties that MJCF cannot
# represent.  Values are from Franka's "Control Interface Specification and
# Robot Limits" for FR3.
FR3_Q_MIN = np.array([-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508])
FR3_Q_MAX = np.array([2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508])
FR3_DQ_MAX = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])
FR3_DDQ_MAX = np.full(7, 10.0)
FR3_DDDQ_MAX = np.full(7, 5000.0)
FR3_DQ_OFFSET = np.array([0.6599, 0.2517, 0.2000, 0.3533, 0.5757, 0.4878, 0.4628])
FR3_DDQ_DEC = np.array([6.0, 2.585, 3.5, 4.0, 17.0, 5.5, 17.0])
FR3_DTAU_MAX = np.full(7, 1000.0)


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

        # Constrained command state. Network commands are targets; these are
        # advanced once per physics step so the simulated FCI never teleports
        # a setpoint through velocity/acceleration/jerk limits.
        self._command_q = self.home_q.copy()
        self._command_dq = np.zeros(7)
        self._command_ddq = np.zeros(7)
        self._command_tau = np.zeros(7)

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

        self._command_q = np.clip(self.home_q, FR3_Q_MIN, FR3_Q_MAX)
        self._command_dq.fill(0.0)
        self._command_ddq.fill(0.0)
        self._command_tau.fill(0.0)

        mujoco.mj_forward(self.model, self.data)
        self._publish_state()
        logger.info("MuJoCo scene built: %s (%d joints)", self.scene_xml_path, len(self.joint_names))

    # -- command interface (mirrors the Genesis backend) ---------------
    def set_control_mode(self, mode: ControlMode):
        if not isinstance(mode, ControlMode):
            raise ValueError(f"Mode must be a ControlMode, got {type(mode)}")
        if mode != self.control_mode and self.data is not None:
            # Start every motion generator from the current commanded pose,
            # as the real FCI requires, without carrying derivatives between
            # incompatible controller modes.
            q = self.data.qpos[self._qpos_adr]
            self._command_q = np.clip(q.copy(), FR3_Q_MIN, FR3_Q_MAX)
            self._command_dq.fill(0.0)
            self._command_ddq.fill(0.0)
        logger.info("Switching control mode to: %s", mode.value)
        self.control_mode = mode

    def update_torques(self, torques):
        self.latest_torques = self._validated_command(torques, "torques")

    def update_joint_positions(self, positions):
        positions = self._validated_command(positions, "joint positions")
        self.latest_joint_positions = np.clip(positions, FR3_Q_MIN, FR3_Q_MAX)

    def update_joint_velocities(self, velocities):
        self.latest_joint_velocities = self._validated_command(velocities, "joint velocities")

    @staticmethod
    def _validated_command(values, name: str) -> np.ndarray:
        command = np.asarray(values, dtype=float)
        if command.shape != (7,):
            raise ValueError(f"{name} must contain exactly 7 values")
        if not np.all(np.isfinite(command)):
            raise ValueError(f"{name} must contain only finite values")
        return command.copy()

    def get_robot_state(self) -> Dict[str, np.ndarray]:
        """Return the latest published state snapshot (lock-free read)."""
        return self._state_snapshot

    # -- physics ---------------------------------------------------------
    @staticmethod
    def _position_velocity_limits(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return FR3 lower/upper velocity bounds at joint positions ``q``."""
        upper = np.minimum(
            FR3_DQ_MAX,
            np.maximum(
                0.0,
                -FR3_DQ_OFFSET
                + np.sqrt(np.maximum(0.0, 2.0 * FR3_DDQ_DEC * (FR3_Q_MAX - q))),
            ),
        )
        lower = np.maximum(
            -FR3_DQ_MAX,
            np.minimum(
                0.0,
                FR3_DQ_OFFSET
                - np.sqrt(np.maximum(0.0, 2.0 * FR3_DDQ_DEC * (q - FR3_Q_MIN))),
            ),
        )
        return lower, upper

    def _advance_joint_command(self, target_dq: np.ndarray) -> None:
        """Advance the desired trajectory by one jerk-limited timestep."""
        lower_dq, upper_dq = self._position_velocity_limits(self._command_q)
        target_dq = np.clip(target_dq, lower_dq, upper_dq)

        target_ddq = np.clip(
            (target_dq - self._command_dq) / self.dt, -FR3_DDQ_MAX, FR3_DDQ_MAX
        )
        max_ddq_step = FR3_DDDQ_MAX * self.dt
        self._command_ddq += np.clip(
            target_ddq - self._command_ddq, -max_ddq_step, max_ddq_step
        )
        self._command_ddq = np.clip(self._command_ddq, -FR3_DDQ_MAX, FR3_DDQ_MAX)

        next_dq = self._command_dq + self._command_ddq * self.dt
        next_dq = np.clip(next_dq, lower_dq, upper_dq)

        # The envelope tightens as the next position approaches a stop. A few
        # fixed-point projections make the *resulting* (q, dq) pair satisfy
        # Franka's equation too, instead of only checking velocity against the
        # position from the beginning of the sample.
        for _ in range(3):
            next_q = np.clip(self._command_q + next_dq * self.dt, FR3_Q_MIN, FR3_Q_MAX)
            next_lower_dq, next_upper_dq = self._position_velocity_limits(next_q)
            next_dq = np.clip(next_dq, next_lower_dq, next_upper_dq)
        self._command_dq = next_dq
        self._command_q = np.clip(
            self._command_q + self._command_dq * self.dt, FR3_Q_MIN, FR3_Q_MAX
        )
        final_lower_dq, final_upper_dq = self._position_velocity_limits(self._command_q)
        self._command_dq = np.clip(self._command_dq, final_lower_dq, final_upper_dq)

        # Do not retain acceleration pushing into a hard position stop.
        stopped = ((self._command_q <= FR3_Q_MIN) & (self._command_dq <= 0.0)) | (
            (self._command_q >= FR3_Q_MAX) & (self._command_dq >= 0.0)
        )
        self._command_dq[stopped] = 0.0
        self._command_ddq[stopped] = 0.0

    def _advance_position_command(self) -> None:
        error = self.latest_joint_positions - self._command_q
        lower_dq, upper_dq = self._position_velocity_limits(self._command_q)
        target_dq = np.clip(error / self.dt, lower_dq, upper_dq)

        # Avoid integrating past a close target. This preserves all dynamic
        # limits while allowing the generator to settle exactly at its goal.
        stopping_speed = np.sqrt(2.0 * FR3_DDQ_MAX * np.abs(error))
        target_dq = np.sign(error) * np.minimum(np.abs(target_dq), stopping_speed)
        self._advance_joint_command(target_dq)

        crossed = (self.latest_joint_positions - self._command_q) * error <= 0.0
        settled = crossed & (np.abs(error) < 1e-4)
        self._command_q[settled] = self.latest_joint_positions[settled]
        self._command_dq[settled] = 0.0
        self._command_ddq[settled] = 0.0

    def _compute_control_torque(self) -> np.ndarray:
        q = self.data.qpos[self._qpos_adr]
        dq = self.data.qvel[self._dof_adr]
        # Gravity + Coriolis/centrifugal bias for the whole model; take our
        # joints' rows so position/velocity impedance control doesn't sag
        # under gravity, matching the real robot's internal compensation.
        bias = self.data.qfrc_bias[self._dof_adr]

        mode = self.control_mode
        if mode == ControlMode.POSITION:
            self._advance_position_command()
            tau = self.kp * (self._command_q - q) + self.kd * (self._command_dq - dq) + bias
        elif mode == ControlMode.VELOCITY:
            lower_dq, upper_dq = self._position_velocity_limits(self._command_q)
            target_dq = np.clip(self.latest_joint_velocities, lower_dq, upper_dq)
            self._advance_joint_command(target_dq)
            tau = self.kd_velocity * (self._command_dq - dq) + bias
        elif mode == ControlMode.TORQUE:
            # Real FCI external-torque control applies the commanded torque
            # directly; the caller is responsible for any compensation.
            tau = np.clip(self.latest_torques, -self.torque_limits, self.torque_limits)
        else:  # NONE: hold the last commanded/measured position
            tau = self.kp * (self._command_q - q) - self.kd * dq + bias

        tau = np.clip(tau, -self.torque_limits, self.torque_limits)
        max_tau_step = FR3_DTAU_MAX * self.dt
        self._command_tau += np.clip(tau - self._command_tau, -max_tau_step, max_tau_step)
        return self._command_tau.copy()

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
            "q_d": self._command_q.copy(),
            "dq_d": self._command_dq.copy(),
            "ddq_d": self._command_ddq.copy(),
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
