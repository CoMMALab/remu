import numpy as np
import pytest

import remu.sim.mujoco_sim as sim_module
from remu.sim.mujoco_sim import (
    FR3_DDQ_MAX,
    FR3_DDDQ_MAX,
    FR3_DQ_MAX,
    FR3_DTAU_MAX,
    FR3_Q_MAX,
    FR3_Q_MIN,
    ControlMode,
)


def test_build_publishes_initial_state_near_home(sim):
    state = sim.get_robot_state()
    assert np.allclose(state["q"], sim.home_q, atol=1e-6)
    assert np.allclose(state["dq"], 0.0, atol=1e-6)


def test_none_mode_holds_position_under_gravity(sim):
    q0 = sim.get_robot_state()["q"].copy()
    for _ in range(200):
        sim.step()
    q1 = sim.get_robot_state()["q"]
    assert np.allclose(q0, q1, atol=0.05)


def test_position_control_tracks_target(sim):
    target = sim.home_q + np.array([0.2, 0, 0, 0, 0, 0, 0])
    sim.update_joint_positions(target)
    sim.set_control_mode(ControlMode.POSITION)
    for _ in range(2000):
        sim.step()
    q = sim.get_robot_state()["q"]
    assert abs(q[0] - target[0]) < 0.02


def test_velocity_control_moves_joint(sim):
    sim.update_joint_velocities(np.array([0.3, 0, 0, 0, 0, 0, 0]))
    sim.set_control_mode(ControlMode.VELOCITY)
    q0 = sim.get_robot_state()["q"].copy()
    for _ in range(500):
        sim.step()
    q1 = sim.get_robot_state()["q"]
    assert q1[0] > q0[0] + 0.05


def test_torque_control_applies_raw_torque(sim):
    sim.update_torques(np.array([5.0, 0, 0, 0, 0, 0, 0]))
    sim.set_control_mode(ControlMode.TORQUE)
    for _ in range(5):
        sim.step()
    state = sim.get_robot_state()
    assert np.isclose(state["tau_J"][0], 5.0)


def test_torque_clipped_to_limits(sim):
    sim.update_torques(np.array([1000.0, 0, 0, 0, 0, 0, 0]))
    sim.set_control_mode(ControlMode.TORQUE)
    sim.step()
    state = sim.get_robot_state()
    assert state["tau_J"][0] <= sim.torque_limits[0] + 1e-6


def test_torque_command_is_slew_rate_limited_each_step(sim):
    sim.update_torques(np.full(7, 1000.0))
    sim.set_control_mode(ControlMode.TORQUE)
    previous = sim.get_robot_state()["tau_J"].copy()

    for _ in range(20):
        sim.step()
        current = sim.get_robot_state()["tau_J"]
        assert np.all(np.abs(current - previous) <= FR3_DTAU_MAX * sim.dt + 1e-9)
        previous = current.copy()


def test_position_command_obeys_joint_derivative_limits(sim):
    sim.update_joint_positions(FR3_Q_MAX + 1.0)
    sim.set_control_mode(ControlMode.POSITION)
    previous_ddq = sim.get_robot_state()["ddq_d"].copy()

    for _ in range(250):
        sim.step()
        state = sim.get_robot_state()
        assert np.all(state["q_d"] >= FR3_Q_MIN - 1e-12)
        assert np.all(state["q_d"] <= FR3_Q_MAX + 1e-12)
        assert np.all(np.abs(state["dq_d"]) <= FR3_DQ_MAX + 1e-12)
        lower_dq, upper_dq = sim._position_velocity_limits(state["q_d"])
        assert np.all(state["dq_d"] >= lower_dq - 1e-12)
        assert np.all(state["dq_d"] <= upper_dq + 1e-12)
        assert np.all(np.abs(state["ddq_d"]) <= FR3_DDQ_MAX + 1e-12)
        assert np.all(
            np.abs(state["ddq_d"] - previous_ddq) <= FR3_DDDQ_MAX * sim.dt + 1e-9
        )
        previous_ddq = state["ddq_d"].copy()


def test_position_based_velocity_limits_match_franka_equations(sim):
    q = FR3_Q_MIN + np.array([0.0, 0.05, 0.2, 0.5, 1.0, 2.0, 3.0]) / 3.0 * (
        FR3_Q_MAX - FR3_Q_MIN
    )
    lower, upper = sim._position_velocity_limits(q)

    expected_upper = np.minimum(
        FR3_DQ_MAX,
        np.maximum(
            0.0,
            -sim_module.FR3_DQ_OFFSET
            + np.sqrt(
                np.maximum(
                    0.0, 2.0 * sim_module.FR3_DDQ_DEC * (FR3_Q_MAX - q)
                )
            ),
        ),
    )
    expected_lower = np.maximum(
        -FR3_DQ_MAX,
        np.minimum(
            0.0,
            sim_module.FR3_DQ_OFFSET
            - np.sqrt(
                np.maximum(
                    0.0, 2.0 * sim_module.FR3_DDQ_DEC * (q - FR3_Q_MIN)
                )
            ),
        ),
    )
    assert np.allclose(upper, expected_upper)
    assert np.allclose(lower, expected_lower)


def test_position_based_velocity_limits_stop_motion_into_joint_limits(sim):
    lower_at_min, _ = sim._position_velocity_limits(FR3_Q_MIN)
    _, upper_at_max = sim._position_velocity_limits(FR3_Q_MAX)
    assert np.array_equal(lower_at_min, np.zeros(7))
    assert np.array_equal(upper_at_max, np.zeros(7))


def test_velocity_command_obeys_position_dependent_envelope(sim):
    sim._command_q = FR3_Q_MAX - 1e-5
    sim.update_joint_velocities(np.full(7, 100.0))
    sim.set_control_mode(ControlMode.VELOCITY)
    # set_control_mode starts from measured q, so place the simulated and
    # commanded state near the stop after the mode transition.
    sim._command_q = FR3_Q_MAX - 1e-5
    sim.step()
    _, upper = sim._position_velocity_limits(FR3_Q_MAX - 1e-5)
    assert np.all(sim.get_robot_state()["dq_d"] <= upper + 1e-12)


@pytest.mark.parametrize("command", [np.zeros(6), np.full(7, np.nan)])
def test_rejects_malformed_commands(sim, command):
    with pytest.raises(ValueError):
        sim.update_joint_positions(command)


def test_ee_pose_is_valid_homogeneous_transform(sim):
    O_T_EE = sim.get_robot_state()["O_T_EE"].reshape(4, 4).T
    assert np.allclose(O_T_EE[3, :], [0, 0, 0, 1])
    R = O_T_EE[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-4)


# Franka's modified (Craig) DH parameters for the FR3: (a_{i-1}, d_i, alpha_{i-1}), joints 1..7,
# then the flange 0.107 m along joint 7 -- the frame libfranka reports as O_T_EE when F_T_EE is
# the identity.
_FR3_DH = (
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -np.pi / 2),
    (0.0, 0.316, np.pi / 2),
    (0.0825, 0.0, np.pi / 2),
    (-0.0825, 0.384, -np.pi / 2),
    (0.0, 0.0, np.pi / 2),
    (0.088, 0.0, np.pi / 2),
)


def _fr3_flange(q):
    T = np.eye(4)
    for (a, d, alpha), theta in zip(_FR3_DH, q):
        ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
        T = T @ np.array([[ct, -st, 0.0, a],
                          [st * ca, ct * ca, -sa, -d * sa],
                          [st * sa, ct * sa, ca, d * ca],
                          [0.0, 0.0, 0.0, 1.0]])
    flange = np.eye(4)
    flange[2, 3] = 0.107
    return T @ flange


def test_ee_pose_is_the_flange(sim):
    """O_T_EE must match libfranka's: the flange, 107 mm past joint 7, not link7's origin."""
    poses = [sim.home_q, sim.home_q + np.array([0.4, 0.2, -0.3, 0.3, 0.5, -0.2, 0.6])]
    for target in poses:
        sim.update_joint_positions(target)
        sim.set_control_mode(ControlMode.POSITION)
        for _ in range(1500):
            sim.step()
        state = sim.get_robot_state()
        O_T_EE = state["O_T_EE"].reshape(4, 4).T
        expected = _fr3_flange(state["q"])
        assert np.allclose(O_T_EE[:3, 3], expected[:3, 3], atol=1e-4)
        assert np.allclose(O_T_EE[:3, :3], expected[:3, :3], atol=1e-4)


def test_velocity_control_tracks_its_integrated_command(sim):
    """In velocity mode the arm follows q_d -- the integrated, limited command -- back to rest,
    instead of drifting off it: the FR3 tracks (q_d, dq_d) with its joint impedance."""
    sim.set_control_mode(ControlMode.VELOCITY)
    sim.update_joint_velocities(np.array([0.4, -0.3, 0.0, 0.5, 0.0, 0.3, 0.0]))
    for _ in range(600):
        sim.step()
    sim.update_joint_velocities(np.zeros(7))
    for _ in range(800):
        sim.step()
    state = sim.get_robot_state()
    assert np.abs(state["q_d"] - sim.home_q).max() > 0.15          # it moved
    assert np.abs(state["q"] - state["q_d"]).max() < 2e-3          # and settled on q_d
