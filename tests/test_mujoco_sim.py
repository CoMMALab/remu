import numpy as np

from remu.sim.mujoco_sim import ControlMode


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


def test_ee_pose_is_valid_homogeneous_transform(sim):
    O_T_EE = sim.get_robot_state()["O_T_EE"].reshape(4, 4).T
    assert np.allclose(O_T_EE[3, :], [0, 0, 0, 1])
    R = O_T_EE[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-4)
