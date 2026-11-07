"""Portal RPC used by i2rt examples/minimum_gello/ClientRobot."""


def build_server(sim, port=1234, host="0.0.0.0"):
    import portal

    server = portal.Server(port, host=host, errors=False)
    for name in ("num_dofs", "get_joint_pos", "command_joint_pos",
                 "command_joint_state", "get_observations", "get_robot_info",
                 "command_joint_vel", "command_joint_torque", "get_end_effector_pose",
                 "gripper_move", "gripper_grasp", "gripper_home", "gripper_stop",
                 "get_gripper_state", "get_robot_model", "get_dynamics", "keepalive", "stop"):
        server.bind(name, getattr(sim, name))
    return server
