import struct

from remu.protocol.robot_state import RobotState


def test_pack_state_size_matches_wire_format():
    state = RobotState()
    packed = state.pack_state()
    # sizeof(research_interface::robot::RobotState) for protocol v9
    # (libfranka 0.15.3). A mismatch here makes libfranka throw
    # ProtocolException("libfranka: incorrect object size") at network.h:141.
    assert len(packed) == 2373


def test_state_fields_are_doubles_at_known_offsets():
    """v9 is double-based; a float layout would silently shift every field."""
    state = RobotState()
    state.state["q"] = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
    packed = state.pack_state()

    # q sits after message_id(8) + 6*16 poses + (m_ee,I_ee,F_x_Cee) +
    # (m_load,I_load,F_x_Cload) + elbow + elbow_d + tau_J,tau_J_d,dtau_J.
    offset = 8 + (6 * 16 + 1 + 9 + 3 + 1 + 9 + 3 + 2 + 2 + 7 * 3) * 8
    assert struct.unpack_from("<7d", packed, offset) == (1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5)


def test_update_increments_message_id_monotonically():
    state = RobotState()
    assert state.state["message_id"] == 0
    state.update()
    assert state.state["message_id"] == 1
    state.update()
    assert state.state["message_id"] == 2


def test_set_motion_generator_and_controller_mode():
    state = RobotState()
    state.set_motion_generator_mode(2)
    state.set_controller_mode(1)
    assert state.state["motion_generator_mode"] == 2
    assert state.state["controller_mode"] == 1
    # Still packs cleanly after mutation.
    assert len(state.pack_state()) == 2373
