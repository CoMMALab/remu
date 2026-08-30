from types import SimpleNamespace

import h5py
import numpy as np

from remu.hdf5 import export_hdf5
from remu.recording import (
    Attachment,
    AsyncRecorder,
    RecordingWriter,
    RunRecorder,
    TOPIC_EVENTS,
    TOPIC_FCI_COMMAND,
    TOPIC_FCI_STATE,
    TOPIC_GRIPPER_COMMAND,
    TOPIC_GRIPPER_STATE,
    TOPIC_RAW,
    TOPIC_SIM_STATE,
    color_topic,
    depth_topic,
    iter_messages,
    merge_recordings,
    read_attachments,
    read_metadata,
    recording_pb2,
)


def test_mcap_round_trip_metadata_attachments_and_rotation(tmp_path):
    recorder = AsyncRecorder(
        tmp_path / "run.mcap",
        metadata={"kind": "test"},
        attachments={"scene.xml": Attachment("application/xml", b"<mujoco/>")},
        rotate_duration_ns=2,
    )
    recorder.start()
    for tick in range(4):
        recorder.record(
            TOPIC_SIM_STATE,
            recording_pb2.SimState(
                tick_index=tick, sim_time_ns=tick, qpos=[tick], qvel=[0]
            ),
            log_time_ns=tick,
            sequence=tick,
        )
    assert recorder.stop() == 4

    assert recorder.outputs == [tmp_path / "run.mcap", tmp_path / "run.0001.mcap"]
    assert [row.proto_msg.tick_index for row in iter_messages(recorder.outputs[0])] == [0, 1]
    assert [row.proto_msg.tick_index for row in iter_messages(recorder.outputs[1])] == [2, 3]
    assert read_metadata(recorder.outputs[0])["kind"] == "test"
    assert read_attachments(recorder.outputs[1])["scene.xml"].data == b"<mujoco/>"


def test_deterministic_merge_orders_equal_timestamps_by_channel_and_frame(tmp_path):
    capture = tmp_path / "capture.mcap"
    shard = tmp_path / "worker.mcap"
    with RecordingWriter(capture) as writer:
        writer.write(
            TOPIC_SIM_STATE,
            recording_pb2.SimState(tick_index=1, sim_time_ns=10, qpos=[0], qvel=[0]),
            log_time_ns=10,
        )
        writer.write(
            TOPIC_EVENTS,
            recording_pb2.Event(tick_index=1, sim_time_ns=10, type="done"),
            log_time_ns=10,
        )
    with RecordingWriter(shard) as writer:
        for frame_index in (1, 0):
            writer.write(
                color_topic("realsense/test"),
                recording_pb2.CameraFrame(
                    camera_name="realsense/test",
                    frame_index=frame_index,
                    actual_sim_time_ns=10,
                ),
                log_time_ns=10,
                sequence=frame_index,
            )

    merged = merge_recordings([capture, shard], tmp_path / "run.mcap")
    rows = list(iter_messages(merged))

    assert [row.topic for row in rows] == [
        TOPIC_SIM_STATE,
        color_topic("realsense/test"),
        color_topic("realsense/test"),
        TOPIC_EVENTS,
    ]
    assert [row.proto_msg.frame_index for row in rows[1:3]] == [0, 1]


def test_run_recorder_emits_decoded_channels_and_atomic_rgbd(tmp_path):
    state = {
        "q": np.arange(7) / 10,
        "dq": np.zeros(7),
        "q_d": np.arange(7) / 10,
        "dq_d": np.zeros(7),
        "ddq_d": np.zeros(7),
        "tau_J": np.ones(7),
    }
    finger = {
        "width": 0.06,
        "q": np.array([0.03, 0.03]),
        "dq": np.zeros(2),
        "contact_body_ids": frozenset(),
    }
    sim = SimpleNamespace(
        enable_gripper=True,
        on_step_callbacks=[],
        get_robot_state=lambda: state,
        get_finger_state=lambda: finger,
    )
    recorder = RunRecorder(tmp_path / "run.mcap", level="debug")
    # Ephemeral gripper traffic outside the FCI session is intentionally ignored.
    recorder.gripper_command({"command": "kMove", "width": 0.08})
    recorder.attach(sim)
    recorder.start()
    recorder.fci_session_start()
    recorder.fci_command({
        "message_id": 5,
        "mode": "position",
        "q_command": np.arange(7),
    })
    recorder.gripper_command({"command": "kMove", "command_id": 7, "width": 0.06})
    recorder.on_step(None, SimpleNamespace(
        time=0.001,
        qpos=np.arange(9, dtype=float),
        qvel=np.zeros(9),
    ))
    camera = SimpleNamespace(
        name="realsense/test",
        depth_scale=0.001,
        color_profile=SimpleNamespace(width=2, height=1),
        depth_profile=SimpleNamespace(width=2, height=1),
    )
    recorder.camera_frame(camera, bytes(6), bytes(4), 1)
    recorder.raw_packet(
        transport="udp", direction="client_to_remu", endpoint="fci", data=b"packet"
    )
    recorder.fci_session_end()
    recorder.stop()
    recorder.gripper_command({"command": "kMove", "width": 0.08})

    topics = [row.topic for row in iter_messages(tmp_path / "run.mcap")]
    for topic in (
        TOPIC_SIM_STATE,
        TOPIC_FCI_COMMAND,
        TOPIC_FCI_STATE,
        TOPIC_GRIPPER_COMMAND,
        TOPIC_GRIPPER_STATE,
        color_topic("realsense/test"),
        depth_topic("realsense/test"),
        TOPIC_RAW,
        TOPIC_EVENTS,
    ):
        assert topic in topics
    assert topics.count(color_topic("realsense/test")) == 1
    assert topics.count(depth_topic("realsense/test")) == 1


def test_hdf5_export_builds_dense_trajectory_and_camera_arrays(tmp_path):
    source = tmp_path / "run.mcap"
    calibration = b'{"realsense/test": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}'
    with RecordingWriter(
        source,
        attachments={"camera_calibration.json": Attachment("application/json", calibration)},
    ) as writer:
        for tick in range(2):
            writer.write(
                TOPIC_SIM_STATE,
                recording_pb2.SimState(
                    tick_index=tick + 1,
                    sim_time_ns=(tick + 1) * 1_000_000,
                    qpos=np.arange(9) + tick,
                    qvel=np.zeros(9),
                ),
                log_time_ns=(tick + 1) * 1_000_000,
            )
        writer.write(
            color_topic("realsense/test"),
            recording_pb2.CameraFrame(
                camera_name="realsense/test",
                frame_index=0,
                trajectory_tick_index=1,
                scheduled_sim_time_ns=1_000_000,
                actual_sim_time_ns=1_000_000,
                encoding="raw_rgb8",
                width=2,
                height=1,
                data=bytes([1, 2, 3, 4, 5, 6]),
            ),
            log_time_ns=1_000_000,
        )
        writer.write(
            depth_topic("realsense/test"),
            recording_pb2.CameraFrame(
                camera_name="realsense/test",
                frame_index=0,
                trajectory_tick_index=1,
                scheduled_sim_time_ns=1_000_000,
                actual_sim_time_ns=1_000_000,
                encoding="raw_z16_le",
                width=2,
                height=1,
                depth_scale=0.001,
                data=np.asarray([100, 200], dtype="<u2").tobytes(),
            ),
            log_time_ns=1_000_000,
        )

    output = export_hdf5(source, tmp_path / "run.h5")
    with h5py.File(output) as dataset:
        assert dataset["trajectory/qpos"].shape == (2, 9)
        assert dataset["cameras/realsense/test/color"].shape == (1, 1, 2, 3)
        assert dataset["cameras/realsense/test/depth"][0].tolist() == [[100, 200]]
        assert dataset["cameras/realsense/test/trajectory_index"][0] == 0

