import importlib.util
import socket
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from remu.camera import CameraServer, EmulatedRgbdCamera, StreamProfile

ROOT = Path(__file__).resolve().parent.parent


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "shim" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _camera(vendor, model, serial, color_size=(4, 3), depth_size=(2, 2)):
    camera = EmulatedRgbdCamera(
        vendor=vendor, model=model, serial=serial,
        device_name="Intel RealSense D435I" if vendor == "realsense" else "Orbbec Femto Mega",
        base_from_optical=np.eye(4),
        color=StreamProfile(*color_size, "rgb8", 50.0),
        depth=StreamProfile(*depth_size, "z16", 60.0),
        fps=15,
    )
    color = np.arange(color_size[0] * color_size[1] * 3, dtype=np.uint8).reshape(
        color_size[1], color_size[0], 3
    )
    depth = np.full((depth_size[1], depth_size[0]), 1000, dtype=np.uint16)
    camera.render = lambda _model, _data: (color, depth)
    return camera


@pytest.fixture
def shim_server(monkeypatch):
    cameras = [
        _camera("realsense", "d435i", "rs-one"),
        _camera("orbbec", "femto_mega", "fm-one"),
    ]
    port = _free_port()
    server = CameraServer(cameras, host="127.0.0.1", port=port).start()
    for camera in cameras:
        server._render_one(camera, None, None)
    monkeypatch.setenv("REMU_CAMERA_ADDR", f"127.0.0.1:{port}")
    yield cameras, server
    server.stop()


def test_shims_enumerate_only_their_vendor(shim_server):
    rs = _load("_test_rs_mixed", "pyrealsense2.py")
    ob = _load("_test_ob_mixed", "pyorbbecsdk.py")
    assert [device.get_info(rs.camera_info.serial_number)
            for device in rs.context().query_devices()] == ["rs-one"]
    devices = ob.Context().query_devices()
    assert devices.get_count() == 1
    assert devices[0].get_device_info().get_serial_number() == "fm-one"

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color)
    pipeline.start(config)
    try:
        frame = pipeline.wait_for_frames(1000).get_color_frame()
        assert frame.get_frame_number() == 1
        assert np.asarray(frame.get_data()).shape == (3, 4, 3)
    finally:
        pipeline.stop()


def test_realsense_rejects_an_undeclared_profile(shim_server):
    rs = _load("_test_rs_strict", "pyrealsense2.py")
    config = rs.config()
    config.enable_device("rs-one")
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    with pytest.raises(RuntimeError, match="not configured"):
        rs.pipeline().start(config)


def test_orbbec_v2_pipeline_profiles_frames_alignment_and_points(shim_server):
    ob = _load("_test_ob_pipeline", "pyorbbecsdk.py")
    pipeline = ob.Pipeline()
    config = ob.Config()
    color = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR).get_default_video_stream_profile()
    depth = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile()
    config.enable_stream(color)
    config.enable_stream(depth)
    pipeline.start(config)
    try:
        frames = pipeline.wait_for_frames(1000)
        assert np.asarray(frames.get_color_frame().get_data()).shape == (3, 4, 3)
        assert np.asarray(frames.get_depth_frame().get_data()).shape == (2, 2)
        assert frames.get_depth_frame().get_depth_scale() == pytest.approx(1.0)
        assert frames.get_depth_frame().get_index() == 1

        aligned = ob.AlignFilter(ob.OBStreamType.COLOR_STREAM).process(frames)
        assert np.asarray(aligned.get_depth_frame().get_data()).shape == (3, 4)
        points_filter = ob.PointCloudFilter()
        points_filter.set_create_point_format(ob.OBFormat.RGB_POINT)
        points = points_filter.calculate(points_filter.process(frames))
        assert points.shape == (12, 6)
    finally:
        pipeline.stop()


def test_orbbec_gemini_device_identity_and_frame_number(monkeypatch):
    camera = _camera("orbbec", "gemini_335", "G335-one")
    camera.device_name = "Orbbec Gemini 335"
    port = _free_port()
    server = CameraServer([camera], host="127.0.0.1", port=port).start()
    server._render_one(camera, None, None)
    monkeypatch.setenv("REMU_CAMERA_ADDR", f"127.0.0.1:{port}")
    ob = _load("_test_ob_gemini", "pyorbbecsdk.py")
    pipeline = None

    try:
        device = ob.Context().query_devices()[0]
        info = device.get_device_info()
        assert info.get_name() == "Orbbec Gemini 335"
        assert info.get_uid() == "G335-one"
        assert info.get_pid() != 0x0669

        pipeline = ob.Pipeline(device)
        pipeline.start()
        frame = pipeline.wait_for_frames(1000).get_color_frame()
        assert frame.get_frame_number() == 1
    finally:
        if pipeline is not None:
            pipeline.stop()
        server.stop()


def test_attach_publishes_an_immutable_snapshot_without_rendering():
    camera = _camera("realsense", "d435i", "snapshot")
    camera.render = lambda _model, _data: pytest.fail("rendered on the physics thread")
    data = SimpleNamespace(
        qpos=np.arange(9, dtype=float),
        mocap_pos=np.arange(3, dtype=float).reshape(1, 3),
        mocap_quat=np.array([[1.0, 0.0, 0.0, 0.0]]),
        time=1.25,
    )
    sim = SimpleNamespace(
        model=None, data=data, on_step_callbacks=[], scene_xml_path=Path("unused.xml")
    )
    server = CameraServer([camera]).attach(sim)

    sim.on_step_callbacks[0](None, data)
    qpos, mocap_pos, mocap_quat, sim_time = server._latest_state
    data.qpos[:] = -1
    data.mocap_pos[:] = -1
    data.mocap_quat[:] = -1

    assert qpos.tolist() == list(range(9))
    assert mocap_pos.tolist() == [[0.0, 1.0, 2.0]]
    assert mocap_quat.tolist() == [[1.0, 0.0, 0.0, 0.0]]
    assert sim_time == 1.25
    server.stop()
    assert sim.on_step_callbacks == []
