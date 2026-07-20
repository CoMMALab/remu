"""End-to-end: MuJoCo -> camera server -> pyrealsense2 shim -> point cloud.

These exercise the shim exactly the way ``perception_common.realsense_worker``
does, so a break in the emulated SDK surface shows up here rather than in the
perception container.
"""

import importlib.util
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from remu.camera import CameraServer, d435i_in_front_of_robot
from remu.sim.mujoco_sim import MujocoSim
from remu.sim.scene import build_scene_xml

SHIM_PATH = Path(__file__).resolve().parent.parent / "shim" / "pyrealsense2.py"


@pytest.fixture(scope="module")
def rs():
    """The shim, loaded by path so it never shadows a real pyrealsense2."""
    spec = importlib.util.spec_from_file_location("_remu_rs_shim", SHIM_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def camera_stack():
    """A running sim + camera server with one D435i in front of the arm."""
    camera = d435i_in_front_of_robot(serial="000000000001")
    scene_path = build_scene_xml(cameras=[camera])
    sim = MujocoSim(scene_path, realtime=True)
    sim.build()

    port = _free_port()
    server = CameraServer([camera], host="127.0.0.1", port=port)
    server.attach(sim)
    server.start(background=True)
    threading.Thread(target=sim.run, daemon=True).start()

    # The camera is deliberately *not* bound here: an OpenGL context belongs
    # to the thread that creates it, and binding on this thread would make
    # every later render from the physics thread fail with EGL_BAD_ACCESS.
    # Let the physics thread bind lazily and wait for its first frame.
    deadline = time.monotonic() + 15.0
    while server._frames[camera.serial].frame_id == 0 and time.monotonic() < deadline:
        if server.last_render_error is not None:
            break
        time.sleep(0.05)

    if server._frames[camera.serial].frame_id == 0:
        server.stop()
        sim.stop()
        Path(scene_path).unlink(missing_ok=True)
        pytest.skip(f"offscreen rendering unavailable: {server.last_render_error}")

    yield camera, port

    server.stop()
    sim.stop()
    Path(scene_path).unlink(missing_ok=True)


@pytest.fixture
def address(rs, camera_stack, monkeypatch):
    _camera, port = camera_stack
    monkeypatch.setenv("REMU_CAMERA_ADDR", f"127.0.0.1:{port}")
    return port


def test_query_devices_reports_the_emulated_camera(rs, camera_stack, address):
    camera, _port = camera_stack
    serials = [d.get_info(rs.camera_info.serial_number)
               for d in rs.context().query_devices()]
    assert serials == [camera.serial]


def test_query_devices_is_empty_with_no_server(rs, monkeypatch):
    """No server must look like no camera plugged in, not an exception --
    the perception code already handles an empty device list."""
    monkeypatch.setenv("REMU_CAMERA_ADDR", f"127.0.0.1:{_free_port()}")
    assert rs.context().query_devices() == []


def test_frames_carry_aligned_color_and_depth(rs, camera_stack, address):
    camera, _port = camera_stack
    pipeline, config = rs.pipeline(), rs.config()
    config.enable_device(camera.serial)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    pipeline.start(config)
    try:
        frames = rs.align(rs.stream.color).process(pipeline.wait_for_frames(timeout_ms=5000))
        color = np.asarray(frames.get_color_frame().get_data())
        depth = np.asarray(frames.get_depth_frame().get_data())

        assert color.shape == (camera.height, camera.width, 3)
        assert color.dtype == np.uint8
        # Alignment is by construction here, so shapes must agree exactly.
        assert depth.shape == color.shape[:2]
        assert depth.dtype == np.uint16
        assert (depth > 0).any(), "every depth pixel was a miss"
    finally:
        pipeline.stop()


def test_decimation_scales_intrinsics_with_the_image(rs, camera_stack, address):
    """Halving resolution without halving fx/ppx would shrink the cloud toward
    the optical axis instead of thinning it."""
    camera, _port = camera_stack
    pipeline, config = rs.pipeline(), rs.config()
    config.enable_device(camera.serial)
    pipeline.start(config)
    try:
        depth_frame = pipeline.wait_for_frames(timeout_ms=5000).get_depth_frame()
        full = depth_frame.profile.as_video_stream_profile().intrinsics

        decimation = rs.decimation_filter()
        decimation.set_option(rs.option.filter_magnitude, 2)
        halved = decimation.process(depth_frame).profile.as_video_stream_profile().intrinsics

        assert (halved.width, halved.height) == (full.width // 2, full.height // 2)
        assert halved.fx == pytest.approx(full.fx / 2)
        assert halved.ppx == pytest.approx(full.ppx / 2)
    finally:
        pipeline.stop()


def test_threshold_filter_clamps_range(rs, camera_stack, address):
    camera, _port = camera_stack
    pipeline, config = rs.pipeline(), rs.config()
    config.enable_device(camera.serial)
    pipeline.start(config)
    try:
        depth_frame = pipeline.wait_for_frames(timeout_ms=5000).get_depth_frame()
        threshold = rs.threshold_filter()
        threshold.set_option(rs.option.min_distance, 0.15)
        threshold.set_option(rs.option.max_distance, 1.2)
        out = np.asarray(threshold.process(depth_frame).get_data())

        kept = out[out > 0]
        assert kept.size, "threshold filter removed everything"
        assert kept.max() <= 1200
        assert kept.min() >= 150
    finally:
        pipeline.stop()


def test_point_cloud_lands_on_the_robot_in_world_coordinates(rs, camera_stack, address):
    """The real end-to-end check: deproject, apply the ground-truth extrinsic,
    and confirm the points sit where the arm actually is.

    This catches every frame-convention error at once -- an axis flip or a
    transposed rotation puts the arm behind the camera or under the floor.
    """
    camera, _port = camera_stack
    pipeline, config = rs.pipeline(), rs.config()
    config.enable_device(camera.serial)
    pipeline.start(config)
    try:
        frames = rs.align(rs.stream.color).process(pipeline.wait_for_frames(timeout_ms=5000))
        color_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()

        cloud = rs.pointcloud()
        cloud.map_to(color_frame)
        points = cloud.calculate(depth_frame)
        verts = np.asarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        texcoords = np.asarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)

        assert verts.shape[0] == camera.width * camera.height
        assert texcoords.shape == (verts.shape[0], 2)
        assert texcoords.min() >= 0.0 and texcoords.max() <= 1.0

        # Same validity test perception_common uses.
        verts = verts[(verts * verts).sum(axis=1) > 1e-4]
        assert verts.shape[0] > 1000

        T = camera.optical_pose()
        world = verts @ T[:3, :3].T + T[:3, 3]

        # Nothing may be below the floor, and the arm must be in front of the
        # camera (toward the origin), not behind it.
        assert world[:, 2].min() > -0.02
        arm = world[world[:, 2] > 0.05]
        assert arm.shape[0] > 500, "no points above the floor -- camera sees nothing"

        centroid = arm.mean(axis=0)
        assert abs(centroid[1]) < 0.15, "arm should be centred on y=0"
        assert -0.2 < centroid[0] < 0.8
        assert 0.1 < centroid[2] < 0.9
    finally:
        pipeline.stop()
