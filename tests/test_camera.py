import math

import numpy as np
import pytest

from remu.camera.d435i import (
    EmulatedD435i,
    Intrinsics,
    d435i_in_front_of_robot,
    look_at_axes,
    optical_pose_to_calibration,
)
from remu.camera.wire import DEPTH_SCALE, encode_frame, read_frame


def test_intrinsics_match_a_real_d435i_color_stream():
    """Derived-from-FOV intrinsics should land on the factory values.

    A real D435i 640x480 color stream reports fx ~= fy ~= 616 and a centred
    principal point; if this drifts, clouds come out systematically scaled.
    """
    intr = Intrinsics.from_fovy(640, 480, 42.5)
    assert intr.fx == pytest.approx(617.2, abs=2.0)
    assert intr.fy == pytest.approx(intr.fx)
    assert (intr.ppx, intr.ppy) == (320.0, 240.0)


def test_look_at_axes_are_right_handed_and_orthonormal():
    axes = look_at_axes(eye=(1.0, 0.0, 0.6), target=(0.0, 0.0, 0.35))
    right, up = axes
    assert np.linalg.norm(right) == pytest.approx(1.0)
    assert np.linalg.norm(up) == pytest.approx(1.0)
    assert np.dot(right, up) == pytest.approx(0.0, abs=1e-12)


def test_look_at_rejects_degenerate_geometry():
    with pytest.raises(ValueError):
        look_at_axes(eye=(1.0, 0.0, 0.0), target=(1.0, 0.0, 0.0))
    # Looking straight down: the view axis is parallel to the default up.
    with pytest.raises(ValueError):
        look_at_axes(eye=(0.0, 0.0, 1.0), target=(0.0, 0.0, 0.0))


def test_optical_pose_uses_realsense_axis_convention():
    """+z must point at the target and +y must point down, not MuJoCo's -z/+y.

    Getting this backwards flips the cloud through the camera centre, which
    still looks plausible in a viewer -- hence pinning it here.
    """
    cam = d435i_in_front_of_robot(distance_m=1.0, height_m=0.6, look_at_height_m=0.35)
    T = cam.optical_pose()

    assert np.allclose(T[:3, 3], [1.0, 0.0, 0.6])

    forward = T[:3, 2]
    to_target = np.array([0.0, 0.0, 0.35]) - np.array([1.0, 0.0, 0.6])
    assert np.dot(forward, to_target / np.linalg.norm(to_target)) == pytest.approx(1.0)

    # +y is "down" in an optical frame, so it must have a negative world-z part.
    assert T[2, 1] < 0.0
    assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0)


def test_depth_quantization_zeroes_out_of_range():
    cam = EmulatedD435i(serial="test", eye=(1.0, 0.0, 0.5))
    depth_m = np.array([[0.05, 0.5, 3.0, 20.0]], dtype=np.float32)
    z16 = cam.quantize_depth(depth_m)

    assert z16.dtype == np.uint16
    # Too near and beyond max range are both "no reading" -> 0, matching the
    # sentinel the perception code rejects with its |v|^2 > 1e-4 test.
    assert z16[0, 0] == 0
    assert z16[0, 3] == 0
    assert z16[0, 1] == 500
    assert z16[0, 2] == 3000
    assert z16[0, 2] * DEPTH_SCALE == pytest.approx(3.0)


def test_mjcf_body_declares_the_camera_and_is_non_colliding():
    cam = d435i_in_front_of_robot(serial="123456789012")
    xml = cam.mjcf_body()
    assert 'name="remu_cam_123456789012"' in xml
    assert f'fovy="{cam.fovy_deg}"' in xml
    # The marker must not be able to push the arm around.
    assert 'contype="0" conaffinity="0"' in xml


def test_calibration_export_uses_perception_keys():
    cams = [d435i_in_front_of_robot(serial="abc")]
    calib = optical_pose_to_calibration(cams)
    assert list(calib) == ["realsense/abc"]
    assert np.allclose(np.array(calib["realsense/abc"]), cams[0].optical_pose())


def test_wire_roundtrip_preserves_payloads():
    color = np.arange(12, dtype=np.uint8).tobytes()
    depth = np.arange(4, dtype=np.uint16).tobytes()
    header = {"serial": "x", "color_bytes": len(color), "depth_bytes": len(depth)}
    blob = encode_frame(header, color, depth)

    class _FakeSock:
        def __init__(self, data):
            self._data = data

        def recv(self, n):
            chunk, self._data = self._data[:n], self._data[n:]
            return chunk

    got_header, got_color, got_depth = read_frame(_FakeSock(blob))
    assert got_header["serial"] == "x"
    assert (got_color, got_depth) == (color, depth)


def test_read_frame_rejects_a_desynced_stream():
    class _FakeSock:
        def __init__(self, data):
            self._data = data

        def recv(self, n):
            chunk, self._data = self._data[:n], self._data[n:]
            return chunk

    with pytest.raises(ConnectionError, match="magic"):
        read_frame(_FakeSock(b"XXXX" + b"\x00" * 64))


def test_horizontal_fov_follows_from_aspect_ratio():
    intr = Intrinsics.from_fovy(640, 480, 42.5)
    hfov = math.degrees(2 * math.atan2(intr.width / 2, intr.fx))
    # 4:3 with square pixels -> ~55 deg horizontal, the D435i's actual RGB FOV.
    assert hfov == pytest.approx(55.0, abs=1.5)
