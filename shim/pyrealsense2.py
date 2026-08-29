"""Drop-in ``pyrealsense2`` replacement backed by remu's emulated cameras.

Put this directory first on ``PYTHONPATH`` and ``import pyrealsense2 as rs``
resolves here instead of the real SDK, so pointcloud_perception talks to a
MuJoCo scene without a single line changed::

    PYTHONPATH=/path/to/remu/shim python visualize_cameras.py

Point it at a non-default server with ``REMU_CAMERA_ADDR=host:port``.

Scope
-----
This implements the subset of the SDK that pointcloud_perception actually
uses -- device enumeration, pipeline/config, aligned frames, the point-cloud
helper, and the depth post-processing filters. It is not a general
pyrealsense2 replacement; anything not listed below is absent, and will fail
loudly with AttributeError rather than silently doing nothing.

Filter fidelity
---------------
``decimation_filter`` and ``threshold_filter`` are implemented for real: they
change the point count and the visible range, which is exactly what the
perception code is tuned against, so stubbing them would make the emulator
lie about the thing being tuned. The disparity/spatial/temporal/hole-filling
filters are identity passes -- they exist to denoise real sensor data, and
rendered depth has none of the noise they remove. Consequence: toggling
``FILTERS["rs_sdk"]`` visibly changes range and density, but not smoothness.

The one deliberate departure from a real camera is that ``wait_for_frames``
returns the same frame twice if the client polls faster than the sim renders,
rather than blocking for a genuinely new one; a real camera is a free-running
clock and the sim is not.
"""

import importlib.util
import os
import socket
import threading

import numpy as np

# The shim ships inside remu's tree but has to run inside the
# pointcloud_perception container, where remu is not installed. Load the wire
# module straight off disk under a private name, so the only hard dependency
# is numpy and nothing generic lands in sys.modules.
_WIRE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "remu", "camera", "wire.py",
)
_spec = importlib.util.spec_from_file_location("_remu_camera_wire", _WIRE_PATH)
if _spec is None:
    raise ImportError(f"remu camera wire module not found at {_WIRE_PATH}")
wire = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wire)


DEFAULT_ADDR = ("127.0.0.1", wire.CAMERA_PORT)


def _server_addr():
    """Server address, from ``$REMU_CAMERA_ADDR`` (``host:port``) or the default."""
    raw = os.environ.get("REMU_CAMERA_ADDR")
    if not raw:
        return DEFAULT_ADDR
    host, _, port = raw.partition(":")
    return (host or DEFAULT_ADDR[0], int(port) if port else DEFAULT_ADDR[1])


def _connect(request, timeout=5.0):
    """Open a control connection and send one request; returns (sock, response)."""
    sock = socket.create_connection(_server_addr(), timeout=timeout)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    wire.send_json(sock, request)
    response = wire.recv_json(sock)
    if "error" in response:
        sock.close()
        raise RuntimeError(f"remu camera server: {response['error']}")
    return sock, response


# ── Enums ─────────────────────────────────────────────────────────────────────
# Plain classes with sentinel attributes. The perception code only ever passes
# these through to the shim's own methods, so identity is all that matters.

class stream:
    depth = "depth"
    color = "color"
    infrared = "infrared"


class format:
    z16 = "z16"
    rgb8 = "rgb8"
    bgr8 = "bgr8"
    y8 = "y8"


class camera_info:
    serial_number = "serial_number"
    name = "name"
    firmware_version = "firmware_version"


class option:
    filter_magnitude = "filter_magnitude"
    min_distance = "min_distance"
    max_distance = "max_distance"
    filter_smooth_alpha = "filter_smooth_alpha"
    filter_smooth_delta = "filter_smooth_delta"
    holes_fill = "holes_fill"


class distortion:
    inverse_brown_conrady = "inverse_brown_conrady"
    none = "none"


# ── Devices ───────────────────────────────────────────────────────────────────

class _Device:
    def __init__(self, info):
        self._info = info

    def get_info(self, key):
        if key == camera_info.serial_number:
            return self._info["serial"]
        if key == camera_info.name:
            return self._info.get("name", "Intel RealSense D435I")
        if key == camera_info.firmware_version:
            return "5.16.0.1 (remu)"
        raise RuntimeError(f"camera_info {key!r} not emulated")

    def first_depth_sensor(self):
        return _DepthSensor(self._info.get("depth_scale", wire.DEPTH_SCALE))


class _DepthSensor:
    def __init__(self, scale):
        self._scale = scale

    def get_depth_scale(self):
        return self._scale


class context:
    """Device enumeration. ``query_devices()`` asks the server what it serves."""

    def query_devices(self):
        try:
            sock, response = _connect({"op": "list_devices", "vendor": "realsense"})
        except OSError:
            # No server running is the emulator's equivalent of no camera
            # plugged in, and the perception code already handles that.
            return []
        try:
            return [_Device(d) for d in response["devices"]]
        finally:
            sock.close()


# ── Intrinsics / profiles ─────────────────────────────────────────────────────

class _Intrinsics:
    def __init__(self, d):
        self.width = d["width"]
        self.height = d["height"]
        self.fx = d["fx"]
        self.fy = d["fy"]
        self.ppx = d["ppx"]
        self.ppy = d["ppy"]
        self.coeffs = list(d["coeffs"])
        self.model = distortion.inverse_brown_conrady


class _VideoStreamProfile:
    def __init__(self, intrin, device=None):
        self.intrinsics = intrin
        self._device = device

    def as_video_stream_profile(self):
        return self

    def get_device(self):
        return self._device


# ── Frames ────────────────────────────────────────────────────────────────────

class _Frame:
    """Common frame behaviour. Truthiness matters: the perception loop does
    ``if not depth_frame or not color_frame: continue``."""

    def __init__(self, data, intrin, timestamp_ms, depth_scale=wire.DEPTH_SCALE, frame_number=0):
        self._data = data
        self._intrin = intrin
        self.timestamp_ms = timestamp_ms
        self._depth_scale = depth_scale
        self._frame_number = frame_number

    def __bool__(self):
        return self._data is not None and self._data.size > 0

    def get_data(self):
        # np.asarray() on this returns the array itself with no copy, which is
        # what the SDK's buffer-protocol object achieves.
        return self._data

    def get_width(self):
        return self._data.shape[1]

    def get_height(self):
        return self._data.shape[0]

    def get_timestamp(self):
        return self.timestamp_ms

    def get_frame_number(self):
        return self._frame_number

    @property
    def profile(self):
        return _VideoStreamProfile(_Intrinsics(self._intrin))


class _DepthFrame(_Frame):
    def as_depth_frame(self):
        return self

    def get_distance(self, x, y):
        return float(self._data[y, x]) * self._depth_scale


class _ColorFrame(_Frame):
    pass


class _Composite:
    """The multi-stream bundle ``wait_for_frames`` returns."""

    def __init__(self, depth, color):
        self._depth = depth
        self._color = color

    def get_depth_frame(self):
        return self._depth

    def get_color_frame(self):
        return self._color


# ── Pipeline ──────────────────────────────────────────────────────────────────

class config:
    def __init__(self):
        self.serial = None
        self.streams = []

    def enable_device(self, serial):
        self.serial = str(serial)

    def enable_stream(self, stream_type, width=None, height=None, fmt=None, fps=None):
        # Resolution and format are fixed by the emulated camera, so these are
        # recorded for introspection but do not reconfigure anything.
        self.streams.append((stream_type, width, height, fmt, fps))

    def disable_all_streams(self):
        self.streams = []


class pipeline:
    def __init__(self, ctx=None):
        self._sock = None
        self._lock = threading.Lock()
        self._latest = None
        self._device = None
        self._enabled = {stream.depth, stream.color}

    def start(self, cfg=None):
        serial = cfg.serial if cfg is not None else None
        streams = [] if cfg is None else [
            {"stream": kind, "width": width, "height": height, "format": fmt, "fps": fps}
            for kind, width, height, fmt, fps in cfg.streams
        ]
        try:
            self._sock, response = _connect({
                "op": "stream", "vendor": "realsense", "serial": serial, "streams": streams,
            }, timeout=5.0)
        except OSError as e:
            raise RuntimeError(f"remu camera server unreachable at {_server_addr()}: {e}") from e
        self._device = _Device(response["device"])
        if streams:
            self._enabled = {request["stream"] for request in streams}
        return _VideoStreamProfile(None, self._device)

    def wait_for_frames(self, timeout_ms=5000):
        """Block for the next frame off the wire.

        Raises ``RuntimeError`` on timeout, matching the SDK -- the
        perception worker catches exactly that and retries.
        """
        if self._sock is None:
            raise RuntimeError("pipeline not started")
        self._sock.settimeout(timeout_ms / 1000.0)
        try:
            header, color_bytes, depth_bytes = wire.read_frame(self._sock)
        except socket.timeout as e:
            raise RuntimeError(f"Frame didn't arrive within {timeout_ms} ms") from e
        except (ConnectionError, OSError) as e:
            raise RuntimeError(f"remu camera stream lost: {e}") from e

        color_meta = header.get("color")
        depth_meta = header.get("depth")
        if color_meta is None:  # protocol v1 compatibility
            h, w = header["height"], header["width"]
            color_meta = depth_meta = {"width": w, "height": h, "intrinsics": header["intrinsics"]}
        color = np.frombuffer(color_bytes, dtype=np.uint8).reshape(
            color_meta["height"], color_meta["width"], 3
        )
        depth = np.frombuffer(depth_bytes, dtype=np.uint16).reshape(
            depth_meta["height"], depth_meta["width"]
        )
        ts = header["timestamp_ms"]
        scale = header.get("depth_scale", wire.DEPTH_SCALE)
        frame_number = header.get("frame_number", 0)
        depth_frame = (
            _DepthFrame(depth, depth_meta["intrinsics"], ts, scale, frame_number)
            if stream.depth in self._enabled else None
        )
        color_frame = (
            _ColorFrame(color, color_meta["intrinsics"], ts, frame_number=frame_number)
            if stream.color in self._enabled else None
        )
        return _Composite(depth_frame, color_frame)

    def stop(self):
        if self._sock is not None:
            self._sock.close()
            self._sock = None


# ── Processing blocks ─────────────────────────────────────────────────────────

class _PassthroughFilter:
    """Base for the filters that are no-ops against rendered depth."""

    def __init__(self, *args, **kwargs):
        self.options = {}

    def set_option(self, key, value):
        self.options[key] = value

    def get_option(self, key):
        return self.options.get(key, 0.0)

    def process(self, frame):
        return frame


class spatial_filter(_PassthroughFilter):
    pass


class temporal_filter(_PassthroughFilter):
    pass


class hole_filling_filter(_PassthroughFilter):
    pass


class disparity_transform(_PassthroughFilter):
    def __init__(self, transform_to_disparity=True):
        super().__init__()
        self.to_disparity = transform_to_disparity


class decimation_filter(_PassthroughFilter):
    """Subsample depth by an integer factor, as the SDK's decimation does.

    The intrinsics are scaled with the image, so deprojecting the decimated
    frame still lands the points in the right place -- getting this wrong
    would shrink the cloud toward the optical axis instead of thinning it.
    """

    def process(self, frame):
        mag = int(self.options.get(option.filter_magnitude, 2))
        if mag <= 1:
            return frame
        data = frame._data[::mag, ::mag]
        intrin = dict(frame._intrin)
        h, w = data.shape[:2]
        for key, value in (("fx", intrin["fx"]), ("fy", intrin["fy"]),
                           ("ppx", intrin["ppx"]), ("ppy", intrin["ppy"])):
            intrin[key] = value / mag
        intrin["width"], intrin["height"] = w, h
        return type(frame)(
            data, intrin, frame.timestamp_ms, frame._depth_scale, frame._frame_number
        )


class threshold_filter(_PassthroughFilter):
    """Zero out depth outside [min_distance, max_distance] metres."""

    def process(self, frame):
        lo = self.options.get(option.min_distance, 0.0) / wire.DEPTH_SCALE
        hi = self.options.get(option.max_distance, 16.0) / wire.DEPTH_SCALE
        data = frame._data
        keep = (data >= lo) & (data <= hi)
        return type(frame)(
            np.where(keep, data, 0).astype(np.uint16), frame._intrin,
            frame.timestamp_ms, frame._depth_scale, frame._frame_number,
        )


class align:
    """Reproject co-located pinhole depth into the selected color profile."""

    def __init__(self, to_stream):
        self.to_stream = to_stream

    def process(self, frames):
        if self.to_stream != stream.color or frames._depth is None or frames._color is None:
            return frames
        depth_frame, color_frame = frames._depth, frames._color
        if depth_frame._data.shape == color_frame._data.shape[:2] and depth_frame._intrin == color_frame._intrin:
            return frames
        depth = depth_frame._data
        source, target = depth_frame._intrin, color_frame._intrin
        yy, xx = np.indices(depth.shape)
        z = depth.astype(np.float32) * depth_frame._depth_scale
        x = (xx + 0.5 - source["ppx"]) / source["fx"] * z
        y = (yy + 0.5 - source["ppy"]) / source["fy"] * z
        valid = z > 0
        u = np.rint(x[valid] / z[valid] * target["fx"] + target["ppx"] - 0.5).astype(int)
        v = np.rint(y[valid] / z[valid] * target["fy"] + target["ppy"] - 0.5).astype(int)
        inside = (u >= 0) & (u < target["width"]) & (v >= 0) & (v < target["height"])
        flat = np.full(target["width"] * target["height"], np.iinfo(np.uint16).max, dtype=np.uint16)
        np.minimum.at(flat, v[inside] * target["width"] + u[inside], depth[valid][inside])
        flat[flat == np.iinfo(np.uint16).max] = 0
        aligned = flat.reshape(target["height"], target["width"])
        return _Composite(
            _DepthFrame(
                aligned, target, depth_frame.timestamp_ms,
                depth_frame._depth_scale, depth_frame._frame_number,
            ),
            color_frame,
        )
        return frames


class _Points:
    def __init__(self, vertices, texcoords):
        self._vertices = vertices
        self._texcoords = texcoords

    def get_vertices(self):
        return self._vertices

    def get_texture_coordinates(self):
        return self._texcoords

    def size(self):
        return self._vertices.shape[0]


class pointcloud:
    """Deprojects a depth frame to 3D and maps color onto it.

    Output matches the SDK's: one vertex per depth pixel in row-major order,
    with invalid (zero) depth deprojecting to the origin. The perception code
    relies on that -- it rejects points by ``|v|^2 > 1e-4`` rather than by a
    validity mask, so dropping the zeros here would misalign vertices against
    texture coordinates.
    """

    def __init__(self):
        self._color_frame = None

    def map_to(self, color_frame):
        self._color_frame = color_frame

    def calculate(self, depth_frame):
        depth = depth_frame._data
        intrin = depth_frame._intrin
        h, w = depth.shape

        z = depth.astype(np.float32) * depth_frame._depth_scale
        # Pixel centres, so a point at the principal point deprojects to
        # exactly (0, 0, z) the way the SDK's rs2_deproject_pixel_to_point does.
        u = np.arange(w, dtype=np.float32) + 0.5
        v = np.arange(h, dtype=np.float32) + 0.5
        uu, vv = np.meshgrid(u, v)

        x = (uu - intrin["ppx"]) / intrin["fx"] * z
        y = (vv - intrin["ppy"]) / intrin["fy"] * z
        vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)
        # Zero depth means "no reading"; the SDK emits the origin for those.
        vertices[z.reshape(-1) == 0] = 0.0

        # Texture coordinates are normalized into the *color* image. Depth is
        # aligned to color here, so a decimated depth pixel maps straight back
        # to its normalized position -- no reprojection needed.
        s = (uu / w).reshape(-1)
        t = (vv / h).reshape(-1)
        texcoords = np.stack([s, t], axis=-1).astype(np.float32)

        return _Points(vertices, texcoords)
