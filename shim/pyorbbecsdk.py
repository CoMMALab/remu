"""Small Orbbec SDK v2-compatible facade backed by remu's camera server.

Place this directory first on ``PYTHONPATH`` so ``import pyorbbecsdk`` uses
the simulated Femto Mega or Gemini devices declared in remu's camera YAML.
"""

import importlib.util
import os
import socket

import numpy as np

_WIRE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "remu", "camera", "wire.py",
)
_spec = importlib.util.spec_from_file_location("_remu_orbbec_wire", _WIRE_PATH)
if _spec is None:
    raise ImportError(f"remu camera wire module not found at {_WIRE_PATH}")
wire = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wire)

DEFAULT_ADDR = ("127.0.0.1", wire.CAMERA_PORT)

#: Port the SDK dials for a network device, matching remu's
#: camera.config.DEFAULT_NET_PORT. This is the *emulated device* address,
#: which has nothing to do with REMU_CAMERA_ADDR -- that is where remu's
#: own frame server lives.
DEFAULT_NET_PORT = 8090


def _with_transport(info, transport):
    """Record how a device handle was opened.

    One unit answers get_connection_type() differently depending on the path
    used to reach it: a Femto Mega reports "Ethernet" through
    create_net_device and "USB2.1" through enumeration, at the same time
    (confirmed live on CL25854007B). Carrying it on the record keeps that
    true through Pipeline.get_device(), which rebuilds a Device from it.
    """
    tagged = dict(info)
    tagged["_transport"] = transport
    return tagged


def _server_addr():
    raw = os.environ.get("REMU_CAMERA_ADDR")
    if not raw:
        return DEFAULT_ADDR
    host, _, port = raw.partition(":")
    return host or DEFAULT_ADDR[0], int(port) if port else DEFAULT_ADDR[1]


def _connect(request, timeout=5.0):
    sock = socket.create_connection(_server_addr(), timeout=timeout)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    wire.send_json(sock, request)
    response = wire.recv_json(sock)
    if "error" in response:
        sock.close()
        raise OBError(response["error"])
    return sock, response


class OBError(RuntimeError):
    pass


class OBFormat:
    UNKNOWN_FORMAT = "unknown"
    RGB = "rgb8"
    Z16 = "z16"
    Y16 = "z16"
    POINT = "point"
    RGB_POINT = "rgb_point"


class OBSensorType:
    COLOR_SENSOR = "color"
    DEPTH_SENSOR = "depth"


class OBStreamType:
    COLOR_STREAM = "color"
    DEPTH_STREAM = "depth"


class OBAlignMode:
    DISABLE = "disable"
    HW_MODE = "hardware"
    SW_MODE = "software"


class OBFrameAggregateOutputMode:
    FULL_FRAME_REQUIRE = "full_frame_require"


class DeviceInfo:
    def __init__(self, info):
        self._info = info

    def get_name(self):
        return self._info["name"]

    def get_serial_number(self):
        return self._info["serial"]

    def get_vid(self):
        return 0x2BC5

    def get_pid(self):
        # Product IDs are informational in the emulator. Keep Gemini distinct
        # so applications do not classify it as a Femto Mega.
        return 0x0800 if self._info["model"].startswith("gemini") else 0x0669

    def get_firmware_version(self):
        return "2.0.0-remu"

    def get_uid(self):
        return self.get_serial_number()

    def get_connection_type(self):
        return self._info.get("_transport", "USB3.0")


class Device:
    def __init__(self, info):
        self._info = info

    def get_device_info(self):
        return DeviceInfo(self._info)


class DeviceList:
    def __init__(self, devices, transport="USB3.0"):
        self._devices = [
            Device(_with_transport(info, transport)) for info in devices
        ]

    def get_count(self):
        return len(self._devices)

    def get_device_by_index(self, index):
        return self._devices[index]

    def get_device_serial_number_by_index(self, index):
        return self._devices[index].get_device_info().get_serial_number()

    def get_device_name_by_index(self, index):
        return self._devices[index].get_device_info().get_name()

    def __len__(self):
        return len(self._devices)

    def __getitem__(self, index):
        return self._devices[index]


class Context:
    def query_devices(self):
        """Enumerate devices reachable over USB.

        An Ethernet-only unit is withheld here on purpose: the real SDK
        surfaces one only after enable_net_device_enumeration() broadcast
        discovery finds it, which needs both hosts on one L2 segment.
        Listing it unconditionally would let a config that works against the
        emulator fail against the hardware.
        """
        try:
            sock, response = _connect({"op": "list_devices", "vendor": "orbbec"})
        except OSError:
            return DeviceList([])
        try:
            return DeviceList(
                [d for d in response["devices"] if not d.get("network_only")],
                transport="USB3.0",
            )
        finally:
            sock.close()

    def create_net_device(self, ip, port=DEFAULT_NET_PORT):
        """Open a device by address, the way Ethernet units are reached.

        Returns ``None`` when nothing answers at that address rather than
        raising, because that is what the real SDK does and what callers
        check for -- see fr3-teleop's OrbbecCamera._acquire_device.
        """
        try:
            sock, response = _connect({"op": "list_devices", "vendor": "orbbec"})
        except OSError:
            return None
        try:
            for info in response["devices"]:
                if info.get("ip") != ip:
                    continue
                if int(info.get("port") or DEFAULT_NET_PORT) != int(port):
                    continue
                return Device(_with_transport(info, "Ethernet"))
            return None
        finally:
            sock.close()


class VideoStreamProfile:
    def __init__(self, stream_type, width, height, fmt, fps, intrinsics=None):
        self.stream_type = stream_type
        self.width = width
        self.height = height
        self.format = fmt
        self.fps = fps
        self._intrinsics = intrinsics

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_fps(self):
        return self.fps

    def get_format(self):
        return self.format

    def get_type(self):
        return self.stream_type

    def get_intrinsic(self):
        return self._intrinsics

    def as_video_stream_profile(self):
        return self


class StreamProfileList:
    def __init__(self, profiles):
        self._profiles = profiles

    def get_count(self):
        return len(self._profiles)

    def get_default_video_stream_profile(self):
        if not self._profiles:
            raise OBError("no stream profiles")
        return self._profiles[0]

    def get_video_stream_profile(self, width=0, height=0, fmt=OBFormat.UNKNOWN_FORMAT, fps=0):
        for profile in self._profiles:
            if width not in (0, profile.width) or height not in (0, profile.height):
                continue
            if fmt not in (OBFormat.UNKNOWN_FORMAT, profile.format):
                continue
            if fps not in (0, profile.fps):
                continue
            return profile
        raise OBError("requested video stream profile is unavailable")

    def __len__(self):
        return len(self._profiles)

    def __getitem__(self, index):
        return self._profiles[index]


class Config:
    def __init__(self):
        self.streams = []
        self.align_mode = OBAlignMode.DISABLE

    def enable_stream(self, profile):
        self.streams = [item for item in self.streams if item.stream_type != profile.stream_type]
        self.streams.append(profile)

    def disable_all_stream(self):
        self.streams = []

    def disable_all_streams(self):
        self.disable_all_stream()

    def set_align_mode(self, mode):
        self.align_mode = mode

    def set_frame_aggregate_output_mode(self, _mode):
        pass

    def get_enabled_stream_profile_list(self):
        return StreamProfileList(list(self.streams))


class _Frame:
    def __init__(self, data, profile, timestamp_ms, frame_number=0):
        self._data = data
        self._profile = profile
        self._timestamp_ms = timestamp_ms
        self._frame_number = frame_number

    def __bool__(self):
        return self._data is not None and self._data.size > 0

    def get_data(self):
        return self._data

    def get_width(self):
        return self._data.shape[1]

    def get_height(self):
        return self._data.shape[0]

    def get_timestamp(self):
        return self._timestamp_ms

    def get_timestamp_us(self):
        return int(self._timestamp_ms * 1000)

    def get_index(self):
        return self._frame_number

    def get_frame_number(self):
        return self._frame_number

    def get_stream_profile(self):
        return self._profile


class DepthFrame(_Frame):
    def __init__(self, data, profile, timestamp_ms, scale, frame_number=0):
        super().__init__(data, profile, timestamp_ms, frame_number)
        self._scale = scale

    def get_depth_scale(self):
        return self._scale * 1000.0


class ColorFrame(_Frame):
    pass


class FrameSet:
    def __init__(self, depth, color):
        self._depth = depth
        self._color = color

    def get_depth_frame(self):
        return self._depth

    def get_color_frame(self):
        return self._color

    def as_frame_set(self):
        return self

    def __bool__(self):
        return self._depth is not None or self._color is not None


def _profiles(info):
    return {
        kind: VideoStreamProfile(
            kind, meta["width"], meta["height"], meta["format"], info["fps"],
            meta.get("intrinsics"),
        )
        for kind, meta in (("color", info["color"]), ("depth", info["depth"]))
    }


class Pipeline:
    def __init__(self, device=None):
        self._sock = None
        self._device_info = device._info if device is not None else None
        self._enabled = {"color", "depth"}
        self._align_mode = OBAlignMode.DISABLE

    def _ensure_device(self):
        if self._device_info is None:
            devices = Context().query_devices()
            if devices.get_count() == 0:
                raise OBError("No device found")
            self._device_info = devices.get_device_by_index(0)._info

    def get_stream_profile_list(self, sensor_type):
        self._ensure_device()
        profiles = _profiles(self._device_info)
        profile = profiles.get(sensor_type)
        return StreamProfileList([] if profile is None else [profile])

    def get_device(self):
        self._ensure_device()
        return Device(self._device_info)

    def get_d2c_depth_profile_list(self, _color_profile, _align_mode):
        return self.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)

    def get_camera_param(self):
        self._ensure_device()
        return {kind: profile.get_intrinsic() for kind, profile in _profiles(self._device_info).items()}

    def enable_frame_sync(self):
        pass

    def start(self, config=None):
        self._ensure_device()
        selected = [] if config is None else config.streams
        requests = [
            {"stream": p.stream_type, "width": p.width, "height": p.height,
             "format": p.format, "fps": p.fps}
            for p in selected
        ]
        self._sock, response = _connect({
            "op": "stream", "vendor": "orbbec",
            "serial": self._device_info["serial"], "streams": requests,
        })
        self._device_info = response["device"]
        self._enabled = {p.stream_type for p in selected} if selected else {"color", "depth"}
        self._align_mode = config.align_mode if config is not None else OBAlignMode.DISABLE

    def wait_for_frames(self, timeout_ms=5000):
        if self._sock is None:
            raise OBError("pipeline not started")
        self._sock.settimeout(timeout_ms / 1000.0)
        try:
            header, color_bytes, depth_bytes = wire.read_frame(self._sock)
        except socket.timeout:
            return None
        except (ConnectionError, OSError) as exc:
            raise OBError(f"camera stream lost: {exc}") from exc
        profiles = _profiles({
            "fps": self._device_info["fps"], "color": header["color"], "depth": header["depth"]
        })
        color_meta, depth_meta = header["color"], header["depth"]
        color_data = np.frombuffer(color_bytes, np.uint8).reshape(
            color_meta["height"], color_meta["width"], 3
        )
        depth_data = np.frombuffer(depth_bytes, np.uint16).reshape(
            depth_meta["height"], depth_meta["width"]
        )
        timestamp = header["timestamp_ms"]
        frame_number = header.get("frame_number", 0)
        color = ColorFrame(color_data, profiles["color"], timestamp, frame_number) if "color" in self._enabled else None
        depth = DepthFrame(
            depth_data, profiles["depth"], timestamp,
            header.get("depth_scale", wire.DEPTH_SCALE), frame_number,
        ) if "depth" in self._enabled else None
        frames = FrameSet(depth, color)
        if self._align_mode != OBAlignMode.DISABLE:
            frames = _align_to_color(frames)
        return frames

    def stop(self):
        if self._sock is not None:
            self._sock.close()
            self._sock = None


def _align_to_color(frames):
    depth_frame, color_frame = frames.get_depth_frame(), frames.get_color_frame()
    if depth_frame is None or color_frame is None:
        return frames
    source = depth_frame.get_stream_profile().get_intrinsic()
    target = color_frame.get_stream_profile().get_intrinsic()
    if depth_frame._data.shape == color_frame._data.shape[:2] and source == target:
        return frames
    depth = depth_frame._data
    yy, xx = np.indices(depth.shape)
    z = depth.astype(np.float32) * depth_frame._scale
    valid = z > 0
    x = (xx + 0.5 - source["ppx"]) / source["fx"] * z
    y = (yy + 0.5 - source["ppy"]) / source["fy"] * z
    u = np.rint(x[valid] / z[valid] * target["fx"] + target["ppx"] - 0.5).astype(int)
    v = np.rint(y[valid] / z[valid] * target["fy"] + target["ppy"] - 0.5).astype(int)
    inside = (u >= 0) & (u < target["width"]) & (v >= 0) & (v < target["height"])
    flat = np.full(target["width"] * target["height"], 65535, np.uint16)
    np.minimum.at(flat, v[inside] * target["width"] + u[inside], depth[valid][inside])
    flat[flat == 65535] = 0
    profile = VideoStreamProfile(
        "depth", target["width"], target["height"], OBFormat.Z16,
        depth_frame._profile.fps, target,
    )
    return FrameSet(DepthFrame(flat.reshape(target["height"], target["width"]), profile,
                               depth_frame._timestamp_ms, depth_frame._scale,
                               depth_frame._frame_number), color_frame)


class AlignFilter:
    def __init__(self, align_to_stream=OBStreamType.COLOR_STREAM):
        self.align_to_stream = align_to_stream

    def process(self, frames):
        return _align_to_color(frames) if self.align_to_stream == OBStreamType.COLOR_STREAM else frames


class _PointFrame(_Frame):
    pass


class PointCloudFilter:
    def __init__(self):
        self._format = OBFormat.POINT

    def set_create_point_format(self, fmt):
        self._format = fmt

    def set_position_data_scaled(self, _scale):
        pass

    def set_camera_param(self, _param):
        pass

    def process(self, frames):
        if self._format == OBFormat.RGB_POINT and isinstance(frames, FrameSet):
            frames = _align_to_color(frames)
        if isinstance(frames, DepthFrame):
            depth_frame, color_frame = frames, None
        else:
            depth_frame, color_frame = frames.get_depth_frame(), frames.get_color_frame()
        if depth_frame is None:
            return None
        depth = depth_frame._data
        intrinsics = depth_frame._profile.get_intrinsic()
        yy, xx = np.indices(depth.shape)
        z = depth.astype(np.float32) * depth_frame._scale
        x = (xx + 0.5 - intrinsics["ppx"]) / intrinsics["fx"] * z
        y = (yy + 0.5 - intrinsics["ppy"]) / intrinsics["fy"] * z
        xyz = np.stack((x, y, z), axis=-1).reshape(-1, 3).astype(np.float32)
        xyz[z.reshape(-1) == 0] = 0
        if self._format == OBFormat.RGB_POINT and color_frame is not None:
            color = color_frame._data
            data = np.concatenate((xyz, color.reshape(-1, 3).astype(np.float32)), axis=1)
        else:
            data = xyz
        return _PointFrame(
            data, depth_frame._profile, depth_frame._timestamp_ms, depth_frame._frame_number
        )

    def calculate(self, frame):
        return frame.get_data()
