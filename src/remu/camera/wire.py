"""Wire format for the emulated-camera transport.

Deliberately trivial: a JSON control request, then a stream of length-prefixed
frames. The point is that the client side (``remu/shim/pyrealsense2.py``) can
speak it with nothing but the standard library and numpy, since it has to run
inside the pointcloud_perception container rather than the remu conda env.

Control (one JSON object per line, request then response):

    {"op": "list_devices"}
        -> {"devices": [{"serial": "...", "name": "..."}, ...]}
    {"op": "stream", "serial": "..."}
        -> {"ok": true}   followed by frames until the client disconnects

Frame::

    b"RMUC" | u32 header_len | header JSON | color bytes | depth bytes

``color`` is H*W*3 uint8 RGB and ``depth`` is H*W uint16 in millimetres --
the same formats ``rs.format.rgb8`` and ``rs.format.z16`` deliver, so the
shim hands the perception code byte-identical buffers to the real SDK's.
"""

import json
import struct

CAMERA_PORT = 1338

MAGIC = b"RMUC"
_PREFIX = struct.Struct("<4sI")

# Millimetres per depth unit, i.e. rs.depth_sensor.get_depth_scale(). Matches
# the D435i default so a client that scales by it gets metres back.
DEPTH_SCALE = 0.001


def encode_frame(header: dict, color_bytes: bytes, depth_bytes: bytes) -> bytes:
    """Serialize one frame. ``header`` must carry the payload sizes."""
    blob = json.dumps(header).encode("utf-8")
    return _PREFIX.pack(MAGIC, len(blob)) + blob + color_bytes + depth_bytes


def read_exactly(sock, size: int) -> bytes:
    """Read exactly ``size`` bytes, or raise ConnectionError on a short read.

    ``recv`` is free to return less than asked for on a stream socket; every
    payload here is fixed-size and known in advance, so a partial read is a
    framing error rather than something to paper over.
    """
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError(f"connection closed with {remaining} of {size} bytes left")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(sock):
    """Read one frame; returns ``(header, color_bytes, depth_bytes)``."""
    magic, header_len = _PREFIX.unpack(read_exactly(sock, _PREFIX.size))
    if magic != MAGIC:
        raise ConnectionError(f"bad frame magic {magic!r} -- stream out of sync")
    header = json.loads(read_exactly(sock, header_len).decode("utf-8"))
    color = read_exactly(sock, header["color_bytes"])
    depth = read_exactly(sock, header["depth_bytes"])
    return header, color, depth


def send_json(sock, obj: dict) -> None:
    sock.sendall((json.dumps(obj) + "\n").encode("utf-8"))


def recv_json(sock) -> dict:
    """Read one newline-terminated JSON object."""
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = sock.recv(1)
        if not chunk:
            raise ConnectionError("connection closed mid-request")
        buf += chunk
    return json.loads(buf.decode("utf-8"))
