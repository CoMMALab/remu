"""TCP server publishing emulated RealSense frames.

The same idea as :class:`~remu.server.franka_server.FrankaFciServer`: rather
than patching the client, stand up something it can connect to and speak its
language. Here the "language" is the pyrealsense2 Python API, which
``remu/shim/pyrealsense2.py`` reimplements on top of this transport -- so
pointcloud_perception runs unmodified, in its own container, against a
simulated camera.

Rendering happens on the physics thread via :meth:`CameraServer.attach`, not
on the client threads. Two reasons: ``MjData`` would otherwise be read while
it is being stepped, and each ``mujoco.Renderer`` owns an OpenGL context that
cannot be used from a thread other than the one that created it. Client
threads only ever ship the most recently rendered bytes.
"""

import logging
import socket
import threading
import time
from typing import Dict, List, Optional, Sequence

from .rgbd import EmulatedRgbdCamera
from .wire import CAMERA_PORT, encode_frame, recv_json, send_json

logger = logging.getLogger(__name__)


class _LatestFrame:
    """Single-slot mailbox for one camera's most recent encoded frame.

    Publishing is a plain attribute assignment, which is atomic under the GIL,
    so the renderer never blocks on a slow client and clients never block the
    physics loop. A client that can't keep up drops frames rather than
    queueing them -- correct for a live camera, where a stale frame is worse
    than a missing one.
    """

    def __init__(self):
        self.payload: Optional[bytes] = None
        self.frame_id = 0
        self._new_frame = threading.Condition()

    def publish(self, payload: bytes) -> None:
        with self._new_frame:
            self.payload = payload
            self.frame_id += 1
            self._new_frame.notify_all()

    def wait_for_next(self, last_seen: int, timeout: float):
        """Block until a frame newer than ``last_seen`` arrives.

        Returns ``(payload, frame_id)``, or ``(None, last_seen)`` on timeout.
        """
        with self._new_frame:
            if self.frame_id <= last_seen:
                self._new_frame.wait(timeout)
            if self.frame_id <= last_seen:
                return None, last_seen
            return self.payload, self.frame_id


class CameraServer:
    """Serves one or more vendor-neutral emulated RGB-D cameras over TCP."""

    def __init__(
        self,
        cameras: Sequence[EmulatedRgbdCamera],
        host: str = "0.0.0.0",
        port: int = CAMERA_PORT,
    ):
        self.cameras: List[EmulatedRgbdCamera] = list(cameras)
        serials = [camera.serial for camera in self.cameras]
        if len(set(serials)) != len(serials):
            raise ValueError("camera serials must be globally unique across vendors")
        self.host = host
        self.port = port

        self._by_serial: Dict[str, EmulatedRgbdCamera] = {c.serial: c for c in self.cameras}
        self._frames: Dict[str, _LatestFrame] = {c.serial: _LatestFrame() for c in self.cameras}
        self._last_render: Dict[str, float] = {c.serial: 0.0 for c in self.cameras}

        self.running = False
        self._sock: Optional[socket.socket] = None
        self._threads: List[threading.Thread] = []
        # Last render failure, kept so callers waiting on a first frame can
        # report *why* none arrived instead of just timing out.
        self.last_render_error: Optional[BaseException] = None

    # -- render side (physics thread) --------------------------------------

    def attach(self, sim):
        """Render each camera at its own fps as part of ``sim``'s step loop."""
        sim.on_step_callbacks.append(self.render_due)
        return self

    def render_due(self, model, data) -> None:
        """Render whichever cameras are due for a new frame. Called per step."""
        now = time.monotonic()
        for cam in self.cameras:
            if now - self._last_render[cam.serial] < 1.0 / cam.fps:
                continue
            self._last_render[cam.serial] = now
            try:
                self._render_one(cam, model, data)
            except Exception as exc:
                # A render failure must not kill the physics loop -- the arm
                # keeps running and the camera stream simply stalls.
                self.last_render_error = exc
                logger.exception("camera %s render failed", cam.serial)

    def _render_one(self, cam: EmulatedRgbdCamera, model, data) -> None:
        color, depth = cam.render(model, data)
        color_bytes = color.tobytes()
        depth_bytes = depth.tobytes()
        header = {
            "protocol_version": 2,
            "vendor": cam.vendor,
            "model": cam.model,
            "serial": cam.serial,
            "frame_number": self._frames[cam.serial].frame_id + 1,
            "timestamp_ms": time.time() * 1000.0,
            "depth_scale": cam.depth_scale,
            "color": cam.color_profile.to_dict(),
            "depth": cam.depth_profile.to_dict(),
            "color_bytes": len(color_bytes),
            "depth_bytes": len(depth_bytes),
        }
        self._frames[cam.serial].publish(encode_frame(header, color_bytes, depth_bytes))

    # -- network side ------------------------------------------------------

    def start(self, background: bool = True):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(8)
        self.running = True
        logger.info(
            "camera server on %s:%d serving %s",
            self.host, self.port, ", ".join(c.serial for c in self.cameras),
        )
        if not background:
            return self._accept_loop()
        thread = threading.Thread(target=self._accept_loop, daemon=True)
        thread.start()
        self._threads.append(thread)
        return self

    def _accept_loop(self) -> None:
        while self.running:
            try:
                client, addr = self._sock.accept()
            except OSError:
                # Expected on stop(): the listening socket is closed out from
                # under accept() to break out of this loop.
                if self.running:
                    logger.exception("camera server accept failed")
                return
            thread = threading.Thread(target=self._serve_client, args=(client, addr), daemon=True)
            thread.start()
            self._threads.append(thread)

    def _serve_client(self, client: socket.socket, addr) -> None:
        try:
            request = recv_json(client)
            op = request.get("op")
            if op == "list_devices":
                vendor = request.get("vendor")
                send_json(client, {"devices": [
                    c.device_dict()
                    for c in self.cameras
                    if vendor is None or c.vendor == vendor
                ]})
            elif op == "stream":
                self._stream(client, request)
            else:
                send_json(client, {"error": f"unknown op {op!r}"})
        except (ConnectionError, OSError) as e:
            logger.debug("camera client %s disconnected: %s", addr, e)
        except Exception:
            logger.exception("camera client %s failed", addr)
        finally:
            client.close()

    def _stream(self, client: socket.socket, request: dict) -> None:
        # A stream request with no serial means "whichever camera you have",
        # matching pyrealsense2's behaviour when config.enable_device is never
        # called. It is only unambiguous with exactly one camera.
        serial = request.get("serial")
        vendor = request.get("vendor")
        candidates = [c for c in self.cameras if vendor is None or c.vendor == vendor]
        if serial is None and len(candidates) == 1:
            serial = candidates[0].serial
        if serial not in self._by_serial:
            send_json(client, {"error": f"no such device {serial!r}"})
            return

        camera = self._by_serial[serial]
        if vendor is not None and camera.vendor != vendor:
            send_json(client, {"error": f"no such {vendor} device {serial!r}"})
            return
        for stream_request in request.get("streams", []):
            kind = stream_request.get("stream")
            profile = getattr(camera, f"{kind}_profile", None)
            if profile is None:
                send_json(client, {"error": f"unsupported stream {kind!r}"})
                return
            expected = (profile.width, profile.height, profile.format, camera.fps)
            actual = (
                stream_request.get("width"), stream_request.get("height"),
                stream_request.get("format"), stream_request.get("fps"),
            )
            if any(value is not None and value != wanted for value, wanted in zip(actual, expected)):
                send_json(client, {
                    "error": f"profile {kind} {actual} is not configured; expected {expected}"
                })
                return

        send_json(client, {"ok": True, "device": camera.device_dict()})
        mailbox = self._frames[serial]
        last_seen = 0
        while self.running:
            payload, last_seen = mailbox.wait_for_next(last_seen, timeout=1.0)
            if payload is None:
                continue
            client.sendall(payload)

    def stop(self) -> None:
        self.running = False
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        # Renderers are deliberately *not* closed here: their OpenGL contexts
        # belong to the physics thread that created them, and freeing a
        # context from another thread crashes rather than erroring. They are
        # released when the process exits, or by whoever owns the sim loop.
        logger.info("camera server stopped")
