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

from .d435i import EmulatedD435i
from .wire import CAMERA_PORT, DEPTH_SCALE, encode_frame, recv_json, send_json

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
    """Serves one or more :class:`EmulatedD435i` over TCP."""

    def __init__(
        self,
        cameras: Sequence[EmulatedD435i],
        host: str = "0.0.0.0",
        port: int = CAMERA_PORT,
    ):
        self.cameras: List[EmulatedD435i] = list(cameras)
        self.host = host
        self.port = port

        self._by_serial: Dict[str, EmulatedD435i] = {c.serial: c for c in self.cameras}
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

    def _render_one(self, cam: EmulatedD435i, model, data) -> None:
        color, depth = cam.render(model, data)
        color_bytes = color.tobytes()
        depth_bytes = depth.tobytes()
        header = {
            "serial": cam.serial,
            "width": cam.width,
            "height": cam.height,
            "timestamp_ms": time.time() * 1000.0,
            "depth_scale": DEPTH_SCALE,
            "intrinsics": cam.intrinsics.to_dict(),
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
                send_json(client, {"devices": [
                    {"serial": c.serial, "name": "Intel RealSense D435I",
                     "width": c.width, "height": c.height, "fps": c.fps}
                    for c in self.cameras
                ]})
            elif op == "stream":
                self._stream(client, request.get("serial"))
            else:
                send_json(client, {"error": f"unknown op {op!r}"})
        except (ConnectionError, OSError) as e:
            logger.debug("camera client %s disconnected: %s", addr, e)
        except Exception:
            logger.exception("camera client %s failed", addr)
        finally:
            client.close()

    def _stream(self, client: socket.socket, serial: Optional[str]) -> None:
        # A stream request with no serial means "whichever camera you have",
        # matching pyrealsense2's behaviour when config.enable_device is never
        # called. It is only unambiguous with exactly one camera.
        if serial is None and len(self.cameras) == 1:
            serial = self.cameras[0].serial
        if serial not in self._by_serial:
            send_json(client, {"error": f"no such device {serial!r}"})
            return

        send_json(client, {"ok": True})
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
