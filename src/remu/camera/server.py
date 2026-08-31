"""TCP server publishing emulated RealSense frames.

The same idea as :class:`~remu.server.franka_server.FrankaFciServer`: rather
than patching the client, stand up something it can connect to and speak its
language. Here the "language" is the pyrealsense2 Python API, which
``remu/shim/pyrealsense2.py`` reimplements on top of this transport -- so
pointcloud_perception runs unmodified, in its own container, against a
simulated camera.

The physics thread only publishes a small, immutable state snapshot. A
dedicated render thread owns a second ``MjData`` plus every OpenGL context,
so GPU rendering, readback, encoding, and recording callbacks do not synchronously block
the 1 kHz control loop. Client threads only ever ship the most recently
rendered bytes.
"""

import logging
import os
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
        on_frame=None,
    ):
        self.cameras: List[EmulatedRgbdCamera] = list(cameras)
        serials = [camera.serial for camera in self.cameras]
        if len(set(serials)) != len(serials):
            raise ValueError("camera serials must be globally unique across vendors")
        self.host = host
        self.port = port
        self.on_frame = on_frame

        self._by_serial: Dict[str, EmulatedRgbdCamera] = {c.serial: c for c in self.cameras}
        self._frames: Dict[str, _LatestFrame] = {c.serial: _LatestFrame() for c in self.cameras}
        self._sim = None
        self._latest_state = None
        self._render_thread: Optional[threading.Thread] = None
        self._render_stop = threading.Event()
        self._render_ready = threading.Event()

        self.running = False
        self._sock: Optional[socket.socket] = None
        self._threads: List[threading.Thread] = []
        # Last render failure, kept so callers waiting on a first frame can
        # report *why* none arrived instead of just timing out.
        self.last_render_error: Optional[BaseException] = None

    # -- render side -------------------------------------------------------

    def attach(self, sim):
        """Publish simulation state to an asynchronous camera renderer."""
        if self._sim is sim:
            return self
        if self._sim is not None:
            raise RuntimeError("camera server is already attached to a simulator")
        self._sim = sim
        self.publish_state(sim.model, sim.data)
        sim.on_step_callbacks.append(self.publish_state)
        return self

    def publish_state(self, _model, data) -> None:
        """Atomically replace the latest render state; called at 1 kHz."""
        self._latest_state = (
            data.qpos.copy(),
            data.mocap_pos.copy(),
            data.mocap_quat.copy(),
            float(data.time),
        )

    @staticmethod
    def _drop_realtime_priority() -> None:
        """Keep camera work from competing with a real-time control thread."""
        try:
            os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
        except (AttributeError, OSError):
            logger.warning("could not set camera render thread to SCHED_OTHER", exc_info=True)

    def _render_loop(self) -> None:
        self._drop_realtime_priority()
        try:
            self._render_loop_inner()
        except Exception as exc:
            self.last_render_error = exc
            logger.exception("camera render worker failed")
        finally:
            self._render_ready.set()

    def _render_loop_inner(self) -> None:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(self._sim.scene_xml_path))
        data = mujoco.MjData(model)
        periods = {camera.serial: 1.0 / camera.fps for camera in self.cameras}
        start = time.perf_counter()
        count = len(self.cameras)
        deadlines = {
            camera.serial: start + index * periods[camera.serial] / count
            for index, camera in enumerate(self.cameras)
        }

        try:
            while self.running and not self._render_stop.is_set():
                now = time.perf_counter()
                due = sorted(
                    (camera for camera in self.cameras if deadlines[camera.serial] <= now),
                    key=lambda camera: deadlines[camera.serial],
                )
                state = self._latest_state
                if due and state is not None:
                    qpos, mocap_pos, mocap_quat, _sim_time = state
                    data.qpos[:] = qpos
                    data.mocap_pos[:] = mocap_pos
                    data.mocap_quat[:] = mocap_quat
                    mujoco.mj_forward(model, data)

                    for camera in due:
                        deadline = deadlines[camera.serial]
                        try:
                            self._render_one(camera, model, data)
                            if all(frame.frame_id for frame in self._frames.values()):
                                self._render_ready.set()
                        except Exception as exc:
                            self.last_render_error = exc
                            logger.exception("camera %s render failed", camera.serial)
                        period = periods[camera.serial]
                        elapsed_periods = max(
                            1, int((time.perf_counter() - deadline) // period) + 1
                        )
                        deadlines[camera.serial] = deadline + elapsed_periods * period
                    continue

                next_deadline = min(deadlines.values())
                timeout = max(0.0, min(next_deadline - now, 0.01))
                self._render_stop.wait(timeout if state is not None else 0.001)
        finally:
            for camera in self.cameras:
                camera.close()

    def _render_one(self, cam: EmulatedRgbdCamera, model, data) -> None:
        color, depth = cam.render(model, data)
        color_bytes = color.tobytes()
        depth_bytes = depth.tobytes()
        frame_number = self._frames[cam.serial].frame_id + 1
        header = {
            "protocol_version": 2,
            "vendor": cam.vendor,
            "model": cam.model,
            "serial": cam.serial,
            "frame_number": frame_number,
            "timestamp_ms": time.time() * 1000.0,
            "depth_scale": cam.depth_scale,
            "color": cam.color_profile.to_dict(),
            "depth": cam.depth_profile.to_dict(),
            "color_bytes": len(color_bytes),
            "depth_bytes": len(depth_bytes),
        }
        if self.on_frame is not None:
            self.on_frame(cam, color_bytes, depth_bytes, frame_number)
        self._frames[cam.serial].publish(encode_frame(header, color_bytes, depth_bytes))

    # -- network side ------------------------------------------------------

    def start(self, background: bool = True):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self.port = self._sock.getsockname()[1]
        self._sock.listen(8)
        self.running = True
        self._render_stop.clear()
        self._render_ready.clear()
        if self._sim is not None:
            self._render_thread = threading.Thread(
                target=self._render_loop, name="remu-camera-render", daemon=True
            )
            self._render_thread.start()
            if not self._render_ready.wait(timeout=15.0):
                logger.warning("camera renderer did not produce its first frames within 15 seconds")
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
        self._render_stop.set()
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        if self._render_thread is not None:
            self._render_thread.join(timeout=5.0)
            if self._render_thread.is_alive():
                logger.warning("camera render thread did not stop within 5 seconds")
            self._render_thread = None
        if self._sim is not None:
            try:
                self._sim.on_step_callbacks.remove(self.publish_state)
            except ValueError:
                pass
            self._sim = None
        logger.info("camera server stopped")
