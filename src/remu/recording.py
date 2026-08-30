"""Canonical MCAP recording primitives shared by persistent and ephemeral modes."""

from __future__ import annotations

import heapq
import os
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from mcap.reader import make_reader
from mcap.writer import CompressionType, Writer as McapWriter
from mcap_protobuf.decoder import DecoderFactory
from mcap_protobuf.schema import register_schema

from remu.proto import recording_pb2

FORMAT_PROFILE = "remu/1"
LIBRARY = "remu"
TOPIC_SIM_STATE = "/remu/sim/state"
TOPIC_FCI_COMMAND = "/remu/fci/command"
TOPIC_FCI_STATE = "/remu/fci/state"
TOPIC_GRIPPER_COMMAND = "/remu/gripper/command"
TOPIC_GRIPPER_STATE = "/remu/gripper/state"
TOPIC_EVENTS = "/remu/events"
TOPIC_RAW = "/remu/debug/raw"


def color_topic(camera_name: str) -> str:
    return f"/remu/camera/{camera_name}/color"


def depth_topic(camera_name: str) -> str:
    return f"/remu/camera/{camera_name}/depth"


@dataclass(frozen=True)
class Attachment:
    media_type: str
    data: bytes


class RecordingWriter:
    """Low-level Protobuf MCAP writer with metadata and attachments."""

    def __init__(
        self,
        output: str | Path,
        *,
        metadata: Mapping[str, str] | None = None,
        attachments: Mapping[str, Attachment] | None = None,
        chunk_size: int = 4 * 1024 * 1024,
    ):
        self.path = Path(output)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.chunk_size = int(chunk_size)
        self._stream = self.path.open("wb")
        self._writer = McapWriter(
            self._stream,
            chunk_size=chunk_size,
            compression=CompressionType.ZSTD,
            enable_crcs=True,
            enable_data_crcs=True,
        )
        self._writer.start(profile=FORMAT_PROFILE, library=LIBRARY)
        self._last_flush_size = self._stream.tell()
        self._channels: dict[tuple[str, str], int] = {}
        self._schemas: dict[str, int] = {}
        self._finished = False
        if metadata:
            self._writer.add_metadata("remu", dict(metadata))
        for name, attachment in (attachments or {}).items():
            self._writer.add_attachment(
                create_time=0,
                log_time=0,
                name=name,
                media_type=attachment.media_type,
                data=attachment.data,
            )

    def write(
        self,
        topic: str,
        message: Any,
        *,
        log_time_ns: int,
        publish_time_ns: int | None = None,
        sequence: int = 0,
    ) -> None:
        message_type = type(message)
        schema_name = message_type.DESCRIPTOR.full_name
        schema_id = self._schemas.get(schema_name)
        if schema_id is None:
            schema_id = register_schema(self._writer, message_type)
            self._schemas[schema_name] = schema_id
        key = (topic, schema_name)
        channel_id = self._channels.get(key)
        if channel_id is None:
            channel_id = self._writer.register_channel(
                topic=topic,
                message_encoding="protobuf",
                schema_id=schema_id,
            )
            self._channels[key] = channel_id
        self._writer.add_message(
            channel_id=channel_id,
            log_time=int(log_time_ns),
            publish_time=int(log_time_ns if publish_time_ns is None else publish_time_ns),
            sequence=int(sequence),
            data=message.SerializeToString(),
        )
        if self.size - self._last_flush_size >= self.chunk_size:
            self._stream.flush()
            self._last_flush_size = self.size

    @property
    def size(self) -> int:
        return self._stream.tell()

    def finish(self) -> None:
        if self._finished:
            return
        self._writer.finish()
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stream.close()
        self._finished = True

    def __enter__(self) -> "RecordingWriter":
        return self

    def __exit__(self, *_exc) -> None:
        self.finish()


@dataclass(frozen=True)
class _QueuedMessage:
    topic: str
    message: Any
    log_time_ns: int
    publish_time_ns: int | None
    sequence: int


@dataclass(frozen=True)
class _QueuedBatch:
    messages: tuple[_QueuedMessage, ...]


class AsyncRecorder:
    """One-writer MCAP recorder; producers only enqueue immutable messages."""

    def __init__(
        self,
        output: str | Path,
        *,
        metadata: Mapping[str, str] | None = None,
        attachments: Mapping[str, Attachment] | None = None,
        chunk_size: int = 4 * 1024 * 1024,
        queue_size: int = 16384,
        rotate_size_bytes: int = 0,
        rotate_duration_ns: int = 0,
    ):
        self.output = Path(output)
        self.metadata = dict(metadata or {})
        self.attachments = dict(attachments or {})
        self.chunk_size = int(chunk_size)
        self.rotate_size_bytes = int(rotate_size_bytes)
        self.rotate_duration_ns = int(rotate_duration_ns)
        self._queue: queue.Queue[_QueuedMessage | _QueuedBatch | None] = queue.Queue(
            maxsize=queue_size
        )
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._started = False
        self.message_count = 0
        self.outputs: list[Path] = []

    def start(self) -> None:
        if self._started:
            raise RuntimeError("recorder can only be started once")
        self._started = True
        self._thread = threading.Thread(target=self._run, name="remu-mcap-writer", daemon=True)
        self._thread.start()

    def record(
        self,
        topic: str,
        message: Any,
        *,
        log_time_ns: int,
        publish_time_ns: int | None = None,
        sequence: int = 0,
        drop_if_full: bool = False,
    ) -> bool:
        if not self._started or self._thread is None:
            raise RuntimeError("recorder has not been started")
        if self._error is not None:
            raise RuntimeError("MCAP writer failed") from self._error
        item = _QueuedMessage(topic, message, int(log_time_ns), publish_time_ns, int(sequence))
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            if drop_if_full:
                return False
            raise RuntimeError("MCAP writer queue exhausted; refusing to lose numeric state")
        self.message_count += 1
        return True

    def record_batch(
        self,
        messages: Sequence[tuple[str, Any, int, int]],
        *,
        drop_if_full: bool = False,
    ) -> bool:
        """Atomically enqueue related messages, such as one RGB-D frame pair."""
        if not self._started or self._thread is None:
            raise RuntimeError("recorder has not been started")
        if self._error is not None:
            raise RuntimeError("MCAP writer failed") from self._error
        batch = _QueuedBatch(tuple(
            _QueuedMessage(topic, message, int(log_time_ns), None, int(sequence))
            for topic, message, log_time_ns, sequence in messages
        ))
        try:
            self._queue.put_nowait(batch)
        except queue.Full:
            if drop_if_full:
                return False
            raise RuntimeError("MCAP writer queue exhausted; refusing to lose message batch")
        self.message_count += len(batch.messages)
        return True

    def stop(self) -> int:
        if self._thread is None:
            return self.message_count
        while self._thread.is_alive():
            try:
                self._queue.put(None, timeout=0.1)
                break
            except queue.Full:
                if self._error is not None:
                    break
        self._thread.join()
        self._thread = None
        if self._error is not None:
            raise RuntimeError("MCAP writer failed") from self._error
        return self.message_count

    def _part_path(self, part: int) -> Path:
        if part == 0:
            return self.output
        return self.output.with_name(f"{self.output.stem}.{part:04d}{self.output.suffix}")

    def _run(self) -> None:
        writer = None
        try:
            part = 0
            part_start_ns = None
            while True:
                queued = self._queue.get()
                if queued is None:
                    break
                items = queued.messages if isinstance(queued, _QueuedBatch) else (queued,)
                for item in items:
                    should_rotate = writer is not None and (
                        (self.rotate_size_bytes and writer.size >= self.rotate_size_bytes)
                        or (
                            self.rotate_duration_ns
                            and part_start_ns is not None
                            and item.log_time_ns - part_start_ns >= self.rotate_duration_ns
                        )
                    )
                    if should_rotate:
                        writer.finish()
                        writer = None
                        part += 1
                    if writer is None:
                        path = self._part_path(part)
                        part_metadata = dict(self.metadata)
                        part_metadata["rotation_part"] = str(part)
                        writer = RecordingWriter(
                            path,
                            metadata=part_metadata,
                            attachments=self.attachments,
                            chunk_size=self.chunk_size,
                        )
                        self.outputs.append(path)
                        part_start_ns = item.log_time_ns
                    writer.write(
                        item.topic,
                        item.message,
                        log_time_ns=item.log_time_ns,
                        publish_time_ns=item.publish_time_ns,
                        sequence=item.sequence,
                    )
            if writer is not None:
                writer.finish()
        except BaseException as exc:
            self._error = exc


class RunRecorder:
    """Translate decoded Remu activity into the canonical channel vocabulary."""

    LEVELS = {"minimal", "standard", "debug"}

    def __init__(
        self,
        output: str | Path,
        *,
        level: str = "standard",
        metadata: Mapping[str, str] | None = None,
        attachments: Mapping[str, Attachment] | None = None,
        chunk_size: int = 4 * 1024 * 1024,
        rotate_size_bytes: int = 0,
        rotate_duration_ns: int = 0,
    ):
        if level not in self.LEVELS:
            raise ValueError(f"recording level must be one of {sorted(self.LEVELS)}")
        recording_metadata = dict(metadata or {})
        recording_metadata.update({
            "format": "remu-mcap",
            "format_version": "1",
            "recording_level": level,
        })
        self.level = level
        self.writer = AsyncRecorder(
            output,
            metadata=recording_metadata,
            attachments=attachments,
            chunk_size=chunk_size,
            rotate_size_bytes=rotate_size_bytes,
            rotate_duration_ns=rotate_duration_ns,
        )
        self.sim = None
        self.tick_index = 0
        self.sim_time_ns = 0
        self.fci_active = False
        self.active = False
        self._sequences: dict[str, int] = {}
        self._sequence_lock = threading.Lock()
        self.dropped_camera_frames = 0

    def start(self) -> None:
        self.writer.start()
        self.active = True
        self.event("recording_started", "MCAP recording started")

    def attach(self, sim) -> "RunRecorder":
        self.sim = sim
        sim.on_step_callbacks.append(self.on_step)
        return self

    def stop(self) -> int:
        event_error = None
        try:
            if self.dropped_camera_frames:
                self.event(
                    "camera_frames_dropped",
                    f"dropped {self.dropped_camera_frames} frames because the writer queue was full",
                    {"count": str(self.dropped_camera_frames)},
                )
            self.event("recording_stopped", "MCAP recording stopped")
        except RuntimeError as exc:
            event_error = exc
        self.active = False
        count = self.writer.stop()
        if event_error is not None:
            raise event_error
        return count

    def _sequence(self, topic: str) -> int:
        with self._sequence_lock:
            value = self._sequences.get(topic, 0)
            self._sequences[topic] = value + 1
            return value

    def _record(self, topic: str, message: Any, *, drop_if_full: bool = False) -> bool:
        if not self.active:
            return False
        return self.writer.record(
            topic,
            message,
            log_time_ns=self.sim_time_ns,
            sequence=self._sequence(topic),
            drop_if_full=drop_if_full,
        )

    def event(
        self,
        event_type: str,
        message: str,
        details: Mapping[str, str] | None = None,
    ) -> None:
        self._record(
            TOPIC_EVENTS,
            recording_pb2.Event(
                tick_index=self.tick_index,
                sim_time_ns=self.sim_time_ns,
                type=event_type,
                message=message,
                details=dict(details or {}),
            ),
        )

    def on_step(self, _model, data) -> None:
        if not self.active:
            return
        self.tick_index += 1
        self.sim_time_ns = int(round(float(data.time) * 1e9))
        self._record(
            TOPIC_SIM_STATE,
            recording_pb2.SimState(
                tick_index=self.tick_index,
                sim_time_ns=self.sim_time_ns,
                qpos=data.qpos,
                qvel=data.qvel,
            ),
        )
        if self.level == "minimal" or self.sim is None:
            return

        state = self.sim.get_robot_state()
        self._record(
            TOPIC_FCI_STATE,
            recording_pb2.FciState(
                tick_index=self.tick_index,
                sim_time_ns=self.sim_time_ns,
                q=state["q"],
                dq=state["dq"],
                q_d=state["q_d"],
                dq_d=state["dq_d"],
                ddq_d=state["ddq_d"],
                tau_j=state["tau_J"],
            ),
        )
        if self.sim.enable_gripper:
            finger = self.sim.get_finger_state()
            self._record(
                TOPIC_GRIPPER_STATE,
                recording_pb2.GripperState(
                    tick_index=self.tick_index,
                    sim_time_ns=self.sim_time_ns,
                    width=float(finger["width"]),
                    max_width=0.08,
                    is_grasped=bool(finger["contact_body_ids"]),
                    q=finger["q"],
                    dq=finger["dq"],
                ),
            )

    def fci_session_start(self) -> None:
        self.fci_active = True
        self.event("fci_session_start", "FCI client connected")

    def fci_session_end(self) -> None:
        self.event("fci_session_end", "FCI client disconnected")
        self.fci_active = False

    def fci_command(self, command: Mapping[str, Any]) -> None:
        if self.level == "minimal":
            return
        self._record(
            TOPIC_FCI_COMMAND,
            recording_pb2.FciCommand(
                tick_index=self.tick_index,
                sim_time_ns=self.sim_time_ns,
                mode=str(command.get("mode", "none")),
                q_command=command.get("q_command", ()),
                dq_command=command.get("dq_command", ()),
                tau_command=command.get("tau_command", ()),
                motion_finished=bool(command.get("motion_finished", False)),
                message_id=int(command.get("message_id", 0)),
            ),
        )

    def gripper_command(self, command: Mapping[str, Any]) -> None:
        if self.level == "minimal":
            return
        self._record(
            TOPIC_GRIPPER_COMMAND,
            recording_pb2.GripperCommand(
                tick_index=self.tick_index,
                sim_time_ns=self.sim_time_ns,
                command=str(command.get("command", "")),
                width=float(command.get("width", 0.0)),
                speed=float(command.get("speed", 0.0)),
                force=float(command.get("force", 0.0)),
                epsilon_inner=float(command.get("epsilon_inner", 0.0)),
                epsilon_outer=float(command.get("epsilon_outer", 0.0)),
                command_id=int(command.get("command_id", 0)),
            ),
        )

    def camera_frame(self, camera, color: bytes, depth: bytes, frame_index: int) -> None:
        if self.level == "minimal":
            return
        common = {
            "camera_name": camera.name,
            "frame_index": frame_index,
            "trajectory_tick_index": self.tick_index,
            "scheduled_sim_time_ns": self.sim_time_ns,
            "actual_sim_time_ns": self.sim_time_ns,
        }
        color_message = recording_pb2.CameraFrame(
            **common,
            encoding="raw_rgb8",
            width=camera.color_profile.width,
            height=camera.color_profile.height,
            data=color,
        )
        depth_message = recording_pb2.CameraFrame(
            **common,
            encoding="raw_z16_le",
            width=camera.depth_profile.width,
            height=camera.depth_profile.height,
            depth_scale=camera.depth_scale,
            data=depth,
        )
        color_path = color_topic(camera.name)
        depth_path = depth_topic(camera.name)
        queued = self.writer.record_batch(
            [
                (color_path, color_message, self.sim_time_ns, self._sequence(color_path)),
                (depth_path, depth_message, self.sim_time_ns, self._sequence(depth_path)),
            ],
            drop_if_full=True,
        )
        if not queued:
            self.dropped_camera_frames += 1

    def raw_packet(
        self,
        *,
        transport: str,
        direction: str,
        endpoint: str,
        data: bytes,
    ) -> None:
        if self.level != "debug":
            return
        self._record(
            TOPIC_RAW,
            recording_pb2.RawPacket(
                tick_index=self.tick_index,
                sim_time_ns=self.sim_time_ns,
                transport=transport,
                direction=direction,
                endpoint=endpoint,
                data=data,
            ),
            drop_if_full=True,
        )


@dataclass(frozen=True)
class DecodedRecord:
    proto_msg: Any
    sequence_count: int
    topic: str
    channel_metadata: Mapping[str, str]
    log_time_ns: int
    publish_time_ns: int


def iter_messages(
    source: str | Path,
    topics: Iterable[str] | str | None = None,
) -> Iterator[DecodedRecord]:
    with Path(source).open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for _schema, channel, message, proto_msg in reader.iter_decoded_messages(
            topics=topics, log_time_order=True
        ):
            yield DecodedRecord(
                proto_msg=proto_msg,
                sequence_count=message.sequence,
                topic=channel.topic,
                channel_metadata=channel.metadata,
                log_time_ns=message.log_time,
                publish_time_ns=message.publish_time,
            )


def read_attachments(source: str | Path) -> dict[str, Attachment]:
    with Path(source).open("rb") as stream:
        reader = make_reader(stream)
        return {
            value.name: Attachment(value.media_type, value.data)
            for value in reader.iter_attachments()
        }


def read_metadata(source: str | Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with Path(source).open("rb") as stream:
        reader = make_reader(stream)
        for value in reader.iter_metadata():
            if value.name == "remu":
                result.update(value.metadata)
    return result


_CHANNEL_ORDER = {
    TOPIC_SIM_STATE: 0,
    TOPIC_FCI_COMMAND: 1,
    TOPIC_FCI_STATE: 2,
    TOPIC_GRIPPER_COMMAND: 3,
    TOPIC_GRIPPER_STATE: 4,
    TOPIC_EVENTS: 7,
}


def _merge_key(record: DecodedRecord, source_index: int) -> tuple:
    topic_order = _CHANNEL_ORDER.get(
        record.topic,
        5 if record.topic.endswith("/color") else 6 if record.topic.endswith("/depth") else 8,
    )
    frame_index = getattr(record.proto_msg, "frame_index", 0)
    return (
        record.log_time_ns,
        topic_order,
        int(frame_index),
        record.topic,
        record.sequence_count,
        source_index,
    )


def _sort_equal_timestamps(
    records: Iterator[DecodedRecord], source_index: int
) -> Iterator[DecodedRecord]:
    group: list[DecodedRecord] = []
    timestamp = None
    for record in records:
        if timestamp is not None and record.log_time_ns != timestamp:
            yield from sorted(group, key=lambda value: _merge_key(value, source_index))
            group = []
        timestamp = record.log_time_ns
        group.append(record)
    if group:
        yield from sorted(group, key=lambda value: _merge_key(value, source_index))


def merge_recordings(
    inputs: Sequence[str | Path],
    output: str | Path,
    *,
    metadata: Mapping[str, str] | None = None,
    attachments: Mapping[str, Attachment] | None = None,
) -> Path:
    """Streaming deterministic merge ordered by time, channel, and frame index."""
    paths = [Path(value) for value in inputs]
    iterators = [
        iter(_sort_equal_timestamps(iter(iter_messages(path)), source_index))
        for source_index, path in enumerate(paths)
    ]
    heap: list[tuple[tuple, int, DecodedRecord]] = []
    for source_index, iterator in enumerate(iterators):
        try:
            record = next(iterator)
        except StopIteration:
            continue
        heapq.heappush(heap, (_merge_key(record, source_index), source_index, record))

    output = Path(output)
    with RecordingWriter(output, metadata=metadata, attachments=attachments) as writer:
        while heap:
            _key, source_index, record = heapq.heappop(heap)
            writer.write(
                record.topic,
                record.proto_msg,
                log_time_ns=record.log_time_ns,
                publish_time_ns=record.publish_time_ns,
                sequence=record.sequence_count,
            )
            try:
                following = next(iterators[source_index])
            except StopIteration:
                continue
            heapq.heappush(
                heap, (_merge_key(following, source_index), source_index, following)
            )
    return output


__all__ = [
    "AsyncRecorder",
    "Attachment",
    "RecordingWriter",
    "RunRecorder",
    "TOPIC_EVENTS",
    "TOPIC_FCI_COMMAND",
    "TOPIC_FCI_STATE",
    "TOPIC_GRIPPER_COMMAND",
    "TOPIC_GRIPPER_STATE",
    "TOPIC_SIM_STATE",
    "TOPIC_RAW",
    "color_topic",
    "depth_topic",
    "iter_messages",
    "merge_recordings",
    "read_attachments",
    "read_metadata",
    "recording_pb2",
]
