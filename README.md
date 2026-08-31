# remu

A MuJoCo-backed emulator for the robots and peripherals of a franka fr3 and realsenses.
`remu` speaks the same TCP/UDP wire protocol as a real Franka robot, so a
controller built on libfranka can connect to `127.0.0.1` exactly as it would connect to hardware, and drive a simulated FR3 and Franka Hand in MuJoCo via position, velocity, torque, and gripper commands.

## Protocol version

`remu` targets **robot server protocol v9** and **gripper server protocol v3** — libfranka **0.15.x**, matching
the 0.15.3 build in `frankabridge`. 

If you move to a newer libfranka, remu rejects the connection with an
incompatible-version response. Protocol v10 changes both command numbering and
the `RobotState` wire layout, so accepting it as v9 would otherwise produce
misleading failures later in client initialization.

## Setup

```bash
conda env create -f environment.yml   # creates the `remu` conda env
conda activate remu
```

(Already created and installed in this checkout: `conda activate remu`.)

## Layout

```
src/remu/
  protocol/   libfranka wire format: Command enums, message structs, RobotState packing
  sim/        MuJoCo physics backend (MujocoSim) + scene composition (build_scene_xml)
  server/     Arm server (TCP 1337) and Franka Hand server (TCP 1338)
  camera/     Configurable RealSense/Orbbec RGB-D rendering + frame server
  viewer/     MujocoPassiveViewer (native) and ViserViewer (browser, via mjviser)
  cli.py      `remu` command-line entry point
  models/     fr3.urdf served to clients via GetRobotModel
shim/         Drop-in pyrealsense2 and pyorbbecsdk Python SDK replacements
scripts/      run_fci_viser.py: the whole stack (physics + FCI + camera + browser)
tests/        pytest suite (protocol, robot_state, scene, sim, server + camera integration)
references/   protocol reference material used by the emulator
```

## Usage

```bash
# Browser-based viewer via mjviser (default)
remu

# Browser-based viewer on a custom port
remu --viewer viser --viser-port 8080

# No viewer, just the physics + FCI server
remu --viewer none

# Native MuJoCo viewer
MUJOCO_GL=glfw remu --viewer mujoco

# Arm-only operation, for a custom model without conventional hand attachment names
remu --no-gripper

# A different robot MJCF / joint names, or a fully custom scene
remu --robot-mjcf /path/to/robot.xml --joint-names j1 j2 j3 j4 j5 j6 j7
remu --scene-mjcf /path/to/complete_scene.xml

# Mixed RealSense and Orbbec cameras declared relative to the robot base
remu --camera-config configs/cameras.example.yaml
```

Then point your libfranka-based controller at IP `127.0.0.1` (the default
FCI command port 1337 and gripper port 1338 match the real robot) — no code
changes are needed to switch between `remu` and real hardware. The Franka Hand
is physics-backed and appears in both the native and Viser viewers by default.

## Camera configuration

Camera YAML version 1 declares any number of mixed `realsense/d435i`,
`orbbec/femto_mega`, and Orbbec Gemini devices. Supported Gemini model keys are
`gemini_2`, `gemini_2_l`, `gemini_330`, `gemini_330l`, `gemini_335`,
`gemini_335l`, `gemini_336`, `gemini_336l`, and the Ethernet-only
`gemini_435le`. A rig can be supplied by `--camera-config`, or placed
under `camera_rig` in the unified run configuration. See
[`configs/cameras.example.yaml`](configs/cameras.example.yaml) and the
end-effector-mounted example in
[`configs/ephemeral_ee_camera.yaml`](configs/ephemeral_ee_camera.yaml).

Each camera entry has:

- `vendor`, `model`, and a globally unique `serial`.
- `ip`, and optionally `port` (default 8090), for a device the client opens
  with `Context.create_net_device(ip, port)` rather than by USB
  enumeration. Required for `gemini_435le`, which has no USB mode at all;
  accepted for `femto_mega`, which offers both; rejected for everything
  else, so a rig that could never reach its device fails at load time
  instead of timing out on the first frame.
- `parent_body`, the MuJoCo body/link that carries this camera. If omitted, it
  inherits the rig-level `robot_base_body`. Different cameras may name
  different links, such as `fr3_link0`, `fr3_link7`, or `fr3_hand`.
- `base_from_optical`, a rigid 4x4 transform from the camera optical frame into
  `parent_body`. Optical coordinates follow the SDK convention: +x right, +y
  down, and +z forward. Despite the historical field name, this transform is
  relative to `parent_body`, not necessarily the robot base.
- One `pipeline` FPS and color/depth stream descriptions. Color is `rgb8`,
  depth is `z16`; the two streams may use different resolutions.

The named parent body must exist in the composed MuJoCo scene. A bad name is
rejected while building the scene rather than silently placing the camera in
world coordinates. Cameras attached to moving links follow those links during
both persistent simulation and ephemeral replay.

## Camera SDK shims

Persistent mode exposes the configured cameras over Remu's TCP camera service
(port 1339 by default). The files in `shim/` are drop-in, pure-Python subsets
of `pyrealsense2` and the Orbbec SDK v2 `pyorbbecsdk` module. Put `shim/`
first on `PYTHONPATH`; client imports then resolve to the shim without changing
the perception application:

```bash
# Camera server on the same host, default port.
PYTHONPATH=/path/to/remu/shim python your_camera_program.py

# Camera server on another host or port.
REMU_CAMERA_ADDR=remu-host:1339 \
  PYTHONPATH=/path/to/remu/shim python your_camera_program.py
```

The shim asks the server to enumerate devices, validates requested profiles
against the YAML, and streams aligned RGB/depth arrays plus intrinsics and
timestamps. Each shim enumerates only its own vendor. RealSense decimation and
threshold filters are implemented; disparity, spatial, temporal, and
hole-filling filters are intentional identity passes because rendered depth
does not contain the sensor noise those filters remove. This is a compatibility
surface for the APIs used by the perception stack, not a complete vendor SDK.

Ephemeral mode does not start the camera service or use the shims: it records
physics first and writes camera arrays directly into the final dataset during
offline rendering.

Remu's non-blocking command/state handling follows the approach used by
[franky-sim](https://github.com/TimSchneider42/franky-sim), while its explicit
gripper backend boundary and protocol-v3 framing also draw from
[libfranka-sim](https://github.com/BarisYazici/libfranka-sim).

Unmodified libfranka 0.15 clients may call `Robot::loadModel()`. Remu handles
the corresponding v9 `LoadModelLibrary` command by selecting a library for the
platform reported by the client. A Linux ARM64 library is bundled. For a Linux
client with the same architecture as the remu host, Remu can also compile a
small native FR3 kinematics library on first request; this requires `cc` (for
example the compiler from `build-essential`). For any other remote platform,
provide a compatible prebuilt library with `--model-library PATH` or the
`REMU_MODEL_LIBRARY` environment variable.

## Persistent MCAP recording

Persistent mode records to one canonical MCAP when `--record` or
`recording.path` is configured. File I/O and Protobuf serialization happen on
one background writer; physics and network threads enqueue decoded messages.

```bash
remu --record datasets/run.mcap --record-level standard

# The same settings in unified YAML:
# recording:
#   path: datasets/run.mcap
#   level: standard
#   chunk_mib: 4
#   rotate_seconds: 1800
#   rotate_mib: 4096
```

MCAP chunks use Zstandard compression and are flushed at the configured bounded
chunk size. Rotation may be disabled with zero or enabled by simulation duration
and/or approximate file size. The first segment uses the requested name and
later segments use `.0001.mcap`, `.0002.mcap`, and so on. Configuration,
composed scene MJCF, camera calibration, and model dimensions/joint names are
stored as metadata and attachments in every segment.

Recording levels are:

- `minimal`: authoritative full-model physics snapshots and lifecycle events.
- `standard`: minimal plus decoded FCI commands/states, gripper commands/states,
  and camera color/depth.
- `debug`: standard plus raw FCI and gripper TCP/UDP packets.

Decoded numeric streams are never silently dropped: exhausting their bounded
queue fails the recording. Persistent RGB-D pairs use one atomic queue item and
may be dropped together if the writer cannot keep up; a lifecycle event records
the total. This prevents recording backpressure from stalling the 1 kHz physics
loop or leaving an unmatched color/depth half-frame.

The canonical topics are:

```text
/remu/sim/state
/remu/fci/command
/remu/fci/state
/remu/gripper/command
/remu/gripper/state
/remu/camera/<vendor>/<serial>/color
/remu/camera/<vendor>/<serial>/depth
/remu/events
/remu/debug/raw                         # debug level only
```

Messages use the versioned Protobuf schema in
[`src/remu/proto/recording.proto`](src/remu/proto/recording.proto). Physics
snapshots contain complete MuJoCo `qpos` and `qvel`, so manipulated objects
and free joints remain authoritative for deterministic visual replay.

## Ephemeral capture and offline rendering

Ephemeral mode is a one-client, quick-setup/quick-teardown path. It starts the
FCI and optional gripper servers, waits for one successful FCI session, records
that session to `run.capture.mcap`, renders after disconnect, writes one final
MCAP, and exits. Persistent mode remains the default when `mode` is omitted.

During FCI execution, Remu records complete MuJoCo `qpos` and `qvel` at every
physics tick along with decoded commands, desired state, gripper state, and
events. No viewer or camera renderer runs in this phase, so a 1 kHz control loop
does not also pay for perception rendering. `viewer.backend` must be `none`.

After the FCI client disconnects, Remu:

1. Selects the first logged physics state at or after each camera's exact frame
   boundary. Cameras with different FPS values get independent schedules.
2. Splits the union of selected trajectory indices into contiguous chunks.
3. Starts up to `render_workers` processes. Each loads its own MuJoCo model and
   EGL context once, restores `qpos` and `qvel`, calls `mj_forward`, and
   renders without physics integration or wall-clock pacing.
4. Writes `worker-000.mcap`, `worker-001.mcap`, and so on, containing
   lossless PNG RGB and 16-bit depth frames.
5. Streaming-merges capture and worker messages by simulation timestamp,
   channel priority, and frame index, then atomically promotes the merged file
   to the requested output.

```bash
remu --config configs/ephemeral_ee_camera.yaml

# Explicit CLI values override YAML.
remu --config run.yaml --initial-q Q1 Q2 Q3 Q4 Q5 Q6 Q7 \
  --render-workers 2 --output datasets/run.mcap --overwrite
```

The unified version-1 YAML groups settings under `robot`, `simulation`,
`network`, `viewer`, `camera_rig`, `recording`, and `ephemeral`.
Relative paths resolve against the configuration file. `robot.initial_q`
overrides a model's `home` keyframe. See the executable
[`configs/ephemeral_ee_camera.yaml`](configs/ephemeral_ee_camera.yaml).

`render_workers` defaults to one because EGL context contention is hardware
dependent. Benchmark the same workload at 1, 2, 4, and so on; throughput often
stops scaling before the CPU core count because workers contend for GPU
resources. If capture or rendering fails, the phase-one `.capture.mcap` is
retained; the final path appears only after a successful deterministic merge.

## Replay and tensor export

fr3-teleop reads canonical MCAP directly, including embedded color images, so
Rerun no longer needs a staged directory containing thousands of files:

```bash
fr3-teleop replay -c fr3_remu datasets/run.mcap --serve
```

For dense tensor slicing and training preprocessing, export MCAP to HDF5:

```bash
remu-export-hdf5 datasets/run.mcap datasets/run.h5
```

The HDF5 export contains `/trajectory/{time_s,qpos,qvel}` and
`/cameras/<vendor>/<serial>/{color,depth,time_s,scheduled_time_s,trajectory_index}`.
MCAP remains the source of truth; exports can align, filter, or reshape streams
without discarding decoded commands, events, attachments, or debug packets.
The older `remu-export-fr3-teleop` staging converter remains available for
legacy HDF5 recordings.
## FR3 command limits

Joint position and velocity commands are filtered at the 1 kHz physics rate.
In particular, the allowed velocity is recomputed for every joint and every
step from the current commanded position using Franka's asymmetric
[position-based velocity limits](https://frankarobotics.github.io/docs/robot_specifications.html#position-based-velocity-limits).
The filter also applies the published joint position, nominal velocity,
acceleration, jerk, torque, and torque-rate limits. The constrained trajectory
is exposed to FCI clients as `q_d`, `dq_d`, and `ddq_d`.

To measure raw physics headroom and sustained 1 kHz pacing on your machine:

```bash
python scripts/benchmark_1khz.py --duration 30
```

The benchmark reports effective frequency, step-interval percentiles, missed
deadlines, state validity, and clean thread shutdown.

## Notes

- Control modes: joint position, joint velocity, and joint torque (external
  controller) are all implemented; Cartesian motion generators are not yet
  wired up (StopMove/AutomaticErrorRecovery/impedance-setting commands are
  acknowledged but not yet enforced).
- The robot's native MJCF actuators are disabled at load time; `MujocoSim`
  applies control uniformly as a joint torque (`qfrc_applied`) computed from
  whichever mode is active, so behavior doesn't depend on what actuators the
  source MJCF happens to define.
- Remu defaults `MUJOCO_GL` to `egl`, so offscreen camera rendering works on a
  headless machine without `$DISPLAY`. An explicitly configured backend is
  preserved; for example, use `MUJOCO_GL=glfw remu --viewer mujoco` for the
  native desktop viewer. Each `mujoco.Renderer` owns a GL context bound to the
  thread that created it, so cameras are bound lazily on the physics thread —
  never call `EmulatedD435i.bind()` from anywhere else, or every later render
  fails with `EGL_BAD_ACCESS`.
