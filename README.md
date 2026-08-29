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
  camera/     Configurable RealSense/Femto Mega RGB-D rendering + frame server
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

Camera YAML version 1 declares a named MuJoCo robot-base body, any number of
mixed `realsense/d435i` and `orbbec/femto_mega` devices, a unique serial, a
`base_from_optical` rigid 4x4 transform, and one RGB-D pipeline per device.
See [`configs/cameras.example.yaml`](configs/cameras.example.yaml). Color is
RGB8, depth is Z16, and the streams may have different resolutions but share
the configured pipeline FPS.

To run unmodified Python camera clients, put `shim/` first on `PYTHONPATH` and
set `REMU_CAMERA_ADDR` when the server is not at `127.0.0.1:1339`:

```bash
PYTHONPATH=/path/to/remu/shim python your_realsense_or_orbbec_program.py
```

The shims target the commonly used RGB-D portions of `pyrealsense2` and the
Orbbec SDK v2 `pyorbbecsdk` API. They enumerate only their own vendor and
reject stream profiles that differ from the YAML declaration.
Its non-blocking command/state handling follows the approach used by
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
