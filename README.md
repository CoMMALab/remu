# remu

remu simulates Franka FR3 and YAM arms and their grippers in MuJoCo.
Franka clients connect through the libfranka TCP/UDP protocol. YAM clients
connect through i2rt-compatible Portal RPCs, with yambridge providing DDS
commands and measured feedback for both physical and simulated YAMs.

## YAM quick start

Install remu and the sibling [yambridge](https://github.com/a123pal/yambridge)
checkout in the same Python 3.11 environment:

```bash
python -m pip install -e '.[yam,viser]' -e ../yambridge
python -m remu.yam.bimanual --viewer viser
```

One command starts both arms in a **shared physics world**, two Portal
endpoints, one browser viewer, and the DDS bridge. The viewer is at
`http://localhost:8080`; left and right Portal endpoints are
`127.0.0.1:1235` and `127.0.0.1:1234`. Ctrl+C stops the session.

Your policy or teleoperation program supplies DDS commands. Add `--demo`
to start the example publisher, which sends the all-zero joint pose.
Use `--viewer mujoco` for one native window or `--viewer none` for headless
operation. YAM commands contain six joint angles in radians followed by
gripper opening, from 0 closed to 1 open.

| Entry point | Purpose |
| --- | --- |
| `remu` | FR3 and Franka Hand through libfranka |
| `remu-yam` | Single YAM follower through Portal |
| `remu-yam-world` | Shared two-arm scene and Portal endpoints |
| `remu-yam-bimanual` | Shared scene, viewer, and yambridge in one session |

The YAM simulator supports position, velocity, external torque, and impedance
control; configurable velocity/acceleration/jerk and effort limits; gripper
move/grasp/home/stop operations; and measured-position watchdog holds.
Both arms can contact each other and the same objects. Configure mounts with
`--left-position` / `--right-position` and the corresponding quaternion flags.
Attach object MJCFs with repeated `--object` flags or supply an environment
with `--scene-mjcf`.

For the extended DDS control interface and emulated RealSense camera:

```bash
# Select per-arm control modes through DDS.
python -m remu.yam.bimanual --viewer viser --command-format control

# Headless shared scene with aligned RGB/depth streaming.
MUJOCO_GL=egl python -m remu.yam.bimanual --viewer none --camera \
  --calibration-out /tmp/yam_calibration.json
```

The camera uses remu's existing RealSense transport on port 1339 and
`shim/pyrealsense2.py`. yambridge publishes measured joint/gripper state,
per-arm validity, timestamps, timing, desired trajectories, end-effector
poses, capabilities, and gripper status. Portal also exposes the source
model and mass/bias/Jacobian queries.

YAM dynamics limits are simulation tuning, not manufacturer ratings. Mesh
contacts are approximate, and joint conventions must match your hardware.
The hardware adapter reports operations its public i2rt API cannot perform.
See the [YAM guide](docs/yam.md) for configuration, DDS schemas, recovery,
camera calibration, and the hardware capability boundary.

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

## Layout

```
src/remu/
  protocol/   libfranka wire format: Command enums, message structs, RobotState packing
  sim/        MuJoCo physics backend (MujocoSim) + scene composition (build_scene_xml)
  server/     Arm server (TCP 1337) and Franka Hand server (TCP 1338)
  camera/     Emulated RealSense D435i: MuJoCo rendering + frame server (TCP 1339)
  viewer/     MujocoPassiveViewer (native) and ViserViewer (browser, via mjviser)
  yam/        YAM physics, shared scenes, control, Portal, and session launchers
  cli.py      `remu` command-line entry point
  models/     FR3 URDF, YAM URDF/meshes, and example objects
shim/         pyrealsense2.py -- drop-in SDK replacement for the perception stack
scripts/      run_fci_viser.py: the whole stack (physics + FCI + camera + browser)
tests/        pytest suite (protocol, robot_state, scene, sim, server + camera integration)
references/   protocol reference material used by the emulator
```

## Franka usage

```bash
# Browser-based viewer (default)
remu

# Native MuJoCo viewer
remu --viewer mujoco

# No rendering, just the physics + FCI server
remu --viewer none

# Arm-only operation, for a custom model without conventional hand attachment names
remu --no-gripper

# A different robot MJCF / joint names, or a fully custom scene
remu --robot-mjcf /path/to/robot.xml --joint-names j1 j2 j3 j4 j5 j6 j7
remu --scene-mjcf /path/to/complete_scene.xml
```

Then point your libfranka-based controller at IP `127.0.0.1` (the default
FCI command port 1337 and gripper port 1338 match the real robot) — no code
changes are needed to switch between `remu` and real hardware. The Franka Hand
is physics-backed and appears in both the native and Viser viewers by default.
Its non-blocking command/state handling follows the approach used by
[franky-sim](https://github.com/TimSchneider42/franky-sim), while its explicit
gripper backend boundary and protocol-v3 framing also draw from
[libfranka-sim](https://github.com/BarisYazici/libfranka-sim).

Unmodified libfranka 0.15 clients may call `Robot::loadModel()`. Remu handles
the corresponding v9 `LoadModelLibrary` command by compiling a small native
FR3 kinematics library on first request. This supports a Linux client with the
same architecture as the remu host and requires `cc` (for example the compiler
from `build-essential`). For a remote client with a different architecture,
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
- Offscreen camera rendering needs a GL platform. On a headless machine set
  `MUJOCO_GL=egl`; without it MuJoCo tries GLFW and fails on the missing
  `$DISPLAY`. Each `mujoco.Renderer` owns a GL context bound to the thread
  that created it, so cameras are bound lazily on the physics thread — never
  call `EmulatedD435i.bind()` from anywhere else, or every later render fails
  with `EGL_BAD_ACCESS`.
