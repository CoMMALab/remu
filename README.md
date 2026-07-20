# remu

A MuJoCo-backed emulator for the Franka FCI (libfranka) network protocol.
`remu` speaks the same TCP/UDP wire protocol as a real Franka robot, so a
controller built on libfranka — RTPDemo included — can connect to `127.0.0.1`
exactly as it would connect to hardware, and drive a simulated FR3 in MuJoCo
via position, velocity, or torque control.

## Protocol version

`remu` targets **robot server protocol v9** — libfranka **0.15.x**, matching
the 0.15.3 build in `frankabridge`. Verified byte-for-byte against the
vendored `frankabridge/libfranka` headers (`kVersion = 9`,
`sizeof(RobotState) == 2373`), and validated end to end by linking a real
libfranka 0.15.3 client against a running emulator.

This is deliberately **not** the v10 protocol used by libfranka >= 0.18.
The two are wire-incompatible in ways that fail loudly:

| | v9 (libfranka 0.15.x) | v10 (libfranka >= 0.18) |
|---|---|---|
| `RobotState` | 2373 B, `double` | 1377 B, `float` |
| accelerometer fields | absent | present |
| `kGetRobotModel` | 12 | 11 |

If you ever move to a newer libfranka, `RobotState` size is the first thing
that breaks: libfranka throws `ProtocolException("libfranka: incorrect
object size")` on the first UDP state packet. Note the handshake will *still*
succeed — `connect()` only checks the status byte and never validates the
version number — so a version mismatch surfaces one step later than you'd
expect.

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
  server/     FrankaFciServer: TCP command channel + UDP 1kHz state channel
  camera/     Emulated RealSense D435i: MuJoCo rendering + frame server (TCP 1338)
  viewer/     MujocoPassiveViewer (native) and ViserViewer (browser, via mjviser)
  cli.py      `remu` command-line entry point
  models/     fr3.urdf served to clients via GetRobotModel
shim/         pyrealsense2.py -- drop-in SDK replacement for the perception stack
scripts/      run_fci_viser.py: the whole stack (physics + FCI + camera + browser)
tests/        pytest suite (protocol, robot_state, scene, sim, server + camera integration)
references/   vendored libfranka-sim reference (Genesis-based); remu adapts its
              protocol/server design onto MuJoCo
```

## Usage

```bash
# Native MuJoCo viewer (default)
remu

# Browser-based viewer via mjviser
remu --viewer viser --viser-port 8080

# No rendering, just the physics + FCI server
remu --viewer none

# A different robot MJCF / joint names, or a fully custom scene
remu --robot-mjcf /path/to/robot.xml --joint-names j1 j2 j3 j4 j5 j6 j7
remu --scene-mjcf /path/to/complete_scene.xml
```

Then point your libfranka-based controller at IP `127.0.0.1` (the default
FCI command port, 1337, matches the real robot) — no code changes needed to
switch between `remu` and real hardware.

## Emulated cameras

An emulated Intel RealSense D435i renders color + depth from a MuJoCo camera
and serves them on TCP 1338. `shim/pyrealsense2.py` is a drop-in replacement
for the RealSense SDK that reads that stream, so **pointcloud_perception runs
unmodified against the simulated scene** — the same trick the FCI server
plays on libfranka, one layer up.

```bash
# Everything at once: physics + FCI (1337) + camera (1338) + browser viewer
MUJOCO_GL=egl python scripts/run_fci_viser.py

# In the perception container (the shim is mounted at /opt/remu/shim by
# docker-compose, and is NOT on PYTHONPATH unless you put it there):
PYTHONPATH=/opt/remu/shim python visualize_cameras.py
```

The camera stands 1 m in front of the robot base at 0.6 m, aimed at the
workspace (`--camera-distance` / `--camera-height`, or `--no-camera`), and is
drawn in the viser scene as an axes triad plus an orange frustum. Point
`$REMU_CAMERA_ADDR` (`host:port`) elsewhere if the emulator isn't local.

Because the emulator knows every extrinsic exactly, the AprilTag calibration
is unnecessary: the run script writes ground truth to `calibration.remu.json`
(keyed `realsense/<serial>`, plus an identity `robot/base`). It deliberately
does **not** touch `pointcloud_perception/calibration.json` — copy the entries
across yourself when you want them.

Fidelity, and where it stops:

- Depth and color are rendered from one MuJoCo camera at one resolution, so
  they are aligned by construction and `rs.align` is an identity pass.
- Intrinsics are *derived* from the MJCF camera's `fovy` rather than
  hardcoded, so they can't drift from what is actually being rendered. At
  640×480 that gives fx = fy ≈ 617, matching a real D435i's factory values.
- Out-of-range depth becomes 0, RealSense's "no reading" sentinel, so the
  perception code's zero-vertex rejection runs the same path as on hardware.
- `decimation_filter` and `threshold_filter` are implemented for real —
  they change point count and range, which is what the perception filters are
  tuned against. The spatial/temporal/hole-filling/disparity filters are
  identity passes: they exist to denoise a real sensor, and rendered depth has
  no noise to remove. So toggling `FILTERS["rs_sdk"]` changes density and
  range, but not smoothness.
- There is no sensor noise model, no IMU, and no exposure/auto-white-balance
  behaviour.

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
