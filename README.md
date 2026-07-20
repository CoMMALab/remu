# remu

A MuJoCo-backed emulator for the robots and peripherals of a franka fr3 and realsenses.
`remu` speaks the same TCP/UDP wire protocol as a real Franka robot, so a
controller built on libfranka can connect to `127.0.0.1` exactly as it would connect to hardware, and drive a simulated FR3 in MuJoCo
via position, velocity, or torque control.

## Protocol version

`remu` targets **robot server protocol v9** — libfranka **0.15.x**, matching
the 0.15.3 build in `frankabridge`. 

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
