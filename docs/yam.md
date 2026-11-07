# YAM simulation and DDS control

remu simulates the supplied six-axis YAM URDF and two-finger gripper. yambridge
connects a DDS controller to either these simulators through Portal or physical
YAMs through i2rt. FR3/libfranka remains available through the separate `remu`
entry point. The YAM interface is at the robot API boundary, not CAN emulation.

## Install and launch

Use Python 3.11 with remu and yambridge in the same environment:

```bash
python -m pip install -e '/path/to/remu[yam,viser]' -e /path/to/yambridge
python -m remu.yam.bimanual --viewer viser
```

One terminal starts a shared MuJoCo world, two Portal endpoints (left 1235,
right 1234), one viewer (browser port 8080), and yambridge. Ctrl+C stops the
session. The arms share physics time, contacts, and objects. `--viewer mujoco`
selects one native window; `--viewer none` is headless. Add `--demo` to publish
the example all-zero joint goals; otherwise supply your own DDS publisher.
After reinstalling the entry points, `remu-yam-bimanual` is equivalent.

To run the scene separately from the bridge:

```bash
remu-yam-world --viewer viser
# Another terminal, or a separately supervised controller process:
yambridge --backend remu
```

`remu-yam` still runs a single arm, including the existing i2rt follower RPCs.
`remu-yam --viewer none` is useful for isolated tests. Separate single-arm
processes remain separate worlds; use the shared launcher for bimanual contact.

## Scene, mounts, and objects

The default mounts are `(0, -0.35, 0)` and `(0, 0.35, 0)` metres. Configure
`--left-position X Y Z`, `--right-position X Y Z`, and each mount's
`--left-quaternion W X Y Z` / `--right-quaternion W X Y Z`. Quaternions are
normalized and cannot have zero norm. The supplied URDF's own mounting
transform is retained beneath these mounts.

Use `--object /path/to/object.xml` repeatedly to attach standalone MJCF
objects, including free bodies. `--scene-mjcf /path/to/environment.xml` attaches
a custom environment; it must include any desired floor because this option
replaces the default ground plane. Scene/object relative mesh and texture
paths resolve against their source files. Robot bodies are added by remu, so
the environment should not contain another copy of the YAM arms.

An example object is bundled at `src/remu/models/yam/objects/box.xml`:

```bash
python -m remu.yam.bimanual --viewer viser \
  --object src/remu/models/yam/objects/box.xml
```

Cross-arm and object contacts are enabled. Each robot's own self-collisions
remain disabled: collision geometry uses approximate convex mesh hulls.
The default source is the exact user-supplied Onshape URDF, with upstream
meshes substituted because the original meshes were unavailable. See
[model provenance](../src/remu/models/yam/README.md). This is not a calibrated
contact, friction, or motor thermal model.

## RealSense camera and perception

```bash
MUJOCO_GL=egl python -m remu.yam.bimanual --viewer none --camera \
  --camera-port 1339 --calibration-out /tmp/yam_calibration.json
```

The shared scene uses remu's existing D435i renderer, aligned RGB8/Z16 stream,
TCP transport, and `shim/pyrealsense2.py`. `MUJOCO_GL=egl` is for headless
rendering; use your normal GL backend on a desktop. Configure `--camera-eye X
Y Z`, `--camera-target X Y Z`, and `--camera-serial SERIAL` as needed.

Point the perception process at `REMU_CAMERA_ADDR=127.0.0.1:1339` and put
remu's `shim` directory on its Python path to use the emulated SDK. The output
calibration contains the camera optical pose, `robot/left_base` and
`robot/right_base` mount poses, and `robot/base` as the shared world frame.
For a single-arm camera scene, `robot/base` is that arm's mount pose. RGB and
depth are aligned by construction. Rendering runs on the physics thread,
as on the Franka path, and can reduce achieved physics frequency.

## Position command compatibility

The existing `BimanualJointConfigurationTopic` and
`YamBridge::BimanualJointConfiguration` are unchanged. Each message contains
seven-value `left` and `right` arrays, `sequence`, and sender Unix `timestamp`.
Each arm uses six joint angles in radians followed by normalized gripper
opening: **0 closed, 1 open**. Physical finger travel is simulated internally
as two slides. No joint sign or offset conversion is applied.

```bash
yambridge --backend remu                 # legacy position DDS input
yambridge-example-publisher --rate 50    # example zero pose, simulation only
```

The export differs from upstream hardware coordinates, including joint 5's
axis sign and the flange frame. Verify mapping against your physical YAMs
before treating simulation and hardware goals as interchangeable.

## Extended control over DDS

Launch with `--command-format control` to select
`BimanualJointControlTopic` / `YamBridge::BimanualJointControl`. The legacy
position topic is not read simultaneously. `--topic NAME` overrides the
selected input topic. The shared launcher accepts the same format selection:

```bash
python -m remu.yam.bimanual --viewer viser --command-format control
# From your controller environment:
yambridge-control-publisher --left-mode velocity --left .1 0 0 0 0 0 0
```

`yambridge.control_publisher.make_arm_control()` builds an `ArmControl`; use
it with `BimanualJointControl(left, right, sequence, timestamp)` in a policy.
Each arm independently selects these mode codes:

| Code | Mode | Meaning |
| --- | --- | --- |
| 0 | position | Position goal; bridge and simulator filter the trajectory |
| 1 | velocity | Velocity goal; simulator integrates a bounded trajectory |
| 2 | torque | External generalized effort; no implicit gravity compensation |
| 3 | impedance | Position/velocity with optional seven-value kp/kd and feedforward effort |
| 4 | hold | Stop at measured position |
| 5 | gripper_move | Move to normalized opening at bounded speed |
| 6 | gripper_grasp | Close with a force cap; evaluate bilateral contact and opening tolerance |
| 7 | gripper_home | Open to the model's known upper stops |
| 8 | gripper_stop | Stop the finger motion at its measured opening |

`ArmControl` has `position`, `velocity`, `effort`, `kp`, `kd` arrays of length
seven and `use_gains`; inactive arrays should be zero-filled. Gripper fields
are `action_id`, `opening`, `speed`, `force`, `epsilon_inner`, `epsilon_outer`,
and `timeout`. Speed is normalized opening per second; force is a per-finger
Newton cap in simulation. Force zero selects the backend default. Repeated
messages with the same gripper mode/action ID keep the operation alive rather
than restarting it. Increment the action ID for a new operation. Continue
publishing while an operation runs so command freshness and watchdogs remain
satisfied. A gripper command following torque mode first holds the arm.

Move/home completion requires settling near the target. A grasp requires
both fingers contacting the same external body and the measured opening in
the requested tolerance band; closing empty fingers is not a successful
grasp. Status includes `moving`, `succeeded`, `grasped`, `lost`, `failed`,
`timeout`, and `stopped`. Homing uses known simulated travel, not hardware
calibration. Collision approximations limit grasp realism.

## Portal API

`YamClient` wraps bounded calls to `command_joint_pos`, `command_joint_vel`,
`command_joint_torque`, `command_joint_state`, `stop`, `get_joint_pos`,
`get_observations`, `get_robot_info`, `get_end_effector_pose`, `get_robot_model`,
`get_dynamics`, `keepalive`, and
the `gripper_move/grasp/home/stop` and `get_gripper_state` methods. Calls to
start gripper operations are asynchronous and return initial state; poll the
state and maintain freshness with `keepalive()` while the motion runs.

`command_joint_state` accepts `mode`, `pos` (`position` alias), `vel`, `eff`,
and optional nonnegative `kp`/`kd`, all seven-value finite arrays. Position
commands are clipped to joint ranges. Effort is Nm for arm joints. Gripper
effort is generalized force conjugate to normalized opening, not motor torque
or per-finger Newtons. Invalid commands are rejected before changing targets.

`get_robot_model()` returns the original URDF XML. `get_dynamics()` returns a
seven-coordinate mass matrix, generalized bias force, world-frame geometric
Jacobian (linear rows then angular rows), and end-effector pose. Finger slides
are projected through their travel into the single normalized gripper coordinate.
Shared-world queries take the same simulation lock as stepping and commands.

## Trajectory limits and watchdog

The simulator filters at its physics timestep (default 1 ms). Position and
velocity trajectories use position-dependent braking envelopes, velocity,
acceleration, and jerk limits. Effort and effort-rate caps apply to the actual
actuator forces, retaining implicit damping for stable low-inertia fingers.
Mode transitions reinitialize position trajectories at measured positions.
Hard position/force caps take precedence if a new configuration creates an
infeasible state; mode changes and watchdog holds reset trajectory derivatives.

Simulation defaults for the six arm joints are 1.5 rad/s, 10 rad/s²,
100 rad/s³, 10 Nm, and 100 Nm/s. Gripper defaults are 2 normalized units/s,
20 units/s², 200 units/s³, with 20 N/finger and 200 N/s/finger. These are
**simulation tuning, not certified YAM hardware ratings**. PD gains and arm
force-cap provenance are documented with the model; do not copy Franka limits.

`--limits FILE.json` overrides seven-element `velocity`, `acceleration`,
`jerk`, `effort`, and `effort_rate` arrays. Unspecified fields keep defaults;
unknown fields and nonpositive/nonfinite limits are rejected. The last effort
and effort-rate entries are in normalized gripper generalized-force units.

The bridge independently limits setpoints. `--max-joint-vel` defaults to 1
rad/s (gripper normalized units/s), and `--trajectory-limits FILE.json` can
override seven-element velocity/acceleration/jerk arrays. The global speed
argument remains an upper cap. Bridge default acceleration/jerk are 10/100 in
the corresponding joint units; these too are tuning, not hardware ratings.
Integration time is capped at 20 ms after a delayed cycle.

CLI simulators default to a 0.5-second monotonic communication watchdog. It
switches to a measured-position hold and cancels gripper motion if inputs
stop, including if the bridge dies. `--watchdog 0` disables it. Manually stepped
`YamSim()` defaults to watchdog disabled for deterministic offline experiments.
A watchdog hold is not an emergency-stop or hardware safety mechanism.

## Feedback and recovery

The existing `BimanualJointFeedbackTopic` remains wire-compatible, publishing
measured state, per-arm validity, acquisition timestamps, age, and timing.
The additional `BimanualRobotFeedbackTopic` / `YamBridge::BimanualRobotFeedback`
contains that same measured record plus desired trajectory, desired effort,
end-effector pose, mode, capabilities, gripper status/ID, watchdog/contact state,
and command errors. `yambridge-feedback` prints this richer stream as JSON.
Override its topic with `--rich-feedback-topic` on the bridge.

Both feedback streams use best effort / volatile / keep last 1, at
`--feedback-rate` (default 50 Hz). Pose matrices flatten row-major; simulation
poses are the URDF gripper-body frame in world coordinates. Missing data is
NaN or explicitly invalid. Hardware may provide desired position/derivatives
from the bridge while lacking desired effort; the combined trajectory validity
stays false in that case. Historical values may remain after disconnection;
always check validity flags and sample age.

Per-arm sample timestamps cover host I/O start/completion; Unix timestamps
are metadata. Freshness uses only the bridge-host monotonic clock. A remote
consumer should use the transmitted age, not compare monotonic epochs.
Loop period, work duration, maximum duration, overruns, arm I/O duration, and
timeout/reconnect counts describe achieved timing, not hard real-time guarantees.

Each arm owns one supervised process with at most one in-flight cycle.
`--rpc-timeout` bounds the command/read cycle (default 0.1 s),
`--startup-timeout` bounds construction/initial read (10 s), and
`--reconnect-delay` controls retries (0.5 s). Failed workers cannot block the
other arm. After reconnect, measured positions initialize setpoints and all
pre-connection goals are discarded. Portal automatic replay is disabled.
Already delivered remote commands cannot be cancelled.

`--command-timeout` (default 0.2 s) uses local monotonic receipt time and is
checked again immediately before robot I/O. Stale position goals stop advancing;
stale velocity, torque, and gripper modes actively transition to hold. The
simulator watchdog provides a separate layer if the bridge disappears.

## Physical hardware capability boundary

```bash
yambridge --backend i2rt --left-channel can_follower_l --right-channel can_follower_r
```

Use interfaces and arm/gripper selections matching your installation. The
public i2rt API checked at revision `fe78bae67334b9af0b79833dd8ba141a61767f2a`
provides position and position/velocity/gain commands. The hardware adapter
supports position, impedance, velocity, normalized gripper move, and stop.
It does not access private command registers to implement unavailable torque,
force-controlled grasp, or homing operations. Unsupported modes/explicit
hardware gripper force requests produce `command_error`, retaining feedback
and holding the affected arm. Hardware pose/contact fields remain invalid
when the driver does not supply them. Check reported capabilities rather
than assuming parity of every device operation. Physical hardware recovery
and these additional hardware modes have not been bench-tested.

The original i2rt minimum-GELLO leaders can still drive single-arm Portal
followers; see upstream `examples/minimum_gello/minimum_gello.py`. Their CAN
leader process is a command source, distinct from yambridge's DDS interface.
YAM motor observations and optional MCAP recording come from upstream i2rt;
this does not imply a downloadable teleoperation dataset exists.

## Benchmarking

```bash
python scripts/benchmark_yam.py --duration 5 --raw-steps 5000
```

The benchmark reports raw shared-world headroom, achieved paced frequency,
interval percentiles, missed intervals, and state finiteness. Camera/rendering
costs and host load affect timing.
