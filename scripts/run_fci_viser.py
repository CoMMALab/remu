#!/usr/bin/env python3
"""Run the Franka FCI emulator with MuJoCo physics, rendered in the browser
via mjviser, with an emulated RealSense D435i watching the workspace.

Starts five things and holds them together until Ctrl+C:

  1. A MuJoCo scene (FR3 arm + ground plane, an emulated D435i, plus any
     ``--object`` MJCFs)
  2. A viser/mjviser server streaming that scene to a browser, with the
     camera drawn as a labelled frustum so its placement is visible
  3. The FCI server on TCP 1337, so a libfranka client connecting to
     ``127.0.0.1`` drives the simulated arm
  4. The libfranka gripper server on TCP 1338.
  5. The camera server on TCP 1339, which ``remu/shim/pyrealsense2.py``
     connects to so pointcloud_perception sees a RealSense

The camera is hardcoded to stand 1 m in front of the robot base at 0.6 m
height, tilted down at the workspace. Override with ``--camera-distance`` /
``--camera-height``, or drop it entirely with ``--no-camera``.

Physics runs in the foreground thread and pushes state to mjviser (and
renders camera frames) on every step; the network servers run in the background.

Usage:
    python scripts/run_fci_viser.py
    python scripts/run_fci_viser.py --viser-port 8080 --port 1337
    python scripts/run_fci_viser.py --object /path/to/box.xml

Then open the printed http://localhost:<viser-port> URL, point your
controller at robot IP 127.0.0.1, and run pointcloud_perception with
``PYTHONPATH=<repo>/remu/shim``.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow running straight from a source checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from remu.camera import (  # noqa: E402
    CAMERA_PORT,
    CameraServer,
    load_camera_config,
    d435i_in_front_of_robot,
    optical_pose_to_calibration,
)
from remu.protocol.franka_protocol import COMMAND_PORT  # noqa: E402
from remu.protocol.gripper_protocol import GRIPPER_COMMAND_PORT  # noqa: E402
from remu.server.franka_server import FrankaFciServer  # noqa: E402
from remu.server.gripper_server import FrankaGripperServer  # noqa: E402
from remu.sim.mujoco_sim import MujocoSim  # noqa: E402
from remu.sim.scene import build_scene_xml  # noqa: E402
from remu.viewer.viser_viewer import ViserViewer  # noqa: E402

# Where the ground-truth extrinsics get written. Deliberately *not*
# pointcloud_perception/calibration.json: that file holds the real rig's
# hard-won AprilTag calibration, and clobbering it would cost a recalibration.
CALIB_OUT = Path(__file__).resolve().parent.parent / "calibration.remu.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot-mjcf", default=None, help="Robot MJCF (default: bundled FR3)")
    parser.add_argument(
        "--object", action="append", default=[], dest="objects",
        help="Extra standalone MJCF to include in the scene (repeatable)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="FCI server bind host")
    parser.add_argument("--port", type=int, default=COMMAND_PORT, help="FCI TCP command port")
    parser.add_argument("--model-library", default=None,
                        help="Prebuilt v9 model library for cross-platform clients")
    parser.add_argument("--gripper-port", type=int, default=GRIPPER_COMMAND_PORT,
                        help="libfranka gripper TCP port")
    parser.add_argument("--no-gripper", action="store_true", help="Run without the Franka Hand")
    parser.add_argument("--viser-host", default="0.0.0.0", help="mjviser bind host")
    parser.add_argument("--viser-port", type=int, default=8080, help="mjviser server port")
    parser.add_argument("--no-camera", action="store_true", help="Run without the emulated D435i")
    parser.add_argument("--camera-config", default=None,
                        help="YAML file declaring RealSense and Orbbec RGB-D cameras")
    parser.add_argument("--camera-port", type=int, default=CAMERA_PORT, help="Camera server TCP port")
    parser.add_argument("--camera-serial", default="934222071887", help="Serial the emulated D435i reports")
    parser.add_argument("--camera-distance", type=float, default=1.0,
                        help="Camera distance in front of the robot base (m)")
    parser.add_argument("--camera-height", type=float, default=0.6, help="Camera height (m)")
    parser.add_argument("--dt", type=float, default=0.001, help="Physics timestep (s)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.camera_config and (
        args.no_camera or args.camera_serial != "934222071887"
        or args.camera_distance != 1.0 or args.camera_height != 0.6
    ):
        raise SystemExit(
            "--camera-config cannot be combined with legacy --no-camera/--camera-* placement flags"
        )
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.camera_config:
        cameras = list(load_camera_config(args.camera_config).cameras)
    else:
        cameras = []
    if not args.camera_config and not args.no_camera:
        cameras.append(d435i_in_front_of_robot(
            serial=args.camera_serial,
            distance_m=args.camera_distance,
            height_m=args.camera_height,
        ))

    scene_path = build_scene_xml(
        robot_mjcf=args.robot_mjcf, extra_object_mjcfs=args.objects, cameras=cameras,
        add_gripper=not args.no_gripper,
    )

    sim = MujocoSim(
        scene_path, dt=args.dt, realtime=True, enable_gripper=not args.no_gripper
    )
    sim.build()

    viewer = ViserViewer(sim.model, sim.data, host=args.viser_host, port=args.viser_port)
    viewer.attach(sim)
    for camera in cameras:
        viewer.add_camera(camera)

    server = FrankaFciServer(
        sim, host=args.host, port=args.port, model_library_path=args.model_library
    )
    gripper_server = (
        None
        if args.no_gripper
        else FrankaGripperServer(sim, host=args.host, port=args.gripper_port)
    )
    camera_server = None
    try:
        server.start(background=True)
        if gripper_server is not None:
            gripper_server.start(background=True)
        if cameras:
            camera_server = CameraServer(cameras, host=args.host, port=args.camera_port)
            # Attach before start so an immediate client finds rendered frames.
            camera_server.attach(sim)
            camera_server.start(background=True)
            # The emulator knows every extrinsic exactly, so write ground truth
            # instead of requiring the perception stack's AprilTag calibration.
            calibration = optical_pose_to_calibration(cameras)
            calibration["robot/base"] = [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
            CALIB_OUT.write_text(json.dumps(calibration, indent=2))

        print()
        print("=" * 68)
        print(f"  viewer         http://localhost:{args.viser_port}")
        print(f"  FCI server     {args.host}:{server.port}  (connect as robot IP 127.0.0.1)")
        if gripper_server:
            print(f"  gripper server {args.host}:{gripper_server.port}")
        if camera_server:
            identities = ", ".join(f"{camera.vendor}/{camera.serial}" for camera in cameras)
            print(f"  camera server  {args.host}:{camera_server.port}  ({identities})")
            print(f"  extrinsics     {CALIB_OUT}")
            print(f"  perception     PYTHONPATH={Path(__file__).resolve().parent.parent / 'shim'}")
        print("  Ctrl+C to stop")
        print("=" * 68)
        print()

        sim.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if camera_server:
            camera_server.stop()
        if gripper_server:
            gripper_server.stop()
        server.stop()
        sim.stop()
        viewer.close()
        # Scene assets use absolute paths, so the composed XML itself is temporary.
        Path(scene_path).unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
