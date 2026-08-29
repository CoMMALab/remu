"""``remu`` command-line entry point: build a scene, start the FCI server, and
step the MuJoCo physics loop in the foreground (with optional rendering).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from remu.camera import CAMERA_PORT, CameraServer, cameras_to_calibration, load_camera_config
from remu.protocol.franka_protocol import COMMAND_PORT
from remu.protocol.gripper_protocol import GRIPPER_COMMAND_PORT
from remu.server.franka_server import FrankaFciServer
from remu.server.gripper_server import FrankaGripperServer
from remu.sim.mujoco_sim import DEFAULT_JOINT_NAMES, MujocoSim
from remu.sim.scene import build_scene_xml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="remu", description="Franka FCI emulator backed by MuJoCo physics."
    )
    parser.add_argument(
        "--robot-mjcf",
        default=None,
        help="Robot MJCF to load (defaults to the bundled FR3, fetched via robot_descriptions)",
    )
    parser.add_argument(
        "--scene-mjcf",
        default=None,
        help="A complete scene MJCF to augment with the Franka Hand unless disabled",
    )
    parser.add_argument(
        "--object", action="append", default=[], dest="objects",
        help="Extra standalone MJCF file to include in the scene (repeatable)",
    )
    parser.add_argument("--urdf", default=None, help="URDF served via GetRobotModel")
    parser.add_argument(
        "--model-library",
        default=None,
        help="Prebuilt v9 model library for cross-platform FCI clients",
    )
    parser.add_argument("--host", default="0.0.0.0", help="FCI server bind host")
    parser.add_argument("--port", type=int, default=COMMAND_PORT, help="FCI TCP command port")
    parser.add_argument(
        "--gripper-port", type=int, default=GRIPPER_COMMAND_PORT,
        help="libfranka gripper TCP command port",
    )
    parser.add_argument("--no-gripper", action="store_true", help="Run without the Franka Hand")
    parser.add_argument(
        "--viewer",
        choices=["mujoco", "viser", "none"],
        default="none",
        help=(
            "Rendering backend: native MuJoCo passive viewer, browser-based mjviser, "
            "or none (default)"
        ),
    )
    parser.add_argument("--viser-port", type=int, default=8080, help="mjviser server port")
    parser.add_argument(
        "--camera-config", default=None,
        help="YAML file declaring simulated RealSense and Orbbec RGB-D cameras",
    )
    parser.add_argument(
        "--camera-port", type=int, default=CAMERA_PORT,
        help="Simulated camera transport port",
    )
    parser.add_argument(
        "--camera-calibration-out", default="calibration.remu.json",
        help="Ground-truth camera calibration output path",
    )
    parser.add_argument(
        "--joint-names",
        nargs=7,
        default=None,
        metavar="JOINT",
        help=f"7 arm joint names in the MJCF (default: {DEFAULT_JOINT_NAMES})",
    )
    parser.add_argument("--dt", type=float, default=0.001, help="Physics timestep (s)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    camera_rig = load_camera_config(args.camera_config) if args.camera_config else None
    cameras = list(camera_rig.cameras) if camera_rig else []

    if args.scene_mjcf:
        scene_path = build_scene_xml(
            robot_mjcf=args.scene_mjcf, add_ground=False, cameras=cameras,
            add_gripper=not args.no_gripper,
        )
    else:
        scene_path = build_scene_xml(
            robot_mjcf=args.robot_mjcf, extra_object_mjcfs=args.objects,
            cameras=cameras, add_gripper=not args.no_gripper,
        )

    sim = MujocoSim(
        scene_path, joint_names=args.joint_names, dt=args.dt,
        enable_gripper=not args.no_gripper,
    )
    sim.build()

    viewer = None
    if args.viewer == "mujoco":
        from remu.viewer.mujoco_viewer import MujocoPassiveViewer

        viewer = MujocoPassiveViewer(sim.model, sim.data).attach(sim)
    elif args.viewer == "viser":
        from remu.viewer.viser_viewer import ViserViewer

        viewer = ViserViewer(sim.model, sim.data, port=args.viser_port).attach(sim)
        for camera in cameras:
            viewer.add_camera(camera)

    server = FrankaFciServer(
        sim,
        host=args.host,
        port=args.port,
        urdf_path=args.urdf,
        model_library_path=args.model_library,
    )
    gripper_server = (
        None
        if args.no_gripper
        else FrankaGripperServer(sim, host=args.host, port=args.gripper_port)
    )
    camera_server = CameraServer(cameras, host=args.host, port=args.camera_port) if cameras else None
    try:
        server.start(background=True)
        if gripper_server is not None:
            gripper_server.start(background=True)
        if camera_server is not None:
            camera_server.attach(sim).start(background=True)
            calibration = cameras_to_calibration(cameras)
            calibration["robot/base"] = [
                [1, 0, 0, 0], [0, 1, 0, 0],
                [0, 0, 1, 0], [0, 0, 0, 1],
            ]
            Path(args.camera_calibration_out).write_text(json.dumps(calibration, indent=2))

        print(f"remu FCI emulator listening on {args.host}:{server.port}")
        if gripper_server is not None:
            print(f"remu gripper emulator listening on {args.host}:{gripper_server.port}")
        if camera_server is not None:
            print(f"remu camera emulator listening on {args.host}:{camera_server.port}")
        print("Point your libfranka client at '127.0.0.1' (or this host's IP) to connect.")
        print("Press Ctrl+C to stop.")
        sim.run()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        if gripper_server is not None:
            gripper_server.stop()
        if camera_server is not None:
            camera_server.stop()
        sim.stop()
        if viewer is not None:
            viewer.close()
        Path(scene_path).unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
