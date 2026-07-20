"""``remu`` command-line entry point: build a scene, start the FCI server, and
step the MuJoCo physics loop in the foreground (with optional rendering).
"""

import argparse
import logging
import sys

from remu.protocol.franka_protocol import COMMAND_PORT
from remu.server.franka_server import FrankaFciServer
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
        help="A complete scene MJCF to use as-is, bypassing scene composition",
    )
    parser.add_argument(
        "--object", action="append", default=[], dest="objects",
        help="Extra standalone MJCF file to include in the scene (repeatable)",
    )
    parser.add_argument("--urdf", default=None, help="URDF served via GetRobotModel")
    parser.add_argument("--host", default="0.0.0.0", help="FCI server bind host")
    parser.add_argument("--port", type=int, default=COMMAND_PORT, help="FCI TCP command port")
    parser.add_argument(
        "--viewer",
        choices=["mujoco", "viser", "none"],
        default="mujoco",
        help="Rendering backend: native MuJoCo passive viewer, browser-based mjviser, or none",
    )
    parser.add_argument("--viser-port", type=int, default=8080, help="mjviser server port")
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

    if args.scene_mjcf:
        scene_path = args.scene_mjcf
    else:
        scene_path = build_scene_xml(
            robot_mjcf=args.robot_mjcf, extra_object_mjcfs=args.objects
        )

    sim = MujocoSim(scene_path, joint_names=args.joint_names, dt=args.dt)
    sim.build()

    viewer = None
    if args.viewer == "mujoco":
        from remu.viewer.mujoco_viewer import MujocoPassiveViewer

        viewer = MujocoPassiveViewer(sim.model, sim.data).attach(sim)
    elif args.viewer == "viser":
        from remu.viewer.viser_viewer import ViserViewer

        viewer = ViserViewer(sim.model, sim.data, port=args.viser_port).attach(sim)

    server = FrankaFciServer(sim, host=args.host, port=args.port, urdf_path=args.urdf)
    server.start(background=True)

    print(f"remu FCI emulator listening on {args.host}:{args.port}")
    print("Point your libfranka client at '127.0.0.1' (or this host's IP) to connect.")
    print("Press Ctrl+C to stop.")

    try:
        sim.run()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        sim.stop()
        if viewer is not None:
            viewer.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
