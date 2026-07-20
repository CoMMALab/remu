#!/usr/bin/env python3
"""Run the Franka FCI emulator with MuJoCo physics, rendered in the browser
via mjviser.

Starts three things and holds them together until Ctrl+C:

  1. A MuJoCo scene (FR3 arm + ground plane, plus any ``--object`` MJCFs)
  2. A viser/mjviser server streaming that scene to a browser
  3. The FCI server on TCP 1337, so a libfranka client connecting to
     ``127.0.0.1`` drives the simulated arm

Physics runs in the foreground thread and pushes state to mjviser on every
step; the FCI server runs in the background.

Usage:
    python scripts/run_fci_viser.py
    python scripts/run_fci_viser.py --viser-port 8080 --port 1337
    python scripts/run_fci_viser.py --object /path/to/box.xml

Then open the printed http://localhost:<viser-port> URL, and point your
controller at robot IP 127.0.0.1.
"""

import argparse
import logging
import sys
import threading
from pathlib import Path

# Allow running straight from a source checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from remu.protocol.franka_protocol import COMMAND_PORT  # noqa: E402
from remu.server.franka_server import FrankaFciServer  # noqa: E402
from remu.sim.mujoco_sim import MujocoSim  # noqa: E402
from remu.sim.scene import build_scene_xml  # noqa: E402
from remu.viewer.viser_viewer import ViserViewer  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot-mjcf", default=None, help="Robot MJCF (default: bundled FR3)")
    parser.add_argument(
        "--object", action="append", default=[], dest="objects",
        help="Extra standalone MJCF to include in the scene (repeatable)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="FCI server bind host")
    parser.add_argument("--port", type=int, default=COMMAND_PORT, help="FCI TCP command port")
    parser.add_argument("--viser-host", default="0.0.0.0", help="mjviser bind host")
    parser.add_argument("--viser-port", type=int, default=8080, help="mjviser server port")
    parser.add_argument("--dt", type=float, default=0.001, help="Physics timestep (s)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    scene_path = build_scene_xml(robot_mjcf=args.robot_mjcf, extra_object_mjcfs=args.objects)

    sim = MujocoSim(scene_path, dt=args.dt, realtime=True)
    sim.build()

    viewer = ViserViewer(sim.model, sim.data, host=args.viser_host, port=args.viser_port)
    viewer.attach(sim)

    server = FrankaFciServer(sim, host=args.host, port=args.port)
    server.start(background=True)

    print()
    print("=" * 62)
    print(f"  viewer     http://localhost:{args.viser_port}")
    print(f"  FCI server {args.host}:{args.port}  (connect as robot IP 127.0.0.1)")
    print("  Ctrl+C to stop")
    print("=" * 62)
    print()

    try:
        sim.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.stop()
        sim.stop()
        viewer.close()
        # build_scene_xml writes next to the robot MJCF so relative meshdirs
        # still resolve; clean up our own file rather than leaving it behind.
        Path(scene_path).unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
