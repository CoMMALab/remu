"""Run a YAM follower simulator for an i2rt leader or Portal client."""

import argparse
import time

from .server import build_server
from .runtime import add_scene_arguments, make_cameras, run_scene
from .sim import DEFAULT_URDF, YamSim


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", default=str(DEFAULT_URDF))
    parser.add_argument("--port", type=int, default=1234,
                        help="Portal follower port (i2rt: right 1234, left 1235)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--viewer", choices=["mujoco", "viser", "none"], default="mujoco")
    parser.add_argument("--viser-port", type=int, default=8080)
    parser.add_argument("--dt", type=float, default=.001)
    add_scene_arguments(parser)
    args = parser.parse_args(argv)
    if args.scene_mjcf or args.object or args.camera:
        from .world import YamWorld
        cameras = make_cameras(args)
        world = YamWorld(args.urdf, dt=args.dt, left_position=[0., 0., 0.],
            scene=args.scene_mjcf, objects=args.object, cameras=cameras,
            limits=args.limits, watchdog=args.watchdog, bimanual=False)
        return run_scene(world, [(world.left, args.port)], args, cameras)
    sim = YamSim(args.urdf, dt=args.dt, limits=args.limits, watchdog=args.watchdog)
    server = build_server(sim, args.port, args.host)
    viewer = None
    try:
        server.start(block=False)
        if args.viewer == "mujoco":
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
            viewer.cam.lookat[:] = [0, 0, .25]
            viewer.cam.distance = 1.3
        elif args.viewer == "viser":
            from remu.viewer.viser_viewer import ViserViewer
            viewer = ViserViewer(sim.model, sim.data, port=args.viser_port)
        print(f"YAM follower Portal server on port {args.port}; 6 arm joints + gripper")
        deadline = time.monotonic()
        next_frame = deadline
        while args.viewer != "mujoco" or viewer.is_running():
            sim.step()
            now = time.monotonic()
            if viewer is not None and now >= next_frame:
                if args.viewer == "viser":
                    viewer.sync(sim.model, sim.data)
                else:
                    viewer.sync()
                next_frame = now + 1 / 60
            deadline += args.dt
            delay = deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -args.dt * 10:
                deadline = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        if viewer is not None:
            viewer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
