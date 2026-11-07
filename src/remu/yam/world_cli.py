"""Run both YAM Portal endpoints in one shared MuJoCo scene."""
import argparse

from .runtime import add_scene_arguments, make_cameras, run_scene
from .sim import DEFAULT_URDF
from .world import YamWorld


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urdf', default=str(DEFAULT_URDF))
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--left-port', type=int, default=1235)
    parser.add_argument('--right-port', type=int, default=1234)
    parser.add_argument('--left-position', nargs=3, type=float, default=[0., -.35, 0.])
    parser.add_argument('--right-position', nargs=3, type=float, default=[0., .35, 0.])
    parser.add_argument('--left-quaternion', nargs=4, type=float, default=[1., 0., 0., 0.])
    parser.add_argument('--right-quaternion', nargs=4, type=float, default=[1., 0., 0., 0.])
    parser.add_argument('--viewer', choices=['mujoco', 'viser', 'none'], default='mujoco')
    parser.add_argument('--viser-port', type=int, default=8080)
    parser.add_argument('--dt', type=float, default=.001)
    add_scene_arguments(parser)
    args = parser.parse_args(argv)
    if args.left_port == args.right_port:
        parser.error('left and right ports must differ')
    cameras = make_cameras(args)
    sim = YamWorld(args.urdf, dt=args.dt, left_position=args.left_position,
        right_position=args.right_position, left_quaternion=args.left_quaternion,
        right_quaternion=args.right_quaternion, scene=args.scene_mjcf, objects=args.object,
        limits=args.limits, watchdog=args.watchdog, cameras=cameras)
    return run_scene(sim, [(sim.left, args.left_port), (sim.right, args.right_port)], args, cameras)


if __name__ == '__main__':
    raise SystemExit(main())
