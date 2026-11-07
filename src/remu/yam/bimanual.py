"""Launch a shared two-arm YAM world and yambridge as one session."""

import argparse
import importlib.util
import os
import signal
import subprocess
import sys
import time

from .client import YamClient
from .runtime import add_scene_arguments
from .sim import DEFAULT_URDF


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--viewer', choices=['mujoco', 'viser', 'none'], default='mujoco')
    parser.add_argument('--left-port', type=int, default=1235)
    parser.add_argument('--right-port', type=int, default=1234)
    parser.add_argument('--viser-port', type=int, default=8080,
                        help='shared browser viewer port')
    parser.add_argument('--command-format', choices=['position', 'control'], default='position')
    parser.add_argument('--demo', action='store_true', help='also publish example zero-pose targets')
    parser.add_argument('--urdf', default=str(DEFAULT_URDF))
    parser.add_argument('--dt', type=float, default=.001)
    parser.add_argument('--left-position', nargs=3, type=float, default=[0., -.35, 0.])
    parser.add_argument('--right-position', nargs=3, type=float, default=[0., .35, 0.])
    parser.add_argument('--left-quaternion', nargs=4, type=float, default=[1., 0., 0., 0.])
    parser.add_argument('--right-quaternion', nargs=4, type=float, default=[1., 0., 0., 0.])
    add_scene_arguments(parser)
    args = parser.parse_args(argv)
    if args.demo and args.command_format != 'position':
        parser.error('--demo publishes position commands; omit it for control mode')
    if args.left_port == args.right_port:
        parser.error('left and right Portal ports must differ')
    if importlib.util.find_spec('yambridge') is None:
        parser.error('install yambridge in this Python environment first')

    processes = []
    handlers = {}

    def start(module, *arguments):
        process = subprocess.Popen([sys.executable, '-m', module, *arguments],
                                   start_new_session=True)
        processes.append(process)
        return process

    def interrupted(_signal, _frame):
        raise KeyboardInterrupt

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, interrupted)
        world_arguments = ['--host', '127.0.0.1', '--left-port', str(args.left_port),
            '--right-port', str(args.right_port), '--viewer', args.viewer,
            '--viser-port', str(args.viser_port), '--urdf', args.urdf, '--dt', str(args.dt),
            '--watchdog', str(args.watchdog)]
        for key in ('left_position', 'right_position', 'left_quaternion', 'right_quaternion'):
            world_arguments += ['--' + key.replace('_', '-'), *map(str, getattr(args, key))]
        for key in ('scene_mjcf', 'limits'):
            if getattr(args, key):
                world_arguments += ['--' + key.replace('_', '-'), getattr(args, key)]
        for path in args.object:
            world_arguments += ['--object', path]
        if args.camera:
            world_arguments += ['--camera', '--camera-port', str(args.camera_port),
                '--camera-serial', args.camera_serial, '--calibration-out', args.calibration_out,
                '--camera-eye', *map(str, args.camera_eye), '--camera-target', *map(str, args.camera_target)]
        process = start('remu.yam.world_cli', *world_arguments)
        addresses = []
        for side, port in [('left', args.left_port), ('right', args.right_port)]:
            address = f'127.0.0.1:{port}'
            addresses.append(address)
            deadline = time.monotonic() + 15
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'{side} simulator exited during startup')
                client = YamClient(address, timeout=.2)
                try:
                    if client.num_dofs() != 7:
                        raise RuntimeError(f'{side} endpoint does not expose seven DOFs')
                    break
                except TimeoutError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f'{side} simulator did not become ready')
                    time.sleep(.05)
                finally:
                    client.close()
        start('yambridge.main', '--backend', 'remu', '--left-address', addresses[0],
              '--right-address', addresses[1], '--command-format', args.command_format)
        if args.demo:
            start('yambridge.example_publisher', '--rate', '50')
        print('Bimanual session running. Ctrl+C stops all processes.', flush=True)
        while True:
            for process in processes:
                if process.poll() is not None:
                    return process.returncode
            time.sleep(.1)
    except KeyboardInterrupt:
        return 0
    finally:
        # Ignore repeated stop requests while reaping all children.
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
        deadline = time.monotonic() + 5
        for process in reversed(processes):
            try:
                process.wait(timeout=max(.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
