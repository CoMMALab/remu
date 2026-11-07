"""Viewer, Portal endpoints and RealSense transport sharing one physics loop."""
import json
from pathlib import Path
import time

import numpy as np

from .server import build_server


def add_scene_arguments(parser):
    parser.add_argument('--scene-mjcf', help='MJCF environment to attach around the YAM arms')
    parser.add_argument('--object', action='append', default=[], help='additional object MJCF; repeatable')
    parser.add_argument('--limits', help='JSON override for YAM simulation trajectory/effort limits')
    parser.add_argument('--watchdog', type=float, default=.5, help='seconds without commands before holding; 0 disables')
    parser.add_argument('--camera', action='store_true', help='enable emulated RealSense RGB/depth stream')
    parser.add_argument('--camera-port', type=int, default=1339)
    parser.add_argument('--camera-eye', nargs=3, type=float, default=[1.2, 0., .8])
    parser.add_argument('--camera-target', nargs=3, type=float, default=[0., 0., .3])
    parser.add_argument('--camera-serial', default='934222071887')
    parser.add_argument('--calibration-out', default='yam_calibration.json')


def make_cameras(args):
    if not args.camera:
        return []
    from remu.camera.d435i import EmulatedD435i
    return [EmulatedD435i(args.camera_serial, args.camera_eye, args.camera_target)]


def run_scene(sim, endpoints, args, cameras=()):
    servers = []
    viewer = None
    camera_server = None
    try:
        for arm, port in endpoints:
            server = build_server(arm, port, args.host)
            servers.append(server)
            server.start(block=False)
        if args.viewer == 'mujoco':
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
            viewer.cam.lookat[:] = [0., 0., .25]
            viewer.cam.distance = 1.7
        elif args.viewer == 'viser':
            from remu.viewer.viser_viewer import ViserViewer
            viewer = ViserViewer(sim.model, sim.data, port=args.viser_port)
            for camera in cameras:
                viewer.add_camera(camera)
        if cameras:
            from remu.camera import CameraServer
            from remu.camera.d435i import optical_pose_to_calibration
            camera_server = CameraServer(cameras, host=args.host, port=args.camera_port).attach(sim)
            camera_server.start(background=True)
            calibration = optical_pose_to_calibration(cameras)
            calibration['robot/base'] = np.eye(4).tolist()
            for arm, _ in endpoints:
                body = sim.model.body(arm.namespace + 'root').id
                pose = np.eye(4)
                pose[:3, :3] = sim.data.xmat[body].reshape(3, 3)
                pose[:3, 3] = sim.data.xpos[body]
                calibration['robot/' + (arm.namespace.rstrip('_') + '_base' if arm.namespace else 'base')] = pose.tolist()
                if len(endpoints) == 1:
                    calibration['robot/base'] = pose.tolist()
            Path(args.calibration_out).write_text(json.dumps(calibration, indent=2) + '\n')
            print(f'RealSense stream on {args.host}:{args.camera_port}; calibration {args.calibration_out}', flush=True)
        print('YAM Portal endpoints: ' + ', '.join(f'{args.host}:{port}' for _, port in endpoints), flush=True)
        deadline = time.monotonic()
        next_frame = deadline
        dt = sim.model.opt.timestep
        while args.viewer != 'mujoco' or viewer.is_running():
            sim.step()
            now = time.monotonic()
            if viewer is not None and now >= next_frame:
                if args.viewer == 'viser':
                    viewer.sync(sim.model, sim.data)
                else:
                    viewer.sync()
                next_frame = now + 1 / 60
            deadline += dt
            delay = deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -dt * 10:
                deadline = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        if camera_server is not None:
            camera_server.stop()
        for server in reversed(servers):
            server.close()
        for camera in cameras:
            camera.close()
        if viewer is not None:
            viewer.close()
    return 0
