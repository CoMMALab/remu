"""``remu`` command-line entry point."""

import argparse
import json
import logging
import sys
import threading
from pathlib import Path

from remu.camera import (
    CAMERA_PORT,
    CameraServer,
    cameras_to_calibration,
    load_camera_config,
    parse_camera_config,
)
from remu.config import LoadedRunConfig, load_run_config
from remu.ephemeral import TrajectoryRecorder, render_offline
from remu.protocol.franka_protocol import COMMAND_PORT
from remu.protocol.gripper_protocol import GRIPPER_COMMAND_PORT
from remu.server.franka_server import FrankaFciServer
from remu.server.gripper_server import FrankaGripperServer
from remu.sim.mujoco_sim import DEFAULT_JOINT_NAMES, MujocoSim
from remu.sim.scene import build_scene_xml


def build_parser(defaults=None) -> argparse.ArgumentParser:
    defaults = defaults or {}

    def value(name, fallback):
        return defaults.get(name, fallback)

    parser = argparse.ArgumentParser(
        prog="remu", description="Franka FCI emulator backed by MuJoCo physics."
    )
    parser.add_argument("--config", default=None, help="Version-1 unified remu run YAML")
    parser.add_argument(
        "--mode", choices=["persistent", "ephemeral"], default=value("mode", "persistent"),
        help="Persistent live serving or one-shot capture/offline rendering",
    )
    parser.add_argument(
        "--robot-mjcf", default=value("robot_mjcf", None),
        help="Robot MJCF to load (defaults to the bundled FR3)",
    )
    parser.add_argument(
        "--scene-mjcf", default=value("scene_mjcf", None),
        help="A complete scene MJCF to augment with the Franka Hand unless disabled",
    )
    parser.add_argument(
        "--object", action="append", default=value("objects", []), dest="objects",
        help="Extra standalone MJCF file to include in the scene (repeatable)",
    )
    parser.add_argument("--urdf", default=value("urdf", None), help="URDF served via GetRobotModel")
    parser.add_argument(
        "--model-library", default=value("model_library", None),
        help="Prebuilt v9 model library for cross-platform FCI clients",
    )
    parser.add_argument("--host", default=value("host", "0.0.0.0"), help="FCI server bind host")
    parser.add_argument(
        "--port", type=int, default=value("port", COMMAND_PORT), help="FCI TCP command port"
    )
    parser.add_argument(
        "--gripper-port", type=int, default=value("gripper_port", GRIPPER_COMMAND_PORT),
        help="libfranka gripper TCP command port",
    )
    parser.add_argument(
        "--no-gripper", action="store_true", default=value("no_gripper", False),
        help="Run without the Franka Hand",
    )
    parser.add_argument(
        "--viewer", choices=["mujoco", "viser", "none"], default=value("viewer", "viser"),
        help="Rendering backend (persistent mode only)",
    )
    parser.add_argument(
        "--viser-port", type=int, default=value("viser_port", 8080), help="mjviser server port"
    )
    parser.add_argument(
        "--camera-config", default=None,
        help="Legacy YAML file declaring simulated RGB-D cameras; replaces inline camera_rig",
    )
    parser.add_argument(
        "--camera-port", type=int, default=value("camera_port", CAMERA_PORT),
        help="Simulated camera transport port",
    )
    parser.add_argument(
        "--camera-calibration-out", default=value("camera_calibration_out", "calibration.remu.json"),
        help="Ground-truth camera calibration output path",
    )
    parser.add_argument(
        "--joint-names", nargs=7, default=value("joint_names", None), metavar="JOINT",
        help=f"7 arm joint names in the MJCF (default: {DEFAULT_JOINT_NAMES})",
    )
    parser.add_argument(
        "--initial-q", nargs=7, type=float, default=value("initial_q", None), metavar="Q",
        help="Seven initial arm joint positions (overrides the MJCF home keyframe)",
    )
    parser.add_argument("--dt", type=float, default=value("dt", 0.001), help="Physics timestep (s)")
    parser.add_argument(
        "--output", default=value("output", "remu_run.h5"),
        help="Ephemeral-mode HDF5 output path",
    )
    parser.add_argument(
        "--render-workers", type=int, default=value("render_workers", 1),
        help="Offline renderer process count (default: 1)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", default=value("overwrite", False),
        help="Replace an existing ephemeral output/partial capture",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return parser


def _parse_args(argv=None) -> tuple[argparse.Namespace, LoadedRunConfig | None]:
    argv = list(sys.argv[1:] if argv is None else argv)
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config")
    known, _ = probe.parse_known_args(argv)
    loaded = load_run_config(known.config) if known.config else None
    args = build_parser(loaded.defaults if loaded else None).parse_args(argv)
    return args, loaded


def _build_cameras(args, loaded):
    if args.camera_config:
        return list(load_camera_config(args.camera_config).cameras)
    if loaded is not None and loaded.camera_rig is not None:
        return list(parse_camera_config(loaded.camera_rig).cameras)
    return []


def _build_scene(args, cameras):
    if args.scene_mjcf and args.robot_mjcf:
        raise ValueError("--scene-mjcf and --robot-mjcf are mutually exclusive")
    if args.scene_mjcf and args.objects:
        raise ValueError("--object cannot be combined with --scene-mjcf")
    if args.scene_mjcf:
        return build_scene_xml(
            robot_mjcf=args.scene_mjcf, add_ground=False, cameras=cameras,
            add_gripper=not args.no_gripper,
        )
    return build_scene_xml(
        robot_mjcf=args.robot_mjcf, extra_object_mjcfs=args.objects,
        cameras=cameras, add_gripper=not args.no_gripper,
    )


def _servers(args, sim, **fci_kwargs):
    server = FrankaFciServer(
        sim, host=args.host, port=args.port, urdf_path=args.urdf,
        model_library_path=args.model_library, **fci_kwargs,
    )
    gripper = None if args.no_gripper else FrankaGripperServer(
        sim, host=args.host, port=args.gripper_port
    )
    return server, gripper


def _write_calibration(path, cameras):
    calibration = cameras_to_calibration(cameras)
    calibration["robot/base"] = [
        [1, 0, 0, 0], [0, 1, 0, 0],
        [0, 0, 1, 0], [0, 0, 0, 1],
    ]
    Path(path).write_text(json.dumps(calibration, indent=2), encoding="utf-8")


def _run_persistent(args, sim, cameras):
    viewer = None
    if args.viewer == "mujoco":
        from remu.viewer.mujoco_viewer import MujocoPassiveViewer

        viewer = MujocoPassiveViewer(sim.model, sim.data).attach(sim)
    elif args.viewer == "viser":
        from remu.viewer.viser_viewer import ViserViewer

        viewer = ViserViewer(sim.model, sim.data, port=args.viser_port).attach(sim)
        for camera in cameras:
            viewer.add_camera(camera)

    server, gripper = _servers(args, sim)
    camera_server = CameraServer(cameras, host=args.host, port=args.camera_port) if cameras else None
    try:
        server.start(background=True)
        if gripper is not None:
            gripper.start(background=True)
        if camera_server is not None:
            camera_server.attach(sim).start(background=True)
            _write_calibration(args.camera_calibration_out, cameras)

        print(f"remu FCI emulator listening on {args.host}:{server.port}")
        if gripper is not None:
            print(f"remu gripper emulator listening on {args.host}:{gripper.port}")
        if camera_server is not None:
            print(f"remu camera emulator listening on {args.host}:{camera_server.port}")
        print("Point your libfranka client at '127.0.0.1' (or this host's IP) to connect.")
        print("Press Ctrl+C to stop.")
        sim.run()
    finally:
        server.stop()
        if gripper is not None:
            gripper.stop()
        if camera_server is not None:
            camera_server.stop()
        sim.stop()
        if viewer is not None:
            viewer.close()


def _run_ephemeral(args, loaded, sim, cameras, scene_path):
    if args.viewer != "none":
        raise ValueError("ephemeral mode requires --viewer none so capture performs no rendering")
    if args.render_workers < 1:
        raise ValueError("--render-workers must be at least 1")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(f"{output_path.stem}.partial{output_path.suffix or '.h5'}")
    for path in (output_path, partial_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"path already exists (use --overwrite): {path}")
        if path.exists():
            path.unlink()

    recorder = TrajectoryRecorder(partial_path, nq=sim.model.nq, nv=sim.model.nv)
    sim.on_step_callbacks.append(recorder.capture)
    session_finished = threading.Event()
    capture_errors = []

    def session_start():
        recorder.start()
        print("FCI client connected; recording every physics tick")

    def session_end():
        sim.stop()
        try:
            recorder.stop()
        except BaseException as exc:
            capture_errors.append(exc)
        finally:
            session_finished.set()

    server, gripper = _servers(
        args, sim, on_session_start=session_start, on_session_end=session_end
    )
    cancelled = False
    try:
        server.start(background=True)
        if gripper is not None:
            gripper.start(background=True)
        print(f"remu ephemeral FCI emulator listening on {args.host}:{server.port}")
        print("Waiting for one FCI session; rendering starts when that client disconnects.")
        try:
            sim.run()
        except KeyboardInterrupt:
            cancelled = True
            raise
        if not session_finished.wait(timeout=30.0):
            raise RuntimeError("physics stopped without a completed FCI session")
    finally:
        server.stop()
        if gripper is not None:
            gripper.stop()
        sim.stop()

    if cancelled:
        return
    if capture_errors:
        raise RuntimeError("trajectory capture failed") from capture_errors[0]

    metadata = {
        "arguments": vars(args),
        "run_config": loaded.source if loaded is not None else None,
    }
    print(f"Captured {recorder.sample_count} ticks; rendering offline with {args.render_workers} worker(s)")
    result = render_offline(
        scene_path=scene_path,
        capture_path=partial_path,
        output_path=output_path,
        cameras=cameras,
        workers=args.render_workers,
        overwrite=args.overwrite,
        metadata=metadata,
    )
    print(
        f"Wrote {result['rendered_frames']} RGB-D frames to {result['output']} "
        f"at {result['frames_per_second']:.1f} frames/s"
    )


def main(argv=None):
    args, loaded = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cameras = _build_cameras(args, loaded)
    scene_path = _build_scene(args, cameras)
    try:
        sim = MujocoSim(
            scene_path, joint_names=args.joint_names, dt=args.dt,
            home_q=args.initial_q, enable_gripper=not args.no_gripper,
        )
        sim.build()
        if args.mode == "ephemeral":
            _run_ephemeral(args, loaded, sim, cameras, scene_path)
        else:
            try:
                _run_persistent(args, sim, cameras)
            except KeyboardInterrupt:
                pass
    except KeyboardInterrupt:
        print("\nEphemeral capture cancelled; no final dataset was written.")
    finally:
        Path(scene_path).unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
