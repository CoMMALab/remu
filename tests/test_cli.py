import os

from remu import _configure_mujoco_gl
from remu.cli import build_parser


def test_viewer_defaults_to_viser():
    assert build_parser().parse_args([]).viewer == "viser"


def test_viewer_can_be_enabled_explicitly():
    assert build_parser().parse_args(["--viewer", "mujoco"]).viewer == "mujoco"
    assert build_parser().parse_args(["--viewer", "none"]).viewer == "none"


def test_mujoco_gl_defaults_to_egl(monkeypatch):
    monkeypatch.delenv("MUJOCO_GL", raising=False)

    _configure_mujoco_gl()

    assert os.environ["MUJOCO_GL"] == "egl"


def test_explicit_mujoco_gl_backend_is_preserved(monkeypatch):
    monkeypatch.setenv("MUJOCO_GL", "glfw")

    _configure_mujoco_gl()

    assert os.environ["MUJOCO_GL"] == "glfw"
