from remu.cli import build_parser


def test_viewer_defaults_to_none():
    assert build_parser().parse_args([]).viewer == "none"


def test_viewer_can_be_enabled_explicitly():
    assert build_parser().parse_args(["--viewer", "mujoco"]).viewer == "mujoco"
    assert build_parser().parse_args(["--viewer", "viser"]).viewer == "viser"
