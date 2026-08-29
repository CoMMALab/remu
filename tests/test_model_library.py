from pathlib import Path

import pytest

from remu.server import model_library


def test_linux_arm64_uses_bundled_library(monkeypatch):
    monkeypatch.setattr(
        model_library,
        "_compile_native_library",
        lambda: pytest.fail("bundled library should not be compiled"),
    )

    result = model_library.model_library_bytes(
        model_library.ARCH_ARM64, model_library.SYSTEM_LINUX
    )

    assert result.startswith(b"\x7fELF")
    assert result[4] == 2  # ELFCLASS64
    assert int.from_bytes(result[18:20], "little") == 183  # EM_AARCH64


def test_native_linux_architecture_is_compiled(monkeypatch, tmp_path):
    expected = b"native-library"
    compiled = tmp_path / "libfcimodels-remu.so"
    compiled.write_bytes(expected)
    monkeypatch.setattr(model_library, "_native_linux_architecture", lambda: 99)
    monkeypatch.setattr(model_library, "_compile_native_library", lambda: compiled)

    assert model_library.model_library_bytes(99, model_library.SYSTEM_LINUX) == expected


@pytest.mark.parametrize(
    ("architecture", "system"),
    [
        (model_library.ARCH_ARM64, model_library.SYSTEM_WINDOWS),
        (model_library.ARCH_ARM, model_library.SYSTEM_LINUX),
    ],
)
def test_unsupported_platform_is_rejected(monkeypatch, architecture, system):
    monkeypatch.setattr(model_library, "_native_linux_architecture", lambda: None)

    with pytest.raises(model_library.ModelLibraryUnavailable):
        model_library.model_library_bytes(architecture, system)


def test_explicit_override_takes_precedence(tmp_path):
    expected = b"explicit-library"
    override = Path(tmp_path) / "custom-model-library"
    override.write_bytes(expected)

    assert (
        model_library.model_library_bytes(
            model_library.ARCH_ARM64,
            model_library.SYSTEM_LINUX,
            override=override,
        )
        == expected
    )
