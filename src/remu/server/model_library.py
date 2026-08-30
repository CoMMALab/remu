"""Build or load the native model library required by robot protocol v9."""

import logging
import os
import platform
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SYSTEM_LINUX = 0
SYSTEM_WINDOWS = 1
ARCH_X64 = 0
ARCH_X86 = 1
ARCH_ARM = 2
ARCH_ARM64 = 3

_SOURCE = Path(__file__).resolve().parent.parent / "models" / "fr3_kinematics.c"
_ARCHITECTURES = {
    "x86_64": ARCH_X64,
    "amd64": ARCH_X64,
    "i386": ARCH_X86,
    "i686": ARCH_X86,
    "arm": ARCH_ARM,
    "armv7l": ARCH_ARM,
    "aarch64": ARCH_ARM64,
    "arm64": ARCH_ARM64,
}
_compile_lock = threading.Lock()
_compiled_library: Optional[Path] = None


class ModelLibraryUnavailable(RuntimeError):
    pass


def _native_linux_architecture() -> Optional[int]:
    if platform.system() != "Linux":
        return None
    return _ARCHITECTURES.get(platform.machine().lower())


def _compile_native_library() -> Path:
    global _compiled_library
    with _compile_lock:
        if _compiled_library is not None and _compiled_library.exists():
            return _compiled_library
        compiler = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
        if compiler is None:
            raise ModelLibraryUnavailable(
                "no C compiler found; install build-essential or pass --model-library"
            )
        output_dir = Path(tempfile.gettempdir()) / "remu-model-library"
        output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
        output = output_dir / "libfcimodels-remu.so"
        temporary = output.with_name(f"{output.name}.{os.getpid()}.tmp")
        command = [
            compiler,
            "-std=c99",
            "-O2",
            "-fPIC",
            "-shared",
            str(_SOURCE),
            "-lm",
            "-o",
            str(temporary),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            temporary.replace(output)
        except (OSError, subprocess.CalledProcessError) as exc:
            temporary.unlink(missing_ok=True)
            detail = (
                exc.stderr.strip()
                if isinstance(exc, subprocess.CalledProcessError)
                else str(exc)
            )
            raise ModelLibraryUnavailable(f"could not compile model library: {detail}") from exc
        _compiled_library = output
        logger.info("Compiled native FR3 model library: %s", output)
        return output


def model_library_bytes(
    architecture: int,
    system: int,
    override: Optional[Path] = None,
) -> bytes:
    """Return a client-loadable model library for a v9 request."""
    if override is not None:
        path = Path(override)
        if not path.is_file():
            raise ModelLibraryUnavailable(f"model library not found: {path}")
        return path.read_bytes()

    native_architecture = _native_linux_architecture()
    if system != SYSTEM_LINUX or native_architecture != architecture:
        raise ModelLibraryUnavailable(
            "built-in model library generation supports native Linux clients only; "
            "pass --model-library with a binary for the requested platform"
        )
    return _compile_native_library().read_bytes()
