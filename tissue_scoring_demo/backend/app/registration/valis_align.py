"""Where the isolated registration environment lives, and whether step 12 can use it.

VALIS itself is no longer used. Step 12 used to register each pair here with VALIS's
feature matching, which failed outright on nearly unstained sections; it now carries
regions through a stored Mattes mutual-information transform instead (see
`slide_registration/transform_warp.py`). The old VALIS driver is in
`storage/decrecated_code/valis_registration/`.

The environment is still needed, though. The transforms are applied with SimpleITK,
which is installed in the separate Python 3.11 venv under
`tissue_scoring_demo/valis_service/` and not in this backend's, so step 12 runs
`apply_transform.py` there as a subprocess. The module keeps its old name because the
folder and the `valis_service_dir` setting do.
"""

from __future__ import annotations

from pathlib import Path

from app.core.config import settings

from .exceptions import RegistrationUnavailable

#: The script step 12 runs in that environment, via `transform_warp.py`.
APPLY_SCRIPT = "apply_transform.py"


def service_dir() -> Path:
    """Where the isolated registration environment lives."""
    if settings.valis_service_dir is not None:
        return Path(settings.valis_service_dir)
    # backend/app/registration/valis_align.py -> tissue_scoring_demo/valis_service
    return Path(__file__).resolve().parents[3] / "valis_service"


def _python() -> Path:
    directory = service_dir()
    candidates = [
        directory / ".venv" / "Scripts" / "python.exe",  # Windows
        directory / ".venv" / "bin" / "python",  # POSIX
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise RegistrationUnavailable(
        f"no registration environment at {directory / '.venv'}. It is a separate Python "
        f"3.11 venv because its dependencies pin numpy<2.0, which will not install "
        f"alongside this backend. See valis_service/README.md to create it."
    )


def available() -> bool:
    """Whether step 12 could carry regions at all, without attempting it."""
    try:
        return _python().exists() and (service_dir() / APPLY_SCRIPT).exists()
    except RegistrationUnavailable:
        return False


__all__ = ["APPLY_SCRIPT", "available", "service_dir"]
