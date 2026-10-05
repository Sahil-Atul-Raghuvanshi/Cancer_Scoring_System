"""List the versioned data trees, switch between them, and start a new one.

Every path in `app.core.config` is fixed when the module is imported, so the backend
serves exactly one version for its whole life. Switching therefore writes the choice
to `storage/ACTIVE_DATA_VERSION` and restarts the process: rewriting `config.py` is a
file change uvicorn's `--reload` (which start.bat always passes) restarts on. The
frontend polls `GET /data-versions` until `active` matches, and tells the user to run
stop.bat + start.bat if it never does.

A version this code can no longer open (an older storage layout) is listed but cannot
be selected; if one was selected anyway - by an older checkout, say - the backend runs
the latest openable version and `fallbackReason` says why, so the page can tell the
viewer. Nothing is deleted or converted either way.

A restart drops whatever pipeline step is in flight, the same as editing the backend
does - the version picker warns before it calls this.
"""

import asyncio
import multiprocessing
import os
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

import data_versions
from app.core import config
from app.schemas.data_version import (
    DataVersionInfo,
    DataVersionSelectRequest,
    DataVersionSelectResponse,
    DataVersionsResponse,
)

router = APIRouter(prefix="/data-versions", tags=["data-versions"])


def _summary(version: str) -> str:
    """The first paragraph under the first `## ` heading of the version's VERSION.md."""
    try:
        text = (data_versions.version_root(version) / "VERSION.md").read_text(encoding="utf-8")
    except OSError:
        return ""
    _, _, after = text.partition("\n## ")
    lines = after.splitlines()[1:]
    paragraph: list[str] = []
    for line in lines:
        if line.strip():
            paragraph.append(line.strip().lstrip("-* ").strip())
        elif paragraph:
            break
    return " ".join(paragraph)


def _info(version: str) -> DataVersionInfo:
    reason = data_versions.why_not_openable(version)
    return DataVersionInfo(
        version=version,
        path=str(data_versions.version_root(version)),
        layout=data_versions.layout_of(version),
        openable=reason is None,
        not_openable_reason=reason,
        has_own_models=(data_versions.version_root(version) / "models").is_dir(),
        models_path=str(data_versions.models_root(version)),
        summary=_summary(version),
    )


def _describe() -> DataVersionsResponse:
    return DataVersionsResponse(
        # What this process was started on - not a fresh read, which would already
        # report a switch whose restart has not happened yet.
        active=config.DATA_VERSION,
        latest=data_versions.latest(),
        requested=config.DATA_RESOLUTION.requested,
        fallback_reason=config.DATA_RESOLUTION.fallback_reason,
        pinned_by_env=config.DATA_RESOLUTION.pinned_by_env,
        shared_data_path=str(data_versions.SHARED_DATA_ROOT),
        versions=[_info(version) for version in data_versions.available()],
    )


async def _restart() -> None:
    # Give the response time to reach the browser before the reloader kills us.
    await asyncio.sleep(0.5)
    # Rewrite the bytes rather than `touch()`: on Windows WatchFiles ignores an
    # mtime-only change, measured - a touch left the old process serving for 40 s.
    path = Path(config.__file__)
    path.write_bytes(path.read_bytes())
    # Under --reload this process is the reloader's child, and the reloader stops it
    # with a CTRL_C_EVENT - which never arrives when the server has no console of its
    # own, measured: "Reloading..." logged and the old process went on serving. So
    # leave on our own; the reloader is blocked joining us and starts the new process
    # as soon as we are gone. Without --reload there is no parent to restart us, so
    # stay up and let the page tell the viewer to restart by hand.
    if multiprocessing.parent_process() is not None:
        await asyncio.sleep(3)
        os._exit(0)


@router.get("", response_model=DataVersionsResponse, summary="Which data version is served")
async def list_versions() -> DataVersionsResponse:
    return _describe()


@router.post(
    "/select",
    response_model=DataVersionSelectResponse,
    summary="Switch the backend to another data version (restarts it)",
)
async def select_version(
    body: DataVersionSelectRequest, background: BackgroundTasks
) -> DataVersionSelectResponse:
    if config.DATA_RESOLUTION.pinned_by_env:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"{data_versions.ENV_VAR} is set, so the version is fixed for this run",
        )
    reason = data_versions.why_not_openable(body.version)
    if reason:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=reason)
    data_versions.select(body.version)
    restarting = body.version != config.DATA_VERSION
    if restarting:
        background.add_task(_restart)
    return DataVersionSelectResponse(selected=body.version, restarting=restarting)


@router.post(
    "",
    response_model=DataVersionInfo,
    status_code=status.HTTP_201_CREATED,
    summary="Start the next version: an empty v<N+1>_data/ that inherits the models",
)
async def create_version() -> DataVersionInfo:
    return _info(data_versions.create_next())
