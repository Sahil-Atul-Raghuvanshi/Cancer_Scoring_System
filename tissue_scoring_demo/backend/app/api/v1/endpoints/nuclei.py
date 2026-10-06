"""Step 11 endpoints - the nuclei inside the carried regions.

Started, polled, then read, like steps 2, 8 and 10: a pass is minutes of CPU and
a request that waits for it is a request that times out.

    GET  /nuclei/capability                    is the segmenter installed
    POST /nuclei/{he}/run                      start (or return the finished one)
    POST /nuclei/{he}/run?restart=1            segment again, ignoring the cache
    GET  /nuclei/{he}/run                      progress
    GET  /nuclei/{he}                          the report
    GET  /nuclei/{he}/regions/{rank}           that region's nuclei, with geometry
    GET  /nuclei/{he}/regions/{rank}/fields/{i}.png?view=raw|input|overlay
    GET  /nuclei/{he}/compare/{rank}.png?view=watershed|rgb

Like step 10's, every route takes the IHC slide as a query parameter: the H&E is
"the slide" the pipeline is keyed on, and the IHC one is which of the case's five
antibodies this result is for.
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Response, status
from fastapi.responses import FileResponse

from app.nuclei import capability as nuclei_capability
from app.schemas.nuclei import NucleiReport, NucleiRun
from app.services.nuclei_service import NucleiError, nuclei_service
from app.services.upload_service import UploadError

router = APIRouter(prefix="/nuclei", tags=["nuclei"])

# Sync handlers throughout, like the other step routers: these open slides and
# read files, and an `async def` doing that stalls the event loop for everyone.

_PNG = {200: {"content": {"image/png": {}}}}
_PNG_HEADERS = {"Cache-Control": "public, max-age=3600"}

_FIELD_VIEWS = {"raw": "raw", "input": "input", "overlay": "over"}


def _failure(exc: NucleiError) -> HTTPException:
    """Step problems are 409: the request is fine, the pipeline is not ready."""
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))


def _upload_failure(exc: UploadError) -> HTTPException:
    missing = "not found" in str(exc)
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND if missing else status.HTTP_409_CONFLICT,
        detail=str(exc),
    )


@router.get("/capability", summary="Can this machine segment nuclei")
def capability() -> dict:
    """Whether the InstanSeg checkpoint is present and reproduces its own test pair.

    Reported rather than discovered on first use, so the screen can say the model
    is missing before somebody waits for a run to tell them the same thing. A
    parity failure reports as unavailable *with its reason*, because "the model
    is installed but segments differently than published" needs a different
    answer from "run setup".
    """
    report = nuclei_capability()
    return {
        "available": report.available,
        "reason": report.reason,
        "modelsDir": report.models_dir,
        "model": report.model_name,
        "version": report.model_version,
        "licence": report.licence,
        "mpp": report.mpp,
        "engine": report.engine,
    }


@router.post("/{he_upload_id}/run", response_model=NucleiRun, summary="Segment the nuclei")
def run(
    he_upload_id: str,
    background: BackgroundTasks,
    ihc_upload_id: str = Query(alias="ihcUploadId"),
    restart: bool = Query(default=False),
) -> NucleiRun:
    try:
        state = nuclei_service.start(he_upload_id, ihc_upload_id, restart=restart)
    except UploadError as exc:
        raise _upload_failure(exc) from exc
    except NucleiError as exc:
        raise _failure(exc) from exc

    if state.state == "queued":
        background.add_task(nuclei_service.execute, he_upload_id, ihc_upload_id)
    return state


@router.get("/{he_upload_id}/run", response_model=NucleiRun, summary="Progress")
def progress(he_upload_id: str, ihc_upload_id: str = Query(alias="ihcUploadId")) -> NucleiRun:
    try:
        return nuclei_service.state(he_upload_id, ihc_upload_id)
    except NucleiError as exc:
        raise _failure(exc) from exc


@router.get("/{he_upload_id}", response_model=NucleiReport, summary="The nuclei report")
def report(he_upload_id: str, ihc_upload_id: str = Query(alias="ihcUploadId")) -> NucleiReport:
    try:
        return nuclei_service.report(he_upload_id, ihc_upload_id)
    except NucleiError as exc:
        raise _failure(exc) from exc


@router.get(
    "/{he_upload_id}/regions/{rank}",
    summary="One region's nuclei, geometry included",
)
def region(
    he_upload_id: str, rank: int, ihc_upload_id: str = Query(alias="ihcUploadId")
) -> Response:
    """Served as a file rather than through a response model on purpose.

    A region's outlines are a few megabytes of vertices and they are already
    stored as exactly the JSON the client wants. Re-validating them through
    pydantic on every fetch would cost more than sending them does, and would
    not catch anything - this process wrote the file.
    """
    path = nuclei_service.artifact(he_upload_id, ihc_upload_id, f"region{rank}", "nuclei.json")
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"region {rank} has no stored nuclei for this pair",
        )
    return FileResponse(path, media_type="application/json", headers=_PNG_HEADERS)


@router.get(
    "/{he_upload_id}/regions/{rank}/fields/{index}.png",
    responses=_PNG,
    response_class=Response,
    summary="One sampled field",
)
def field_image(
    he_upload_id: str,
    rank: int,
    index: int,
    ihc_upload_id: str = Query(alias="ihcUploadId"),
    view: str = Query(default="overlay", pattern="^(raw|input|overlay)$"),
) -> Response:
    """`raw` is the field, `input` is what the model saw, `overlay` is its answer.

    Three views rather than one because no single picture makes the step legible:
    raw against input shows what taking the DAB out did, raw against overlay
    shows what the model did, and a viewer needs both comparisons.
    """
    path = nuclei_service.artifact(
        he_upload_id, ihc_upload_id, f"region{rank}", f"f{index}_{_FIELD_VIEWS[view]}.png"
    )
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no {view} image for field {index} of region {rank}",
        )
    return FileResponse(path, media_type="image/png", headers=_PNG_HEADERS)


@router.get(
    "/{he_upload_id}/compare/{rank}.png",
    responses=_PNG,
    response_class=Response,
    summary="The comparison panels",
)
def compare(
    he_upload_id: str,
    rank: int,
    ihc_upload_id: str = Query(alias="ihcUploadId"),
    view: str = Query(default="watershed", pattern="^(watershed|rgb)$"),
) -> Response:
    """`watershed` is the touching-nuclei case; `rgb` is detection on the raw stain."""
    path = nuclei_service.artifact(he_upload_id, ihc_upload_id, "compare", f"{rank}_{view}.png")
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no {view} comparison for region {rank}",
        )
    return FileResponse(path, media_type="image/png", headers=_PNG_HEADERS)
