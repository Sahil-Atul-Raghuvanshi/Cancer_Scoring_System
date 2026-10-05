"""Step 10 endpoints - carrying the ROI onto the IHC slide.

Started, polled, then read, like steps 2 and 8: registering two whole-slide
images is minutes of CPU in another process, and a request that waits for it
is a request that times out.

    GET  /ihc-alignment/capability     can registration run on this machine
    POST /ihc-alignment/{he}/run       start (or return the finished one)
    POST /ihc-alignment/{he}/run?restart=1   register again, ignoring the cache
    GET  /ihc-alignment/{he}/run       progress
    GET  /ihc-alignment/{he}           the report
    POST /ihc-alignment/{he}/confirm   the blocking check a person signs off
    GET  /ihc-alignment/{he}/panels/{name}.png   he_borders | ihc_borders
    GET  /ihc-alignment/{he}/crops/{rank}.png    a region, cropped from the IHC slide

Every route takes the IHC slide as a query parameter rather than putting both
ids in the path: the H&E is "the slide" every other step is keyed on, and the
IHC one is which of the case's five antibodies this alignment is for.
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Response, status

from app.registration.valis_align import available as registration_available
from app.registration.valis_align import service_dir
from app.schemas.ihc_alignment import AlignmentReport, AlignmentRun
from app.services.ihc_alignment_service import AlignmentError, ihc_alignment_service
from app.services.upload_service import UploadError

router = APIRouter(prefix="/ihc-alignment", tags=["ihc alignment"])

# Sync handlers throughout, like the other step routers: these open slides and
# read files, and an `async def` doing that stalls the event loop for everyone.

_PNG = {200: {"content": {"image/png": {}}}}
_PNG_HEADERS = {"Cache-Control": "public, max-age=3600"}


def _failure(exc: AlignmentError) -> HTTPException:
    """Alignment problems are 409: the request is fine, the pipeline is not ready."""
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))


def _upload_failure(exc: UploadError) -> HTTPException:
    missing = "not found" in str(exc)
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND if missing else status.HTTP_409_CONFLICT,
        detail=str(exc),
    )


def _png(payload: bytes) -> Response:
    return Response(content=payload, media_type="image/png", headers=_PNG_HEADERS)


@router.get("/capability", summary="Can this machine register two slides")
def capability() -> dict:
    """Whether the isolated registration environment is present, and where it is looked for.

    Reported rather than discovered on first use, so the screen can say
    "registration is not installed" before somebody waits five minutes to be
    told the same thing.
    """
    return {
        "available": registration_available(),
        "serviceDir": str(service_dir()),
        "reason": None
        if registration_available()
        else (
            "no registration environment found. It is a separate Python 3.11 venv, "
            "because its dependencies pin numpy<2.0, which cannot be installed alongside "
            "this backend. "
            "See valis_service/README.md."
        ),
    }


@router.post("/{he_upload_id}/run", response_model=AlignmentRun, summary="Align the two slides")
def run(
    he_upload_id: str,
    background: BackgroundTasks,
    ihc_upload_id: str = Query(alias="ihcUploadId"),
    restart: bool = Query(default=False),
) -> AlignmentRun:
    try:
        state = ihc_alignment_service.start(he_upload_id, ihc_upload_id, restart=restart)
    except UploadError as exc:
        raise _upload_failure(exc) from exc
    except AlignmentError as exc:
        raise _failure(exc) from exc

    if state.state == "queued":
        background.add_task(ihc_alignment_service.execute, he_upload_id, ihc_upload_id)
    return state


@router.get("/{he_upload_id}/run", response_model=AlignmentRun, summary="Progress")
def progress(
    he_upload_id: str, ihc_upload_id: str = Query(alias="ihcUploadId")
) -> AlignmentRun:
    try:
        return ihc_alignment_service.state(he_upload_id, ihc_upload_id)
    except AlignmentError as exc:
        raise _failure(exc) from exc


@router.get("/{he_upload_id}", response_model=AlignmentReport, summary="The alignment report")
def report(
    he_upload_id: str, ihc_upload_id: str = Query(alias="ihcUploadId")
) -> AlignmentReport:
    try:
        return ihc_alignment_service.report(he_upload_id, ihc_upload_id)
    except AlignmentError as exc:
        raise _failure(exc) from exc


@router.post(
    "/{he_upload_id}/confirm",
    response_model=AlignmentReport,
    summary="Sign off (or withdraw) the alignment",
)
def confirm(
    he_upload_id: str,
    ihc_upload_id: str = Query(alias="ihcUploadId"),
    confirmed: bool = Query(default=True),
) -> AlignmentReport:
    """The blocking check. Nothing downstream should use these regions without it.

    Always records a *person* - a request to this endpoint came from somebody
    looking at the screen. The machine stamp is reachable only from the batch
    runner, which is the one caller that genuinely cannot look.
    """
    try:
        return ihc_alignment_service.confirm(
            he_upload_id, ihc_upload_id, confirmed=confirmed, by="person"
        )
    except AlignmentError as exc:
        raise _failure(exc) from exc


@router.get(
    "/{he_upload_id}/panels/{name}.png",
    responses=_PNG,
    response_class=Response,
    summary="One slide with the regions outlined",
)
def panel(
    he_upload_id: str, name: str, ihc_upload_id: str = Query(alias="ihcUploadId")
) -> Response:
    try:
        return _png(ihc_alignment_service.panel(he_upload_id, ihc_upload_id, name))
    except AlignmentError as exc:
        raise _failure(exc) from exc


@router.get(
    "/{he_upload_id}/crops/{rank}.png",
    responses=_PNG,
    response_class=Response,
    summary="One carried region, cropped from the slide",
)
def crop(
    he_upload_id: str,
    rank: int,
    ihc_upload_id: str = Query(alias="ihcUploadId"),
    source: str = Query(default="ihc", pattern="^(ihc|he)$"),
) -> Response:
    """`source=ihc` is the point of the step; `source=he` is the same region on
    the H&E, for a viewer comparing the two."""
    try:
        return _png(ihc_alignment_service.crop(he_upload_id, ihc_upload_id, rank, source=source))
    except AlignmentError as exc:
        raise _failure(exc) from exc
