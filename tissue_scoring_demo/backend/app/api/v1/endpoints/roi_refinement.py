"""Step 11 endpoints - BEETLE on the chosen regions, one at a time.

    POST /roi-refinement/{id}/run           refine every selected region not yet done
    POST /roi-refinement/{id}/run?rebuild=1 refine them all again
    POST /roi-refinement/{id}/run  {roiIds}  refine only these - the retry
    POST /roi-refinement/{id}/cancel        stop after the region in progress
    GET  /roi-refinement/{id}/run           how far the pass has got
    GET  /roi-refinement/{id}/run?paintedSince=N  ...and what it has painted since N
    GET  /roi-refinement/{id}               the report: one row per selected region
    GET  /roi-refinement/{id}/regions/{roiId}/{name}.png
                                            input | tile | beetle_mask | beetle_overlay
    GET  /roi-refinement/{id}/panels/{name}.png   coarse | refined

A run is started, polled, then read - like step 8, and for step 8's reason. This is
minutes of CPU and a request that holds a connection open for it is a request that times
out.

**The report is readable while the pass is running, and that is the point of the route
layout.** Every region writes its own directory the moment it finishes, so `GET
/roi-refinement/{id}` returns the finished rows alongside the pending ones and the
pictures for a finished region are already there. A screen polls `/run` for the position
and re-reads the report to draw whatever has landed; it never has to wait for the last
region to show the first.

**A retry is the same endpoint with `roiIds`.** Not a separate route, because it is the
same action with a narrower scope, and because expressing it as a scope is what stops it
from becoming a second code path that could refine a region step 10 never ticked - the
ids are intersected with the selection either way.
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Response, status

from app.schemas.roi_refinement import (
    RefinementReport,
    RefinementRequest,
    RefinementRun,
)
from app.services.roi_refinement_service import RefinementError, roi_refinement_service
from app.services.roi_selection_service import RoiSelectionError
from app.services.tissue_type_service import TissueTypeError
from app.services.upload_service import UploadError

router = APIRouter(prefix="/roi-refinement", tags=["roi refinement"])

# Plain `def`, not `async def`: these block on file reads and on starting a worker.

_PNG = {200: {"content": {"image/png": {}}}}

#: A region's pictures are immutable until that region is refined again, which writes
#: new bytes under the same URL - so this is not `immutable`.
_PNG_HEADERS = {"Cache-Control": "public, max-age=3600"}

_REGION_PANELS = ("input", "tile", "beetle_mask", "beetle_overlay")
_SLIDE_PANELS = ("coarse", "refined")

_PAINTED_SINCE = Query(
    default=None,
    ge=0,
    alias="paintedSince",
    description=(
        "Also return the windows the region in progress has segmented since this "
        "position in its paint log, each with the pixel mask BEETLE produced for it. "
        "The position is per region and resets when the pass moves on, so a client "
        "watches `paint.roiId` rather than the cursor to know when to start a new "
        "canvas. Omit it and the response is the position alone."
    ),
)

_REBUILD = Query(
    default=False,
    description=(
        "Refine regions that already finished. Off by default, which is what makes a "
        "restart cheap: a complete region's boundary is on disk and re-running it would "
        "spend minutes reproducing it."
    ),
)


def _upload_failure(exc: UploadError) -> HTTPException:
    missing = "not found" in str(exc)
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND if missing else status.HTTP_409_CONFLICT,
        detail=str(exc),
    )


def _conflict(exc: Exception) -> HTTPException:
    """409: the request is fine, the slide or an earlier step is not ready."""
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))


@router.post(
    "/{upload_id}/run",
    response_model=RefinementRun,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Step 11 - run BEETLE on the selected regions",
)
def start(
    upload_id: str,
    background: BackgroundTasks,
    request: RefinementRequest | None = None,
    rebuild: bool = _REBUILD,
) -> RefinementRun:
    """Queue a pass over whatever is selected and not already refined.

    Returns immediately with the run's state. When every selected region is already on
    disk at this geometry the state comes back `ready` without a worker being started at
    all - that is the resume path, and it is why revisiting this screen after a finished
    pass costs nothing.
    """
    body = request or RefinementRequest()
    try:
        run = roi_refinement_service.start(
            upload_id, roi_ids=body.roi_ids, rebuild=rebuild or body.rebuild
        )
    except UploadError as exc:
        raise _upload_failure(exc) from exc
    except (RefinementError, RoiSelectionError, TissueTypeError) as exc:
        raise _conflict(exc) from exc

    if run.state == "queued":
        background.add_task(roi_refinement_service.execute, upload_id)
    return run


@router.post(
    "/{upload_id}/cancel",
    response_model=RefinementRun,
    summary="Ask a running pass to stop",
)
def cancel(upload_id: str) -> RefinementRun:
    """Stop after the region in progress, keeping every region already finished.

    After rather than during: a half-segmented region is not a result, and the value of
    cancelling a pass like this is precisely that the finished regions survive it and are
    not recomputed next time.
    """
    try:
        return roi_refinement_service.cancel(upload_id)
    except UploadError as exc:
        raise _upload_failure(exc) from exc
    except RefinementError as exc:
        raise _conflict(exc) from exc


@router.get(
    "/{upload_id}/run",
    response_model=RefinementRun,
    summary="How far the pass has got",
)
def state(
    upload_id: str,
    painted_since: int | None = _PAINTED_SINCE,
) -> RefinementRun:
    """The position: which region, how many regions, how many windows.

    Three counters rather than one, because they answer different questions. Regions
    done is what the list ticks off; windows done is what stops a single large region
    from looking stalled for the ten minutes it takes; and `progress` carries the same
    pair per region, which is what lets every row draw its own bar instead of sharing
    one.

    With `paintedSince` it is also the paint feed: the windows the region in progress
    has been through since that point, each with the mask BEETLE drew inside it. That is
    what turns the right-hand panel from an empty frame into the segmentation arriving -
    and it is the same data `beetle_mask.png` is rendered from when the region finishes,
    rather than an animation standing in for it.
    """
    try:
        return roi_refinement_service.state(upload_id, painted_since=painted_since)
    except UploadError as exc:
        raise _upload_failure(exc) from exc
    except RefinementError as exc:
        raise _conflict(exc) from exc


@router.get(
    "/{upload_id}",
    response_model=RefinementReport,
    summary="The refined regions: one row per selected candidate",
)
def report(upload_id: str) -> RefinementReport:
    """Every selected region with its state, its boundary and its before/after numbers.

    Safe to poll while the pass runs. Rows land as their regions finish, so a client can
    draw a comparison the moment there is one to draw.
    """
    try:
        return roi_refinement_service.report(upload_id)
    except UploadError as exc:
        raise _upload_failure(exc) from exc
    except RefinementError as exc:
        raise _conflict(exc) from exc


@router.get(
    "/{upload_id}/regions/{roi_id}/{name}.png",
    responses=_PNG,
    response_class=Response,
    summary="One region's before/after pictures",
)
def region_panel(upload_id: str, roi_id: str, name: str) -> Response:
    """`tile` and `beetle_overlay` are the comparison; `beetle_mask` is the evidence.

    The two overlays are the same crop with different geometry drawn on it, which is what
    makes them a fair pair. `beetle_mask` is the one that cannot flatter the result: it
    shows what the network said about every pixel of the box, so a refinement that had
    simply returned the rectangle would be obvious at a glance.
    """
    if name not in _REGION_PANELS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"unknown picture {name!r}; expected one of {', '.join(_REGION_PANELS)}",
        )
    try:
        payload = roi_refinement_service.asset(upload_id, roi_id, f"{name}.png")
    except UploadError as exc:
        raise _upload_failure(exc) from exc
    except RefinementError as exc:
        raise _conflict(exc) from exc
    return Response(content=payload, media_type="image/png", headers=_PNG_HEADERS)


@router.get(
    "/{upload_id}/panels/{name}.png",
    responses=_PNG,
    response_class=Response,
    summary="The whole section, before and after",
)
def slide_panel(upload_id: str, name: str) -> Response:
    """`coarse` is the chosen squares; `refined` is what BEETLE drew inside them.

    The same drawing code with different geometry, so the difference between the two
    panels is the difference between the two answers and not between two renderers.
    """
    if name not in _SLIDE_PANELS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"unknown panel {name!r}; expected one of {', '.join(_SLIDE_PANELS)}",
        )
    try:
        payload = roi_refinement_service.panel(upload_id, name)
    except UploadError as exc:
        raise _upload_failure(exc) from exc
    except RefinementError as exc:
        raise _conflict(exc) from exc
    return Response(content=payload, media_type="image/png", headers=_PNG_HEADERS)
