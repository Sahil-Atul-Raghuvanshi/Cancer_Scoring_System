"""Step 2 endpoints - quality control.

A run is started, polled, then read. It is not a single request: GrandQC's
artefact pass over a whole slide is minutes of CPU work, and a request that
holds a connection open for that long is a request that times out.

    GET  /qc/capability            can step 2 run here, and what is missing
    POST /qc/{id}/run              start a run (or return the cached one)
    POST /qc/{id}/run?restart=1    throw the cached one away and run again
    POST /qc/{id}/cancel           ask a running pass to stop
    GET  /qc/{id}/run              progress
    GET  /qc/{id}                  the finished report
    GET  /qc/{id}/grid             per-patch metrics, for client-side heatmaps
    GET  /qc/{id}/overlay.png      the slide with artefacts tinted
    GET  /qc/{id}/tissue.png       pass 1's tissue map
    GET  /qc/{id}/classes.png      the class map alone
    GET  /qc/{id}/mask.png         indexed-colour mask, for download
    GET  /qc/{id}/heatmap/{metric}.png
    GET  /qc/{id}/explain          one region, re-measured finely
    GET  /qc/{id}/explain.png      that region's pixels
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Response, status

from app.schemas.qc import QCCapability, QCGrid, QCRegionExplain, QCReport, QCRun
from app.services.qc_service import QCError, qc_service
from app.services.upload_service import UploadError

router = APIRouter(prefix="/qc", tags=["quality control"])

# Every handler here is a plain `def`, not `async def`, and that is deliberate.
# All of them block: /capability imports torch the first time it is called,
# /explain opens a slide and runs filters over it, and the image routes read
# files. Declared `async`, each would stall the event loop - and with it every
# other request - for as long as that took. A sync handler is dispatched to the
# threadpool instead.

_PNG = {200: {"content": {"image/png": {}}}}

#: Cached QC artefacts are immutable for a given upload and parameter set, so
#: they cache hard. A re-run writes new bytes under the same URL, which is why
#: this is not `immutable`.
_PNG_HEADERS = {"Cache-Control": "public, max-age=3600"}


def _upload_failure(exc: UploadError) -> HTTPException:
    missing = "not found" in str(exc)
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND if missing else status.HTTP_409_CONFLICT,
        detail=str(exc),
    )


def _qc_failure(exc: QCError) -> HTTPException:
    """QC problems are 409, not 400: the request is fine, the server is not ready."""
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))


def _png(payload: bytes) -> Response:
    return Response(content=payload, media_type="image/png", headers=_PNG_HEADERS)


@router.get(
    "/capability",
    response_model=QCCapability,
    summary="Whether step 2 can run, and exactly what is missing if not",
)
def capability() -> QCCapability:
    """Report the torch stack, the checkpoints on disk, and where to get the rest.

    Deliberately never raises. A missing checkpoint is the normal state of a
    fresh clone, and the UI needs to be able to say so rather than break.
    """
    return qc_service.capability()


@router.post(
    "/{upload_id}/run",
    response_model=QCRun,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a QC run over an uploaded slide",
)
def start_run(
    upload_id: str,
    background: BackgroundTasks,
    model_mpp: float | None = Query(
        default=None,
        alias="modelMpp",
        description=(
            "Which GrandQC artefact model to run, by the resolution it was trained at: "
            "2.0 = 5x, 1.5 = 7x, 1.0 = 10x. Coarser is faster and slightly less accurate; "
            "on a CPU-only machine that trade is worth making."
        ),
    ),
    restart: bool = Query(
        default=False,
        description=(
            "Discard the cached run and start again. Without this, a request whose "
            "settings match a finished run returns that run - which is what makes "
            "revisiting the step instant instead of another few minutes."
        ),
    ),
) -> QCRun:
    """Queue the two GrandQC passes, or hand back a cached run with the same settings.

    `restart` is what makes "run it again" possible: this endpoint otherwise returns
    the cache whenever the settings match, which is right for revisiting the step and
    wrong when the viewer is deliberately asking for a fresh run.
    """
    try:
        run = (
            qc_service.restart(upload_id, model_mpp=model_mpp)
            if restart
            else qc_service.start(upload_id, model_mpp=model_mpp)
        )
    except UploadError as exc:
        raise _upload_failure(exc) from exc
    except QCError as exc:
        raise _qc_failure(exc) from exc

    if run.state == "queued":
        background.add_task(qc_service.execute, upload_id)
    return run


@router.post(
    "/{upload_id}/cancel",
    response_model=QCRun,
    summary="Ask a running QC pass to stop",
)
def cancel_run(upload_id: str) -> QCRun:
    """Stop the run, promptly, and report the state that leaves.

    Prompt because the flag is read by the progress callback, which runs after every
    patch, rather than being polled once per pass. Cancelling nothing returns the
    current state rather than a conflict - a viewer who clicks as the last patch
    lands has done nothing wrong.
    """
    try:
        return qc_service.cancel(upload_id)
    except UploadError as exc:
        raise _upload_failure(exc) from exc
    except QCError as exc:
        raise _qc_failure(exc) from exc


@router.get(
    "/{upload_id}/run",
    response_model=QCRun,
    summary="Progress of a QC run",
)
def run_state(upload_id: str) -> QCRun:
    """Poll a run. `state` is idle, queued, running, ready or failed."""
    return qc_service.state(upload_id)


@router.get(
    "/{upload_id}",
    response_model=QCReport,
    summary="Step 2 - the finished QC report",
)
def report(upload_id: str) -> QCReport:
    """Artefact shares, tissue accounting, the QC on/off comparison, and the metrics."""
    try:
        return qc_service.report(upload_id)
    except QCError as exc:
        raise _qc_failure(exc) from exc


@router.get(
    "/{upload_id}/grid",
    response_model=QCGrid,
    summary="Per-patch classical metrics",
)
def grid(upload_id: str) -> QCGrid:
    """Every measured patch, so the client can draw its own heatmaps."""
    try:
        return qc_service.grid(upload_id)
    except QCError as exc:
        raise _qc_failure(exc) from exc


@router.get(
    "/{upload_id}/explain",
    response_model=QCRegionExplain,
    summary="Why one region was called what it was called",
)
def explain(
    upload_id: str,
    x: int = Query(default=0, ge=0, description="Left edge in level-0 coordinates"),
    y: int = Query(default=0, ge=0, description="Top edge in level-0 coordinates"),
    size: int = Query(
        default=2048, ge=64, le=16384, description="Region extent in level-0 pixels"
    ),
    target_mpp: float | None = Query(
        default=None,
        gt=0,
        le=8,
        alias="targetMpp",
        description=(
            "Resolution to re-measure at. Defaults to the pipeline's working mpp, which is "
            "finer than the grid - focus is a claim about fine detail."
        ),
    ),
) -> QCRegionExplain:
    """Re-measure one region and set its metrics against this slide's clean tissue."""
    try:
        return qc_service.explain_region(upload_id, x=x, y=y, size=size, target_mpp=target_mpp)
    except UploadError as exc:
        raise _upload_failure(exc) from exc
    except QCError as exc:
        raise _qc_failure(exc) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"cannot read region: {exc}") from exc


# --- images -----------------------------------------------------------------


@router.get(
    "/{upload_id}/overlay.png",
    summary="The slide with its artefacts tinted",
    response_class=Response,
    responses=_PNG,
)
def overlay_png(upload_id: str) -> Response:
    """Only artefacts are painted - what is highlighted is what step 3 will not get."""
    try:
        return _png(qc_service.asset(upload_id, "overlay.png"))
    except QCError as exc:
        raise _qc_failure(exc) from exc


@router.get(
    "/{upload_id}/tissue.png",
    summary="Pass 1 - the tissue map",
    response_class=Response,
    responses=_PNG,
)
def tissue_png(upload_id: str) -> Response:
    """The tissue detector's own output, kept separate so the two models stay two."""
    try:
        return _png(qc_service.asset(upload_id, "tissue.png"))
    except QCError as exc:
        raise _qc_failure(exc) from exc


@router.get(
    "/{upload_id}/classes.png",
    summary="The artefact class map, with no slide under it",
    response_class=Response,
    responses=_PNG,
)
def classes_png(upload_id: str) -> Response:
    try:
        return _png(qc_service.asset(upload_id, "flat.png"))
    except QCError as exc:
        raise _qc_failure(exc) from exc


@router.get(
    "/{upload_id}/mask.png",
    summary="The mask as indexed colour, for download",
    response_class=Response,
    responses=_PNG,
)
def mask_png(upload_id: str) -> Response:
    """One byte per pixel, GrandQC's class ids preserved in the palette index."""
    try:
        return _png(qc_service.asset(upload_id, "mask.png"))
    except QCError as exc:
        raise _qc_failure(exc) from exc


@router.get(
    "/{upload_id}/heatmap/{metric}.png",
    summary="One classical metric over the patch grid",
    response_class=Response,
    responses=_PNG,
)
def heatmap_png(upload_id: str, metric: str) -> Response:
    """A feature map - the explanation a red blob on the overlay is asking for."""
    try:
        return _png(qc_service.heatmap(upload_id, metric))
    except QCError as exc:
        raise _qc_failure(exc) from exc


@router.get(
    "/{upload_id}/explain.png",
    summary="The pixels of an inspected region",
    response_class=Response,
    responses=_PNG,
)
def explain_png(
    upload_id: str,
    x: int = Query(default=0, ge=0),
    y: int = Query(default=0, ge=0),
    size: int = Query(default=2048, ge=64, le=16384),
    out: int = Query(default=512, ge=64, le=2048, description="Rendered size in pixels"),
) -> Response:
    try:
        return _png(qc_service.region_png(upload_id, x=x, y=y, size=size, out=out))
    except UploadError as exc:
        raise _upload_failure(exc) from exc
    except QCError as exc:
        raise _qc_failure(exc) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"cannot read region: {exc}") from exc
