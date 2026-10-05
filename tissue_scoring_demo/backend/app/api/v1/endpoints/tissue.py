"""Step 3 endpoints - the tissue mask.

One request, no job. A run is a thumbnail read and a few numpy passes, so unlike
step 2 there is nothing to poll.

    GET /tissue/{id}                    the report, Otsu's threshold by default
    GET /tissue/{id}?threshold=N        the same run at a hand-set cut
    GET /tissue/{id}/panels/{name}.png  thumbnail | saturation | mask | overlay

`threshold` is what makes the slider work. Every drag re-runs the real
thresholding and the real morphology rather than approximating it in the
browser; the cached saturation channel is what keeps that instant.
"""

from fastapi import APIRouter, HTTPException, Query, Response, status

from app.pipeline.step03_tissue_mask.mask import TissueMaskError
from app.schemas.tissue import TissuePanel, TissueReport
from app.services.tissue_service import tissue_service
from app.services.upload_service import UploadError

router = APIRouter(prefix="/tissue", tags=["tissue mask"])

# Every handler here is a plain `def`. All of them block - reading a slide,
# running scipy's morphology, encoding a PNG - so declaring them `async` would
# stall the event loop, and with it every other request, for the duration. A
# sync handler is dispatched to the threadpool instead.

_PNG = {200: {"content": {"image/png": {}}}}

_THRESHOLD = Query(
    default=None,
    ge=0,
    le=255,
    description=(
        "Saturation level to cut at, 0-255. Omit to let Otsu choose. A value makes the "
        "run a manual comparison; Otsu's own answer is reported either way, so the two "
        "are always on screen together."
    ),
)

_TARGET_MPP = Query(
    default=None,
    gt=0,
    le=32,
    alias="targetMpp",
    description=(
        "Resolution to build the mask at. Defaults to the server setting (2 um/px). "
        "The achieved resolution can be coarser - the mask's longest edge is capped."
    ),
)


def _upload_failure(exc: UploadError) -> HTTPException:
    missing = "not found" in str(exc)
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND if missing else status.HTTP_409_CONFLICT,
        detail=str(exc),
    )


def _tissue_failure(exc: TissueMaskError) -> HTTPException:
    """409, not 400: the request is fine, the slide does not carry what the step needs."""
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))


@router.get(
    "/{upload_id}",
    response_model=TissueReport,
    summary="Step 3 - separate tissue from empty glass",
)
def report(
    upload_id: str,
    threshold: int | None = _THRESHOLD,
    target_mpp: float | None = _TARGET_MPP,
) -> TissueReport:
    """Threshold the saturation channel, clean up with morphology, report every step."""
    try:
        return tissue_service.report(
            upload_id, threshold=threshold, target_mpp=target_mpp
        )
    except UploadError as exc:
        raise _upload_failure(exc) from exc
    except TissueMaskError as exc:
        raise _tissue_failure(exc) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"cannot read slide: {exc}") from exc


@router.get(
    "/{upload_id}/panels/{name}.png",
    summary="One of the four panels the step is meant to be read as",
    response_class=Response,
    responses=_PNG,
)
def panel(
    upload_id: str,
    name: TissuePanel,
    threshold: int | None = _THRESHOLD,
    target_mpp: float | None = _TARGET_MPP,
) -> Response:
    """`thumbnail`, `saturation`, `mask` or `overlay`.

    The first two do not depend on the threshold and cache hard. The last two do,
    so they are rendered per request and must not be cached by an intermediary
    that ignores the query string.
    """
    try:
        png = tissue_service.panel(
            upload_id, name.value, threshold=threshold, target_mpp=target_mpp
        )
    except UploadError as exc:
        raise _upload_failure(exc) from exc
    except TissueMaskError as exc:
        raise _tissue_failure(exc) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"cannot read slide: {exc}") from exc

    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )
