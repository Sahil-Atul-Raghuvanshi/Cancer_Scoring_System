"""Step 4 endpoints - white calibration.

One request, no job. Given step 3's mask a run is a thumbnail read and a few
numpy passes, so unlike step 2 there is nothing to poll.

    GET /calibration/{id}                       the report, at step 3's own cut
    GET /calibration/{id}?threshold=N           the same run over a hand-cut mask
    GET /calibration/{id}/panels/{name}.png     thumbnail | glass | field | corrected
    GET /calibration/{id}/swatch.png            I0 as a flat square
    GET /calibration/compare?uploadIds=a,b      two or more slides' I0 side by side

`threshold` belongs to step 3 and is passed straight through to it. Step 4 does
not interpret it and does not own a threshold of its own: the viewer moves one
slider, on step 3's screen, and the glass this step samples is whatever mask that
produced. Two places deciding what tissue is would be one place too many.

`compare` is the step's argument rather than its output. One slide's I0 in
isolation is a number; two slides' I0 next to each other is the reason the number
has to be measured per slide, and it is registered before `/{upload_id}` so the
literal path is not swallowed by the id parameter.
"""

from fastapi import APIRouter, HTTPException, Query, Response, status

from app.pipeline.step04_white_calibration.calibration import CalibrationError
from app.schemas.calibration import (
    CalibrationComparison,
    CalibrationPanel,
    CalibrationReport,
)
from app.services.calibration_service import calibration_service
from app.services.upload_service import UploadError

router = APIRouter(prefix="/calibration", tags=["white calibration"])

# Every handler here is a plain `def`. All of them block - reading a slide,
# running a least squares, encoding a PNG - so declaring them `async` would stall
# the event loop, and with it every other request, for the duration. A sync
# handler is dispatched to the threadpool instead.

_PNG = {200: {"content": {"image/png": {}}}}

_THRESHOLD = Query(
    default=None,
    ge=0,
    le=255,
    description=(
        "Step 3's saturation cut, 0-255, passed through to it. Omit to use whichever rule "
        "step 3's histogram selects. A different cut is a different tissue mask and "
        "therefore a different set of glass pixels, so I0 moves with it."
    ),
)

_PERCENTILE = Query(
    default=None,
    gt=0.0,
    le=100.0,
    description=(
        "Which percentile of the glass to take as I0. Defaults to the server setting "
        "(the 95th). The report carries the whole ladder either way, so this is for "
        "seeing the plateau move rather than for tuning."
    ),
)


def _upload_failure(exc: UploadError) -> HTTPException:
    missing = "not found" in str(exc)
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND if missing else status.HTTP_409_CONFLICT,
        detail=str(exc),
    )


def _calibration_failure(exc: CalibrationError) -> HTTPException:
    """409, not 400: the request is fine, the slide does not carry what the step needs.

    "No glass left to sample" is the case this exists for. Nothing is wrong with
    the query - the slide is cropped to its section, or step 3's threshold has
    claimed the whole scan as tissue - and a 400 would send the caller looking at
    their own parameters.
    """
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))


@router.get(
    "/compare",
    response_model=CalibrationComparison,
    summary="Two or more slides' white points, and the OD cost of confusing them",
)
def compare(
    upload_ids: str = Query(
        alias="uploadIds",
        description="Comma-separated upload ids, at least two",
    ),
) -> CalibrationComparison:
    """Why calibration is per slide, as one response.

    Each slide is calibrated at its own default settings, so the comparison is
    between slides and not between parameter choices.
    """
    ids = [part.strip() for part in upload_ids.split(",") if part.strip()]

    try:
        return calibration_service.compare(ids)
    except UploadError as exc:
        raise _upload_failure(exc) from exc
    except CalibrationError as exc:
        raise _calibration_failure(exc) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"cannot read slide: {exc}") from exc


@router.get(
    "/{upload_id}",
    response_model=CalibrationReport,
    summary="Step 4 - estimate I0 from this slide's own glass",
)
def report(
    upload_id: str,
    threshold: int | None = _THRESHOLD,
    percentile: float | None = _PERCENTILE,
) -> CalibrationReport:
    """Sample the glass, fit the illumination field, and report both plus the choice."""
    try:
        return calibration_service.report(
            upload_id, threshold=threshold, percentile=percentile
        )
    except UploadError as exc:
        raise _upload_failure(exc) from exc
    except CalibrationError as exc:
        raise _calibration_failure(exc) from exc
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
    name: CalibrationPanel,
    threshold: int | None = _THRESHOLD,
    percentile: float | None = _PERCENTILE,
) -> Response:
    """`thumbnail`, `glass`, `field` or `corrected`.

    All four depend on which pixels are glass, so all four depend on step 3's cut
    and none may be cached by an intermediary that ignores the query string.
    """
    try:
        png = calibration_service.panel(
            upload_id, name.value, threshold=threshold, percentile=percentile
        )
    except UploadError as exc:
        raise _upload_failure(exc) from exc
    except CalibrationError as exc:
        raise _calibration_failure(exc) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"cannot read slide: {exc}") from exc

    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get(
    "/{upload_id}/swatch.png",
    summary="I0 as a flat square of colour",
    response_class=Response,
    responses=_PNG,
)
def swatch(
    upload_id: str,
    threshold: int | None = _THRESHOLD,
    percentile: float | None = _PERCENTILE,
) -> Response:
    """The white point itself, as an image.

    A separate endpoint rather than a fifth panel because it is not a picture of
    the slide - it is a picture of one number, and it is the artefact to put next
    to another slide's.
    """
    try:
        png = calibration_service.swatch(
            upload_id, threshold=threshold, percentile=percentile
        )
    except UploadError as exc:
        raise _upload_failure(exc) from exc
    except CalibrationError as exc:
        raise _calibration_failure(exc) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"cannot read slide: {exc}") from exc

    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )
