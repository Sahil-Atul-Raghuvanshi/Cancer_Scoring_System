"""Step 6 endpoints - colour deconvolution.

One request, no job. The step's own work is two inverses of a 3x3 and two matrix
multiplies over one tile, and its input comes from step 5's already-memoised
hand-off, so there is nothing to poll.

    GET /deconvolution/{id}                              both bases, in one report
    GET /deconvolution/{id}?x=N&y=M                      the same, on a chosen tile
    GET /deconvolution/{id}/panels/{name}.png            tile | haematoxylin | dab | residual
    GET /deconvolution/{id}/panels/{name}.png?basis=estimated

Every query parameter here belongs to an earlier step and is passed straight
through: `threshold` is step 3's, `percentile` step 4's, `x` and `y` step 5's.
Step 6 owns none of them, and it deliberately has no parameter of its own for the
one thing it does decide - which stain vectors to project onto - because making
that a query parameter would make the DAB scale the caller's choice, and a
measurement whose units depend on the request is not comparable to anything.

`basis` is the one exception, and only on the panels. It exists so the screen can
show what choosing a per-image estimate would look like, side by side with the
published vectors. The report always carries both, so the two scores can be put on
one screen without a second request.
"""

from fastapi import APIRouter, HTTPException, Query, Response, status

from app.pipeline.step05_optical_density.density import DensityError
from app.pipeline.step05_optical_density.tiles import TileError
from app.pipeline.step06_colour_deconvolution.deconvolution import DeconvolutionError
from app.schemas.deconvolution import (
    DeconvolutionBasis,
    DeconvolutionPanel,
    DeconvolutionReport,
)
from app.services.deconvolution_service import deconvolution_service
from app.services.upload_service import UploadError

router = APIRouter(prefix="/deconvolution", tags=["colour deconvolution"])

# Every handler here is a plain `def`. All of them block - un-mixing a quarter of
# a million pixels, encoding a PNG, and on a cold cache reading a tile off the
# pyramid behind step 5 - so declaring them `async` would stall the event loop,
# and with it every other request, for the duration.

_PNG = {200: {"content": {"image/png": {}}}}

_THRESHOLD = Query(
    default=None,
    ge=0,
    le=255,
    description=(
        "Step 3's saturation cut, 0-255, passed through to it. It decides which pixels are "
        "glass, therefore what I0 is, therefore the optical density this step un-mixes."
    ),
)

_PERCENTILE = Query(
    default=None,
    gt=0.0,
    le=100.0,
    description="Which percentile of the glass step 4 takes as I0. Defaults to the server setting.",
)

_X = Query(
    default=None,
    ge=0,
    description=(
        "Level-0 x of the tile to un-mix. Step 5's control, passed through, so both steps "
        "stand on the same pixels. Omit to take the tile step 5's screening chose."
    ),
)

_Y = Query(default=None, ge=0, description="Level-0 y of the tile to un-mix")

_BASIS = Query(
    default=DeconvolutionBasis.FIXED,
    description=(
        "Which stain matrix the panel is drawn from. 'fixed' is Ruifrok & Johnston's "
        "published vectors and is what the measurement branch uses; 'estimated' is "
        "Macenko's per-image pair, served so the screen can show what choosing it would "
        "cost. Both bases' numbers are in the report either way."
    ),
)


def _upload_failure(exc: UploadError) -> HTTPException:
    missing = "not found" in str(exc)
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND if missing else status.HTTP_409_CONFLICT,
        detail=str(exc),
    )


def _step_failure(exc: DeconvolutionError | DensityError | TileError) -> HTTPException:
    """409, not 400: the request is fine, the slide does not carry what the step needs.

    Steps 5's failures reach here unchanged, because step 6's input is step 5's
    output - "no block of this slide holds enough tissue to measure a density on"
    is as true a reason for step 6 to have no answer as for step 5, and rewording
    it here would hide which step actually stopped.
    """
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))


@router.get(
    "/{upload_id}",
    response_model=DeconvolutionReport,
    summary="Step 6 - un-mix haematoxylin from DAB, on both bases at once",
)
def report(
    upload_id: str,
    threshold: int | None = _THRESHOLD,
    percentile: float | None = _PERCENTILE,
    x: int | None = _X,
    y: int | None = _Y,
) -> DeconvolutionReport:
    """Separate the stains, and report what the choice of stain vectors costs.

    Both bases in one response deliberately. The comparison is the argument of
    this step - the same tile, the same arithmetic, the same absolute cut, two
    different scores - and a screen that had to fetch the second one separately
    could show the two halves of that argument a second apart, describing
    different runs.
    """
    try:
        return deconvolution_service.report(
            upload_id, threshold=threshold, percentile=percentile, x=x, y=y
        )
    except UploadError as exc:
        raise _upload_failure(exc) from exc
    except (DeconvolutionError, DensityError, TileError) as exc:
        raise _step_failure(exc) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"cannot read slide: {exc}") from exc


@router.get(
    "/{upload_id}/panels/{name}.png",
    summary="One channel of one basis, as a picture",
    response_class=Response,
    responses=_PNG,
)
def panel(
    upload_id: str,
    name: DeconvolutionPanel,
    basis: DeconvolutionBasis = _BASIS,
    threshold: int | None = _THRESHOLD,
    percentile: float | None = _PERCENTILE,
    x: int | None = _X,
    y: int | None = _Y,
) -> Response:
    """`tile`, `haematoxylin`, `dab` or `residual`.

    All four depend on the tile, the white point behind it and the basis, so all
    four carry the whole query string and none may be cached by an intermediary
    that ignores it. `tile` ignores `basis` on purpose - the input to the step is
    the input to the step, and serving a different picture per basis would suggest
    the two un-mixed different pixels.
    """
    try:
        png = deconvolution_service.panel(
            upload_id,
            name.value,
            basis=basis.value,
            threshold=threshold,
            percentile=percentile,
            x=x,
            y=y,
        )
    except UploadError as exc:
        raise _upload_failure(exc) from exc
    except (DeconvolutionError, DensityError, TileError) as exc:
        raise _step_failure(exc) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"cannot read slide: {exc}") from exc

    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )
