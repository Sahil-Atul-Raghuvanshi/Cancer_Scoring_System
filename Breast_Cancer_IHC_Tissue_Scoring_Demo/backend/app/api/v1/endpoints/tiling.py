"""Step 7 endpoints - tiling.

One request, no job. The index is one pass over the grid against a mask step 3
has already cached, so there is nothing to poll.

    GET /tiling/{id}                        the funnel, the geometry, a sample of tiles
    GET /tiling/{id}?overlap=0.5            the same grid at a different overlap
    GET /tiling/{id}/panels/{name}.png      grid | sample

`threshold` belongs to step 3 and is passed straight through: it decides what is
tissue, and what is tissue decides which tiles survive.

`overlap` is step 7's own control and the only one. The tile size and the working
resolution are step 1's - the working magnification is one decision for the whole
pipeline - and the two gates are thresholds whose values are arguments rather than
preferences a caller should be moving. Overlap is different: it is a genuine
trade, seam quality against compute, and a viewer moving it should watch the tile
count climb.
"""

from fastapi import APIRouter, HTTPException, Query, Response, status

from app.pipeline.step07_tiling.index import TilingError
from app.schemas.tiling import TilingPanel, TilingReport
from app.services.tiling_service import tiling_service
from app.services.upload_service import UploadError

router = APIRouter(prefix="/tiling", tags=["tiling"])

# Every handler here is a plain `def`. All of them block - scoring tens of
# thousands of grid cells, encoding a PNG, and on a cold cache building step 3's
# mask behind it - so declaring them `async` would stall the event loop, and with
# it every other request, for the duration.

_PNG = {200: {"content": {"image/png": {}}}}

_THRESHOLD = Query(
    default=None,
    ge=0,
    le=255,
    description=(
        "Step 3's saturation cut, 0-255, passed through to it. It decides what is tissue, "
        "and what is tissue decides which tiles survive the first gate."
    ),
)

_OVERLAP = Query(
    default=None,
    ge=0.0,
    lt=1.0,
    description=(
        "Fraction of its own extent a tile shares with its neighbour. Defaults to the "
        "server setting. Overlap buys reliable tile edges - a model has no context past "
        "one - at a quadratic cost in tiles, and every tile is a forward pass at step 8."
    ),
)


def _upload_failure(exc: UploadError) -> HTTPException:
    missing = "not found" in str(exc)
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND if missing else status.HTTP_409_CONFLICT,
        detail=str(exc),
    )


def _tiling_failure(exc: TilingError) -> HTTPException:
    """409, not 400: the request is fine, the slide does not carry what the step needs.

    "No tile of this slide is enough tissue and clear enough of artefacts" is the
    case this exists for. Nothing is wrong with the query - step 3's threshold has
    claimed almost nothing, or step 2 flagged most of what there was - and a 400
    would send the caller looking at their own parameters instead of at the two
    steps upstream.
    """
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))


@router.get(
    "/{upload_id}",
    response_model=TilingReport,
    summary="Step 7 - cut the tissue into patches the model can take",
)
def report(
    upload_id: str,
    threshold: int | None = _THRESHOLD,
    overlap: float | None = _OVERLAP,
) -> TilingReport:
    """Build the tile index, and report what each gate removed."""
    try:
        return tiling_service.report(upload_id, threshold=threshold, overlap=overlap)
    except UploadError as exc:
        raise _upload_failure(exc) from exc
    except TilingError as exc:
        raise _tiling_failure(exc) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"cannot read slide: {exc}") from exc


@router.get(
    "/{upload_id}/panels/{name}.png",
    summary="The grid on the slide, or one tile as the model receives it",
    response_class=Response,
    responses=_PNG,
)
def panel(
    upload_id: str,
    name: TilingPanel,
    threshold: int | None = _THRESHOLD,
    overlap: float | None = _OVERLAP,
) -> Response:
    """`grid` or `sample`.

    Both carry the whole query string, because both depend on the mask the index
    was built against and on the overlap that set the grid - and `sample` reaches
    through step 6 for its pixels, so it depends on the white point behind that
    too.
    """
    try:
        png = tiling_service.panel(
            upload_id, name.value, threshold=threshold, overlap=overlap
        )
    except UploadError as exc:
        raise _upload_failure(exc) from exc
    except TilingError as exc:
        raise _tiling_failure(exc) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"cannot read slide: {exc}") from exc

    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )
