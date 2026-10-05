"""Step 5 endpoints - optical density.

One request, no job. Given step 4's white point a run is one tile read off the
pyramid and a handful of numpy passes, so unlike step 2 there is nothing to poll.

    GET /density/{id}                              the report, on the best tile
    GET /density/{id}?x=N&y=M                      the same run on a chosen tile
    GET /density/{id}/panels/{name}.png            map | tile | density | scatter | limits
    GET /density/{id}/candidates/{col}/{row}.png   one scored block, as a thumbnail

`threshold` and `percentile` belong to steps 3 and 4 and are passed straight
through to them. Step 5 interprets neither and owns neither: the viewer moves
those controls on those steps' screens, and the density this step reports is
whatever the resulting white point produces. Two places deciding what I0 is would
be one place too many, and I0 is the denominator of every number here.

`x` and `y` are step 5's own control, and the only one. They are level-0 pixel
coordinates - the frame of reference that does not move when a resolution changes -
and they snap to the nearest block the tile chooser actually scored, rather than
reading an unscored tile and reporting a score belonging to its neighbour.
"""

from fastapi import APIRouter, HTTPException, Query, Response, status

from app.pipeline.step05_optical_density.density import DensityError
from app.pipeline.step05_optical_density.tiles import TileError
from app.schemas.density import DensityPanel, DensityReport
from app.services.density_service import density_service
from app.services.upload_service import UploadError

router = APIRouter(prefix="/density", tags=["optical density"])

# Every handler here is a plain `def`. All of them block - reading a tile off the
# pyramid, eigendecomposing a point cloud, encoding a PNG - so declaring them
# `async` would stall the event loop, and with it every other request, for the
# duration. A sync handler is dispatched to the threadpool instead.

_PNG = {200: {"content": {"image/png": {}}}}

_THRESHOLD = Query(
    default=None,
    ge=0,
    le=255,
    description=(
        "Step 3's saturation cut, 0-255, passed through to it. It decides which pixels are "
        "glass, therefore what I0 is, therefore every density here - and also which blocks "
        "are eligible to be the tile."
    ),
)

_PERCENTILE = Query(
    default=None,
    gt=0.0,
    le=100.0,
    description=(
        "Which percentile of the glass step 4 takes as I0. Defaults to the server setting. "
        "Moving it moves every density on this screen by a constant, because density is a "
        "logarithm of a ratio."
    ),
)

_X = Query(
    default=None,
    ge=0,
    description=(
        "Level-0 x of the tile to transform. Snaps to the nearest scored block. Omit to "
        "take the block that scored highest on stain x mixing."
    ),
)

_Y = Query(default=None, ge=0, description="Level-0 y of the tile to transform")


def _upload_failure(exc: UploadError) -> HTTPException:
    missing = "not found" in str(exc)
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND if missing else status.HTTP_409_CONFLICT,
        detail=str(exc),
    )


def _density_failure(exc: DensityError | TileError) -> HTTPException:
    """409, not 400: the request is fine, the slide does not carry what the step needs.

    "No block holds enough tissue to measure a density on" is the case this exists
    for. Nothing is wrong with the query - step 3's threshold has claimed almost
    nothing as tissue, or step 2 excluded what there was - and a 400 would send the
    caller looking at their own parameters instead of at the two steps upstream.
    """
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))


@router.get(
    "/{upload_id}",
    response_model=DensityReport,
    summary="Step 5 - turn one tile's colour into how much stain",
)
def report(
    upload_id: str,
    threshold: int | None = _THRESHOLD,
    percentile: float | None = _PERCENTILE,
    x: int | None = _X,
    y: int | None = _Y,
) -> DensityReport:
    """Transform a tile to optical density, and report the geometry that appears."""
    try:
        return density_service.report(
            upload_id, threshold=threshold, percentile=percentile, x=x, y=y
        )
    except UploadError as exc:
        raise _upload_failure(exc) from exc
    except (DensityError, TileError) as exc:
        raise _density_failure(exc) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"cannot read slide: {exc}") from exc


@router.get(
    "/{upload_id}/panels/{name}.png",
    summary="One of the five panels the step is meant to be read as",
    response_class=Response,
    responses=_PNG,
)
def panel(
    upload_id: str,
    name: DensityPanel,
    threshold: int | None = _THRESHOLD,
    percentile: float | None = _PERCENTILE,
    x: int | None = _X,
    y: int | None = _Y,
) -> Response:
    """`map`, `tile`, `density`, `scatter` or `limits`.

    All five depend on which tile was chosen and on the white point it was divided
    by, so all five carry the query string and none may be cached by an
    intermediary that ignores it.
    """
    try:
        png = density_service.panel(
            upload_id, name.value, threshold=threshold, percentile=percentile, x=x, y=y
        )
    except UploadError as exc:
        raise _upload_failure(exc) from exc
    except (DensityError, TileError) as exc:
        raise _density_failure(exc) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"cannot read slide: {exc}") from exc

    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get(
    "/{upload_id}/candidates/{col}/{row}.png",
    summary="One scored block, as a thumbnail for the contact sheet",
    response_class=Response,
    responses=_PNG,
)
def candidate(
    upload_id: str,
    col: int,
    row: int,
    threshold: int | None = _THRESHOLD,
    percentile: float | None = _PERCENTILE,
) -> Response:
    """The block at `col`, `row` cropped out of the screening thumbnail.

    Takes `threshold` and `percentile` but not `x` and `y`, and both halves of
    that are deliberate. The two it takes decide which blocks are candidates at
    all - the mask says what is tissue, and I0 sets the stain gate every block is
    admitted or refused by - so a thumbnail fetched under one and shown beside
    figures computed under another would be a picture of a different slide. The
    two it omits are the *choice*, and a choice does not change what the
    alternatives look like: the contact sheet is the same twelve pictures whether
    the viewer is standing on the first or the ninth, so leaving x and y out is
    what lets the browser keep them all cached across a pick.
    """
    try:
        png = density_service.candidate(
            upload_id, col=col, row=row, threshold=threshold, percentile=percentile
        )
    except UploadError as exc:
        raise _upload_failure(exc) from exc
    except DensityError as exc:
        # 404 and not 409, unlike every other handler here: an unscored block is a
        # resource that does not exist, not a slide that cannot answer. The rest of
        # the step's failures are the slide's, and stay 409.
        if "no scored block" in str(exc):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
            ) from exc
        raise _density_failure(exc) from exc
    except TileError as exc:
        raise _density_failure(exc) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"cannot read slide: {exc}") from exc

    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )
