"""Step 7 endpoints - tiling.

One request, no job. The index is one pass over the grid against a mask step 3
has already cached, so there is nothing to poll.

    GET  /tiling/{id}/branches              what this slide can be shown as, and why not
    POST /tiling/{id}/selection              commit a branch and a field of view
    GET  /tiling/{id}                        the funnel, the geometry, a sample of tiles
    GET  /tiling/{id}?overlap=0.5            the same grid at a different overlap
    GET  /tiling/{id}/panels/{name}.png      grid | sample

**The commit is a POST and the reads are GETs, and that split is load-bearing.** The
choice a viewer commits is what step 8, step 9 and the runner rebuild their grid from,
so writing it from a GET would make "the viewer looked at 672 um" indistinguishable from
"the viewer chose 672 um" - and every panel request, which carries the same query string,
would rewrite it.

`branch` is accepted on the two GETs so a screen can preview a choice before committing
it. Previewing is free; it lays a grid and prices it and changes nothing downstream.

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
from app.pipeline.step08_tissue_type_segmentation.branches import (
    SERVED_BRANCHES,
    ModelBranch,
)
from app.schemas.tiling import (
    TilingBranchName,
    TilingBranches,
    TilingPanel,
    TilingReport,
    TilingSelection,
    TilingSelectionIn,
)
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


_FOV = Query(
    default=None,
    gt=0.0,
    description=(
        "Microns of slide across one square. One of the offered fields of view - see "
        "`fields_of_view` on the report - and anything else is refused rather than "
        "snapped to the nearest, because this is the parameter that decides which "
        "checkpoint step 8 runs. Defaults to the server setting."
    ),
)


_BRANCH = Query(
    default=None,
    description=(
        "Which of step 7's three options to lay the grid for: `h_channel`, `he` or "
        "`beetle`. Defaults to whatever this slide's viewer committed, and to "
        "`h_channel` when nobody has. `beetle` is refused - it is named on the screen "
        "so the option is visible as not built, and there is no grid to lay for it."
    ),
)


def _refuse_undeveloped(branch: TilingBranchName | None) -> None:
    """A branch this pipeline names but has not built is a 409, not a 422: the request
    is well formed and the codebase is what is missing, which is a statement about us
    rather than about the caller's query.

    Nothing currently trips this - `SERVED_BRANCHES` holds all three since BEETLE was
    built - and it is kept rather than deleted because `ModelBranch` is the wire enum
    and a fourth name added there before its serving path exists would otherwise reach
    `tiling_service` and fail somewhere far less legible. The offered list is read out
    of `SERVED_BRANCHES` for the same reason: a hard-coded "the two that work" was
    already wrong once.
    """
    if branch is None:
        return
    if ModelBranch(branch.value) not in SERVED_BRANCHES:
        offered = ", ".join(f"`{served.value}`" for served in SERVED_BRANCHES)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"`{branch.value}` is not developed yet, so there is no grid to lay "
                f"for it. The options that work are {offered}."
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
    "/{upload_id}/branches",
    response_model=TilingBranches,
    summary="What this slide can be shown as, and why an option is unavailable",
)
def branches(upload_id: str) -> TilingBranches:
    """The branch screen's payload, before any grid exists.

    Cheap by construction: published manifests plus step 5's cached staining verdict.
    No slide pixels, no grid, and no torch - so this answers on a machine where step 8
    cannot run at all, which is the same property that lets the field-of-view picker
    show what is available before anything is computed.
    """
    try:
        return tiling_service.branches(upload_id)
    except UploadError as exc:
        raise _upload_failure(exc) from exc
    except TilingError as exc:
        raise _tiling_failure(exc) from exc


@router.post(
    "/{upload_id}/selection",
    response_model=TilingSelection,
    summary="Commit the branch and field of view step 8 will run on",
)
def commit_selection(upload_id: str, body: TilingSelectionIn) -> TilingSelection:
    """Record the choice, and return the geometry it resolves to.

    What makes this a write rather than a query: step 8's worker, step 9 and the
    pipeline runner all ask step 7 for the grid without naming one, and this record is
    what they get. Before it existed they got the *configured default* instead, so a
    viewer could choose 448 um, watch step 7 price that grid, and have step 8 classify
    a 224 um one underneath them.
    """
    _refuse_undeveloped(body.branch)
    try:
        return tiling_service.commit_selection(
            upload_id,
            branch=body.branch.value,
            fov=body.fov,
            overlap=body.overlap,
            threshold=body.threshold,
        )
    except UploadError as exc:
        raise _upload_failure(exc) from exc
    except TilingError as exc:
        raise _tiling_failure(exc) from exc


@router.get(
    "/{upload_id}",
    response_model=TilingReport,
    summary="Step 7 - cut the tissue into patches the model can take",
)
def report(
    upload_id: str,
    threshold: int | None = _THRESHOLD,
    overlap: float | None = _OVERLAP,
    fov: float | None = _FOV,
    branch: TilingBranchName | None = _BRANCH,
) -> TilingReport:
    """Build the tile index, and report what each gate removed."""
    _refuse_undeveloped(branch)
    try:
        return tiling_service.report(
            upload_id,
            threshold=threshold,
            overlap=overlap,
            fov=fov,
            branch=branch.value if branch else None,
        )
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
    fov: float | None = _FOV,
    branch: TilingBranchName | None = _BRANCH,
) -> Response:
    """`grid` or `sample`.

    Both carry the whole query string, because both depend on the mask the index
    was built against and on the overlap that set the grid - and `sample` reaches
    through step 6 for its pixels, so it depends on the white point behind that
    too.

    `branch` is part of that string for `sample` in particular: the two branches show
    the model different things, and the panel is captioned "one square, as the model
    receives it". A cached URL that ignored the branch would hand the H&E screen a
    picture of a deconvolved density plane the H&E model never sees.
    """
    _refuse_undeveloped(branch)
    try:
        png = tiling_service.panel(
            upload_id,
            name.value,
            threshold=threshold,
            overlap=overlap,
            fov=fov,
            branch=branch.value if branch else None,
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
