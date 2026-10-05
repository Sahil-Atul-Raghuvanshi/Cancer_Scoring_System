"""Step 8 endpoints - tissue-type segmentation.

A run is started, polled, then read. It is not a single request: a whole-slide pass is
tens of thousands of ResNet18 forward passes, which on a CPU is tens of minutes, and a
request that holds a connection open for that long is a request that times out.

    GET  /tissue-type/capability          can step 8 run here, which checkpoints exist
    POST /tissue-type/{id}/run            start a run (or return the cached one)
    POST /tissue-type/{id}/run?restart=1  throw the cached one away and run again
    POST /tissue-type/{id}/cancel         ask a running pass to stop
    GET  /tissue-type/{id}/run            progress
    GET  /tissue-type/{id}/run?paintedSince=0   progress, and the classes decided since
    GET  /tissue-type/{id}                the finished report
    GET  /tissue-type/{id}/panels/{name}.png    map | flat | scored | confidence | uncertainty

`classes` on the panel routes is the guide's per-class opacity toggle. It is applied on
the server rather than in the browser so that an unselected class is *absent* from the
picture rather than recoloured: a viewer switching fat off should watch the tissue leave
the map and the denominator with it, which is the point the guide asks this screen to
make.

The trained options draw a **fourth** class, `3 uncertain`, which no head emits: a
post-pass over the finished map recolours the in-situ calls it will not stand behind.
It can only ever land on class 1, which is excluded from the score anyway, so nothing
it does moves `tumourContent` or step 9. `uncertainty.png` is that layer's score before
it is thresholded, and it is the one panel the BEETLE option does not serve - asking for
it there is a 409 rather than a blank picture.

`overlap` and `model` are the only two things a caller may choose, and neither is a
preference. Overlap is a real trade - seam quality against a compute bill roughly linear
in the window count - and defaults to step 7's, so the choice priced on the tiling
screen is the one this step runs at; the checkpoint is a **licence** decision as much as
an accuracy one: the default is
trained partly on non-commercial data and the alternative is commercially clean and
cannot separate in-situ disease from invasive. The window size and the resolution are
deliberately absent, because they are properties of the checkpoint rather than of this
app; serving the model a different field of view is the same class of error as serving
it a different colour channel.
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Response, status

from app.pipeline.step07_tiling.index import TilingError
from app.pipeline.step08_tissue_type_segmentation.classes import ClassOrderError
from app.pipeline.step08_tissue_type_segmentation.inference import SegmentationError
from app.schemas.tissue_type import (
    TissueTypeCapability,
    TissueTypePanel,
    TissueTypeReport,
    TissueTypeRun,
)
from app.services.tissue_type_service import TissueTypeError, tissue_type_service
from app.services.upload_service import UploadError

router = APIRouter(prefix="/tissue-type", tags=["tissue type"])

# Every handler here is a plain `def`, not `async def`. All of them block:
# /capability imports torch and hashes nothing but reads manifests, the panel routes
# read files or redraw a picture, and /run resolves step 7's index behind it. Declared
# `async`, each would stall the event loop - and with it every other request - for as
# long as that took. A sync handler is dispatched to the threadpool instead.

_PNG = {200: {"content": {"image/png": {}}}}

#: A cached panel is immutable for a given upload and parameter set, but a re-run
#: writes new bytes under the same URL - so this is not `immutable`.
_PNG_HEADERS = {"Cache-Control": "public, max-age=3600"}

_OVERLAP = Query(
    default=None,
    ge=0.0,
    lt=1.0,
    description=(
        "Fraction of its own field of view a window shares with its neighbour. Defaults "
        "to step 7's overlap, which is the one parameter the two grids share. Overlap "
        "buys reliable window edges - a model has no context past one - at a cost in "
        "forward passes roughly linear in the window count."
    ),
)

_BRANCH = Query(
    default=None,
    description=(
        "The step 7 option this request believes is committed - `h_channel` or `he`. "
        "Checked against the record rather than applied: a disagreement is a 409, "
        "because running a grid other than the one step 7 priced is how the two steps "
        "come apart. Omit it to run whatever was committed."
    ),
)

_FOV = Query(
    default=None,
    gt=0.0,
    description=(
        "The field of view this request believes is committed, in microns. Checked, "
        "not applied - see `branch`."
    ),
)

_MODEL = Query(
    default=None,
    description=(
        "Which published checkpoint scores the slide. Defaults to the server setting. "
        "This is a licence decision as well as an accuracy one - see /capability, which "
        "reports each checkpoint's effective licence track."
    ),
)

_RESTART = Query(
    default=False,
    description=(
        "Discard the cached pass and run again. Without this, a request whose "
        "parameters match a finished pass returns that pass - which is what makes "
        "revisiting the screen instant instead of another half hour."
    ),
)

_PAINTED_SINCE = Query(
    default=None,
    ge=0,
    alias="paintedSince",
    description=(
        "Ask for the windows classified after this point in the run's paint log, so a "
        "screen can paint the pass onto the slide as it happens. The reply carries "
        "them as `paintedCells` and says how far it got in `paintedCursor`; pass that "
        "back next time. Omit it to poll the numbers alone - the feed is a whole "
        "slide's grid over the life of a pass, and a caller that does not ask should "
        "not carry any of it."
    ),
)

_CLASSES = Query(
    default=None,
    description=(
        "Comma-separated class ids to draw - `2` for the scored class alone on a trained "
        "option, `3` on the BEETLE one. Omit to draw all of them. The valid ids depend "
        "on which option ran: the two trained options draw four classes - the three the "
        "model emits plus `3 uncertain`, which the post-pass writes over in-situ calls "
        "it will not stand behind - and BEETLE draws its own five. Only `map` and "
        "`flat` honour this; `scored`, `confidence` and `uncertainty` are fixed views."
    ),
)


def _upload_failure(exc: UploadError) -> HTTPException:
    missing = "not found" in str(exc)
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND if missing else status.HTTP_409_CONFLICT,
        detail=str(exc),
    )


def _conflict(exc: Exception) -> HTTPException:
    """409, not 400: the request is fine, the server or the slide is not ready.

    "No checkpoint is published", "step 8 has not run yet" and "no window of this slide
    lands on tissue step 7 kept" are the cases this exists for. Nothing is wrong with
    the query in any of them, and a 400 would send the caller looking at their own
    parameters instead of at the step upstream.
    """
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))


def _parse_classes(raw: str | None, emitted: tuple[str, ...]) -> frozenset[int] | None:
    """`"0,2"` to a class filter, refusing anything that is not a class of this pass.

    Rejected at the boundary rather than ignored: a typo that silently drew every class
    would make a viewer believe they had switched one off and were watching the
    denominator change when they were not.

    **`emitted` is the pass's own class list, not a constant**, because step 8 no longer
    has one. The two trained options emit three classes and the BEETLE option emits its
    own five, so `4` is a valid filter on one slide and a refusal on another - and it
    has to be a refusal there, since the alternative is the silent no-op above.
    `tissue_type_service.emitted_classes` reads which pass ran.
    """
    if raw is None:
        return None

    names = ", ".join(f"{index} {name}" for index, name in enumerate(emitted))
    wanted: set[int] = set()
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            label = int(token)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"{token!r} is not a class id; this pass emits {names}",
            ) from exc
        if not 0 <= label < len(emitted):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"class {label} does not exist on this slide's pass, which emits "
                    f"{names}"
                ),
            )
        wanted.add(label)

    return frozenset(wanted)


@router.get(
    "/capability",
    response_model=TissueTypeCapability,
    summary="Can step 8 run here, and under what licence",
)
def capability() -> TissueTypeCapability:
    """Which checkpoints are published, which one would run, and what it leaves behind.

    Deliberately reports every checkpoint rather than only the selected one, because
    the difference between them in this project is not accuracy - it is whether the
    result may be sold.
    """
    return tissue_type_service.capability()


@router.post(
    "/{upload_id}/run",
    response_model=TissueTypeRun,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Step 8 - classify every patch of tissue",
)
def start(
    upload_id: str,
    background: BackgroundTasks,
    overlap: float | None = _OVERLAP,
    model: str | None = _MODEL,
    branch: str | None = _BRANCH,
    fov: float | None = _FOV,
    restart: bool = _RESTART,
) -> TissueTypeRun:
    """Queue a pass, or hand back the cached one when the parameters already match.

    `restart` is what makes "run it again" possible at all: without it this endpoint
    returns the cache whenever the parameters match, which is exactly right for
    revisiting the screen and exactly wrong when the viewer is asking for a fresh
    pass. It is a separate flag rather than a second endpoint because it is the same
    action with one difference - what happens to the previous answer.

    `branch` and `fov` are **assertions, not overrides**. This step runs on the choice
    step 7 committed; naming one here that disagrees is a 409 rather than a run on a
    different grid, because the caller and the viewer would then be looking at numbers
    from two different pipelines. They exist so a client can state what it believes it
    is asking for and be told when that belief is stale - which is what happens if the
    viewer went back and changed step 7 in another tab.
    """
    try:
        run = (
            tissue_type_service.restart(
                upload_id, overlap=overlap, name=model, branch=branch, fov=fov
            )
            if restart
            else tissue_type_service.start(
                upload_id, overlap=overlap, name=model, branch=branch, fov=fov
            )
        )
    except UploadError as exc:
        raise _upload_failure(exc) from exc
    except (TissueTypeError, TilingError, ClassOrderError) as exc:
        raise _conflict(exc) from exc

    if run.state == "queued":
        background.add_task(tissue_type_service.execute, upload_id)
    return run


@router.post(
    "/{upload_id}/cancel",
    response_model=TissueTypeRun,
    summary="Ask a running pass to stop",
)
def cancel(upload_id: str) -> TissueTypeRun:
    """Stop the pass, promptly, and report the state that leaves.

    Prompt because the flag is read by the progress callback, which runs after every
    block of 64 patches - about two seconds - rather than being polled once per
    slide. A cancel that took tens of minutes to land would be a cancel in name only.

    Cancelling nothing returns the current state rather than a conflict: a viewer who
    clicks as the last block finishes has done nothing wrong, and an error there
    reports the race rather than the outcome. A *finished* cached pass is never
    discarded by this.
    """
    try:
        return tissue_type_service.cancel(upload_id)
    except UploadError as exc:
        raise _upload_failure(exc) from exc
    except TissueTypeError as exc:
        raise _conflict(exc) from exc


@router.get(
    "/{upload_id}/run",
    response_model=TissueTypeRun,
    summary="How far the pass has got",
)
def progress(
    upload_id: str,
    painted_since: int | None = _PAINTED_SINCE,
) -> TissueTypeRun:
    """Live job state if one is running, else whatever is cached, else idle.

    With `paintedSince` it is also the paint feed: the windows classified since that
    point, each with the class the model gave it. That is what turns this step's screen
    from a bar into a picture of the pass - the model's answers landing on a thumbnail
    of the slide, in the order the sweep produced them - and it is the same data the
    finished class map is drawn from rather than an animation of it.
    """
    try:
        return tissue_type_service.state(upload_id, painted_since=painted_since)
    except UploadError as exc:
        raise _upload_failure(exc) from exc


@router.get(
    "/{upload_id}",
    response_model=TissueTypeReport,
    summary="The class map, what it is measured on, and what it cannot do",
)
def report(upload_id: str) -> TissueTypeReport:
    """The finished report, or a 409 naming the run that has not happened."""
    try:
        return tissue_type_service.report(upload_id)
    except UploadError as exc:
        raise _upload_failure(exc) from exc
    except TissueTypeError as exc:
        raise _conflict(exc) from exc


@router.get(
    "/{upload_id}/panels/{name}.png",
    summary="The class map on the slide, alone, gated to the scored class, or as certainty",
    response_class=Response,
    responses=_PNG,
)
def panel(
    upload_id: str,
    name: TissueTypePanel,
    classes: str | None = _CLASSES,
) -> Response:
    """One panel, redrawn when a class filter is asked for and served from disk if not."""
    wanted = _parse_classes(classes, tissue_type_service.emitted_classes(upload_id))
    try:
        png = tissue_type_service.panel(upload_id, name.value, classes=wanted)
    except UploadError as exc:
        raise _upload_failure(exc) from exc
    except (TissueTypeError, SegmentationError) as exc:
        raise _conflict(exc) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"cannot draw the class map: {exc}") from exc

    return Response(content=png, media_type="image/png", headers=_PNG_HEADERS)
