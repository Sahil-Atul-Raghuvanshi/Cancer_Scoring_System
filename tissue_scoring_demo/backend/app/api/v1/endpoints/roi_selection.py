"""Step 10 endpoints - reviewing and selecting the candidate regions.

    POST /roi-selection/{id}/build           build the candidate list and its cards
    POST /roi-selection/{id}/build?rebuild   redraw them from a fresh class map
    GET  /roi-selection/{id}                 the candidates and what is ticked
    PUT  /roi-selection/{id}/selection       replace the ticked set
    GET  /roi-selection/{id}/cards/{roiId}.png   one candidate's picture

No run/poll pair, unlike steps 8 and 11: there is no model here. A cold build is one
padded slide read per candidate, which is seconds, and a cached read is immediate.

**The selection is a `PUT` and it replaces rather than merges.** That is what makes
"deselect all" expressible at all - a merging `PATCH` has no way to say "none of them" -
and it means the client and the server never hold two halves of one answer. It is also
why unknown ids are a 422 rather than being dropped: a caller who sent a typo and got a
200 would believe it had ticked a region it had not, and would find out only as a
missing result at the end of step 11.
"""

from fastapi import APIRouter, HTTPException, Query, Response, status

from app.schemas.roi_selection import (
    RoiSelectionReport,
    RoiSelectionRequest,
)
from app.services.roi_selection_service import RoiSelectionError, roi_selection_service
from app.services.roi_service import RoiServiceError
from app.services.tissue_type_service import TissueTypeError
from app.services.upload_service import UploadError

router = APIRouter(prefix="/roi-selection", tags=["roi selection"])

# Plain `def`, not `async def`: every handler here blocks on a slide read, numpy or a
# file. Declared async, each would stall the event loop for as long as that took.

_PNG = {200: {"content": {"image/png": {}}}}

#: A card is immutable for a given upload and class map, but a rebuild writes new bytes
#: under the same URL - so this is not `immutable`.
_PNG_HEADERS = {"Cache-Control": "public, max-age=3600"}

_REBUILD = Query(
    default=False,
    description=(
        "Redraw the candidates even when a cached list matches the class map on disk. "
        "Rarely needed: the cache is keyed on the class map, so a re-run of step 8 "
        "already invalidates it."
    ),
)


def _upload_failure(exc: UploadError) -> HTTPException:
    missing = "not found" in str(exc)
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND if missing else status.HTTP_409_CONFLICT,
        detail=str(exc),
    )


def _conflict(exc: Exception) -> HTTPException:
    """409: the request is fine, the slide is not ready.

    "Step 8 has not run" and "step 9 has not built a region" are what this is for.
    Nothing is wrong with the query in either case, and a 400 would send the caller
    looking at their own parameters instead of at the step upstream.
    """
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))


@router.post(
    "/{upload_id}/build",
    response_model=RoiSelectionReport,
    summary="Step 10 - list the candidate invasive regions",
)
def build(upload_id: str, rebuild: bool = _REBUILD) -> RoiSelectionReport:
    """Offer step 9's invasive patches as a list somebody can tick.

    Returns the cached list when it was built against the class map currently on disk.
    A selection already recorded survives this: the candidates and the selection are
    stored apart precisely so redrawing the cards does not throw away somebody's answer.
    """
    try:
        return roi_selection_service.build(upload_id, rebuild=rebuild)
    except UploadError as exc:
        raise _upload_failure(exc) from exc
    except (RoiSelectionError, RoiServiceError, TissueTypeError) as exc:
        raise _conflict(exc) from exc


@router.get(
    "/{upload_id}",
    response_model=RoiSelectionReport,
    summary="The candidates, and which of them are ticked",
)
def report(upload_id: str) -> RoiSelectionReport:
    """The cached list. Raises rather than quietly building one.

    Reading and building are separate verbs for step 9's reason one step on: a GET that
    built the candidate list would make "what is on offer" and "work out what is on
    offer" the same request, and the second one opens the slide twenty times.
    """
    try:
        return roi_selection_service.report(upload_id)
    except UploadError as exc:
        raise _upload_failure(exc) from exc
    except (RoiSelectionError, TissueTypeError) as exc:
        raise _conflict(exc) from exc


@router.put(
    "/{upload_id}/selection",
    response_model=RoiSelectionReport,
    summary="Choose which regions go to BEETLE",
)
def select(upload_id: str, request: RoiSelectionRequest) -> RoiSelectionReport:
    """Replace the ticked set outright, and report what that now commits to.

    The whole report comes back rather than an acknowledgement, because ticking a region
    changes the numbers underneath the list - the selected area, the share of the tumour
    it covers, the windows it costs - and a client that had to re-fetch to learn them
    would render a stale cost beside a fresh selection.
    """
    try:
        return roi_selection_service.select(upload_id, request.selected)
    except UploadError as exc:
        raise _upload_failure(exc) from exc
    except RoiSelectionError as exc:
        # 422, not 409: an unknown id *is* a problem with the request, and the caller
        # can fix it by sending ids that exist.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except TissueTypeError as exc:
        raise _conflict(exc) from exc


@router.get(
    "/{upload_id}/cards/{roi_id}.png",
    responses=_PNG,
    response_class=Response,
    summary="One candidate's picture: the crop with its tile boundary on it",
)
def card(upload_id: str, roi_id: str) -> Response:
    """The slide's own pixels around one candidate, with step 8's staircase drawn on.

    The boundary is drawn as the staircase it is rather than smoothed - see
    `step10_roi_selection/overlay.py`. A curve here would claim a precision the tile
    model does not have and would make step 11 look cosmetic.
    """
    try:
        payload = roi_selection_service.card(upload_id, roi_id)
    except UploadError as exc:
        raise _upload_failure(exc) from exc
    except RoiSelectionError as exc:
        raise _conflict(exc) from exc
    return Response(content=payload, media_type="image/png", headers=_PNG_HEADERS)
