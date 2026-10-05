"""Step 12 endpoints - which cells belong in the denominator.

    GET  /cell-typing/{he}                  the mix at the shipped thresholds
    GET  /cell-typing/{he}?lymphocyteMaxAreaUm2=30&...   the mix at other ones
    GET  /cell-typing/{he}/regions/{rank}   a class per nucleus id, for colouring

**No run, no poll.** Unlike steps 8, 10 and 11 this one opens no slide and runs
no model - it applies five thresholds to numbers step 11 already stored, which
is well under a second. That is what lets the thresholds be live: the screen
puts them on sliders and re-asks, which is the only honest way to present
parameters that were reasoned rather than fitted.
"""

from fastapi import APIRouter, HTTPException, Query, Response, status
from fastapi.responses import FileResponse

from app.schemas.cell_typing import CellTypingReport, TypingRules
from app.services.cell_typing_service import CellTypingError, cell_typing_service

router = APIRouter(prefix="/cell-typing", tags=["cell typing"])


def _failure(exc: CellTypingError) -> HTTPException:
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))


@router.get("/{he_upload_id}", response_model=CellTypingReport, summary="The cell mix")
def report(
    he_upload_id: str,
    ihc_upload_id: str = Query(alias="ihcUploadId"),
    lymphocyte_max_area_um2: float | None = Query(default=None, alias="lymphocyteMaxAreaUm2"),
    lymphocyte_min_circularity: float | None = Query(
        default=None, alias="lymphocyteMinCircularity"
    ),
    lymphocyte_min_darkness: float | None = Query(default=None, alias="lymphocyteMinDarkness"),
    spindle_min_eccentricity: float | None = Query(
        default=None, alias="spindleMinEccentricity"
    ),
    spindle_max_area_um2: float | None = Query(default=None, alias="spindleMaxAreaUm2"),
) -> CellTypingReport:
    """The mix, and the sensitivity of that mix to each threshold.

    Every threshold is optional and overrides one field of the default. Sent as
    query parameters rather than a POST body because this is a *read* - it
    computes nothing that was not already implied by step 11's output, and a
    viewer dragging a slider should be able to reload the page and see the same
    answer.
    """
    defaults = TypingRules()
    rules = TypingRules(
        lymphocyte_max_area_um2=lymphocyte_max_area_um2
        if lymphocyte_max_area_um2 is not None
        else defaults.lymphocyte_max_area_um2,
        lymphocyte_min_circularity=lymphocyte_min_circularity
        if lymphocyte_min_circularity is not None
        else defaults.lymphocyte_min_circularity,
        lymphocyte_min_darkness=lymphocyte_min_darkness
        if lymphocyte_min_darkness is not None
        else defaults.lymphocyte_min_darkness,
        spindle_min_eccentricity=spindle_min_eccentricity
        if spindle_min_eccentricity is not None
        else defaults.spindle_min_eccentricity,
        spindle_max_area_um2=spindle_max_area_um2
        if spindle_max_area_um2 is not None
        else defaults.spindle_max_area_um2,
    )

    try:
        return cell_typing_service.report(he_upload_id, ihc_upload_id, rules=rules)
    except CellTypingError as exc:
        raise _failure(exc) from exc


@router.get(
    "/{he_upload_id}/regions/{rank}",
    summary="A class per nucleus id, for the overlay to colour by",
)
def region(
    he_upload_id: str, rank: int, ihc_upload_id: str = Query(alias="ihcUploadId")
) -> Response:
    """Served as the stored file: it is a flat id-to-class map and nothing more.

    Kept separate from the report for the same reason step 11 keeps its
    geometry separate - the summary is fetched constantly and this is only
    needed for the region a viewer has open.
    """
    path = cell_typing_service.artifact(
        he_upload_id, ihc_upload_id, f"region{rank}", "types.json"
    )
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"region {rank} has not been typed for this pair",
        )
    return FileResponse(
        path, media_type="application/json", headers={"Cache-Control": "no-cache"}
    )
