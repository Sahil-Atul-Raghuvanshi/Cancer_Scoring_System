"""Step 15 endpoints - cells sorted into levels, with the cuts shown on screen.

    GET /binning/{he}?ihcUploadId=...         the bins, the histogram, the cut lines
    GET /binning/{he}/cells?ihcUploadId=...   one bin per cell, for the slide overlay

**The cut points are not a query parameter.** The width at step 13 is one,
because neither default is established fact and sweeping it is the point. A cut
point is different: it is the thing being calibrated, and a `?odCuts=` would be
a way to reach a reported number by choosing the threshold that produces it.
They are read from `backend/config/marker_cuts.v1.json` by the antibody letter.
"""

from fastapi import APIRouter, HTTPException, Query, status

from app.schemas.binning import BinningReport
from app.services.binning_service import BinningError, binning_service

router = APIRouter(prefix="/binning", tags=["binning"])


@router.get(
    "/{he_upload_id}",
    response_model=BinningReport,
    summary="Bin every measured cell against this antibody's cuts",
)
def report(
    he_upload_id: str,
    ihc_upload_id: str = Query(alias="ihcUploadId"),
) -> BinningReport:
    try:
        return binning_service.report(he_upload_id, ihc_upload_id)
    except BinningError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc


@router.get(
    "/{he_upload_id}/cells",
    summary="One bin per cell, for colouring them on the slide",
)
def cells(
    he_upload_id: str,
    ihc_upload_id: str = Query(alias="ihcUploadId"),
) -> dict:
    """Separate from the report because the report is fetched on every visit.

    These run to tens of thousands of rows and are only wanted by the screen that
    draws the cells. The bin and the positivity verdict are computed on this side
    so that the colours on the slide and the percentage in the table come from one
    application of the rule - see `BinningService.cells`.
    """
    try:
        rows = binning_service.cells(he_upload_id, ihc_upload_id)
    except BinningError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    return {"cells": rows, "total": len(rows)}
