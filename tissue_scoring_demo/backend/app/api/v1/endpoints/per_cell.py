"""Step 14 endpoints - one measurement row per tumour cell.

    GET /per-cell/{he}?ihcUploadId=...          the summary, histogram and scatter
    GET /per-cell/{he}/cells?ihcUploadId=...    every row, for an export
    GET /per-cell/{he}/cells/{id}.png           the crop behind one dot

**Nothing here is a parameter.** Which compartment, which second number and
which cut points all resolve from the antibody letter the case was loaded with,
on the server. A `?compartment=` would be a way to measure a cytoplasmic marker
in a ring from a URL, and the cleanest way to prevent that is for it not to
exist.
"""

from fastapi import APIRouter, HTTPException, Query, Response, status
from fastapi.responses import FileResponse

from app.schemas.per_cell import PerCellReport
from app.services.nuclei_service import nuclei_service
from app.services.per_cell_service import PerCellError, per_cell_service

router = APIRouter(prefix="/per-cell", tags=["per-cell"])


def _failure(exc: PerCellError) -> HTTPException:
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))


@router.get(
    "/{he_upload_id}",
    response_model=PerCellReport,
    summary="Measure every tumour cell's compartment",
)
def report(
    he_upload_id: str,
    ihc_upload_id: str = Query(alias="ihcUploadId"),
) -> PerCellReport:
    try:
        return per_cell_service.report(he_upload_id, ihc_upload_id)
    except PerCellError as exc:
        raise _failure(exc) from exc


@router.get("/{he_upload_id}/cells", summary="Every measured cell, unsampled")
def cells(
    he_upload_id: str,
    ihc_upload_id: str = Query(alias="ihcUploadId"),
) -> dict:
    """The full rows rather than the scatter's sample.

    Separate from the report because the report is fetched every time the screen
    opens and these can run to tens of thousands of rows. This is for an export
    or a check, not for drawing.
    """
    try:
        rows = per_cell_service.rows(he_upload_id, ihc_upload_id)
    except PerCellError as exc:
        raise _failure(exc) from exc
    return {"cells": rows, "total": len(rows)}


@router.get(
    "/{he_upload_id}/fields/{rank}/{index}.png",
    responses={200: {"content": {"image/png": {}}}},
    response_class=Response,
    summary="The field one dot's cell was measured in",
)
def field_image(
    he_upload_id: str,
    rank: int,
    index: int,
    ihc_upload_id: str = Query(alias="ihcUploadId"),
) -> Response:
    """Step 11's stored field, as it really looks.

    Served from step 11's artefacts rather than re-cropped here: it is the same
    field, at the same scale, and a second crop would be a second chance for the
    picture beside a number to be of somewhere else.
    """
    path = nuclei_service.artifact(
        he_upload_id, ihc_upload_id, f"region{rank}", f"f{index}_raw.png"
    )
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no stored field {index} in region {rank} for this pair",
        )
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-cache"})
