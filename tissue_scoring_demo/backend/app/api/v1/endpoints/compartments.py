"""Step 13 endpoints - where in each cell the brown is supposed to be.

    GET /compartments/{he}                      the geometry at the marker's own width
    GET /compartments/{he}?widthUm=5            the same at another width
    GET /compartments/{he}/regions/{rank}/geometry         one region's outlines
    GET /compartments/{he}/regions/{rank}/fields/{i}.png   one field, drawn

**The compartment kind is not a parameter.** Which shape a marker gets - a
membrane ring or a cytoplasm band - is read from `app.panel` by the antibody
letter and cannot be overridden from a request. Putting a cytoplasmic marker
through the membrane path is the regression the guide names explicitly, and the
cleanest way to prevent it is to make it unreachable rather than to validate
against it.

The *width* is a parameter, because it is a genuine unknown: neither default is
established fact, and the screen sweeps it.
"""

from fastapi import APIRouter, HTTPException, Query, Response, status
from fastapi.responses import FileResponse

from app.schemas.compartments import CompartmentsReport
from app.services.compartment_service import CompartmentError, compartment_service

router = APIRouter(prefix="/compartments", tags=["compartments"])


def _failure(exc: CompartmentError) -> HTTPException:
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))


@router.get("/{he_upload_id}", response_model=CompartmentsReport, summary="The compartments")
def report(
    he_upload_id: str,
    ihc_upload_id: str = Query(alias="ihcUploadId"),
    width_um: float | None = Query(default=None, alias="widthUm", gt=0.0, le=25.0),
    voronoi: bool = Query(default=True),
    tumour_only: bool = Query(default=True, alias="tumourOnly"),
) -> CompartmentsReport:
    """`voronoi=false` is for the demo only - it shows what neighbours colliding costs."""
    try:
        return compartment_service.report(
            he_upload_id,
            ihc_upload_id,
            width_um=width_um,
            voronoi=voronoi,
            tumour_only=tumour_only,
        )
    except CompartmentError as exc:
        raise _failure(exc) from exc


@router.get(
    "/{he_upload_id}/regions/{rank}/geometry",
    summary="One region's compartments, as outlines in slide coordinates",
)
def region_geometry(
    he_upload_id: str, rank: int, ihc_upload_id: str = Query(alias="ihcUploadId")
) -> Response:
    """Served as the stored file, like step 11's nuclei and for the same reasons.

    It is a few megabytes of vertices that this process wrote, so re-validating
    it through a response model on every fetch would cost more than sending it
    and could not catch anything. Kept out of the report because the report is
    fetched every time the screen opens and on every drag of the width slider.

    `no-cache` rather than the hour the images get: the width is adjustable, so
    the outlines for a region change while the reader is looking at them.
    """
    path = compartment_service.artifact(
        he_upload_id, ihc_upload_id, f"region{rank}", "rings.json"
    )
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"region {rank} has no stored compartment outlines for this pair. "
                "Ask for the report first - building it is what writes them."
            ),
        )
    return FileResponse(
        path, media_type="application/json", headers={"Cache-Control": "no-cache"}
    )


@router.get(
    "/{he_upload_id}/regions/{rank}/fields/{index}.png",
    responses={200: {"content": {"image/png": {}}}},
    response_class=Response,
    summary="One field's compartments, drawn",
)
def field_image(
    he_upload_id: str,
    rank: int,
    index: int,
    ihc_upload_id: str = Query(alias="ihcUploadId"),
) -> Response:
    path = compartment_service.artifact(
        he_upload_id, ihc_upload_id, f"region{rank}", f"f{index}_compartments.png"
    )
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no compartment image for field {index} of region {rank}",
        )
    return FileResponse(
        path, media_type="image/png", headers={"Cache-Control": "no-cache"}
    )
