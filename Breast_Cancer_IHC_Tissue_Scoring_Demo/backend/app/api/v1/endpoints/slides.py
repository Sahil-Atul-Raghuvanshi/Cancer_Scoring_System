"""Slide endpoints: upload policy, and step 1 against a real uploaded scan."""

from fastapi import APIRouter, HTTPException, Query, Response, status

from app.api.deps import SlideServiceDep
from app.pipeline.step01_read_slide.pipeline import slide_reader_service
from app.schemas.slide import SlideReadout, UploadCapability
from app.services.tile_service import tile_service
from app.services.upload_service import UploadError

router = APIRouter(prefix="/slides", tags=["slides"])


def _fail(exc: UploadError) -> HTTPException:
    is_missing = "not found" in str(exc)
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND if is_missing else status.HTTP_409_CONFLICT,
        detail=str(exc),
    )


@router.get(
    "/upload-capability",
    response_model=UploadCapability,
    summary="Whether slide upload is accepted, and the limits that apply",
)
async def get_upload_capability(service: SlideServiceDep) -> UploadCapability:
    """Report upload availability so the UI can configure its drop zone."""
    return service.upload_capability()


@router.get(
    "/{upload_id}/readout",
    response_model=SlideReadout,
    summary="Step 1 - read the slide",
)
async def get_slide_readout(
    upload_id: str,
    target_mpp: float | None = Query(
        default=None,
        gt=0,
        le=64,
        alias="targetMpp",
        description="Resolution the pipeline should work at. Defaults to the server setting.",
    ),
    mpp_override: float | None = Query(
        default=None,
        gt=0,
        le=64,
        alias="mppOverride",
        description=(
            "Supply the slide's own microns-per-pixel, for files that record none or "
            "record it wrongly. Reported back as mppSource='override' - it is an "
            "assertion by the caller, not a measurement."
        ),
    ),
) -> SlideReadout:
    """Open an uploaded slide and report its pyramid, resolution and working level.

    This is pipeline step 1. The value that matters is `mpp`: every later step
    is defined in microns per pixel, and `workingLevel` is derived from it
    rather than hard-coded, because the same level index is a different
    resolution on a different scanner.
    """
    try:
        return slide_reader_service.readout(
            upload_id, target_mpp=target_mpp, mpp_override=mpp_override
        )
    except UploadError as exc:
        raise _fail(exc) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"cannot read slide: {exc}") from exc


@router.get(
    "/{upload_id}/thumbnail",
    summary="Whole-slide overview as PNG",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}, "description": "Slide overview"}},
)
async def get_slide_thumbnail(
    upload_id: str,
    max_size: int = Query(default=1024, ge=64, le=4096, alias="maxSize"),
) -> Response:
    """Render an overview from the pyramid.

    Never served from the file's associated images: on scanner output those can
    be a half-resolution copy of the whole slide, or a photograph of the label
    carrying case identifiers.
    """
    try:
        png = slide_reader_service.thumbnail_png(upload_id, max_size=max_size)
    except UploadError as exc:
        raise _fail(exc) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"cannot read slide: {exc}") from exc

    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get(
    "/{upload_id}/region",
    summary="One region of the slide as PNG",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}, "description": "Slide region"}},
)
async def get_slide_region(
    upload_id: str,
    x: int = Query(default=0, ge=0, description="Left edge in level-0 coordinates"),
    y: int = Query(default=0, ge=0, description="Top edge in level-0 coordinates"),
    level: int = Query(default=0, ge=0, description="Pyramid level to read from"),
    width: int = Query(default=512, ge=1, le=2048),
    height: int = Query(default=512, ge=1, le=2048),
) -> Response:
    """Read a single region, the way the tiler will once later steps are wired up."""
    try:
        png = slide_reader_service.region_png(
            upload_id, x=x, y=y, level=level, width=width, height=height
        )
    except UploadError as exc:
        raise _fail(exc) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"cannot read region: {exc}") from exc

    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


# --- Deep Zoom -------------------------------------------------------------
#
# These two routes are what make a gigapixel slide pannable: the viewer only
# ever fetches the tiles currently on screen. Their paths follow the DZI
# convention OpenSeadragon expects - `<name>.dzi` for the descriptor and
# `<name>_files/<level>/<col>_<row>.jpeg` for tiles - so they are declared last,
# after the `/{upload_id}/...` routes, to keep the matching unambiguous.


@router.get(
    "/{upload_id}.dzi",
    summary="Deep Zoom descriptor",
    response_class=Response,
    responses={200: {"content": {"application/xml": {}}, "description": "DZI descriptor"}},
)
async def get_slide_dzi(upload_id: str) -> Response:
    """The image descriptor a Deep Zoom viewer reads before requesting tiles."""
    try:
        xml = tile_service.descriptor(upload_id)
    except UploadError as exc:
        raise _fail(exc) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"cannot read slide: {exc}") from exc

    return Response(content=xml, media_type="application/xml")


@router.get(
    "/{upload_id}_files/{level}/{col}_{row}.jpeg",
    summary="One Deep Zoom tile",
    response_class=Response,
    responses={200: {"content": {"image/jpeg": {}}, "description": "Slide tile"}},
)
async def get_slide_tile(upload_id: str, level: int, col: int, row: int) -> Response:
    """Render a single tile on the fly from the nearest pyramid level."""
    try:
        jpeg = tile_service.tile_jpeg(upload_id, level=level, col=col, row=row)
    except UploadError as exc:
        raise _fail(exc) from exc
    except ValueError as exc:
        # Out-of-range tiles are normal at the edges of a viewport, not errors.
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=422, detail=f"cannot read tile: {exc}") from exc

    return Response(
        content=jpeg,
        media_type="image/jpeg",
        # Tiles are immutable for a given upload id, so they cache hard.
        headers={"Cache-Control": "public, max-age=86400, immutable"},
    )
