"""Step 9 endpoints - the ROI mask.

    POST /roi/{id}/build          build the region (or return the cached one)
    POST /roi/{id}/build?rebuild  throw the cached one away and build again
    GET  /roi/{id}                the finished report, with the polygons
    GET  /roi/{id}/panels/{name}.png   seed | smoothed | binary | region | outline |
                                       borders | borders_on_slide
    GET  /roi/{id}/top/{class}/{rank}.png  the rank-th largest dcis or invasive crop
    GET  /roi/{id}/qupath.geojson  every border-class region, as a QuPath import file

No run/poll pair, unlike step 8. The region takes about a tenth of a second; a cold
build is roughly twenty, almost all of it drawing the panels, and a cached read is
immediate. Twenty seconds is a slow request and still not a job - there is no model to
wait on, and a client polling for it would learn nothing a single call did not.

**Building and reading are separate verbs on purpose.** A GET that quietly built a
region would let a client change the denominator every later step measures in by
looking at it, and would make "what is the ROI" and "make me an ROI" the same request.

The parameters are all query arguments on `build` because every one of them changes
the region, and a region is a claim: two callers passing different closing radii must
not both be told they are looking at the ROI. `build` compares the parameters against
the cached report and rebuilds when they differ rather than serving the cache.

`protectInSitu` is the one worth reading the description of. It is not a smoothing
preference - it is Rule 5 defended against the closing radius sitting next to it.
"""

from fastapi import APIRouter, HTTPException, Query, Response, status

from app.pipeline.step09_roi_mask.mask import RoiError
from app.schemas.roi import RoiBorderClass, RoiPanel, RoiReport, RoiTopClass
from app.services.roi_service import RoiServiceError, roi_service
from app.services.tissue_type_service import TissueTypeError
from app.services.upload_service import UploadError

router = APIRouter(prefix="/roi", tags=["roi mask"])

# Plain `def`, not `async def`: every handler here blocks on numpy, scipy or a file
# read. Declared async, each would stall the event loop for as long as that took.

_PNG = {200: {"content": {"image/png": {}}}}

#: A cached panel is immutable for a given upload and parameter set, but a rebuild
#: writes new bytes under the same URL - so this is not `immutable`.
_PNG_HEADERS = {"Cache-Control": "public, max-age=3600"}

_SIGMA = Query(
    default=None,
    ge=0.0,
    le=16.0,
    description=(
        "Gaussian blur applied to P(invasive) before thresholding, in grid cells. "
        "Smoothing runs on probabilities rather than on labels because averaging "
        "argmaxes is a vote, and a vote has already thrown away the margin."
    ),
)

_THRESHOLD = Query(
    default=None,
    ge=0.0,
    le=1.0,
    description="Where the smoothed invasive probability becomes region.",
)

_CLOSE = Query(
    default=None,
    ge=0,
    le=32,
    description=(
        "Closing radius in grid cells - how far apart two patches of tumour may be and "
        "still be called one focus. Larger merges more and invents more: the area it "
        "adds is reported separately in the ledger for that reason."
    ),
)

_PROTECT = Query(
    default=None,
    ge=0.0,
    le=1.0,
    description=(
        "In-situ probability at which a window is carved back out of the closed region. "
        "Rule 5: in-situ carcinoma is not scored, and a closing radius large enough to "
        "merge neighbouring tumour is large enough to swallow the in-situ ducts between "
        "them - which is the in-situ that matters, because it is the in-situ next to "
        "invasive disease. Set 0 to disable and get the plain recipe."
    ),
)

_MIN_AREA = Query(
    default=None,
    ge=0.0,
    le=100.0,
    description=(
        "Components below this many mm2 are dropped as speckle. In mm2 rather than "
        "pixels so the cutoff survives a change of scanner."
    ),
)

_KEEP_LARGEST = Query(
    default=None,
    ge=1,
    le=64,
    description=(
        "Keep only the N largest foci, matching how a pathologist circles one or two. "
        "Omit to keep every component that cleared the area cutoff."
    ),
)


def _handle(error: Exception) -> HTTPException:
    """Map a service error to a status a client can act on."""
    if isinstance(error, UploadError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error))
    if isinstance(error, TissueTypeError):
        # Step 8 has not run. A precondition, not a bad request.
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error))
    if isinstance(error, RoiError):
        return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error))
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error))


@router.post("/{upload_id}/build", response_model=RoiReport)
def build_roi(
    upload_id: str,
    rebuild: bool = Query(default=False, description="Discard the cached region and build again"),
    sigma: float | None = _SIGMA,
    threshold: float | None = _THRESHOLD,
    close_cells: int | None = _CLOSE,
    protect_in_situ: float | None = _PROTECT,
    min_area_mm2: float | None = _MIN_AREA,
    keep_largest: int | None = _KEEP_LARGEST,
) -> RoiReport:
    """Build the invasive-tumour region this slide will be scored inside."""
    try:
        params = roi_service.resolve_params(
            sigma=sigma,
            threshold=threshold,
            close_cells=close_cells,
            protect_in_situ=protect_in_situ,
            min_area_mm2=min_area_mm2,
            keep_largest=keep_largest,
        )
        return roi_service.build(upload_id, params, rebuild=rebuild)
    except (RoiError, RoiServiceError, TissueTypeError, UploadError) as error:
        raise _handle(error) from error


@router.get("/{upload_id}", response_model=RoiReport)
def get_roi(upload_id: str) -> RoiReport:
    """The cached region. 404 until it has been built."""
    try:
        return roi_service.report(upload_id)
    except (RoiServiceError, UploadError) as error:
        raise _handle(error) from error


@router.get("/{upload_id}/panels/{name}.png", responses=_PNG, response_class=Response)
def get_panel(upload_id: str, name: RoiPanel) -> Response:
    """One of the panels the step is read as."""
    try:
        return Response(
            content=roi_service.panel(upload_id, name.value),
            media_type="image/png",
            headers=_PNG_HEADERS,
        )
    except (RoiServiceError, UploadError) as error:
        raise _handle(error) from error


@router.get("/{upload_id}/top/{class_name}/{rank}.png", responses=_PNG, response_class=Response)
def get_top_crop(upload_id: str, class_name: RoiTopClass, rank: int) -> Response:
    """The `rank`-th largest DCIS or invasive region, cropped from the slide itself.

    `rank` is 1-based (1 = largest) because it is what the URL a viewer builds reads
    as, not an index into a list they never see.
    """
    try:
        return Response(
            content=roi_service.top_crop(upload_id, class_name.value, rank),
            media_type="image/png",
            headers=_PNG_HEADERS,
        )
    except (RoiServiceError, UploadError) as error:
        raise _handle(error) from error


@router.get("/{upload_id}/region/{class_name}/{index}.png", responses=_PNG, response_class=Response)
def get_region_crop(upload_id: str, class_name: RoiBorderClass, index: int) -> Response:
    """One region a viewer selected off the borders panel, enlarged from the slide.

    `index` is 0-based, matching the `index` already on every region in the report's
    `classRegions[className]` - the region a viewer just clicked, not a fresh rank.
    Unlike `top_crop`, this is not limited to DCIS and invasive's largest three:
    `class_name` also takes `uncertain`, and any index the report actually listed.
    """
    try:
        return Response(
            content=roi_service.region_crop(upload_id, class_name.value, index),
            media_type="image/png",
            headers=_PNG_HEADERS,
        )
    except (RoiServiceError, UploadError, TissueTypeError) as error:
        raise _handle(error) from error


@router.get("/{upload_id}/qupath.geojson")
def get_qupath(upload_id: str) -> Response:
    """Every invasive/DCIS/uncertain region as one QuPath-importable GeoJSON file."""
    try:
        return Response(
            content=roi_service.qupath(upload_id),
            media_type="application/geo+json",
            headers={
                "Content-Disposition": f'attachment; filename="{upload_id}_qupath.geojson"'
            },
        )
    except (RoiServiceError, UploadError) as error:
        raise _handle(error) from error
