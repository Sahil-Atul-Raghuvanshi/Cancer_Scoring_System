"""Chunked slide-upload endpoints.

A resumable 3-call protocol - init / chunk / complete - plus status and abort.
`complete` schedules a background job that reassembles and validates the file;
clients poll `GET /uploads/{id}` until the state is `ready` or `failed`.

A whole-slide image is several gigabytes, which is why the transfer is chunked
and the reassembly happens off the request path.
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status

from app.core.config import settings
from app.schemas.upload import UploadInit, UploadStatus
from app.services import upload_service
from app.services.upload_service import UploadError

router = APIRouter(prefix="/uploads", tags=["uploads"])


def _fail(exc: UploadError) -> HTTPException:
    """Missing rows are 404; everything else is the client's to correct."""
    is_missing = "not found" in str(exc)
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND if is_missing else status.HTTP_400_BAD_REQUEST,
        detail=str(exc),
    )


@router.post(
    "",
    response_model=UploadStatus,
    status_code=status.HTTP_201_CREATED,
    summary="Open an upload",
)
async def init(body: UploadInit) -> UploadStatus:
    """Reserve an upload id, validating extension and size before any bytes move."""
    try:
        result = upload_service.init_upload(
            filename=body.filename,
            total_size=body.total_size,
            num_chunks=body.num_chunks,
            chunk_size=body.chunk_size,
            sha256=body.sha256,
        )
    except UploadError as exc:
        raise _fail(exc) from exc
    return UploadStatus(**result)


@router.put(
    "/{upload_id}/chunks/{index}",
    response_model=UploadStatus,
    summary="Upload one chunk",
)
async def put_chunk(upload_id: str, index: int, request: Request) -> UploadStatus:
    """Store one part. The body is the raw bytes, not multipart.

    Re-sending an index is safe: the write is atomic and idempotent, so a client
    can retry a failed chunk without restarting the transfer.
    """
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="empty chunk body")

    try:
        result = upload_service.save_chunk(upload_id=upload_id, index=index, data=data)
    except UploadError as exc:
        raise _fail(exc) from exc
    return UploadStatus(**result)


@router.get("/{upload_id}", response_model=UploadStatus, summary="Upload status")
async def get_status(upload_id: str) -> UploadStatus:
    """Poll after `complete`, and read `received` to resume an interrupted transfer."""
    try:
        return UploadStatus(**upload_service.upload_status(upload_id=upload_id))
    except UploadError as exc:
        raise _fail(exc) from exc


@router.post(
    "/{upload_id}/complete",
    response_model=UploadStatus,
    summary="Finish an upload and start reassembly",
)
async def complete(upload_id: str, background: BackgroundTasks) -> UploadStatus:
    """Check every chunk arrived, then reassemble and validate in the background."""
    try:
        result = upload_service.complete_upload(upload_id=upload_id)
    except UploadError as exc:
        raise _fail(exc) from exc

    if result["state"] == "processing":
        background.add_task(upload_service.finalize_upload, upload_id)

    return UploadStatus(**result)


@router.delete("/{upload_id}", response_model=UploadStatus, summary="Abort an upload")
async def abort(upload_id: str) -> UploadStatus:
    """Discard the staged chunks for an upload that is no longer wanted."""
    try:
        return UploadStatus(**upload_service.abort_upload(upload_id=upload_id))
    except UploadError as exc:
        raise _fail(exc) from exc


@router.get("/config/limits", summary="Upload limits the client should honour")
async def limits() -> dict:
    """Chunk size and ceilings, so the client does not hard-code them."""
    return {
        "chunkSize": settings.upload_chunk_size,
        "maxFileSizeBytes": settings.max_upload_bytes,
        "acceptedFormats": sorted(settings.allowed_slide_ext),
    }
