"""Resolve a case folder to its slides, and load one marker's H&E + IHC pair.

This is the demo's step 0: pick a biomarker, point at a case folder, and get
back the two upload ids step 1 onward already know how to work with. Separate
from `uploads.py`'s chunked protocol because these files already live on the
server's disk (`data/original/oncostem_slides/<CASE_ID>/`) - no bytes move.
"""

from fastapi import APIRouter, HTTPException, status

from app.schemas.case import (
    CaseLoadRequest,
    CaseLoadResponse,
    CaseResolveRequest,
    CaseResolveResponse,
)
from app.services import case_service
from app.services.case_service import CaseError

router = APIRouter(prefix="/cases", tags=["cases"])


def _fail(exc: CaseError) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.post(
    "/resolve",
    response_model=CaseResolveResponse,
    summary="Find the H&E and IHC slides in a case folder",
)
async def resolve(body: CaseResolveRequest) -> CaseResolveResponse:
    try:
        resolution = case_service.resolve_case(body.case_path)
    except CaseError as exc:
        raise _fail(exc) from exc
    return CaseResolveResponse(
        case_id=resolution.case_id,
        case_path=resolution.case_path,
        found=resolution.found,
        missing=resolution.missing,
    )


@router.post(
    "/load",
    response_model=CaseLoadResponse,
    summary="Register the case's H&E slide and one marker's IHC slide",
)
async def load(body: CaseLoadRequest) -> CaseLoadResponse:
    try:
        session = case_service.load_case(body.case_path, body.marker)
    except CaseError as exc:
        raise _fail(exc) from exc
    return CaseLoadResponse(
        case_id=session.case_id,
        marker=session.marker,
        case_path=session.case_path,
        he_upload_id=session.he_upload_id,
        ihc_upload_id=session.ihc_upload_id,
    )
