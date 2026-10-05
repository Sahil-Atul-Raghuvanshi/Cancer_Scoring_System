"""Step 16 endpoints - the deliverable.

    GET /scores/{he}?ihcUploadId=...   one marker's pair, and the arithmetic
    GET /scores/case/{caseId}          the case's ten numbers, as the grid sent

The case endpoint gathers whichever markers have been scored rather than
refusing until all five are. A partial grid with four markers filled and one
saying why it is missing is more useful than a 409, and it is also the honest
picture of a run in progress.
"""

from fastapi import APIRouter, HTTPException, Query, status

from app.schemas.scores import CaseScoreReport, ScoreReport
from app.services.case_score_service import case_score_service
from app.services.score_service import ScoreError, score_service

router = APIRouter(prefix="/scores", tags=["scores"])


@router.get(
    "/case/{case_id}",
    response_model=CaseScoreReport,
    summary="The case's ten numbers, as the grid OncoStem receives",
)
def case(case_id: str) -> CaseScoreReport:
    return case_score_service.report(case_id)


@router.get(
    "/{he_upload_id}",
    response_model=ScoreReport,
    summary="One marker's percent positive and intensity",
)
def report(
    he_upload_id: str,
    ihc_upload_id: str = Query(alias="ihcUploadId"),
) -> ScoreReport:
    try:
        return score_service.report(he_upload_id, ihc_upload_id)
    except ScoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
