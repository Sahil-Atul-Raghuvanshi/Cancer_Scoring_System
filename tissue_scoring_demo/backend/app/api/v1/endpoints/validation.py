"""Step 17 endpoints - agreement with pathologists, or an honest absence of it.

    GET /validation                     every case this system has scored
    GET /validation?caseId=CAN_00270    one case

Answers 200 with `available: false` and a reason rather than 404 when there is
no reader sheet. "Nothing to validate against" is a result of this step, not a
missing resource - and it is the result today, because the 120 readings are not
in this repository.
"""

from fastapi import APIRouter, Query

router = APIRouter(prefix="/validation", tags=["validation"])


@router.get("", summary="Agreement against pathologist readings")
def report(case_id: str | None = Query(default=None, alias="caseId")) -> dict:
    from app.services.validation_service import validation_service

    return validation_service.report(case_id)
