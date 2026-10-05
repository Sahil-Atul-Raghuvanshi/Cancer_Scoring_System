"""Step 17 - Validation.

INPUT
    Step 16's scores + pathologist scores
OUTPUT
    An agreement report, or an honest statement that there is nothing to compare
    against

See `agreement.py` for why the standard is inter-pathologist agreement rather
than a ground truth, and for what this step refuses to do when the reader sheet
is absent.
"""

from __future__ import annotations

from app.pipeline.contract import PipelineContext, StepResult

STAGE_ID = "validation"
TITLE = "Validation"


def run(context: PipelineContext) -> StepResult:
    """Compare this case's scores against the readers', if any are on disk."""
    context.require("aggregate")

    from app.services.validation_service import validation_service

    case_id = context.artifacts.get("case-id")
    return StepResult(
        stage_id=STAGE_ID,
        output=validation_service.report(case_id),
    )
