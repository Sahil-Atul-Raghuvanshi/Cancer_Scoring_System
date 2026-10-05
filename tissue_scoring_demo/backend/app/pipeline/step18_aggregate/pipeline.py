"""Step 16 - Aggregate into a score.

INPUT
    Step 14's rows and step 15's cut points
OUTPUT
    The deliverable: percent positive and intensity, for this antibody

Last, because by now every hard decision has been made. `score.py` holds the
arithmetic and imports nothing from the service layer, so a disputed number can
be re-derived from the stored rows without the half of the pipeline that
produced them.
"""

from __future__ import annotations

from app.pipeline.contract import PipelineContext, PipelineError, StepResult

STAGE_ID = "aggregate"
TITLE = "Aggregate into a score"


def run(context: PipelineContext) -> StepResult:
    """Apply the rulebook and produce the two numbers that leave the system."""
    context.require("intensity-binning")

    ihc_upload_id = context.artifacts.get("ihc-upload-id")
    if not ihc_upload_id:
        raise PipelineError(
            "step 16 reports one marker's pair, so it needs to know which marker. Load a "
            "case (POST /cases/load) and put the IHC upload id in "
            "context.artifacts['ihc-upload-id']."
        )

    from app.services.score_service import score_service

    return StepResult(
        stage_id=STAGE_ID,
        output=score_service.report(context.upload_id, ihc_upload_id),
    )
