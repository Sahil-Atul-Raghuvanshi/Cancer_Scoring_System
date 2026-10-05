"""Step 15 - Intensity binning.

INPUT
    Step 14's per-cell measurements
OUTPUT
    Cells binned 0 / 1+ / 2+ / 3+, on this antibody's own absolute cut points

Two things happen here and `binning.py` explains why they are kept apart: the
per-cell bin is internal machinery, and the slide's reported intensity is a
different scale decided once at step 16.
"""

from __future__ import annotations

from app.pipeline.contract import PipelineContext, PipelineError, StepResult

STAGE_ID = "intensity-binning"
TITLE = "Intensity binning"


def run(context: PipelineContext) -> StepResult:
    """Sort every measured cell into a level, against this antibody's cuts."""
    context.require("per-cell-measurement")

    ihc_upload_id = context.artifacts.get("ihc-upload-id")
    if not ihc_upload_id:
        raise PipelineError(
            "step 15 applies one antibody's cut points, so it needs to know which slide - "
            "and therefore which antibody - the cells came from. Load a case "
            "(POST /cases/load) and put the IHC upload id in "
            "context.artifacts['ihc-upload-id']."
        )

    from app.services.binning_service import binning_service

    return StepResult(
        stage_id=STAGE_ID,
        output=binning_service.report(context.upload_id, ihc_upload_id),
    )
