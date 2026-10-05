"""Step 14 - Per-cell measurement.

INPUT
    Step 13's compartments + the calibrated DAB channel
OUTPUT
    One measurement row per tumour cell: mean DAB optical density, and the
    second number this antibody gets

This is the first step that measures anything. Everything before it was deciding
what to measure. See `measure.py` for the fork - ring completeness for A, F and
R; stained fraction of the band for U and W - and for why a cytoplasmic marker
is not merely spared the completeness calculation but never has it run.
"""

from __future__ import annotations

from app.pipeline.contract import PipelineContext, PipelineError, StepResult

STAGE_ID = "per-cell-measurement"
TITLE = "Per-cell measurement"


def run(context: PipelineContext) -> StepResult:
    """Measure the DAB in every tumour cell's own compartment."""
    context.require("compartments")

    ihc_upload_id = context.artifacts.get("ihc-upload-id")
    if not ihc_upload_id:
        raise PipelineError(
            "step 14 measures the stain on the IHC slide, and which part of each cell it "
            "measures depends on which antibody that slide carries. Load a case "
            "(POST /cases/load) and put the IHC upload id in "
            "context.artifacts['ihc-upload-id']."
        )

    from app.services.per_cell_service import per_cell_service

    return StepResult(
        stage_id=STAGE_ID,
        output=per_cell_service.report(context.upload_id, ihc_upload_id),
    )
