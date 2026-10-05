"""The orchestrator-facing entry point for step 3.

Thin, like step 2's, and for the same reason: `app.services.tissue_service` is
step 3's implementation, complete with the basis caching the threshold slider
needs, so this module adapts that to `run(context)` rather than becoming a
second place a tissue mask gets computed.

The one thing it adds is the ordering check. `require("quality-control")` is not
bookkeeping - a tissue mask built before the artefact map exists has let pen ink
into the histogram that chose its threshold, which is the failure the pipeline
order exists to prevent. The service tolerates a missing QC run and says so on
screen, because a viewer may open step 3 on its own; the runner does not, because
in an end-to-end run there is no reason for step 2 not to have gone first.
"""

from __future__ import annotations

from app.pipeline.contract import PipelineContext, StepResult
from app.services.tissue_service import tissue_service

STAGE_ID = "tissue-mask"


def run(context: PipelineContext) -> StepResult:
    """Threshold tissue against glass, after quality control and before everything else."""
    context.require("quality-control")
    report = tissue_service.report(context.upload_id)
    return StepResult(stage_id=STAGE_ID, output=report)
