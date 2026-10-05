"""The orchestrator-facing entry point for step 9.

Thin, like every step: `app.services.roi_service` is step 9's implementation, so this
module adapts it to `run(context)` rather than becoming a second place a region gets
built.

One requirement, and it is the whole of the step's input. `require("tissue-type-
segmentation")` is what makes this step meaningful rather than a threshold on a
picture: the ROI is built from *class probabilities*, and both of the decisions this
step makes - where the region ends, and which islands inside it are excluded - need
the classes, not a brightness map. A region derived without step 8 would be a
silhouette of dark tissue, which is what the pipeline spent eight steps not doing.

**This builds to completion rather than starting a job.** Unlike step 8 there is no
model to wait on: the region takes about a tenth of a second, and the twenty a cold
build spends are the thumbnail and the panels being drawn. So there is no queue and
no progress to report.
"""

from __future__ import annotations

from app.pipeline.contract import PipelineContext, StepResult

STAGE_ID = "roi-mask"
TITLE = "Build the ROI mask"


def run(context: PipelineContext) -> StepResult:
    """Turn step 8's class map into the one region every later step measures inside."""
    context.require("tissue-type-segmentation")

    from app.services.roi_service import roi_service

    return StepResult(stage_id=STAGE_ID, output=roi_service.build(context.upload_id))
