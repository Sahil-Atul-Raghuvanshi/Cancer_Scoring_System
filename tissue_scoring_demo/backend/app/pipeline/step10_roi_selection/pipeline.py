"""The orchestrator-facing entry point for step 10.

Thin, like every step: `app.services.roi_selection_service` is step 10's
implementation, so this module adapts it to `run(context)`.

One requirement, and it is the step's whole input. `require("roi-mask")` is not because
this step reads step 9's *mask* - it reads step 9's per-class regions, which are a
different product of the same class map - but because the two must describe one slide.
Step 9 is where this pipeline decides that a region below `roi_min_area_mm2` is speckle
and that in-situ disease is carved back out; offering candidates without having made
those decisions would put a region on the review screen that the scored region had
already thrown away.

**This builds to completion rather than starting a job.** There is no model here. The
cost is one padded slide read per candidate for its card - twenty reads of a few
megapixels each - which is seconds, not the tens of minutes that make steps 8, 11 and 12
job-shaped.

**A run does not choose anything.** `build` returns the candidates with the coverage
rule's own default ticked, and a headless run of the pipeline proceeds on that default -
which is exactly the selection step 12 would have made for itself before this step
existed, so an unattended run is unchanged by this step's existence. A person selecting
differently is a `POST /roi-selection/{id}/select`, and step 11 records which it got.
"""

from __future__ import annotations

from app.pipeline.contract import PipelineContext, StepResult

STAGE_ID = "roi-selection"
TITLE = "Review and select candidate regions"


def run(context: PipelineContext) -> StepResult:
    """Offer step 9's invasive patches as a list somebody can tick."""
    context.require("roi-mask")

    from app.services.roi_selection_service import roi_selection_service

    return StepResult(
        stage_id=STAGE_ID, output=roi_selection_service.build(context.upload_id)
    )
