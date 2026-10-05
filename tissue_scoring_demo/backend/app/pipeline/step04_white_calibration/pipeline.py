"""The orchestrator-facing entry point for step 4.

Thin, like steps 2 and 3's, and for the same reason:
`app.services.calibration_service` is step 4's implementation, complete with the
thumbnail caching the demo needs, so this module adapts that to `run(context)`
rather than becoming a second place a white point gets estimated.

The one thing it adds is the ordering check. `require("tissue-mask")` is not
bookkeeping - step 4's entire method is "measure the part of the slide that is
not tissue", and without a tissue mask there is no such part. Nor is the order
between them reversible: a white point cannot be used to find tissue, because
step 3 thresholds saturation, which is a ratio between channels and so is
already independent of the illumination level this step measures.

The service tolerates step 2 not having run and says so on screen, because a
viewer may open step 4 on its own; it does not tolerate step 3 missing, because
the mask is the input rather than a refinement of it.
"""

from __future__ import annotations

from app.pipeline.contract import PipelineContext, StepResult
from app.services.calibration_service import calibration_service

STAGE_ID = "white-calibration"


def run(context: PipelineContext) -> StepResult:
    """Estimate I0 from this slide's own glass, after the tissue mask and before OD."""
    context.require("tissue-mask")
    report = calibration_service.report(context.upload_id)
    return StepResult(stage_id=STAGE_ID, output=report)
