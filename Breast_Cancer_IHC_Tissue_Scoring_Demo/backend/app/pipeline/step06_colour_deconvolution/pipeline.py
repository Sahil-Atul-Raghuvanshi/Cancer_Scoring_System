"""The orchestrator-facing entry point for step 6.

Thin, like steps 2 to 5's, and for the same reason:
`app.services.deconvolution_service` is step 6's implementation, so this module
adapts it to `run(context)` rather than becoming a second place the stains get
separated.

The ordering check is the interesting line. `require("optical-density")` is not
bookkeeping: colour deconvolution is a *linear* inverse, and a mixture of stains
is only linear in optical density. Run on RGB - a product of transmissions - the
same matrix produces three confident numbers that are not concentrations of
anything. So the requirement is the validity condition of the arithmetic rather
than a data dependency, and it is the one Rule 3 turns on.

The order does not reverse either. Step 5 needs a white point to divide by, which
step 4 measures from glass, which step 3 finds by knowing where tissue is not.
Each step's input is the previous step's answer to a different question, and this
is where that chain stops being about acquisition and starts being about dyes.
"""

from __future__ import annotations

from app.pipeline.contract import PipelineContext, StepResult
from app.services.deconvolution_service import deconvolution_service

STAGE_ID = "colour-deconvolution"


def run(context: PipelineContext) -> StepResult:
    """Separate haematoxylin from DAB - once, for both branches of the fork."""
    context.require("optical-density")
    report = deconvolution_service.report(context.upload_id)
    return StepResult(stage_id=STAGE_ID, output=report)
