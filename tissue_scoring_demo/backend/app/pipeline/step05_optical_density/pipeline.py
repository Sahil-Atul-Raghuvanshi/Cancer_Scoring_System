"""The orchestrator-facing entry point for step 5.

Thin, like steps 2, 3 and 4's, and for the same reason:
`app.services.density_service` is step 5's implementation, complete with the tile
choice the demo needs, so this module adapts that to `run(context)` rather than
becoming a second place a density gets computed.

The one thing it adds is the ordering check, and here it is not bookkeeping at
all. `require("white-calibration")` states the definition: optical density *is*
`-log10(I / I0)`, so a density computed before something has said what I0 is has
not been computed at a lower quality - it has not been computed. The order is also
not reversible in the other direction, and for a reason worth naming: step 4
measures the glass, so it needs to know where the tissue is not, which step 3
supplies; step 5 measures the tissue, and needs to know what the glass measures
as, which step 4 supplies. Each step's input is the previous step's answer to a
different question.

The service tolerates step 2 not having run and says so on screen, because a
viewer may open step 5 on its own; it does not tolerate steps 3 or 4 missing,
because a mask and a white point are its inputs rather than refinements of them.
"""

from __future__ import annotations

from app.pipeline.contract import PipelineContext, StepResult
from app.services.density_service import density_service

STAGE_ID = "optical-density"


def run(context: PipelineContext) -> StepResult:
    """Transform one tile into optical density, after calibration and before the fork."""
    context.require("white-calibration")
    report = density_service.report(context.upload_id)
    return StepResult(stage_id=STAGE_ID, output=report)
