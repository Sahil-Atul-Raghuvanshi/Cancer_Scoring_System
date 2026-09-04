"""Step 14 - Intensity binning. NOT IMPLEMENTED.

INPUT
    Per-cell measurements
OUTPUT
    Cells binned 0 / 1+ / 2+ / 3+

See README.md in this folder for status and plan.
"""

from __future__ import annotations

from app.pipeline.contract import PipelineContext, StepNotImplementedError, StepResult

STAGE_ID = "intensity-binning"
TITLE = "Intensity binning"


def run(context: PipelineContext) -> StepResult:  # noqa: ARG001 - stub keeps the contract shape
    raise StepNotImplementedError(STAGE_ID, TITLE)
