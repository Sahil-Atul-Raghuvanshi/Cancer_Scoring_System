"""Step 9 - Build the ROI mask. NOT IMPLEMENTED.

INPUT
    Per-tile class map
OUTPUT
    Invasive tumour ROI polygon

See README.md in this folder for status and plan.
"""

from __future__ import annotations

from app.pipeline.contract import PipelineContext, StepNotImplementedError, StepResult

STAGE_ID = "roi-mask"
TITLE = "Build the ROI mask"


def run(context: PipelineContext) -> StepResult:  # noqa: ARG001 - stub keeps the contract shape
    raise StepNotImplementedError(STAGE_ID, TITLE)
