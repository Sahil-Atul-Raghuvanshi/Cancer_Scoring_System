"""Step 15 - Aggregate into a score. NOT IMPLEMENTED.

INPUT
    Binned tumour cells
OUTPUT
    Percent positive, H-score, category

See README.md in this folder for status and plan.
"""

from __future__ import annotations

from app.pipeline.contract import PipelineContext, StepNotImplementedError, StepResult

STAGE_ID = "aggregate"
TITLE = "Aggregate into a score"


def run(context: PipelineContext) -> StepResult:  # noqa: ARG001 - stub keeps the contract shape
    raise StepNotImplementedError(STAGE_ID, TITLE)
