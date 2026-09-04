"""Step 13 - Per-cell measurement. NOT IMPLEMENTED.

INPUT
    Compartments + DAB channel
OUTPUT
    One measurement row per cell

See README.md in this folder for status and plan.
"""

from __future__ import annotations

from app.pipeline.contract import PipelineContext, StepNotImplementedError, StepResult

STAGE_ID = "per-cell-measurement"
TITLE = "Per-cell measurement"


def run(context: PipelineContext) -> StepResult:  # noqa: ARG001 - stub keeps the contract shape
    raise StepNotImplementedError(STAGE_ID, TITLE)
