"""Step 10 - Nuclei segmentation. NOT IMPLEMENTED.

INPUT
    ROI + 40x tiles
OUTPUT
    Nucleus instances with boundaries

See README.md in this folder for status and plan.
"""

from __future__ import annotations

from app.pipeline.contract import PipelineContext, StepNotImplementedError, StepResult

STAGE_ID = "nuclei-segmentation"
TITLE = "Nuclei segmentation"


def run(context: PipelineContext) -> StepResult:  # noqa: ARG001 - stub keeps the contract shape
    raise StepNotImplementedError(STAGE_ID, TITLE)
