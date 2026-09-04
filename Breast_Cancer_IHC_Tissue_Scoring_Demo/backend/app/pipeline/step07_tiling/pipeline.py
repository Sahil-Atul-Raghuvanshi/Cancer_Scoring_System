"""The orchestrator-facing entry point for step 7.

Thin, like every step before it: `app.services.tiling_service` is step 7's
implementation, so this module adapts it to `run(context)` rather than becoming a
second place a tile index gets built.

Two requirements, and they are the step's whole justification rather than
bookkeeping. `require("tissue-mask")` is what makes the funnel possible - without
a mask every tile of the canvas is a candidate, and two thirds of an Aperio frame
is empty glass. `require("colour-deconvolution")` is subtler: the index this step
hands on is addresses, and the pixels those addresses resolve to are the
haematoxylin channel. Requiring the fork means the tile index cannot be built in a
pipeline where nothing has yet defined what a tile's pixels *are*.

Quality control is deliberately not required. The artefact gate passes everything
when step 2 has not run and the report says so on screen, because a viewer may
open this step on its own - but a mask and a deconvolution are inputs rather than
refinements, so those two are not optional.
"""

from __future__ import annotations

from app.pipeline.contract import PipelineContext, StepResult
from app.services.tiling_service import tiling_service

STAGE_ID = "tiling"


def run(context: PipelineContext) -> StepResult:
    """Cut the tissue into patches, and count what the gates removed."""
    context.require("tissue-mask")
    context.require("colour-deconvolution")
    report = tiling_service.report(context.upload_id)
    return StepResult(stage_id=STAGE_ID, output=report)
