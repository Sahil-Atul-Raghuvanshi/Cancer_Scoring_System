"""The orchestrator-facing entry point for step 8.

Thin, like every step before it: `app.services.tissue_type_service` is step 8's
implementation, so this module adapts it to `run(context)` rather than becoming a
second place a class map gets produced.

Two requirements, and both are the step's justification rather than bookkeeping.
`require("tiling")` is what makes the pass affordable - without step 7's audited list
of tiles every window of the canvas is a candidate, and two thirds of an Aperio frame
is empty glass. `require("colour-deconvolution")` is the harder one: what the model
reads is the haematoxylin channel, so requiring the fork means the class map cannot be
produced in a pipeline where nothing has yet defined what a tile's pixels *are*. The
guide is explicit that training and inference must call the same deconvolution or the
immunostained slides will score nothing like the H&E, and this is the step where that
matters most, because this is the step with a fitted model behind it.

**This runs the job to completion rather than starting one.** The service is job-shaped
because an HTTP handler cannot hold a connection open for tens of minutes; a runner
driving the pipeline end to end in one call has no such constraint, and a step that
returned a job id would make `run_pipeline` return before its own output existed.
"""

from __future__ import annotations

from app.pipeline.contract import PipelineContext, StepResult

STAGE_ID = "tissue-type-segmentation"


def run(context: PipelineContext) -> StepResult:
    """Classify every window of the kept tissue, and report what the classes cost."""
    context.require("tiling")
    context.require("colour-deconvolution")

    # Imported here rather than at module scope: the service pulls in torch through
    # `model.load_pinned`, and `app.pipeline.runner` imports every step at start-up.
    # A machine with no torch installed must still be able to import the pipeline and
    # run steps 1 to 7, which is the same argument step 2's package makes.
    from app.services.tissue_type_service import tissue_type_service

    run_state = tissue_type_service.start(context.upload_id)
    if run_state.state == "queued":
        tissue_type_service.execute(context.upload_id)

    return StepResult(stage_id=STAGE_ID, output=tissue_type_service.report(context.upload_id))
