"""The orchestrator-facing entry point for step 11.

Thin, like every step: `app.services.roi_refinement_service` is step 11's
implementation.

Two requirements, and they are different in kind. `require("roi-selection")` is the
input - the ticked candidates are what this step runs on, and without step 10 there is
no selection, not even an empty one. `require("tissue-type-segmentation")` looks
redundant behind it and is not: this step narrows *step 8's grid*, so it needs the class
map's geometry and its gates, and asking for it by name is what stops a refinement from
being attempted against a class map that was discarded after the selection was made.

**This runs the job to completion rather than starting one**, like step 8 and for its
reason: the service is job-shaped because an HTTP handler cannot hold a connection open
for the minutes this takes, and a runner driving the pipeline end to end has no such
constraint. A step that returned a job id would make `run_pipeline` return before its own
output existed.

A headless run refines whatever step 10 left ticked, which with nobody to tick anything
is the coverage rule's default - the same regions step 12 would have carried across
before this step existed.
"""

from __future__ import annotations

from app.pipeline.contract import PipelineContext, StepResult

STAGE_ID = "roi-refinement"
TITLE = "Refine the regions per pixel"


def run(context: PipelineContext) -> StepResult:
    """Run BEETLE over each selected region and trace what it found."""
    context.require("roi-selection")
    context.require("tissue-type-segmentation")

    # Imported here rather than at module scope: the service pulls in torch through
    # `beetle.load`, and `app.pipeline.runner` imports every step at start-up. A machine
    # with no torch installed must still be able to import the pipeline.
    from app.services.roi_refinement_service import roi_refinement_service

    run_state = roi_refinement_service.start(context.upload_id)
    if run_state.state in {"queued", "running"}:
        roi_refinement_service.execute(context.upload_id)

    return StepResult(
        stage_id=STAGE_ID, output=roi_refinement_service.report(context.upload_id)
    )
