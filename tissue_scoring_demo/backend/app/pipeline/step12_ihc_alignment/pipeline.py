"""Step 10 - Align the ROI to the IHC slide.

INPUT
    Step 9's invasive regions (H&E level-0 pixels) + the case's IHC slide
OUTPUT
    The same regions in the IHC slide's level-0 pixels, plus the two panels a
    person confirms them on

See README.md in this folder for why registration is on the main path here at
all, and what it refuses to do.
"""

from __future__ import annotations

from app.pipeline.contract import PipelineContext, PipelineError, StepResult

STAGE_ID = "ihc-alignment"
TITLE = "Align the ROI to the IHC slide"


def run(context: PipelineContext) -> StepResult:
    """Register the pair and move step 9's regions across.

    Unlike steps 1-9 this one needs *two* slides, and `PipelineContext` carries
    one. The runner drives a single slide end to end, so the IHC half arrives
    through `context.artifacts["ihc-upload-id"]`, which the case-level caller
    sets. Without it there is nothing to align to, and saying so is better than
    quietly skipping the step that moves the mask.
    """
    context.require("roi-mask")

    ihc_upload_id = context.artifacts.get("ihc-upload-id")
    if not ihc_upload_id:
        raise PipelineError(
            "step 10 needs the case's IHC slide as well as its H&E. Load a case "
            "(POST /cases/load) so both slides exist, and put the IHC upload id in "
            "context.artifacts['ihc-upload-id']."
        )

    # Imported here rather than at module scope: the service pulls in the
    # registration bridge and the slide readers, and steps 1-9 must stay
    # importable on a machine that has no registration environment at all.
    from app.services.ihc_alignment_service import ihc_alignment_service

    ihc_alignment_service.start(context.upload_id, ihc_upload_id)
    ihc_alignment_service.execute(context.upload_id, ihc_upload_id)
    report = ihc_alignment_service.report(context.upload_id, ihc_upload_id)

    return StepResult(stage_id=STAGE_ID, output=report)
