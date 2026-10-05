"""Step 12 - Cell typing.

INPUT
    Step 11's nuclei, with their size, shape and stain darkness
OUTPUT
    A class per nucleus, and the share of them that are tumour

See README.md in this folder for why this is a rule rather than a fitted model,
and for the check that refuses to present a mix when the nuclei underneath are
too small to be cells.
"""

from __future__ import annotations

from app.pipeline.contract import PipelineContext, PipelineError, StepResult

STAGE_ID = "cell-typing"
TITLE = "Cell typing"


def run(context: PipelineContext) -> StepResult:
    """Sort step 11's nuclei into tumour, immune and support cells."""
    context.require("nuclei-segmentation")

    ihc_upload_id = context.artifacts.get("ihc-upload-id")
    if not ihc_upload_id:
        raise PipelineError(
            "step 12 sorts the nuclei step 11 found on the IHC slide, so it needs to "
            "know which slide that was. Load a case (POST /cases/load) and put the IHC "
            "upload id in context.artifacts['ihc-upload-id']."
        )

    from app.services.cell_typing_service import cell_typing_service

    return StepResult(
        stage_id=STAGE_ID,
        output=cell_typing_service.report(context.upload_id, ihc_upload_id),
    )
