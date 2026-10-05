"""Step 11 - Nuclei segmentation.

INPUT
    Step 10's carried regions, in the IHC slide's level-0 pixels, plus that slide
OUTPUT
    Every nucleus in a sample of fields inside those regions, outlined and measured

See README.md in this folder for why this step samples rather than exhausts, and
for the two measurements that set its defaults against the guide's advice.
"""

from __future__ import annotations

from app.pipeline.contract import PipelineContext, PipelineError, StepResult

STAGE_ID = "nuclei-segmentation"
TITLE = "Nuclei segmentation"


def run(context: PipelineContext) -> StepResult:
    """Segment the nuclei inside the three regions step 10 carried across.

    Like step 10 this needs *two* slides and `PipelineContext` carries one, so the
    IHC half arrives through `context.artifacts["ihc-upload-id"]`. Unlike step 10
    it does not register anything - the regions are already in the IHC slide's
    coordinates by the time they get here, which is the whole point of the step
    before it.
    """
    context.require("ihc-alignment")

    ihc_upload_id = context.artifacts.get("ihc-upload-id")
    if not ihc_upload_id:
        raise PipelineError(
            "step 11 segments inside the regions step 10 put on the IHC slide, so it "
            "needs that slide. Load a case (POST /cases/load) so both slides exist, "
            "and put the IHC upload id in context.artifacts['ihc-upload-id']."
        )

    # Imported here rather than at module scope: the service pulls in torch and
    # the slide readers, and steps 1-10 must stay importable on a machine with
    # no segmentation model installed at all.
    from app.services.nuclei_service import nuclei_service

    nuclei_service.start(context.upload_id, ihc_upload_id)
    nuclei_service.execute(context.upload_id, ihc_upload_id)
    report = nuclei_service.report(context.upload_id, ihc_upload_id)

    return StepResult(stage_id=STAGE_ID, output=report)
