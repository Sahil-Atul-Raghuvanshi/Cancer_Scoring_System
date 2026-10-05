"""Step 13 - Build compartments.

INPUT
    Step 11's nucleus instances + step 12's classes + the antibody letter
OUTPUT
    The region of each tumour cell where the marker is supposed to be

See README.md in this folder for the fork - membrane ring for A/F/R, cytoplasm
band for U/W - and why running one through the other's path is the specific
regression this step exists to prevent.
"""

from __future__ import annotations

from app.pipeline.contract import PipelineContext, PipelineError, StepResult

STAGE_ID = "compartments"
TITLE = "Build compartments"


def run(context: PipelineContext) -> StepResult:
    """Grow each tumour cell outward into its marker's compartment."""
    context.require("cell-typing")

    ihc_upload_id = context.artifacts.get("ihc-upload-id")
    if not ihc_upload_id:
        raise PipelineError(
            "step 13 builds compartments on the IHC slide's cells, and the shape it "
            "builds depends on which antibody that slide was stained with. Load a case "
            "(POST /cases/load) and put the IHC upload id in "
            "context.artifacts['ihc-upload-id']."
        )

    from app.services.compartment_service import compartment_service

    return StepResult(
        stage_id=STAGE_ID,
        output=compartment_service.report(context.upload_id, ihc_upload_id),
    )
