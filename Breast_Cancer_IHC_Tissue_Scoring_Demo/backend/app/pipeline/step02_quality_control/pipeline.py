"""The orchestrator-facing entry point for step 2.

Step 2's actual work - GrandQC inference, the classical feature maps, the
disk-cached report - is `app.services.qc_service`: it already is step 2's
implementation, complete with the job/caching behaviour a multi-minute
CPU inference run needs. This module only adapts that to the `run(context)`
shape the runner expects, so there remains exactly one place QC is computed.
"""

from __future__ import annotations

from app.pipeline.contract import PipelineContext, StepResult
from app.services.qc_service import qc_service

STAGE_ID = "quality-control"


def run(context: PipelineContext) -> StepResult:
    """Run QC to completion and return its report.

    `qc_service.execute` is the same synchronous pass the background job
    calls; the runner just waits for it inline instead of polling.
    """
    qc_service.execute(context.upload_id)
    report = qc_service.report(context.upload_id)
    return StepResult(stage_id=STAGE_ID, output=report)
