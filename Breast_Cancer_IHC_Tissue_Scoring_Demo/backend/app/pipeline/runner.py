"""Sequences the 16 steps in the fixed order the pipeline guide requires.

Nothing here decides *how* a step works - that is the step package's job.
This module only decides *when* each one runs: in catalogue order, feeding
each step's result into the context the next one sees, and stopping the
moment a step is not built yet.

This mirrors, rather than replaces, the FastAPI layer: `app/services/*.py`
still serve steps 1 and 2 to the UI directly, because a multi-minute QC run
is a polled background job, not a function call an HTTP handler can await.
The runner is for driving the pipeline end-to-end in one call - a script, a
batch job, a test - where that job/poll shape does not apply.
"""

from __future__ import annotations

from app.pipeline.contract import PipelineContext, StepNotImplementedError
from app.pipeline.step01_read_slide.pipeline import run as _step01
from app.pipeline.step02_quality_control.pipeline import run as _step02
from app.pipeline.step03_tissue_mask.pipeline import run as _step03
from app.pipeline.step04_white_calibration.pipeline import run as _step04
from app.pipeline.step05_optical_density.pipeline import run as _step05
from app.pipeline.step06_colour_deconvolution.pipeline import run as _step06
from app.pipeline.step07_tiling.pipeline import run as _step07
from app.pipeline.step08_tissue_type_segmentation.pipeline import run as _step08
from app.pipeline.step09_roi_mask.pipeline import run as _step09
from app.pipeline.step10_nuclei_segmentation.pipeline import run as _step10
from app.pipeline.step11_cell_typing.pipeline import run as _step11
from app.pipeline.step12_compartments.pipeline import run as _step12
from app.pipeline.step13_per_cell_measurement.pipeline import run as _step13
from app.pipeline.step14_intensity_binning.pipeline import run as _step14
from app.pipeline.step15_aggregate.pipeline import run as _step15
from app.pipeline.step16_validation.pipeline import run as _step16

STEP_FUNCTIONS = (
    _step01,
    _step02,
    _step03,
    _step04,
    _step05,
    _step06,
    _step07,
    _step08,
    _step09,
    _step10,
    _step11,
    _step12,
    _step13,
    _step14,
    _step15,
    _step16,
)


def run_pipeline(context: PipelineContext, *, up_to: str | None = None) -> PipelineContext:
    """Run each step in catalogue order, stopping at the first unbuilt one.

    `up_to` stops early at a given stage id (inclusive) even if later steps
    are implemented - useful for driving the pipeline only as far as step 2
    while the rest are stubs. Without it, the run naturally stops at
    `StepNotImplementedError`, which today is every step after quality control.
    """
    for step in STEP_FUNCTIONS:
        result = step(context)
        context.artifacts[result.stage_id] = result.output
        if up_to is not None and result.stage_id == up_to:
            break
    return context


__all__ = ["run_pipeline", "StepNotImplementedError"]
