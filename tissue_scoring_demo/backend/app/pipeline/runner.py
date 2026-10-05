"""Sequences the 19 steps in the fixed order the pipeline guide requires.

Nothing here decides *how* a step works - that is the step package's job.
This module only decides *when* each one runs: in catalogue order, feeding
each step's result into the context the next one sees, and stopping the
moment a step is not built yet.

**One caveat about what "in order" means here.** The pipeline is a Y, not a line:
steps 1 to 6 are a per-slide prefix that the demo runs once on each slide of a case,
steps 7 to 11 run on the H&E only, step 12 joins the two, and 13 to 18 run on the
immunostained slide. Each stage declares which in `runs_on` - see
`app/data/pipeline_steps.py`. This runner drives *one* `upload_id` straight through,
because that is what a script or a batch job wants; the second slide's prefix is
reached through the same services with the other id, which is exactly what steps 13
and 16 already do when they ask step 4 for the IHC slide's white point. A runner that
tried to express the Y would have to own the case, and the case is the API layer's.

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
from app.pipeline.step10_roi_selection.pipeline import run as _step10
from app.pipeline.step11_roi_refinement.pipeline import run as _step11
from app.pipeline.step12_ihc_alignment.pipeline import run as _step12
from app.pipeline.step13_nuclei_segmentation.pipeline import run as _step13
from app.pipeline.step14_cell_typing.pipeline import run as _step14
from app.pipeline.step15_compartments.pipeline import run as _step15
from app.pipeline.step16_per_cell_measurement.pipeline import run as _step16
from app.pipeline.step17_intensity_binning.pipeline import run as _step17
from app.pipeline.step18_aggregate.pipeline import run as _step18
from app.pipeline.step19_validation.pipeline import run as _step19

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
    _step17,
    _step18,
    _step19,
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
