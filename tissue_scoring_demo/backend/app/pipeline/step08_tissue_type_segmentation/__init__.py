"""Step 8 - Tissue-type segmentation.

Implemented, and inference only. See README.md in this folder,
`docs/guides/demo-pipeline-guide.md` (step 9 there) and `app.data.pipeline_steps`
(id="tissue-type-segmentation") for what this step does and why it runs here.

`input.py` is the load-bearing module and the one to read first: it is the single
definition of what the model sees, shared with the training folder rather than copied
into it. `model.py` refuses a checkpoint whose manifest disagrees with it.

`uncertainty.py` is the step's second stage: a post-pass over the finished class map
that gives it a fourth *display* class, "cannot determine", over the in-situ calls it
will not stand behind. Read its docstring before changing the rule - three formulations
that look right are each wrong against a slide this project measured. It can only ever
recolour class 1, which Rule 5 already excludes, so it cannot move the score.

Nothing here imports torch at module scope, so a machine with no torch installed can
still import the pipeline and run steps 1 to 7.
"""

from .classes import (
    CANDIDATE,
    CLASS_COLOURS,
    CLASS_LABELS,
    CLASS_MEANING,
    CLASS_NAMES,
    CLASS_PLAIN,
    DISPLAY_NAMES,
    RIVAL,
    SCORED,
    UNCERTAIN,
    ClassOrderError,
    verify_order,
)
from .inference import (
    OUTSIDE,
    ClassMap,
    SegmentationError,
    WindowGrid,
    build_grid,
    classify,
    plan_block,
)
from .uncertainty import UncertaintyLayer, UncertaintyParams, derive

__all__ = [
    "CANDIDATE",
    "CLASS_COLOURS",
    "CLASS_LABELS",
    "CLASS_MEANING",
    "CLASS_NAMES",
    "CLASS_PLAIN",
    "DISPLAY_NAMES",
    "OUTSIDE",
    "RIVAL",
    "SCORED",
    "UNCERTAIN",
    "ClassMap",
    "ClassOrderError",
    "SegmentationError",
    "UncertaintyLayer",
    "UncertaintyParams",
    "WindowGrid",
    "build_grid",
    "classify",
    "derive",
    "plan_block",
    "verify_order",
]
