"""Step 8 - Tissue-type segmentation.

Implemented, and inference only. See README.md in this folder,
`docs/demo-pipeline-guide.md` (step 9 there) and `app.data.pipeline_steps`
(id="tissue-type-segmentation") for what this step does and why it runs here.

`input.py` is the load-bearing module and the one to read first: it is the single
definition of what the model sees, shared with the training folder rather than copied
into it. `model.py` refuses a checkpoint whose manifest disagrees with it.

Nothing here imports torch at module scope, so a machine with no torch installed can
still import the pipeline and run steps 1 to 7.
"""

from .classes import (
    CLASS_COLOURS,
    CLASS_LABELS,
    CLASS_MEANING,
    CLASS_NAMES,
    CLASS_PLAIN,
    SCORED,
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

__all__ = [
    "CLASS_COLOURS",
    "CLASS_LABELS",
    "CLASS_MEANING",
    "CLASS_NAMES",
    "CLASS_PLAIN",
    "OUTSIDE",
    "SCORED",
    "ClassMap",
    "ClassOrderError",
    "SegmentationError",
    "WindowGrid",
    "build_grid",
    "classify",
    "plan_block",
    "verify_order",
]
