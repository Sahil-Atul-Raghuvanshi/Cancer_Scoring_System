"""Step 9 - Build the ROI mask.

Implemented, and classical: no model, no training, no slide access. See README.md in
this folder, `docs/guides/demo-pipeline-guide.md` (step 10 there) and `app.data.pipeline_steps`
(id="roi-mask").

`mask.py` is the load-bearing module and the one to read first. Its `protect_in_situ`
parameter is the step's one real idea: morphological closing cannot tell a blood vessel
inside a tumour from a duct of in-situ carcinoma inside a tumour, and Rule 5 says those
must be treated in opposite ways. The class map is the only thing that can tell them
apart, so the class map is what decides.

Nothing here imports torch, and nothing here opens a slide - the input is step 8's
stored class map and the output is a boolean grid on step 8's own geometry.
"""

from .mask import (
    IN_SITU,
    RoiError,
    RoiMask,
    RoiParams,
    RoiRegion,
    build,
)
from .overlay import PANELS

__all__ = [
    "IN_SITU",
    "PANELS",
    "RoiError",
    "RoiMask",
    "RoiParams",
    "RoiRegion",
    "build",
]
