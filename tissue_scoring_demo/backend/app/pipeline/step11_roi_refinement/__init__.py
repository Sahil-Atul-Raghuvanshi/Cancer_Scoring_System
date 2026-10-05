"""Step 11 - Refine each chosen region per pixel.

Implemented. This is where BEETLE runs, and the whole design of the step is in *where it
does not run*: not over the slide, but over the regions a person ticked on step 10.

The three modules, in the order worth reading them:

`crop.py` is the performance argument expressed as geometry. It narrows step 8's own
window grid to the windows whose cores meet one padded candidate box. Narrowing rather
than rebuilding is the safety property - every window it returns is one the slide-wide
pass would also have run, gates and all.

`contours.py` is the coordinate integrity. It turns BEETLE's raster answer back into
polygons in H&E level-0 pixels, which is the form step 12's registration takes, so the
refined boundary and the coarse one are the same kind of object and step 12 needs no
fork to warp either.

`overlay.py` is the comparison. Both panels of a region come from one slide read and one
drawing path, so any difference between them is a difference between the two answers.

**What this step produces becomes the tumour mask.** After it, `roi_refinement_service`
holds the invasive boundary the rest of the pipeline measures inside, and step 12 refuses
to carry step 9's tile staircase across once a refinement exists. That refusal is the
point: a square ROI and a pixel ROI give coherent-looking scores on different
denominators, and nothing downstream could tell which it had.

Nothing here imports torch at module scope - `beetle` is imported inside the service's
worker - so a machine with no torch installed can still import the pipeline.
"""

from .contours import RefinedRegion, clean, regions, to_level0
from .crop import RefinementGeometryError, core_bounds, padded_box, restrict, window_count

__all__ = [
    "RefinedRegion",
    "RefinementGeometryError",
    "clean",
    "core_bounds",
    "padded_box",
    "regions",
    "restrict",
    "to_level0",
    "window_count",
]
