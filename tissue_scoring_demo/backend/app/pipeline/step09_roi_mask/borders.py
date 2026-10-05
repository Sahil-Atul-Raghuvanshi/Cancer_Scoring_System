"""Step 9's second product: one region per patch of same-class tile, not one per slide.

`mask.build()` answers "where is scoring allowed" - a single smoothed, closed,
in-situ-protected region built from *probabilities*. This module answers a different
question: "where did the model draw a boundary", read straight off step 8's own
per-tile calls with no smoothing and no morphology. Adjacent tiles of the same class
merge into one region because they are one region on the slide; nothing else changes
the shape the model actually output.

**Stroma is excluded on purpose, and it is the only exclusion.** `non_epithelium` is
the bulk of every slide and the class nobody circles - drawing a border around every
patch of it would bury the three classes a reader is actually looking for under
thousands of stroma outlines. Invasive, in-situ and the derived `uncertain` overlay are
exactly the three a pathologist would want boundaries on, and `class_map.display_labels`
already carries `uncertain` (see `classes.py`), so no new class-derivation logic is
needed here.

**Borders are drawn a hair inside their own region.** Two regions of different
classes that touch share their cell edge exactly, so tracing both truthfully puts two
borders on one line and a reader cannot tell whose edge it is - invasive against
in-situ, in-situ against uncertain, any pair of the three. `BORDER_GAP_CELLS` pulls each
one inward so a gap of twice it always separates them. The areas reported here are
still the cell count, untouched by that.

**No area cutoff, unlike `mask.build()`.** A single stray tile is still a region here,
because the point of this pass is to show the model's raw call, speckle included - the
scored region's `min_area_mm2` exists to drop what closing invented, and closing does
not happen here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from app.pipeline.step08_tissue_type_segmentation.classes import SCORED
from app.pipeline.step08_tissue_type_segmentation.inference import ClassMap
from app.pipeline.step08_tissue_type_segmentation.uncertainty import UNCERTAIN

from .mask import BORDER_GAP_CELLS, IN_SITU, level0_mapper, trace_rings

#: The three classes this module draws borders around. Never `0` (non_epithelium) -
#: see the module docstring.
BORDER_CLASSES: tuple[int, ...] = (IN_SITU, SCORED, UNCERTAIN)


@dataclass(frozen=True)
class ClassRegion:
    """One connected patch of one class, and the rings that draw it.

    `index` ranks this region against the others of the *same* class, largest first -
    it is what "top 3" means, and what the crop and QuPath exporters key their output
    on.
    """

    class_id: int
    index: int
    cells: int
    area_mm2: float
    #: Outer ring first, then holes - a hole here is a different class fully enclosed
    #: by this one, same convention as `RoiRegion.rings`.
    rings: tuple[tuple[tuple[int, int], ...], ...]

    #: Mean probability the model gave *this region's own class* over the windows the
    #: region is made of. Defaulted rather than required, so nothing that builds a
    #: `ClassRegion` by hand - the tests, and step 12's outer-ring stand-in - has to
    #: invent one.
    #:
    #: **It describes the tile model's call, not the tissue.** A region can be 0.97
    #: confident and still be the glass rim that `min_tissue_share` exists to remove;
    #: this number is offered on step 10's cards so a person choosing between
    #: candidates can see which ones the model was equivocal about, and it is not a
    #: gate anywhere.
    confidence: float = 0.0

    @property
    def holes(self) -> int:
        return max(0, len(self.rings) - 1)


def class_regions(class_map: ClassMap) -> dict[int, list[ClassRegion]]:
    """Every region of every border class, sorted largest first within each class.

    Pure and parameter-free: a function of `class_map.display_labels` alone, so a
    caller cannot get a different answer by passing different knobs - unlike the
    scored region, there is nothing here to tune.
    """
    grid = class_map.grid
    labels = class_map.display_labels
    inside = np.asarray(grid.inside, dtype=bool)
    to_level0 = level0_mapper(grid)

    out: dict[int, list[ClassRegion]] = {}
    for class_id in BORDER_CLASSES:
        pieces = _components(labels, inside, class_id)
        certainty = _certainty(class_map, class_id)

        out[class_id] = [
            ClassRegion(
                class_id=class_id,
                index=index,
                cells=cells,
                area_mm2=round(cells * grid.cell_mm2, 4),
                rings=trace_rings(piece, to_level0, gap_cells=BORDER_GAP_CELLS),
                confidence=round(float(certainty[piece].mean()), 4) if cells else 0.0,
            )
            for index, (cells, piece) in enumerate(pieces)
        ]

    return out


def class_region_cells(class_map: ClassMap, class_id: int) -> list[np.ndarray]:
    """The grid cells of each region of `class_id`, in `class_regions`' own order.

    The rings a `ClassRegion` carries are a *boundary*, and a boundary cannot be asked
    which windows it is made of - so a caller that needs the windows themselves (step 10
    lists them as a candidate's tile coordinates, step 11 turns them into the box BEETLE
    reads) would otherwise have to re-label the class map and hope its labelling matched
    the one the regions came from.

    It does match, because both go through `_components`. That is the only reason this
    function is safe, and it is why the labelling lives there rather than being written
    out twice.
    """
    labels = class_map.display_labels
    inside = np.asarray(class_map.grid.inside, dtype=bool)
    return [piece for _, piece in _components(labels, inside, class_id)]


def _components(
    labels: np.ndarray, inside: np.ndarray, class_id: int
) -> list[tuple[int, np.ndarray]]:
    """One class's connected patches, largest first. The one definition of the order.

    8-connectivity, because two windows touching at a corner are one patch of tissue -
    and because `trace_rings`' branch rule is written to agree with exactly that choice.
    The sort is what makes `ClassRegion.index` mean "rank by size", which the crops, the
    QuPath export and step 10's candidate ids all key on.
    """
    mask = (labels == class_id) & inside
    labelled, count = ndimage.label(mask, structure=np.ones((3, 3)))

    pieces = [
        (int((labelled == component).sum()), labelled == component)
        for component in range(1, count + 1)
    ]
    pieces.sort(key=lambda item: item[0], reverse=True)
    return pieces


def _certainty(class_map: ClassMap, class_id: int) -> np.ndarray:
    """(rows, cols) of how sure the model was, per window, for one border class.

    Two cases, because `UNCERTAIN` is not one of the model's classes. For the two that
    are, the honest number is the probability the model assigned to *that* class - a
    window called invasive at 0.51 and one called invasive at 0.99 are the same colour
    on the map and are not the same claim. For the derived class there is no such
    column, so the winning probability is used instead; it is the only thing that
    exists, and on those windows it is the number the uncertainty layer decided was
    not enough.
    """
    probabilities = np.asarray(class_map.probabilities, dtype=np.float32)
    if class_id < probabilities.shape[-1]:
        return probabilities[..., class_id]
    return probabilities.max(axis=-1)


__all__ = [
    "BORDER_CLASSES",
    "ClassRegion",
    "class_region_cells",
    "class_regions",
]
