"""Which of the segmented nuclei belong in the denominator.

Not every cell inside a tumour region is a tumour cell. Lymphocytes,
fibroblasts and endothelial cells are mixed in, and if they land in the
denominator the percentage is diluted by however many of them there are - which
varies from slide to slide, so it is not even a constant bias.

**This is a rule, not a fitted model, and the difference is stated everywhere it
shows.** The guide's recipe is to fit a small classifier on a few hundred
hand-checked nuclei. Nobody has labelled any nuclei in this project, so there is
nothing to fit on, and a model fitted on invented labels would be worse than a
rule: it would carry the same judgements with a number attached that implies
they were measured. So the judgements are written out as thresholds, in physical
units, with the sensitivity of the answer to each of them published beside the
answer. When a few hundred nuclei do get labelled, `fit()` in a future revision
replaces `classify()` and the thresholds become its baseline.

**Every threshold is in microns or dimensionless.** Never pixels: at this
project's 0.2222 um/px a 7 um nucleus is 31 px and on another scanner it is not.

The three classes are the three the guide names, and the separation it names:
lymphocytes are small, round and very dark; tumour nuclei are large and
irregular; spindle cells - fibroblasts and endothelium - are elongated.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class CellType(IntEnum):
    """The three classes, and the one that is the denominator."""

    TUMOUR = 0
    LYMPHOCYTE = 1
    SPINDLE = 2


#: Display names, and the colours the overlay draws each class in.
#:
#: Red, amber, green - tumour, immune, support. Chosen to be distinguishable
#: from the two stains on screen (brown and blue), so a class colour is never
#: confused for something on the slide, and to put the class that decides the
#: denominator in the one colour nobody has to consult a key to find.
#:
#: These are the same values the browser uses. The overlay PNGs are rendered
#: here and the slide overlay is drawn there, and a palette written down twice
#: is a legend that eventually disagrees with the picture beside it.
CELL_TYPE_NAMES: dict[int, str] = {
    CellType.TUMOUR: "tumour",
    CellType.LYMPHOCYTE: "lymphocyte",
    CellType.SPINDLE: "spindle",
}

CELL_TYPE_LABELS: dict[int, str] = {
    CellType.TUMOUR: "Tumour cell",
    CellType.LYMPHOCYTE: "Immune cell",
    CellType.SPINDLE: "Support cell",
}

CELL_TYPE_COLOURS: dict[int, tuple[int, int, int]] = {
    CellType.TUMOUR: (248, 113, 113),
    CellType.LYMPHOCYTE: (251, 191, 36),
    CellType.SPINDLE: (52, 211, 153),
}


@dataclass(frozen=True)
class Rules:
    """The thresholds, gathered so they can be swept and reported in one object.

    Defaults come from the physical size of the cells involved, not from a fit,
    and each carries its reasoning:

    `lymphocyte_max_area_um2`
        A resting lymphocyte nucleus is 6-7 um across, so about 30-38 um2. 35 is
        the middle of that. A breast tumour nucleus starts around 50.

    `lymphocyte_min_circularity`
        Lymphocyte nuclei are close to spherical and section as discs. 0.72 keeps
        the round ones and lets a small but irregular nucleus stay tumour.

    `lymphocyte_min_darkness`
        How much darker than this field's median a lymphocyte has to be, as a
        ratio of haematoxylin. They are dense with chromatin and section very
        dark, which is what separates a small lymphocyte from a small fragment.
        Relative to the field rather than absolute, because stain uptake varies
        per slide and an absolute cutoff would mean a different thing on each.

    `spindle_min_eccentricity`
        Fibroblast and endothelial nuclei are cigar-shaped. 0.85 is elongated
        enough that a tumour nucleus rarely reaches it.

    `spindle_max_area_um2`
        A large elongated object is more likely a mis-segmented pair of tumour
        nuclei than one fibroblast, so elongation alone does not make a cell
        spindle.
    """

    lymphocyte_max_area_um2: float = 35.0
    lymphocyte_min_circularity: float = 0.72
    lymphocyte_min_darkness: float = 1.05
    spindle_min_eccentricity: float = 0.85
    spindle_max_area_um2: float = 90.0


@dataclass(frozen=True)
class Typed:
    """The classes, and what the answer rests on."""

    #: One `CellType` per input nucleus, in the order they were given.
    labels: np.ndarray
    #: Median haematoxylin across the counted nuclei - the darkness the
    #: lymphocyte rule is relative to.
    darkness_reference: float
    counts: dict[int, int]
    shares: dict[int, float]


def classify(
    areas: np.ndarray,
    circularity: np.ndarray,
    eccentricity: np.ndarray,
    haematoxylin: np.ndarray,
    *,
    rules: Rules | None = None,
) -> Typed:
    """Label each nucleus. Pure: arrays in, arrays out, no slide and no I/O.

    Order matters and is deliberate. **Lymphocyte is tested first**, because it
    is the most specific rule - small *and* round *and* dark, three conditions -
    and because the cost of the two mistakes is not symmetric: a lymphocyte left
    in the denominator dilutes the percentage, which is the error this step
    exists to remove. **Spindle is tested second**, and only on cells that are
    not already lymphocytes. **Tumour is what is left**, which is the right
    default for a region step 9 already called invasive carcinoma: inside it,
    the prior is that a cell is tumour unless it looks like something else.
    """
    settings_ = rules or Rules()

    areas = np.asarray(areas, dtype=np.float64)
    circularity = np.asarray(circularity, dtype=np.float64)
    eccentricity = np.asarray(eccentricity, dtype=np.float64)
    haematoxylin = np.asarray(haematoxylin, dtype=np.float64)

    if areas.size == 0:
        return Typed(
            labels=np.empty(0, dtype=np.int8),
            darkness_reference=0.0,
            counts={int(t): 0 for t in CellType},
            shares={int(t): 0.0 for t in CellType},
        )

    reference = float(np.median(haematoxylin))
    # A field with no measurable counterstain has no darkness to be relative to;
    # fall back to admitting the size and shape rules alone rather than dividing
    # by something near zero and calling everything dark.
    darkness = haematoxylin / reference if reference > 1e-6 else np.ones_like(haematoxylin)

    labels = np.full(areas.shape, int(CellType.TUMOUR), dtype=np.int8)

    lymphocyte = (
        (areas <= settings_.lymphocyte_max_area_um2)
        & (circularity >= settings_.lymphocyte_min_circularity)
        & (darkness >= settings_.lymphocyte_min_darkness)
    )
    spindle = (
        ~lymphocyte
        & (eccentricity >= settings_.spindle_min_eccentricity)
        & (areas <= settings_.spindle_max_area_um2)
    )

    labels[lymphocyte] = int(CellType.LYMPHOCYTE)
    labels[spindle] = int(CellType.SPINDLE)

    counts = {int(t): int(np.count_nonzero(labels == int(t))) for t in CellType}
    total = max(1, int(labels.size))
    shares = {key: value / total for key, value in counts.items()}

    return Typed(
        labels=labels, darkness_reference=reference, counts=counts, shares=shares
    )


def sensitivity(
    areas: np.ndarray,
    circularity: np.ndarray,
    eccentricity: np.ndarray,
    haematoxylin: np.ndarray,
    *,
    rules: Rules | None = None,
) -> list[dict]:
    """How much the tumour share moves when each threshold is moved.

    Published beside the answer because the thresholds are judgements rather
    than measurements, and the guide's own standard for a parameter like this is
    that its sensitivity travels with the number it produced. A share that swings
    ten points between two defensible values of a threshold is a share with a
    hidden parameter in it, and the only honest thing to do is show it.
    """
    base = rules or Rules()
    sweeps = {
        "lymphocyte_max_area_um2": (25.0, 30.0, 35.0, 40.0, 45.0),
        "lymphocyte_min_circularity": (0.60, 0.66, 0.72, 0.78, 0.84),
        "lymphocyte_min_darkness": (0.95, 1.00, 1.05, 1.10, 1.15),
        "spindle_min_eccentricity": (0.75, 0.80, 0.85, 0.90, 0.95),
    }

    out: list[dict] = []
    for name, values in sweeps.items():
        points = []
        for value in values:
            variant = Rules(**{**base.__dict__, name: value})
            typed = classify(areas, circularity, eccentricity, haematoxylin, rules=variant)
            points.append(
                {"value": value, "tumourShare": round(typed.shares[int(CellType.TUMOUR)], 4)}
            )
        shares = [point["tumourShare"] for point in points]
        out.append(
            {
                "parameter": name,
                "baseline": getattr(base, name),
                "points": points,
                "swing": round(max(shares) - min(shares), 4),
            }
        )

    return out


__all__ = [
    "CELL_TYPE_COLOURS",
    "CELL_TYPE_LABELS",
    "CELL_TYPE_NAMES",
    "CellType",
    "Rules",
    "Typed",
    "classify",
    "sensitivity",
]
