"""The five panels step 8 is meant to be read as.

  map          the slide with the class map over it, one colour per class
  flat         the class map alone, no slide under it
  scored       only the class the score is gated on, lit; everything else dark
  confidence   the top-class probability, as a heat map
  uncertainty  the in-situ windows, shaded by how far they are from being trusted

The first is the guide's own picture. The second exists because an overlay always
flatters itself - a reader cannot tell how much of what they are seeing is the model
and how much is the scan showing through - and the two side by side settle it.

`uncertainty` is the same argument applied to the purple class. `map` shows a window as
flagged or not, which is a threshold on a continuous score, and a threshold hides how
close a call was: a component at 0.51 and one at 0.99 are the same purple and are not
the same finding. This panel is the score before the threshold, so a viewer can see
whether a region is emphatically flagged or sitting on the line - and, when the defaults
are wrong for a laboratory's slides, which way to move them.

**`scored` is the argument of the whole step, as a picture.** The guide asks for a
per-class opacity toggle so a viewer can switch fat off and *watch the denominator
change*; `map` with a `classes` filter is that toggle, and `scored` is where it ends
up. Rule 1 takes the fat out and Rule 5 takes the in-situ disease out, and what is
left is a good deal smaller than "the tumour" - which is the entire point of running a
learned step here rather than thresholding brightness at step 3.

`confidence` is here because a class map is an argmax and an argmax hides its own
margin. A window at 0.34/0.33/0.33 and a window at 0.99 are the same colour on `map`
and are not the same claim, and step 10 - which thresholds probabilities rather than
labels - needs the difference visible before it smooths it away.

**The class map is drawn at its own geometry, never by stretching an array.** A window
is `span` wide but only responsible for its `stride`-wide core, so the cell a
thumbnail pixel belongs to is `floor((position - (span - stride) / 2) / stride)`.
Resizing the label array to the thumbnail instead would shift the whole map by half a
window - 28 um at the model's geometry, which at step 3's resolution is fourteen
visible pixels - and every boundary in the picture would sit next to the tissue it
describes rather than on it.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from app.pipeline.step05_optical_density.overlay import GROUND
from app.pipeline.step06_colour_deconvolution.overlay import ramp

from .classes import CANDIDATE, CLASS_COLOURS, DISPLAY_NAMES, SCORED
from .inference import OUTSIDE, ClassMap

#: Longest edge of a panel served for display. Steps 5, 6 and 7's number.
MAX_SIZE = 1600

#: The panels this step serves, in reading order.
PANELS: tuple[str, ...] = ("map", "flat", "scored", "confidence", "uncertainty")

#: How strongly the class colours cover the scan on `map`. Enough to read the classes
#: at a glance, light enough that the tissue under them is still visible - which is
#: what lets a viewer disagree with the model rather than only observe it.
CLASS_ALPHA = 0.55

#: The scan is dimmed under the overlay so the colours read as an overlay rather than
#: as part of the section.
SLIDE_DIM = 0.6

#: Colour for tissue that is not the scored class on the `scored` panel. Nearly the
#: ground, so the lit region is unmistakable.
EXCLUDED_TINT = (38, 46, 60)

#: Confidence ramp: red where the model is barely choosing, through amber, to green
#: where it is sure. Starts at 1/3 because that is what a three-class softmax bottoms
#: out at - a ramp starting at zero would spend its first third on values that cannot
#: occur and make every window look confident.
CONFIDENCE_RAMP: tuple[tuple[float, tuple[int, int, int]], ...] = (
    (0.0, (127, 29, 29)),
    (0.25, (220, 38, 38)),
    (0.55, (251, 191, 36)),
    (0.8, (163, 230, 53)),
    (1.0, (34, 197, 94)),
)


def _uncertainty_ramp(
    threshold: float,
) -> tuple[tuple[float, tuple[int, int, int]], ...]:
    """The ramp for panel 5, with the layer's own threshold as a visible step.

    Built per call rather than declared as a constant because the threshold is a setting
    and a fixed ramp would draw the step in the wrong place the moment it moved - which
    is precisely when someone is looking at this panel.

    Brightness rises within each half and the hue jumps between them, so the picture says
    two things at once: which side of the line a region is on, and how far from it. A
    single continuous ramp would show the second and hide the first.
    """
    edge = max(0.0, min(1.0, float(threshold)))
    return (
        (0.0, (30, 64, 130)),
        (max(0.0, edge - 1e-3), (125, 175, 245)),
        (edge, (109, 40, 165)),
        (1.0, (216, 160, 254)),
    )


def _to_png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _fit(image: Image.Image, longest: int = MAX_SIZE) -> Image.Image:
    width, height = image.size
    scale = min(1.0, longest / max(1, max(width, height)))
    if scale >= 1.0:
        return image
    return image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.LANCZOS,
    )


def cell_field(class_map: ClassMap, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Which grid cell each pixel of a `shape` picture belongs to.

    Returns `(rows, cols)` index arrays plus, through `OUTSIDE` in the caller, the
    means to blank the pixels that fall outside the grid. Computed from the *core*
    rather than the span - see the module docstring for the half-window shift this
    avoids - and from pixel centres rather than corners, so a cell's colour lands on
    the tissue it was measured from.
    """
    height, width = shape
    grid = class_map.grid

    # Level-0 position of each display pixel's centre.
    xs = (np.arange(width, dtype=np.float64) + 0.5) * grid.slide_width / max(1, width)
    ys = (np.arange(height, dtype=np.float64) + 0.5) * grid.slide_height / max(1, height)

    offset = (grid.span - grid.stride) / 2.0
    cols = np.floor((xs - offset) / grid.stride).astype(np.int64)
    rows = np.floor((ys - offset) / grid.stride).astype(np.int64)

    return rows, cols


def label_image(class_map: ClassMap, shape: tuple[int, int]) -> np.ndarray:
    """The class map resampled onto a `shape` canvas, `OUTSIDE` off the grid."""
    rows, cols = cell_field(class_map, shape)
    grid = class_map.grid

    valid_rows = (rows >= 0) & (rows < grid.rows)
    valid_cols = (cols >= 0) & (cols < grid.cols)

    labels = class_map.display_labels
    field = labels[
        np.clip(rows, 0, grid.rows - 1)[:, None], np.clip(cols, 0, grid.cols - 1)[None, :]
    ]
    return np.where(valid_rows[:, None] & valid_cols[None, :], field, OUTSIDE)


def _colourise(field: np.ndarray, *, keep: frozenset[int], background: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Class colours for a label field, and the mask of where a class was drawn.

    `keep` is the class filter - the per-class toggle the guide asks for. A class left
    out is not recoloured, it is *not drawn*, so the scan shows through where it was
    and a viewer watching the denominator change sees tissue leave the map rather than
    change colour.
    """
    canvas = np.zeros((*field.shape, 3), dtype=np.uint8)
    canvas[:] = background

    drawn = np.zeros(field.shape, dtype=bool)
    for label in range(len(DISPLAY_NAMES)):
        if label not in keep:
            continue
        here = field == label
        canvas[here] = CLASS_COLOURS[label]
        drawn |= here

    return canvas, drawn


def map_png(
    thumbnail: np.ndarray, class_map: ClassMap, *, classes: frozenset[int] | None = None
) -> bytes:
    """Panel 1 - the slide with the class map over it.

    `classes` is the per-class toggle. Filtering here rather than in the browser is
    what makes the toggle honest: an unselected class is absent from the picture, so
    the tissue it covered is visibly unclaimed rather than quietly recoloured.
    """
    keep = classes if classes is not None else frozenset(range(len(DISPLAY_NAMES)))

    display = _fit(Image.fromarray(thumbnail, mode="RGB"))
    width, height = display.size

    faded = np.asarray(display, dtype=np.float32) * SLIDE_DIM
    field = label_image(class_map, (height, width))
    colours, drawn = _colourise(field, keep=keep, background=GROUND)

    alpha = (drawn[..., None] * CLASS_ALPHA).astype(np.float32)
    blended = faded * (1.0 - alpha) + colours.astype(np.float32) * alpha

    return _to_png(Image.fromarray(blended.clip(0, 255).astype(np.uint8), mode="RGB"))


def flat_png(
    class_map: ClassMap, shape: tuple[int, int], *, classes: frozenset[int] | None = None
) -> bytes:
    """Panel 2 - the class map with no slide under it.

    The honest companion to `map`. An overlay always flatters itself, because a reader
    cannot separate what the model claimed from what the scan is showing through; this
    is the claim alone.
    """
    keep = classes if classes is not None else frozenset(range(len(DISPLAY_NAMES)))
    height, width = _display_shape(shape)

    field = label_image(class_map, (height, width))
    colours, _ = _colourise(field, keep=keep, background=GROUND)

    return _to_png(Image.fromarray(colours, mode="RGB"))


def scored_png(thumbnail: np.ndarray, class_map: ClassMap) -> bytes:
    """Panel 3 - only the tissue the score is gated on.

    Rules 1 and 5 as one picture. Everything the two rules exclude is drawn in the
    excluded tint rather than left transparent, because "this was looked at and left
    out" and "this was never looked at" are different facts and the difference is the
    step's whole contribution.
    """
    display = _fit(Image.fromarray(thumbnail, mode="RGB"))
    width, height = display.size

    faded = np.asarray(display, dtype=np.float32) * SLIDE_DIM
    field = label_image(class_map, (height, width))

    excluded = (field >= 0) & (field != SCORED)
    lit = field == SCORED

    canvas = faded.copy()
    canvas[excluded] = (
        faded[excluded] * (1.0 - CLASS_ALPHA)
        + np.asarray(EXCLUDED_TINT, dtype=np.float32) * CLASS_ALPHA
    )
    canvas[lit] = (
        faded[lit] * (1.0 - CLASS_ALPHA)
        + np.asarray(CLASS_COLOURS[SCORED], dtype=np.float32) * CLASS_ALPHA
    )

    return _to_png(Image.fromarray(canvas.clip(0, 255).astype(np.uint8), mode="RGB"))


def confidence_png(class_map: ClassMap, shape: tuple[int, int]) -> bytes:
    """Panel 4 - how sure the model was, window by window.

    Drawn through a ramp starting at 1/3 rather than at 0, because 1/3 is where a
    three-class softmax bottoms out: a ramp from zero would spend its first third on
    values that cannot occur and make an undecided window look confident.
    """
    height, width = _display_shape(shape)
    rows, cols = cell_field(class_map, (height, width))
    grid = class_map.grid

    valid = (rows >= 0)[:, None] & (rows < grid.rows)[:, None]
    valid = valid & (cols >= 0)[None, :] & (cols < grid.cols)[None, :]

    top = class_map.confidence()
    field = top[
        np.clip(rows, 0, grid.rows - 1)[:, None], np.clip(cols, 0, grid.cols - 1)[None, :]
    ]

    floor = 1.0 / class_map.probabilities.shape[-1]
    normalised = np.clip((field - floor) / (1.0 - floor), 0.0, 1.0)

    canvas = ramp(normalised, CONFIDENCE_RAMP)
    canvas[~(valid & (field > 0.0))] = GROUND

    return _to_png(Image.fromarray(canvas, mode="RGB"))


def uncertainty_png(class_map: ClassMap, shape: tuple[int, int]) -> bytes:
    """Panel 5 - the uncertainty score over the in-situ windows, before the threshold.

    Only the candidates are drawn. The other two classes are never scored by the layer,
    so shading them would put a number on tissue the layer has no opinion about; they sit
    at the excluded tint here, the same way `scored` treats everything it is not lit for.

    The ramp runs from the class colour at zero to the flag colour at one, so where a
    region lands between "still in-situ" and "cannot be determined" reads directly off
    the picture. The threshold is drawn as a step in the ramp rather than left implicit:
    a viewer needs to see which side of it a region is on, which is exactly what `map`
    cannot show them.
    """
    height, width = _display_shape(shape)
    rows, cols = cell_field(class_map, (height, width))
    grid = class_map.grid

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = GROUND

    layer = class_map.uncertainty
    if layer is None:
        return _to_png(Image.fromarray(canvas, mode="RGB"))

    valid = (rows >= 0)[:, None] & (rows < grid.rows)[:, None]
    valid = valid & (cols >= 0)[None, :] & (cols < grid.cols)[None, :]

    row_index = np.clip(rows, 0, grid.rows - 1)[:, None]
    col_index = np.clip(cols, 0, grid.cols - 1)[None, :]

    score = layer.score[row_index, col_index]
    candidate = class_map.labels[row_index, col_index] == CANDIDATE
    classified = class_map.grid.inside[row_index, col_index] & (
        class_map.labels[row_index, col_index] >= 0
    )

    canvas[valid & classified] = EXCLUDED_TINT
    here = valid & candidate
    if bool(here.any()):
        shaded = ramp(np.clip(score, 0.0, 1.0), _uncertainty_ramp(layer.params.threshold))
        canvas[here] = shaded[here]

    return _to_png(Image.fromarray(canvas, mode="RGB"))


def _display_shape(shape: tuple[int, int]) -> tuple[int, int]:
    """A thumbnail-shaped canvas, capped at `MAX_SIZE` on its longest edge."""
    height, width = int(shape[0]), int(shape[1])
    scale = min(1.0, MAX_SIZE / max(1, max(height, width)))
    return max(1, round(height * scale)), max(1, round(width * scale))


def empty_png(size: int = 512) -> bytes:
    """A plain ground, for when there is no class map to draw."""
    return _to_png(Image.new("RGB", (size, size), GROUND))


__all__ = [
    "CONFIDENCE_RAMP",
    "MAX_SIZE",
    "PANELS",
    "cell_field",
    "confidence_png",
    "empty_png",
    "flat_png",
    "label_image",
    "map_png",
    "scored_png",
    "uncertainty_png",
]
