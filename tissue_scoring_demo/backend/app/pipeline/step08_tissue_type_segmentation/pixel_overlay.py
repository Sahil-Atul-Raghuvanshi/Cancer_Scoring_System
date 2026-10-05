"""The four panels a BEETLE pass is read as. The same four names, drawn from pixels.

  map          the slide with the pixel mask over it, one colour per class
  flat         the pixel mask alone, no slide under it
  scored       only invasive carcinoma, lit; the rest of the tissue dark
  confidence   how sure the network was, per window, as a heat map

Deliberately the same four names `overlay.py` serves for the ResNet branches, because
they answer the same four questions and a viewer switching options on step 7 should not
have to learn a new screen. What changes is the *source*: there, a label per window
painted over its cell; here, a mask with a class per pixel.

**That difference is the branch's whole argument, and `flat` is where it shows.** A
window-per-cell map is a grid however good the model is - a duct either fills its cell
or it does not - and this one traces the duct. Putting the two side by side on one slide
is the comparison step 7's third option exists to make possible.

**A separate module rather than a branch inside `overlay.py`.** That module's geometry
is `cell_field` - which grid cell each display pixel belongs to, offset by half a window
because a cell's core is centred in its span - and none of it applies to an array that
is already a picture of the slide. Sharing the file would mean every function taking
either a `ClassMap` or a `PixelMap` and choosing, which is two renderers in one
namespace rather than one renderer with an argument.

The colours, labels and the scored class are `beetle.py`'s, imported rather than
restated: a legend that could disagree with the pixels it describes eventually will.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from app.pipeline.step05_optical_density.overlay import GROUND
from app.pipeline.step06_colour_deconvolution.overlay import ramp

from . import beetle
from .pixels import OUTSIDE, PixelMap

#: Longest edge of a panel served for display. Steps 5, 6, 7 and 8's number.
MAX_SIZE = 1600

#: The panels this branch serves, in reading order. `overlay.PANELS` **minus
#: `uncertainty`**, which is the one panel the two branches do not share.
#:
#: That layer flags in-situ windows this pipeline's class 1 would not stand behind, and
#: it indexes our three classes: BEETLE emits five of its own, answers per pixel rather
#: than per window, and has no connected components on a window grid to measure. So it
#: is not that the panel is unimplemented here - there is nothing for it to describe.
#: `tissue_type_service.panel` refuses the name on this branch rather than serving a
#: file that was never written.
PANELS: tuple[str, ...] = ("map", "flat", "scored", "confidence")

#: How strongly the class colours cover the scan on `map`. `overlay.CLASS_ALPHA`, so the
#: live paint, the ResNet panels and these read as one picture rather than three
#: intensities of the same idea.
CLASS_ALPHA = 0.55

#: The scan is dimmed under the overlay so the colours read as an overlay rather than as
#: part of the section.
SLIDE_DIM = 0.6

#: Colour for tissue that is not the scored class on `scored`. Nearly the ground, so the
#: lit region is unmistakable.
EXCLUDED_TINT = (38, 46, 60)

#: Confidence ramp: red where the network is barely choosing, through amber, to green
#: where it is sure. **Starts at 1/5, not 1/3**, because that is where a five-class
#: softmax bottoms out - `overlay.py`'s ramp starts at a third for its three classes,
#: and reusing that here would make an undecided pixel look half-confident.
CONFIDENCE_RAMP: tuple[tuple[float, tuple[int, int, int]], ...] = (
    (0.0, (127, 29, 29)),
    (0.25, (220, 38, 38)),
    (0.55, (251, 191, 36)),
    (0.8, (163, 230, 53)),
    (1.0, (34, 197, 94)),
)


def _to_png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _display_shape(shape: tuple[int, int]) -> tuple[int, int]:
    """A thumbnail-shaped canvas, capped at `MAX_SIZE` on its longest edge."""
    height, width = int(shape[0]), int(shape[1])
    scale = min(1.0, MAX_SIZE / max(1, max(height, width)))
    return max(1, round(height * scale)), max(1, round(width * scale))


def mask_at(pixel_map: PixelMap, shape: tuple[int, int]) -> np.ndarray:
    """The pixel mask resampled onto a `shape` canvas, `OUTSIDE` where nothing ran.

    Nearest-neighbour, and here that is right where it was wrong inside the pass. The
    reduction in `pixels.reduce_probabilities` is an 8x area reduction from the network's
    own spacing, where picking one pixel in 64 would make a duct wall arbitrarily thicker
    or thinner; this is a further 4x or so from a mask whose features are already tens of
    pixels across, and there are no probabilities left to average - only labels, and
    interpolating a label is meaningless.
    """
    height, width = shape
    if pixel_map.mask.shape == (height, width):
        return pixel_map.mask
    return np.asarray(
        Image.fromarray(pixel_map.mask, mode="L").resize(
            (width, height), Image.Resampling.NEAREST
        ),
        dtype=np.uint8,
    )


def _fit(image: Image.Image, longest: int = MAX_SIZE) -> Image.Image:
    width, height = image.size
    scale = min(1.0, longest / max(1, max(width, height)))
    if scale >= 1.0:
        return image
    return image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.LANCZOS,
    )


def _colourise(
    field: np.ndarray, *, keep: frozenset[int], background: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Class colours for a mask, and where a class was actually drawn.

    `keep` is the class filter. A class left out is not recoloured, it is **not drawn**,
    so the scan shows through where it was and a viewer switching one off watches tissue
    leave the map rather than change colour - which is what makes the toggle a statement
    about the denominator rather than a palette change.
    """
    canvas = np.zeros((*field.shape, 3), dtype=np.uint8)
    canvas[:] = background

    drawn = np.zeros(field.shape, dtype=bool)
    for code in range(len(beetle.PIXEL_CLASSES)):
        if code not in keep:
            continue
        here = field == code
        canvas[here] = beetle.CLASS_COLOURS[code]
        drawn |= here

    return canvas, drawn


def map_png(
    thumbnail: np.ndarray, pixel_map: PixelMap, *, classes: frozenset[int] | None = None
) -> bytes:
    """Panel 1 - the slide with the pixel mask over it."""
    keep = classes if classes is not None else frozenset(range(len(beetle.PIXEL_CLASSES)))

    display = _fit(Image.fromarray(thumbnail, mode="RGB"))
    width, height = display.size

    faded = np.asarray(display, dtype=np.float32) * SLIDE_DIM
    field = mask_at(pixel_map, (height, width))
    colours, drawn = _colourise(field, keep=keep, background=GROUND)

    alpha = (drawn[..., None] * CLASS_ALPHA).astype(np.float32)
    blended = faded * (1.0 - alpha) + colours.astype(np.float32) * alpha

    return _to_png(Image.fromarray(blended.clip(0, 255).astype(np.uint8), mode="RGB"))


def flat_png(
    pixel_map: PixelMap, shape: tuple[int, int], *, classes: frozenset[int] | None = None
) -> bytes:
    """Panel 2 - the pixel mask with no slide under it.

    The honest companion to `map`, and on this branch the more interesting of the two:
    an overlay always flatters itself because a reader cannot separate what the model
    claimed from what the scan is showing through, and this is the claim alone. It is
    also where a per-pixel map and a per-window one stop looking alike.
    """
    keep = classes if classes is not None else frozenset(range(len(beetle.PIXEL_CLASSES)))
    height, width = _display_shape(shape)

    field = mask_at(pixel_map, (height, width))
    colours, _ = _colourise(field, keep=keep, background=GROUND)

    return _to_png(Image.fromarray(colours, mode="RGB"))


def scored_png(thumbnail: np.ndarray, pixel_map: PixelMap) -> bytes:
    """Panel 3 - only the tissue a score would be measured on.

    Rules 1 and 5 as one picture. Everything the rules exclude is drawn in the excluded
    tint rather than left transparent, because "this was looked at and left out" and
    "this was never looked at" are different facts and the difference is the step's whole
    contribution. Glass - BEETLE's `unannotated` - is left as ground rather than tinted:
    it was looked at and found to be nothing, which is a third fact again.
    """
    display = _fit(Image.fromarray(thumbnail, mode="RGB"))
    width, height = display.size

    faded = np.asarray(display, dtype=np.float32) * SLIDE_DIM
    field = mask_at(pixel_map, (height, width))

    excluded = np.isin(field, [code for code in beetle.TISSUE_CODES if code != beetle.SCORED_CODE])
    lit = field == beetle.SCORED_CODE

    canvas = faded.copy()
    canvas[excluded] = (
        faded[excluded] * (1.0 - CLASS_ALPHA)
        + np.asarray(EXCLUDED_TINT, dtype=np.float32) * CLASS_ALPHA
    )
    canvas[lit] = (
        faded[lit] * (1.0 - CLASS_ALPHA)
        + np.asarray(beetle.CLASS_COLOURS[beetle.SCORED_CODE], dtype=np.float32)
        * CLASS_ALPHA
    )

    return _to_png(Image.fromarray(canvas.clip(0, 255).astype(np.uint8), mode="RGB"))


def confidence_png(pixel_map: PixelMap, shape: tuple[int, int]) -> bytes:
    """Panel 4 - how sure the network was, window by window.

    **Per window and not per pixel, and that is a deliberate limitation stated rather
    than hidden.** A per-pixel confidence would be a second gigapixel array to reduce
    and store for a panel; what is kept is each window's mean top probability, which is
    the number that answers "how much room does a smoothing step have here". So this
    panel is the one place a BEETLE pass is still drawn on the window grid, and it is
    visibly blockier than the other three.
    """
    height, width = _display_shape(shape)
    grid = pixel_map.grid

    top = pixel_map.confidence()

    # The window grid resampled onto the canvas, the same way `overlay.cell_field` does
    # it: from each display pixel's centre, offset to the cell's core rather than its
    # span, so a cell's colour lands on the tissue it was measured from.
    xs = (np.arange(width, dtype=np.float64) + 0.5) * grid.slide_width / max(1, width)
    ys = (np.arange(height, dtype=np.float64) + 0.5) * grid.slide_height / max(1, height)
    offset = (grid.span - grid.stride) / 2.0
    cols = np.floor((xs - offset) / grid.stride).astype(np.int64)
    rows = np.floor((ys - offset) / grid.stride).astype(np.int64)

    valid = (rows >= 0)[:, None] & (rows < grid.rows)[:, None]
    valid = valid & (cols >= 0)[None, :] & (cols < grid.cols)[None, :]

    field = top[
        np.clip(rows, 0, grid.rows - 1)[:, None], np.clip(cols, 0, grid.cols - 1)[None, :]
    ]

    floor = 1.0 / len(beetle.PIXEL_CLASSES)
    normalised = np.clip((field - floor) / (1.0 - floor), 0.0, 1.0)

    canvas = ramp(normalised, CONFIDENCE_RAMP)
    canvas[~(valid & (field > 0.0))] = GROUND

    return _to_png(Image.fromarray(canvas, mode="RGB"))


def legend() -> list[dict]:
    """What each colour on these panels means, derived from `beetle.py` and not restated."""
    return [
        {
            "code": code,
            "name": name,
            "label": beetle.CLASS_LABELS[code],
            "rgb": list(beetle.CLASS_COLOURS[code]),
            "hex": "#%02x%02x%02x" % beetle.CLASS_COLOURS[code],
            "tissue": code in beetle.TISSUE_CODES,
            "scored": code == beetle.SCORED_CODE,
        }
        for code, name in enumerate(beetle.PIXEL_CLASSES)
    ]


__all__ = [
    "CONFIDENCE_RAMP",
    "MAX_SIZE",
    "PANELS",
    "confidence_png",
    "flat_png",
    "legend",
    "map_png",
    "mask_at",
    "scored_png",
]
