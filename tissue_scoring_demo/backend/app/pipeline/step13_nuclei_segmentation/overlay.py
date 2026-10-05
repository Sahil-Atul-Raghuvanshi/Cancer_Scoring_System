"""Step 11's pictures: outlines over the tissue, never fills.

Same argument as step 10's overlay module, one scale down. The viewer is judging
whether an outline sits on a nucleus, and a filled shape hides the pixels that
question is about. So every nucleus is drawn as a one-pixel boundary over the
field as it really looks - the raw RGB, not the DAB-stripped image the model was
shown - because "did it find the nuclei" is a question about the slide, not about
the intermediate.

Three pictures per field, and the set is the argument:

  raw       the field, untouched
  input     what the segmenter actually saw, with the brown removed
  overlay   the outlines drawn back onto the raw field

A viewer who flips between the first two sees what deconvolution did; between the
first and third, what the model did. Neither is legible without the other.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image
from scipy import ndimage

#: Outline colour. Green because the two stains on screen are brown and blue, so
#: it is the one hue that cannot be mistaken for something on the slide.
OUTLINE = (0, 230, 64)

#: Nuclei excluded from the count are still drawn, in grey, so the picture and
#: the number can be reconciled by eye rather than taken on trust.
OUTLINE_UNCOUNTED = (150, 150, 150)

#: Where the border band is drawn, so "why was that one not counted" answers
#: itself.
BAND = (255, 170, 0)


def _png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def field_png(rgb: np.ndarray) -> bytes:
    """One field, as it is."""
    return _png(Image.fromarray(np.asarray(rgb, dtype=np.uint8)))


def boundaries(labels: np.ndarray) -> np.ndarray:
    """A boolean mask of every instance's outline, one pixel wide.

    Erosion-and-difference on the label map rather than per object: a pixel is on
    a boundary when its 3x3 neighbourhood is not all one label, which finds the
    seam between two *touching* nuclei as well as the edge against background.
    Drawing per object would paint one over the other and hide exactly the case
    this step exists to show.
    """
    eroded = ndimage.grey_erosion(labels, size=(3, 3))
    dilated = ndimage.grey_dilation(labels, size=(3, 3))
    return (eroded != dilated) & (labels > 0)


def overlay_array(
    rgb: np.ndarray,
    labels: np.ndarray,
    *,
    counted: set[int] | None = None,
    border_px: int = 0,
) -> np.ndarray:
    """The field with every nucleus outlined, counted ones in green.

    Returns the array rather than a PNG, so a caller composing two of these side
    by side does not have to encode and decode them to do it.
    """
    canvas = np.asarray(rgb, dtype=np.uint8).copy()
    edge = boundaries(labels)

    if counted is None:
        canvas[edge] = OUTLINE
    else:
        member = np.isin(labels, list(counted)) if counted else np.zeros_like(edge)
        canvas[edge & member] = OUTLINE
        canvas[edge & ~member] = OUTLINE_UNCOUNTED

    if border_px > 0:
        height, width = labels.shape
        rows = np.zeros(height, dtype=bool)
        cols = np.zeros(width, dtype=bool)
        rows[border_px] = rows[max(0, height - border_px - 1)] = True
        cols[border_px] = cols[max(0, width - border_px - 1)] = True
        canvas[rows, :] = BAND
        canvas[:, cols] = BAND

    return canvas


def overlay_png(
    rgb: np.ndarray,
    labels: np.ndarray,
    *,
    counted: set[int] | None = None,
    border_px: int = 0,
) -> bytes:
    """`overlay_array`, encoded."""
    return _png(
        Image.fromarray(overlay_array(rgb, labels, counted=counted, border_px=border_px))
    )


def side_by_side_png(left: np.ndarray, right: np.ndarray, *, gap: int = 8) -> bytes:
    """Two fields on one canvas, for the touching-nuclei comparison."""
    left = np.asarray(left, dtype=np.uint8)
    right = np.asarray(right, dtype=np.uint8)
    height = max(left.shape[0], right.shape[0])
    width = left.shape[1] + gap + right.shape[1]

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] + gap :] = right
    return _png(Image.fromarray(canvas))


__all__ = [
    "BAND",
    "OUTLINE",
    "OUTLINE_UNCOUNTED",
    "boundaries",
    "field_png",
    "overlay_array",
    "overlay_png",
    "side_by_side_png",
]
