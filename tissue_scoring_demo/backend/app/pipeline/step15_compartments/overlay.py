"""Step 13's picture: one cell's three regions, drawn so the fork is obvious.

Filled here, unlike steps 10 and 11, and for the opposite reason. Those screens
ask "is this outline on the right tissue", which a fill would hide. This one asks
"which part of the cell is being measured", and that is a question about *area* -
an outline of a 4 micron ring at screen resolution is two lines a viewer has to
mentally fill in anyway.

The three colours are ordered so the measured compartment is the one that reads
first: the nucleus recedes, the cell body sits behind, and the band the marker is
scored in is the bright one. That ordering is the argument - the brown is not in
the nucleus, and the picture should say so before the caption does.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

#: Nucleus: muted, because it is the one part that is *not* measured for any
#: marker in this panel.
NUCLEUS = (70, 84, 120)

#: The rest of the cell the marker does not live in.
CELL = (30, 42, 62)

#: The measured compartment - ring or band. The only bright colour on the panel.
MEASURED_MEMBRANE = (56, 189, 248)
MEASURED_CYTOPLASM = (167, 139, 250)

BACKGROUND = (10, 14, 21)


def _png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def compartments_png(built, *, membrane: bool) -> bytes:
    """Nucleus, cell body and measured compartment of every cell in one field."""
    shape = built.nucleus.shape
    canvas = np.zeros((*shape, 3), dtype=np.uint8)
    canvas[:] = BACKGROUND

    canvas[built.cell > 0] = CELL
    canvas[built.measured > 0] = MEASURED_MEMBRANE if membrane else MEASURED_CYTOPLASM
    canvas[built.nucleus > 0] = NUCLEUS

    return _png(Image.fromarray(canvas))


def comparison_png(constrained: np.ndarray, overlaps: np.ndarray) -> bytes:
    """Where plain dilation would have had two cells claim the same pixel.

    `overlaps` counts how many nuclei reached each pixel; anything above one is a
    pixel two cells would both have counted. Drawn in red over the constrained
    result, so the demo can show the error rather than assert it.
    """
    shape = constrained.shape
    canvas = np.zeros((*shape, 3), dtype=np.uint8)
    canvas[:] = BACKGROUND
    canvas[constrained > 0] = CELL
    canvas[overlaps > 1] = (248, 113, 113)
    return _png(Image.fromarray(canvas))


__all__ = [
    "BACKGROUND",
    "CELL",
    "MEASURED_CYTOPLASM",
    "MEASURED_MEMBRANE",
    "NUCLEUS",
    "comparison_png",
    "compartments_png",
]
