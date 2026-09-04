"""The panels step 6 is meant to be read as.

  tile          the mixed picture that went in
  haematoxylin  how much blue counterstain each pixel carries
  dab           how much brown marker each pixel carries
  residual      what neither stain explains

That is the pipeline guide's own three-panel figure, with the input in front of
it so a reader can see which parts of the mixture went where. Read left to right
it makes the claim by itself: the nuclei are bright in one picture, the positive
membranes or nuclei are bright in the next, and the third is nearly empty - which
is what "two stains explain this tile" looks like.

**Each stain gets its own hue, and the hue is the stain's own.** Haematoxylin is
drawn blue and DAB brown, because those are the colours the dyes are and a reader
should not have to consult a legend to know which picture is which. Both are
single progressions from the page's ground through that hue to near-white, so more
ink on screen means more dye on the slide - the same rule step 5's heatmaps
follow, and for the same reason: a multi-hue ramp invents boundaries the data does
not have.

**The residual is drawn as a magnitude, in grey, and says so.** It is signed and
can go either way, so a diverging ramp would be the honest drawing - but a
diverging ramp is two hues, and the whole point of this panel is that a reader
should see *nothing much*. Grey by absolute value keeps "empty is good" legible.

**Both bases are drawn against the same scale, and that is load-bearing.** The
stretch is the fixed basis's 99.5th percentile, used for the estimated basis too.
Stretching each basis to its own range would rescale the picture along with the
numbers, and the toggle - the one place this step shows what choosing a per-image
basis costs - would show two identical-looking images under two different scores.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from app.pipeline.step05_optical_density.overlay import GROUND, tile_png

from .deconvolution import CHANNELS, Separation

#: Longest edge of a panel served for display. Step 5's number, for the reason
#: given there: a browser drawing this at 500 px does not need four thousand.
MAX_SIZE = 1600

#: Haematoxylin: ground, deep indigo, the app's blue, near-white. The dye's own
#: colour, so the panel needs no caption to be identified.
HAEMATOXYLIN_RAMP: tuple[tuple[float, tuple[int, int, int]], ...] = (
    (0.00, GROUND),
    (0.28, (34, 44, 104)),
    (0.66, (96, 122, 233)),
    (1.00, (238, 242, 255)),
)

#: DAB: ground, deep umber, the dye's own mid brown, warm near-white.
DAB_RAMP: tuple[tuple[float, tuple[int, int, int]], ...] = (
    (0.00, GROUND),
    (0.28, (70, 40, 16)),
    (0.66, (181, 121, 63)),
    (1.00, (255, 244, 228)),
)

#: The residual: neutral, so an empty panel reads as empty rather than as a
#: colour a reader has to interpret.
RESIDUAL_RAMP: tuple[tuple[float, tuple[int, int, int]], ...] = (
    (0.00, GROUND),
    (0.35, (48, 56, 70)),
    (0.72, (140, 152, 172)),
    (1.00, (244, 247, 252)),
)

RAMPS = {
    "haematoxylin": HAEMATOXYLIN_RAMP,
    "dab": DAB_RAMP,
    "residual": RESIDUAL_RAMP,
}

#: The panels this step serves, in reading order.
PANELS: tuple[str, ...] = ("tile", *CHANNELS)


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


def ramp(
    normalised: np.ndarray, stops: tuple[tuple[float, tuple[int, int, int]], ...]
) -> np.ndarray:
    """Map values in [0, 1] through a piecewise-linear colour ramp.

    Public because step 7 draws its sample tile through this function and
    `HAEMATOXYLIN_RAMP` above. That is deliberate rather than convenient: a second
    blue ramp over there would be a second answer to "how much counterstain is
    that", shown on the next screen along, and the two would drift.
    """
    values = np.clip(np.asarray(normalised, dtype=np.float32), 0.0, 1.0)
    positions = np.array([stop for stop, _ in stops], dtype=np.float32)
    colours = np.array([colour for _, colour in stops], dtype=np.float32)

    output = np.empty(values.shape + (3,), dtype=np.float32)
    for channel in range(3):
        output[..., channel] = np.interp(values, positions, colours[:, channel])
    return output.astype(np.uint8)


def panel_png(separation: Separation, name: str, *, basis: str) -> bytes:
    """One panel of one basis, as PNG.

    `tile` ignores `basis`, because the input to the step is the input to the step:
    both bases un-mix exactly the same pixels, and serving a different picture for
    each would suggest otherwise.
    """
    if name == "tile":
        return tile_png(separation.rgb)

    if name not in CHANNELS:
        raise ValueError(f"unknown panel {name!r}; expected one of {list(PANELS)}")

    channels = separation.channels_for(basis)
    values = channels.by_name(name)
    high = separation.scale[CHANNELS.index(name)]

    # The residual is signed - a pixel can need a negative amount of the direction
    # no stain occupies - so it is drawn by magnitude. The two stain channels are
    # clamped at zero rather than shown signed, because a negative amount of a dye
    # is not a faint amount of it, and the *count* of those pixels is reported in
    # the reader's own numbers rather than smuggled into a picture.
    scaled = np.abs(values) / high if name == "residual" else np.maximum(values, 0.0) / high

    return _to_png(_fit(Image.fromarray(ramp(scaled, RAMPS[name]), mode="RGB")))
