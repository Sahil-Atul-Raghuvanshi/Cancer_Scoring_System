"""The four panels step 3 is meant to be read as, and the arrays it caches.

  thumbnail    what went in
  saturation   the channel the decision is actually made on
  mask         the decision, alone
  overlay      the decision, back over the slide

They are rendered here rather than assembled in the browser because the mask is
computed on the server and a mask redrawn from a JPEG of itself is not the same
mask. Tissue is the tinted side, matching step 2's overlay: what is highlighted
is what the next step receives.

**Storing and showing are separate, and the difference is not cosmetic.** The
service caches the thumbnail and the saturation channel on disk and re-reads
them on every threshold change. If those files were the display-sized ones, a
re-threshold would run at a different resolution from the one the report quotes,
and every physical cutoff - stated in microns and divided by that resolution -
would quietly mean something else. So `store_*` never resamples and `*_png`
always may.

The tint is deliberately not step 2's blue. Step 2's tissue map and step 3's
tissue mask answer nearly the same question by very different means, and two
pictures that look identical would invite the reader to assume one is the other.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

#: Green, to sit apart from step 2's blue tissue map and GrandQC's artefact
#: palette, neither of which this mask is.
TISSUE_TINT = (52, 211, 153)

TISSUE_ALPHA = 0.42

#: Longest edge of a panel served for display. Nothing in the UI shows more than
#: a couple of thousand pixels, and the mask can be twice that.
MAX_SIZE = 1600


def _to_png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _fit(image: Image.Image, longest: int) -> Image.Image:
    width, height = image.size
    scale = min(1.0, longest / max(width, height))
    if scale >= 1.0:
        return image
    return image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.LANCZOS,
    )


# --- storage: byte-exact, never resampled ------------------------------------


def store_grey(values: np.ndarray) -> bytes:
    """A uint8 channel as cached. Full resolution, lossless."""
    return _to_png(Image.fromarray(values, mode="L"))


def store_binary(mask: np.ndarray) -> bytes:
    """A boolean mask as cached, one bit per pixel."""
    return _to_png(Image.fromarray(mask.astype(np.uint8) * 255, mode="L").convert("1"))


# --- display -----------------------------------------------------------------


def thumbnail_png(thumbnail: np.ndarray) -> bytes:
    """Panel 1 - the slide as it arrived, fitted to the display size.

    This is also the form the service caches, because it is threshold-independent
    and resampling a 4,096 px thumbnail on every request costs more than the rest
    of the overlay put together. Unlike the saturation channel, this is display
    only - nothing is measured from it - so a resampled copy is not the trap it
    would be there.
    """
    return _to_png(_fit(Image.fromarray(thumbnail, mode="RGB"), MAX_SIZE))


def saturation_png(saturation: np.ndarray) -> bytes:
    """Panel 2 - the saturation channel, greyscale, unstretched.

    Not contrast-stretched on purpose. The histogram panel beside it is drawn
    from these same values, so brightening the image here would make the picture
    and the histogram disagree about where the two humps are.
    """
    return _to_png(_fit(Image.fromarray(saturation, mode="L"), MAX_SIZE))


def mask_png(mask: np.ndarray) -> bytes:
    """Panel 3 - the binary mask on its own, white for tissue.

    Served at the mask's own resolution and one bit per pixel, so this is the
    artefact to download and hand to another tool alongside the slide. Fitting
    it to the display size would make the downloadable thing disagree with the
    numbers reported next to it.
    """
    return store_binary(mask)


def overlay_png(
    display: np.ndarray,
    mask: np.ndarray,
    *,
    alpha: float = TISSUE_ALPHA,
) -> bytes:
    """Panel 4 - the mask back over the slide, tissue tinted, glass dimmed.

    `display` is the already-fitted thumbnail from `thumbnail_png`; the mask is
    resampled down to match it with the same filter. So what the blend receives
    is the *fraction* of each display pixel the mask covers, and a boundary pixel
    is tinted in proportion to how much tissue it actually holds. That is a truer
    account of a mask viewed below its own resolution than a hard in-or-out edge,
    and it is what makes this cheap enough to re-render on every threshold
    change: the arithmetic is over the display's two million pixels rather than
    the mask's seventeen.
    """
    base = Image.fromarray(display, mode="RGB")
    coverage = Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
        base.size, Image.Resampling.LANCZOS
    )

    # Lanczos overshoots at a hard edge, so a fraction can land outside 0..1.
    fraction = np.clip(np.asarray(coverage, dtype=np.float32) / 255.0, 0.0, 1.0)[..., None]

    rgb = np.asarray(base, dtype=np.float32)
    tint = np.array(TISSUE_TINT, dtype=np.float32)

    tissue = rgb * (1.0 - alpha) + tint * alpha
    glass = rgb * 0.55
    blended = glass + (tissue - glass) * fraction

    return _to_png(Image.fromarray(blended.astype(np.uint8), mode="RGB"))
