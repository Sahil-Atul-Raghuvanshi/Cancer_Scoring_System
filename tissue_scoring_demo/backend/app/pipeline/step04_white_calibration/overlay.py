"""The four panels step 4 is meant to be read as, and the array it caches.

  thumbnail    what went in
  glass        where I0 was sampled from - the patches marked in green
  field        the fitted illumination surface, as a heatmap
  corrected    the slide flat-fielded by that surface

The middle panel is the one the pipeline guide asks for by name, and it carries
the argument on its own: a reader who sees green squares only along one edge
understands why the surface fit is a weaker claim on that slide without being
told, and a reader who sees them scattered evenly understands why it is a
strong one.

The last two exist as a pair, and only the pair is honest. A heatmap of the
fitted field is a picture of a hypothesis; the same field divided back out of
the slide is the test of it. If the surface is real, the corrected panel is
visibly flatter across the glass than the thumbnail; if it is not, the two look
the same and the reader has caught the step overfitting. Step 4 reports its own
SNR test for exactly this, but a number is easier to accept than a picture is,
so both are shown.

**Green, and specifically step 3's green.** The guide says green, and there is no
reason to fight it - but it is also the same tint step 3 uses for tissue, which
would normally be the collision this codebase avoids. Here it is safe because the
two are complementary by construction: step 3 tints what it keeps, step 4 marks
what step 3 threw away. Nothing is ever green in both panels, so the shared
colour reads as "the part being used at this step" rather than as two pictures of
one thing.

Storing and showing stay separate, as in step 3: `store_rgb` never resamples,
because the RGB thumbnail is what I0 is measured from and a percentile over
averaged pixels is not the same percentile. Everything named `*_png` may.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

from .calibration import Calibration, Patch, evaluate_surface

#: The guide's green, which is also step 3's - see the module docstring.
SAMPLE_TINT = (52, 211, 153)

#: Patches that held too little glass to vote. Drawn, faintly, because where the
#: glass is not is part of the picture.
REJECTED_TINT = (120, 134, 156)

#: Longest edge of a panel served for display.
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


def _display_shape(shape: tuple[int, int]) -> tuple[int, int]:
    """`shape` scaled so its longest edge is at most `MAX_SIZE`, aspect kept.

    Used by the two panels that *generate* their pixels rather than resampling
    them. The fitted surface is an analytic quadratic in coordinates normalised to
    the frame, so evaluating it on this grid is exact rather than approximate -
    which is what makes it correct to shrink the grid instead of the picture.
    """
    height, width = shape
    scale = min(1.0, MAX_SIZE / max(1, max(height, width)))
    return (max(1, round(height * scale)), max(1, round(width * scale)))


# --- storage: byte-exact, never resampled ------------------------------------


def store_rgb(rgb: np.ndarray) -> bytes:
    """The RGB thumbnail as cached. Full resolution, lossless.

    Lossless and unresampled is not fussiness. I0 is a high percentile of these
    pixels: JPEG would move it by a few levels, and resampling would average
    neighbouring pixels, which pulls any percentile towards the mean. Either
    would leave the cached slide measuring a different white from the one the
    report quotes.
    """
    return _to_png(Image.fromarray(rgb, mode="RGB"))


def store_binary(mask: np.ndarray) -> bytes:
    """A boolean mask as cached, one bit per pixel."""
    return _to_png(Image.fromarray(mask.astype(np.uint8) * 255, mode="L").convert("1"))


# --- display -----------------------------------------------------------------


def thumbnail_png(rgb: np.ndarray) -> bytes:
    """Panel 1 - the slide as it arrived, fitted to the display size."""
    return _to_png(_fit(Image.fromarray(rgb, mode="RGB"), MAX_SIZE))


def glass_png(rgb: np.ndarray, calibration: Calibration) -> bytes:
    """Panel 2 - the sampled glass patches marked in green, over the slide.

    Three things at once, in one picture:

      the glass       every pixel that survived the four exclusions, tinted.
      the patches     a green outline per patch that voted for the surface fit.
      the rejects     a faint outline per patch that held too little glass.

    The pixel tint and the patch outlines are both drawn because they answer
    different questions. The tint shows what the flat white point was measured
    over - which is all of it - and the outlines show what the *surface* was
    fitted from, which is a much coarser and more uneven thing. Showing only the
    patches would overstate the sample; showing only the tint would hide that the
    surface has no support wherever the tissue is.
    """
    display = _fit(Image.fromarray(rgb, mode="RGB"), MAX_SIZE)

    # The glass mask, resampled to the display with the same filter as step 3's
    # overlay, so a boundary pixel is tinted in proportion to how much glass it
    # actually holds rather than by a hard in-or-out test.
    coverage = Image.fromarray(calibration.glass.glass.astype(np.uint8) * 255, mode="L").resize(
        display.size, Image.Resampling.LANCZOS
    )
    fraction = np.clip(np.asarray(coverage, dtype=np.float32) / 255.0, 0.0, 1.0)[..., None]

    base = np.asarray(display, dtype=np.float32)
    tint = np.array(SAMPLE_TINT, dtype=np.float32)

    sampled = base * 0.62 + tint * 0.38
    ignored = base * 0.5
    blended = ignored + (sampled - ignored) * fraction

    canvas = Image.fromarray(blended.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")

    scale = display.size[0] / max(1, calibration.shape[1])
    for patch in calibration.patches:
        box = (
            patch.x * scale,
            patch.y * scale,
            (patch.x + patch.width) * scale,
            (patch.y + patch.height) * scale,
        )
        if patch.used:
            draw.rectangle(box, outline=(*SAMPLE_TINT, 235), width=2)
        elif patch.glass_pixels > 0:
            # Held some glass but not enough. Worth drawing: it is the difference
            # between "no glass here" and "not enough glass here", and the second
            # is the one a reader should be suspicious of.
            draw.rectangle(box, outline=(*REJECTED_TINT, 90), width=1)

    return _to_png(canvas)


def field_png(calibration: Calibration) -> bytes:
    """Panel 3 - the fitted illumination field, as a heatmap.

    Rendered from the field's *own* range rather than from 0-255, because the
    whole point is a swing of a few percent and on an absolute scale a few
    percent is invisible. The report carries the real numbers - the swing in
    intensity units, as a fraction, and as optical density - so the stretch here
    is a way of seeing the shape, not of judging the size. The panel is captioned
    to say so.

    Returns an empty PNG-shaped grey field when no surface was fitted, so the API
    has something to serve rather than a special case. The report already says
    the mode is flat; a panel that 404s would make a reported fact look like an
    error.
    """
    if calibration.surface is None:
        flat = np.full(_display_shape(calibration.shape), 128, dtype=np.uint8)
        return _to_png(Image.fromarray(flat, mode="L"))

    # Evaluated at *display* resolution rather than the slide's. That is exact and
    # not an approximation: the surface is an analytic quadratic in coordinates
    # normalised to the frame, so sampling it on a coarser grid of the same frame
    # returns the same function. Evaluating at 4,096 px square and then throwing
    # nine tenths of it away in the fit cost six seconds a request and bought
    # nothing.
    values = evaluate_surface(calibration.surface, _display_shape(calibration.shape))

    # Luminance of the field, so one picture carries all three channels. The
    # per-channel coefficients are in the report for anyone who needs them apart.
    luminance = values @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

    low, high = float(luminance.min()), float(luminance.max())
    span = max(high - low, 1e-6)
    normalised = (luminance - low) / span

    # A two-stop ramp, dim blue to warm white: it reads as an illumination field
    # rather than as a data heatmap, and it stays legible on the app's dark ground.
    cold = np.array([28, 54, 92], dtype=np.float32)
    warm = np.array([255, 244, 214], dtype=np.float32)
    ramp = cold + (warm - cold) * normalised[..., None]

    return _to_png(Image.fromarray(ramp.astype(np.uint8), mode="RGB"))


def corrected_png(rgb: np.ndarray, calibration: Calibration) -> bytes:
    """Panel 4 - the slide flat-fielded by the field that was actually chosen.

    Each pixel is divided by the local white point and re-scaled to the field's
    own mean, so the picture keeps the slide's overall brightness and loses only
    its gradient. That is the correct thing to show: the panel is a test of
    whether the *shape* was real, and rescaling to 255 would also change the
    level, which is a different claim.

    When the flat white point is in force this divides by a constant and rescales
    by that same constant's mean, so the panel is the thumbnail again, pixel for
    pixel. That is the honest output and it is the reason the panel exists: it is
    how a reader sees that this step declined to invent a correction.
    """
    # Fit the slide first and evaluate the field onto *that* grid, rather than
    # dividing at full resolution and shrinking the result. Same reason as
    # `field_png`: the surface is analytic in normalised coordinates, so this is
    # exact, and it divides two million pixels instead of seventeen million.
    display = _fit(Image.fromarray(rgb, mode="RGB"), MAX_SIZE)
    shape = (display.size[1], display.size[0])

    if calibration.uses_surface:
        assert calibration.surface is not None
        reference = evaluate_surface(calibration.surface, shape)
    else:
        reference = np.asarray(calibration.white.rgb, dtype=np.float32)

    level = float(np.mean(reference))
    scaled = np.asarray(display, dtype=np.float32) / np.maximum(reference, 1.0) * level

    return _to_png(Image.fromarray(np.clip(scaled, 0, 255).astype(np.uint8), mode="RGB"))


def swatch_png(rgb: tuple[float, float, float], *, size: int = 96) -> bytes:
    """I0 itself, as a flat square of colour.

    Small and unglamorous, and the single most persuasive artefact this step
    produces once there are two of them side by side: two slides' swatches are
    visibly different colours, and that difference is the entire argument for
    calibrating per slide rather than once.
    """
    colour = tuple(int(round(max(0.0, min(255.0, value)))) for value in rgb)
    return _to_png(Image.new("RGB", (size, size), colour))  # type: ignore[arg-type]


def patch_bounds(patch: Patch, *, shape: tuple[int, int]) -> tuple[float, float, float, float]:
    """A patch's bounds as fractions of the slide, for drawing it in the browser.

    Fractions rather than pixels so the frontend can overlay the patch grid on
    whatever size it renders the thumbnail at, without needing to know the mask's
    resolution. The server draws the definitive version in `glass_png`; this is
    for the interactive one, where hovering a patch shows its own white point.
    """
    height, width = shape
    return (
        patch.x / width,
        patch.y / height,
        patch.width / width,
        patch.height / height,
    )
