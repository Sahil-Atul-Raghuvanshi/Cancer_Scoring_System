"""The five panels step 5 is meant to be read as.

  map        where on the slide this tile is, and which tiles were considered
  tile       the RGB that went in
  density    the same tile as a density heatmap - the physical quantity
  scatter    the densities as a point cloud, with the two arms in it
  limits     the pixels where the transform is not a measurement

The middle three are the pipeline guide's triptych, in its order, and the
sequence carries the argument by itself: colour, then how much stain, then the
geometry that makes un-mixing possible. The reader does not have to be told that
the third picture explains the next step - two arms and two published stain
vectors lying along them says it.

**Nothing here uses a rainbow.** A density is an ordered quantity, and a
multi-hue ramp adds boundaries the data does not have - a reader sees the green
band as a feature when it is an artefact of the palette. Both heatmaps are single
progressions from the page's own ground through one hue to white, so more ink on
screen means more ink on the slide and nothing else.

**The heatmaps are stretched, and say so.** The range drawn is zero to the tile's
99th percentile, not zero to its maximum: one saturated pixel would otherwise
flatten the rest of the picture into the bottom eighth of the ramp. The stretch is
reported as a number beside the panel, so the picture is a way of seeing the shape
and the report is where the size is read.

**Negative densities are not drawn as zero.** A pixel brighter than I0 has
negative density, and clamping it into the ramp would make an impossible reading
look like a faint one. The density panel clamps at zero for legibility and the
`limits` panel marks every clamped pixel, so the reader is never shown a
correction they were not told about.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

from .density import Density, direction_is_stable
from .tiles import Candidate

#: Longest edge of a panel served for display. Step 4's number, for the same
#: reason: a browser drawing this at 500 px does not need four thousand.
MAX_SIZE = 1600

#: Edge of one candidate's thumbnail on the contact sheet, in pixels. Sets the
#: resolution the block is read at - `tile_um / CANDIDATE_SIZE` - so it is a
#: statement about how much of a field of view a reader needs to see to judge it,
#: not a display size. At 192 a `tile_um` block reads off a coarse pyramid level
#: in a few milliseconds and still resolves nuclei as nuclei.
CANDIDATE_SIZE = 192

#: Side of the scatter panel. Square by necessity - the plot's two axes carry the
#: same scale, because the angles on it are the content.
SCATTER_SIZE = 960

#: Bins per axis in the scatter's density grid. 320 over 960 px is three panel
#: pixels a bin: fine enough to show the wedge's edges, coarse enough that a
#: quarter of a million points fill it rather than speckling it. The browser draws
#: this without smoothing, on purpose - a bin is a fact about the data, and
#: interpolating between bins would draw density that was never counted.
SCATTER_BINS = 320

#: The page's own ground, so a panel sits on the app rather than on a white card.
GROUND = (10, 14, 21)

#: The density ramp: ground, deep blue, the app's accent cyan, warm white. One
#: progression, no hue reversals - see the module docstring.
DENSITY_RAMP: tuple[tuple[float, tuple[int, int, int]], ...] = (
    (0.00, GROUND),
    (0.25, (23, 47, 92)),
    (0.62, (56, 189, 248)),
    (1.00, (246, 252, 255)),
)

#: The scatter's ramp. Warmer than the density one on purpose: the scatter is a
#: plot and not a picture of the slide, and the two should not be mistaken for
#: each other at a glance.
CLOUD_RAMP: tuple[tuple[float, tuple[int, int, int]], ...] = (
    (0.00, GROUND),
    (0.22, (46, 34, 84)),
    (0.58, (198, 114, 74)),
    (1.00, (255, 240, 214)),
)

#: Arms, references, and the four ways a density stops being a measurement.
ARM_TINT = (56, 189, 248)
REFERENCE_TINT = (232, 238, 248)
AXIS_TINT = (52, 66, 90)
FLOOR_TINT = (248, 113, 113)
NEGATIVE_TINT = (251, 191, 36)
TRANSPARENT_TINT = (110, 124, 148)
UNSTABLE_TINT = (167, 139, 250)
CHOSEN_TINT = (56, 189, 248)
CANDIDATE_TINT = (120, 134, 156)


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


def _ramp(
    normalised: np.ndarray, stops: tuple[tuple[float, tuple[int, int, int]], ...]
) -> np.ndarray:
    """Map values in [0, 1] through a piecewise-linear colour ramp.

    Written out rather than pulled from a plotting library because the panels are
    the only thing that would need one, and a ramp defined here is a ramp a reader
    of this file can check against what they see on screen.
    """
    values = np.clip(np.asarray(normalised, dtype=np.float32), 0.0, 1.0)
    positions = np.array([stop for stop, _ in stops], dtype=np.float32)
    colours = np.array([colour for _, colour in stops], dtype=np.float32)

    output = np.empty(values.shape + (3,), dtype=np.float32)
    for channel in range(3):
        output[..., channel] = np.interp(values, positions, colours[:, channel])
    return output.astype(np.uint8)


# --- 1. where the tile is ----------------------------------------------------


def map_png(
    thumbnail: np.ndarray,
    *,
    candidates: list[Candidate],
    chosen: Candidate,
) -> bytes:
    """Panel 1 - the slide, with the tile drawn on it and its rivals faint.

    Both are drawn, and the faint ones matter as much as the bright one. A reader
    shown only the chosen tile has to take the choice on trust; a reader shown the
    twelve best sees that the winner sits inside tissue among other tissue, and
    that the score picked a field of view rather than an edge or a fold.
    """
    display = _fit(Image.fromarray(thumbnail, mode="RGB"))
    # Dimmed so the drawn boxes read as an overlay and not as part of the slide.
    faded = (np.asarray(display, dtype=np.float32) * 0.72).astype(np.uint8)
    canvas = Image.fromarray(faded, mode="RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")

    width, height = canvas.size

    def box(candidate: Candidate) -> tuple[float, float, float, float]:
        return (
            candidate.fx * width,
            candidate.fy * height,
            (candidate.fx + candidate.fw) * width,
            (candidate.fy + candidate.fh) * height,
        )

    for candidate in candidates:
        if candidate.col == chosen.col and candidate.row == chosen.row:
            continue
        draw.rectangle(box(candidate), outline=(*CANDIDATE_TINT, 110), width=1)

    draw.rectangle(box(chosen), fill=(*CHOSEN_TINT, 46), outline=(*CHOSEN_TINT, 245), width=3)
    return _to_png(canvas)


def candidate_png(rgb: np.ndarray) -> bytes:
    """One candidate block, for the contact sheet. Unadjusted, like the tile.

    Read off the pyramid rather than cropped out of the screening thumbnail,
    which is the opposite of what cost would suggest and is the only version that
    works. The thumbnail is step 3's grid, and step 3 caps its own longest edge -
    on a large scan that leaves a `tile_um` block about forty pixels across, and
    forty pixels upscaled to a contact-sheet square is a coloured smudge. The
    reader is being asked to judge *what is in a field of view* before clicking
    it, and cannot do that from a smudge.

    Written without `optimize` deliberately. These are photographs, where the flag
    buys a couple of percent for four times the encode - and twelve of them are
    encoded in one request.
    """
    buffer = BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


# --- 2. what went in ---------------------------------------------------------


def tile_png(rgb: np.ndarray) -> bytes:
    """Panel 2 - the tile as it was read. No adjustment of any kind.

    Deliberately untouched. Every other panel here is derived from these pixels,
    so a contrast curve applied for legibility would make the reader compare a
    density against a picture that is not its input.
    """
    return _to_png(_fit(Image.fromarray(rgb, mode="RGB")))


# --- 3. how much stain -------------------------------------------------------


def density_png(density: Density) -> bytes:
    """Panel 3 - the mean optical density as a heatmap.

    The guide asks for exactly this panel, and the thing to notice about it is not
    the colour but that the *nuclei* are bright. In the RGB tile a dark nucleus and
    a dark shadow look alike; in density they do not, because density is a physical
    amount of absorber and a shadow has none. That is the whole reason both
    branches of the pipeline fork from here rather than from the RGB.
    """
    mean_od = density.mean_od
    high = max(density.stats.mean_p99, 1e-3)
    return _to_png(_fit(Image.fromarray(_ramp(mean_od / high, DENSITY_RAMP), mode="RGB")))


# --- 4. the two arms ---------------------------------------------------------


def _dashed(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    fill: tuple[int, int, int, int],
    width: int,
    dash: float = 9.0,
) -> None:
    """A dashed segment, because PIL draws only solid ones.

    Dashes are load-bearing here, not decoration: the arms are measured from this
    slide and the reference vectors are published constants, and the reader has to
    be able to tell which is which without reading a legend.
    """
    span = np.hypot(end[0] - start[0], end[1] - start[1])
    if span < 1e-6:
        return

    steps = max(1, int(span / dash))
    for index in range(steps):
        if index % 2:
            continue
        a = index / steps
        b = min(1.0, (index + 1) / steps)
        draw.line(
            (
                start[0] + (end[0] - start[0]) * a,
                start[1] + (end[1] - start[1]) * a,
                start[0] + (end[0] - start[0]) * b,
                start[1] + (end[1] - start[1]) * b,
            ),
            fill=fill,
            width=width,
        )


def scatter_png(density: Density) -> bytes:
    """Panel 4 - the densities as a point cloud, with the arms drawn on it.

    A two-dimensional histogram and not a scatter of marks, because there are a
    quarter of a million points and overplotting would turn the wedge into a solid
    block whose *shape* - the one thing being shown - is invisible. Counts are
    log-scaled for the same reason: the origin end of each ray holds orders of
    magnitude more pixels than the far end, and on a linear scale the arms would
    be a bright dot at the origin and nothing else.

    Drawn self-contained - origin cross, arms solid, published vectors dashed - so
    the panel means something on its own. The report carries the same geometry in
    plot coordinates, and the browser redraws it labelled and interactive over
    this image; the two cannot disagree because both come from `Cloud`.

    Returns a plain ground when no cloud could be built. That is a real state - a
    tile with no stain has no arms - and the report says so in words; a panel that
    404s would make a reported fact look like a failure.
    """
    cloud = density.cloud
    if cloud is None:
        return _to_png(Image.new("RGB", (SCATTER_SIZE, SCATTER_SIZE), GROUND))

    counts, _, _ = np.histogram2d(
        cloud.points[:, 0],
        cloud.points[:, 1],
        bins=SCATTER_BINS,
        range=((cloud.x_low, cloud.x_high), (cloud.y_low, cloud.y_high)),
    )

    scaled = np.log1p(counts)
    peak = float(scaled.max())
    normalised = scaled / peak if peak > 0 else scaled

    # Transposed and flipped: histogram2d indexes [x, y] with y increasing upward,
    # an image indexes [row, column] with row increasing downward.
    grid = _ramp(np.flipud(normalised.T), CLOUD_RAMP)

    canvas = Image.fromarray(grid, mode="RGB").resize(
        (SCATTER_SIZE, SCATTER_SIZE), Image.Resampling.NEAREST
    )
    draw = ImageDraw.Draw(canvas, "RGBA")

    def to_pixels(point: tuple[float, float]) -> tuple[float, float]:
        x_span = max(cloud.x_high - cloud.x_low, 1e-9)
        y_span = max(cloud.y_high - cloud.y_low, 1e-9)
        return (
            (point[0] - cloud.x_low) / x_span * SCATTER_SIZE,
            (1.0 - (point[1] - cloud.y_low) / y_span) * SCATTER_SIZE,
        )

    origin = to_pixels((0.0, 0.0))

    # The origin is zero stain, and every ray starts there. Marking it is not
    # chrome: without it the wedge is a shape floating in a box, and with it the
    # picture is two rays from "no stain" - which is the geometry step 6 inverts.
    draw.line((0, origin[1], SCATTER_SIZE, origin[1]), fill=(*AXIS_TINT, 190), width=1)
    draw.line((origin[0], 0, origin[0], SCATTER_SIZE), fill=(*AXIS_TINT, 190), width=1)

    for reference in cloud.references:
        _dashed(draw, origin, to_pixels(reference.plot), fill=(*REFERENCE_TINT, 150), width=2)

    for arm in cloud.arms:
        draw.line((*origin, *to_pixels(arm.plot)), fill=(*ARM_TINT, 240), width=3)

    return _to_png(canvas)


# --- 5. where it is not a measurement ----------------------------------------


def limits_png(density: Density) -> bytes:
    """Panel 5 - the pixels whose density is not a reading, marked by reason.

    Four marks, and they are four different faults rather than four shades of one:

      grey    the pixel carries almost no stain, so its density has a magnitude
              but no trustworthy direction. Kept out of the scatter.
      violet  the opposite case: so dark that a channel has almost nothing left,
              where one 8-bit level swings the direction by more than the
              tolerance. Also kept out of the scatter, for the mirror-image
              reason - and worth its own colour precisely because a reader would
              otherwise assume "excluded" meant "faint".
      amber   the pixel is brighter than I0, so its density is negative. Negative
              stain does not exist; I0 is slightly low here.
      red     a channel hit the intensity floor. No light was recorded, so the
              density is a lower bound - the true value is larger and unknowable.

    Drawn in that order, so the rarer and more serious faults paint over the
    commoner ones rather than being buried under them.

    Drawn over a heavily dimmed tile so the marks are locatable rather than
    abstract - a reader can see whether the amber sits on glass inside the tile,
    which is the benign case, or across the tissue, which is not, and whether the
    violet traces the dark membranes, which is expected, or scatters everywhere,
    which is a scan with its shadows crushed.

    This panel exists because the alternative is a step that silently repairs its
    own inputs. Every mark here is a place a later step must not treat as data,
    and the only way to keep that promise is to say where they are.
    """
    base = (np.asarray(density.rgb, dtype=np.float32) * 0.30).astype(np.float32)

    mean_od = density.mean_od
    observed = np.maximum(density.rgb.astype(np.float32), np.float32(density.floor))
    floor_hit = np.any(density.rgb.astype(np.float32) < density.floor, axis=-1)
    negative = mean_od < 0.0
    transparent = (mean_od >= 0.0) & (mean_od < density.limits.beta)
    unstable = (mean_od >= density.limits.beta) & ~direction_is_stable(
        density.od, observed, tolerance_deg=density.tolerance_deg
    )

    for mask, tint, weight in (
        (transparent, TRANSPARENT_TINT, 0.42),
        (unstable, UNSTABLE_TINT, 0.72),
        (negative, NEGATIVE_TINT, 0.78),
        (floor_hit, FLOOR_TINT, 0.88),
    ):
        if not np.any(mask):
            continue
        colour = np.array(tint, dtype=np.float32)
        base[mask] = base[mask] * (1.0 - weight) + colour * weight

    return _to_png(_fit(Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")))
