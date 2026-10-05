"""Step 3's algorithm: is there tissue at this pixel, or is it empty glass?

Four moves, in this order, and each one is here for a stated reason:

  1. work at ~2 um/px. Tissue-versus-glass is a question about millimetres of
     tissue, so full resolution buys nothing and costs a thousandfold.
  2. threshold the *saturation* channel, not brightness. Glass is bright and
     colourless, pale tissue is dim and coloured; they share brightness and do
     not share saturation.
  3. pick the cut from the histogram rather than by hand, so the number is a
     property of this slide and not of whoever tuned it. Which *rule* does the
     picking depends on the histogram's shape - see `choose_threshold`.
  4. clean up with morphology, with every distance stated in microns and every
     area in mm^2, so nothing here changes meaning on a different scanner.

Two rules from the pipeline guide are load-bearing and are enforced here rather
than left to a comment:

  Rule 1 - fat is tissue. Nothing in this module removes a region for being
  pale. Fat leaves the analysis at step 8, as a named class the viewer can
  toggle, never as a brightness threshold nobody can audit. What this module
  does instead is `_fill_small_holes`, which reclaims interior low-saturation
  regions - which is mostly what fat is - and reports exactly how much it
  reclaimed so the reclaim is auditable too.

  Order - QC runs first. Pen ink is darker and far more saturated than any
  stain, so a saturation threshold does not merely include a pen mark, it lets
  the pen mark drag the threshold. So step 2's artefact map is subtracted
  *before* the histogram is built, not after the mask is made.

One thing the pipeline guide does not anticipate, and which this slide taught
us: Otsu is not always the right rule. On a CD44 slide with a weak haematoxylin
counterstain the section is so faintly coloured that three quarters of the frame
sits at saturation exactly zero, and the rest is a monotone tail with no valley
in it. Otsu's criterion is then maximised far out in that tail - measured against
step 2's tissue map, it kept 33% of the section. Zack's triangle rule, which is
derived for exactly that spike-and-tail shape, kept 95%. So both rules are
computed, the shape of the histogram decides which is used, and the report
carries the other one plus the statistic the decision turned on.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from PIL import Image

from app.common.imaging import (
    between_class_variance,
    histogram_256,
    modal_share,
    otsu_from_histogram,
    triangle_from_histogram,
)


class TissueMaskError(ValueError):
    """A precondition step 3 cannot supply for itself."""


# --- reading the slide -------------------------------------------------------


def thumbnail_at_mpp(
    reader: Any, *, base_mpp: float, target_mpp: float, max_px: int
) -> tuple[np.ndarray, float]:
    """A whole-slide thumbnail at roughly `target_mpp`, plus the mpp achieved.

    `max_px` caps the longest edge, and the cap really does bind: a 28 mm slide
    at 2 um/px is 14,000 px across, and the morphology below is iterative, so
    the whole step would cost more than the models it exists to make cheaper.
    Small specimens land at the requested resolution; large ones land coarser.
    Either way the achieved mpp is returned and reported, because every physical
    threshold in this module is divided by it.
    """
    width, height = reader.dimensions
    wanted = max(1, int(round(max(width, height) * base_mpp / target_mpp)))
    longest = min(wanted, max_px)

    image = reader.thumbnail_pil(longest).convert("RGB")
    achieved = base_mpp * (width / max(1, image.width))
    return np.asarray(image), achieved


def artefact_footprint(
    artefact_mask: np.ndarray, *, artefact_ids: tuple[int, ...], shape: tuple[int, int]
) -> np.ndarray:
    """Step 2's artefacts, resampled onto step 3's grid.

    Nearest-neighbour, never bilinear: these are class ids, and the average of
    "pen" and "clean tissue" is not a class. The mask arrives padded to the
    slide's own proportions, so a plain resize aligns the two grids.
    """
    flagged = np.isin(artefact_mask, artefact_ids).astype(np.uint8)
    resized = Image.fromarray(flagged).resize(
        (shape[1], shape[0]), Image.Resampling.NEAREST
    )
    return np.asarray(resized).astype(bool)


# --- the cleanup -------------------------------------------------------------


@dataclass(frozen=True)
class Cleanup:
    """One move of the cleanup, and the mask it left behind.

    Recorded per move rather than only at the end, because the four panels the
    demo shows are useless without the arithmetic between them: the viewer
    should be able to see that closing added area and opening took it away, and
    by how much.
    """

    key: str
    label: str
    what: str
    #: Extent in microns for a morphological move, else None.
    extent_um: float | None
    pixels: int


@dataclass(frozen=True)
class Components:
    """What connected-component filtering kept and dropped."""

    found: int
    kept: int
    dropped: int
    kept_pixels: int
    dropped_pixels: int
    largest_pixels: int
    min_area_mm2: float
    min_area_px: int


@dataclass(frozen=True)
class Choice:
    """Which threshold rule was used, what the alternatives said, and why.

    Both rules are always computed and always reported. The step is allowed to
    pick one, but it is not allowed to pick one quietly: the reader gets the
    other number, the shape statistic the choice turned on, and the cutoff it
    was compared against.
    """

    value: int
    #: "otsu", "triangle" or "manual".
    rule: str
    otsu: int
    triangle: int
    #: The busiest level and its share of the histogram - the shape statistic.
    modal_level: int
    modal_share: float
    #: Share above which the histogram counts as a spike rather than a hump.
    spike_share: float
    #: The chord the triangle rule measured against, for drawing it.
    triangle_from: int
    triangle_to: int

    @property
    def bimodal(self) -> bool:
        """Whether Otsu's two-hump assumption holds well enough to use it."""
        return self.modal_share < self.spike_share


def choose_threshold(histogram: np.ndarray, *, spike_share: float) -> Choice:
    """Pick a threshold rule from the histogram's shape, and show your working.

    Otsu when the histogram is two humps, which is what it was derived for.
    Zack's triangle when it is one spike and a tail, which is what a weakly
    counterstained IHC slide produces and what Otsu cannot read - there, Otsu's
    criterion is maximised far out in the tail and the mask loses most of the
    section.

    The test is the modal level's share of the mass. It is the assumption
    violation stated directly: a level holding most of the pixels is a class
    with no variance, and Otsu's criterion is a ratio of variances.
    """
    counts = np.asarray(histogram, dtype=np.float64)
    otsu = otsu_from_histogram(counts)
    triangle = triangle_from_histogram(counts)
    level, share = modal_share(counts)

    occupied = np.nonzero(counts)[0]
    chord_to = int(occupied[-1]) if occupied.size else 0

    choice = Choice(
        value=otsu,
        rule="otsu",
        otsu=otsu,
        triangle=triangle,
        modal_level=level,
        modal_share=share,
        spike_share=spike_share,
        triangle_from=level,
        triangle_to=chord_to,
    )
    if choice.bimodal:
        return choice

    return replace(choice, value=triangle, rule="triangle")


@dataclass
class TissueMask:
    """Everything step 3 produced for one slide at one threshold."""

    #: The final mask, at `mpp`.
    mask: np.ndarray
    #: The saturation channel it was thresholded from, uint8.
    saturation: np.ndarray
    #: Pixels QC left in play - the histogram's domain, not the whole image.
    considered: np.ndarray
    mpp: float

    #: The cut used, and both rules' answers alongside it.
    choice: Choice
    #: 256 counts over `considered` only.
    histogram: np.ndarray
    #: Otsu's criterion at every level, so the demo can draw the curve.
    criterion: np.ndarray

    stages: list[Cleanup] = field(default_factory=list)
    components: Components | None = None
    holes_filled_pixels: int = 0
    artefact_pixels: int = 0

    @property
    def tissue_pixels(self) -> int:
        return int(np.count_nonzero(self.mask))


def _radius_px(extent_um: float, mpp: float) -> int:
    """A physical extent as a pixel radius on this grid, at least 1 px."""
    return max(1, int(round(extent_um / mpp)))


def _area_px(area_mm2: float, mpp: float) -> int:
    """An area in mm^2 as a pixel count on this grid."""
    return max(1, int(round(area_mm2 * 1_000_000 / (mpp * mpp))))


def _filter_components(
    mask: np.ndarray, *, min_px: int, min_area_mm2: float
) -> tuple[np.ndarray, Components]:
    """Keep components at or above `min_px`; report what went.

    A dust speck and a genuine 200 um tissue fragment are both small islands,
    so the cutoff is a judgement, and it is stated in mm^2 for exactly that
    reason - "half a hundredth of a square millimetre" is arguable, "40 pixels"
    is not even checkable without knowing the scanner.
    """
    from scipy import ndimage

    labels, found = ndimage.label(mask)
    empty = Components(
        found=0,
        kept=0,
        dropped=0,
        kept_pixels=0,
        dropped_pixels=0,
        largest_pixels=0,
        min_area_mm2=min_area_mm2,
        min_area_px=min_px,
    )
    if found == 0:
        return mask, empty

    sizes = np.bincount(labels.ravel(), minlength=found + 1)
    sizes[0] = 0  # label 0 is the background, not a component

    keep = sizes >= min_px
    keep[0] = False

    return keep[labels], Components(
        found=int(found),
        kept=int(keep.sum()),
        dropped=int(found - keep.sum()),
        kept_pixels=int(sizes[keep].sum()),
        dropped_pixels=int(sizes[1:][~keep[1:]].sum()),
        largest_pixels=int(sizes.max()),
        min_area_mm2=min_area_mm2,
        min_area_px=min_px,
    )


def _fill_small_holes(mask: np.ndarray, *, max_px: int) -> tuple[np.ndarray, int]:
    """Fill enclosed low-saturation regions below `max_px`. Returns the mask and area filled.

    This is Rule 1 made mechanical. A fat lobule inside a block of tissue is
    pale, so the saturation threshold punches it out; leaving it out would be
    removing fat at the tissue-mask step, which is the mistake the guide names
    first. Filling it back in keeps the tissue footprint whole.

    Two limits keep this from becoming a licence to invent tissue. A hole that
    touches the image border is not enclosed - it is the outside world, reached
    around the specimen - so it is never filled. And a hole above `max_px` is
    not a vacuole, it is a genuine gap between two pieces of tissue, so it is
    left alone. Fat at the *edge* of the section is still lost, and no amount of
    morphology recovers it; that is the honest limit of a threshold, and it is
    why fat is a semantic class at step 8 rather than a geometry problem here.
    """
    from scipy import ndimage

    holes, found = ndimage.label(~mask)
    if found == 0:
        return mask, 0

    sizes = np.bincount(holes.ravel(), minlength=found + 1)

    outside = np.unique(
        np.concatenate([holes[0, :], holes[-1, :], holes[:, 0], holes[:, -1]])
    )

    fillable = sizes <= max_px
    fillable[0] = False
    fillable[outside[outside > 0]] = False

    filled = fillable[holes]
    return mask | filled, int(np.count_nonzero(filled))


def build_mask(
    *,
    saturation: np.ndarray,
    considered: np.ndarray,
    mpp: float,
    threshold: int | None,
    close_um: float,
    open_um: float,
    min_component_mm2: float,
    fill_hole_max_mm2: float,
    spike_share: float,
) -> TissueMask:
    """Threshold and clean up, recording the mask after every move.

    `considered` is the set of pixels step 2 left in play. It bounds the
    histogram as well as the mask, which is the whole point of running QC first:
    a pen mark left in the histogram pulls the cut towards the ink and, on a
    faintly stained slide, can push it clean past the tissue.

    `threshold` of None means "let the histogram's shape choose the rule" - see
    `choose_threshold`. Passing one makes the run a manual comparison. Either
    way both rules' answers are computed and reported, so a hand-set cut can
    never appear on screen without the automatic ones beside it.
    """
    from scipy import ndimage

    histogram = histogram_256(saturation[considered])
    criterion = between_class_variance(histogram)
    choice = choose_threshold(histogram, spike_share=spike_share)

    if threshold is not None:
        choice = replace(choice, value=max(0, min(255, int(threshold))), rule="manual")

    cut = choice.value
    stages: list[Cleanup] = []

    mask = (saturation > cut) & considered
    stages.append(
        Cleanup(
            key="threshold",
            label="Keep the coloured pixels",
            what=(
                f"Keeps every pixel with more colour than {cut} out of 255. Empty glass is"
                " bright but has almost no colour, so it drops out here."
            ),
            extent_um=None,
            pixels=int(np.count_nonzero(mask)),
        )
    )

    # A 3x3 structure applied N times dilates by an N-pixel radius, and is
    # markedly faster than building one large element - the same trick step 2's
    # feature pass uses, and for the same reason.
    square = np.ones((3, 3), dtype=bool)

    close_px = _radius_px(close_um, mpp)
    mask = ndimage.binary_closing(mask, structure=square, iterations=close_px)
    stages.append(
        Cleanup(
            key="closing",
            label="Close small gaps",
            what=(
                "Joins up narrow gaps, so one piece of tissue is not counted as several"
                " separate pieces."
            ),
            extent_um=close_um,
            pixels=int(np.count_nonzero(mask)),
        )
    )

    open_px = _radius_px(open_um, mpp)
    mask = ndimage.binary_opening(mask, structure=square, iterations=open_px)
    stages.append(
        Cleanup(
            key="opening",
            label="Remove specks",
            what=(
                "Removes tiny dots such as dust and stain splashes. They have colour, but"
                " they are not tissue."
            ),
            extent_um=open_um,
            pixels=int(np.count_nonzero(mask)),
        )
    )

    min_px = _area_px(min_component_mm2, mpp)
    mask, components = _filter_components(
        mask, min_px=min_px, min_area_mm2=min_component_mm2
    )
    stages.append(
        Cleanup(
            key="components",
            label="Drop tiny pieces",
            what=(
                f"Keeps only pieces of at least {min_component_mm2} mm2. "
                f"Dropped {components.dropped} of {components.found}."
            ),
            extent_um=None,
            pixels=int(np.count_nonzero(mask)),
        )
    )

    fill_px = _area_px(fill_hole_max_mm2, mpp)
    mask, filled = _fill_small_holes(mask, max_px=fill_px)
    stages.append(
        Cleanup(
            key="fill",
            label="Fill small holes",
            what=(
                f"Fills pale holes smaller than {fill_hole_max_mm2} mm2, such as fat and"
                " gland spaces. These are still tissue, so they are put back."
            ),
            extent_um=None,
            pixels=int(np.count_nonzero(mask)),
        )
    )

    return TissueMask(
        mask=mask,
        saturation=saturation,
        considered=considered,
        mpp=mpp,
        choice=choice,
        histogram=histogram,
        criterion=criterion,
        stages=stages,
        components=components,
        holes_filled_pixels=filled,
        artefact_pixels=int(considered.size - np.count_nonzero(considered)),
    )
