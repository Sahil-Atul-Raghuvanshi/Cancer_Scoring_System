"""What the segmenter is shown: the counterstain, with the DAB taken out.

**The rule this module exists to enforce.** On an IHC slide the brown is the
thing being measured, so it must not decide where cells are thought to be.
Detect on the haematoxylin and the denominator is independent of the numerator;
detect on raw RGB and a strongly stained cell becomes a more findable cell,
which biases the percentage upward by a mechanism no later step can see or
correct. That is a feedback loop, not a bias term, and it is the reason this is
the first thing in the package.

**But the model takes three channels, not one.** InstanSeg's declared input is
RGB at 0.5 um/px, trained on brightfield sections. Handing it a single-channel
density map, or the same map replicated three times, is out of distribution in a
way nobody has measured. So "run it on the H channel" is implemented as: un-mix
the tile, take the haematoxylin amount, and *draw it back* as a blue-on-white
brightfield image of the same field. That is both what the rule asks for and what
the model was trained on. `step08.../input.py` makes the same argument one step
earlier for its own reasons; this is its counterpart for a model whose input
contract is fixed by somebody else.

**Which basis un-mixes, and a measurement that contradicts the guide.** The guide
recommends estimating stain vectors per slide with Macenko, on the reasoning that
a fixed matrix is at its worst when one stain dominates - which is exactly this
panel. Measured on CAN_00270's CD44 slide over six fields, it is the other way
round: the fixed Ruifrok basis finds **695** nuclei against Macenko's **581**, a
20% shortfall.

The reason is visible in the estimated vectors rather than inferred. Macenko
returns the two angular extremes of the density cloud, and on this slide they are
DAB - recovered well, 3.3 degrees off the published direction - and an
**achromatic** arm at (0.577, 0.577, 0.577), 18.7 degrees off haematoxylin. With
a weak counterstain under heavy DAB there is no second *colour* in the cloud to
find, so the method returns the darkness axis instead. A hybrid taking DAB from
Macenko and haematoxylin from Ruifrok was tried and scored 574, no better.

So the condition the guide says Macenko protects against - one stain dominating -
is the condition that breaks Macenko's estimate of the *other* stain. The default
is therefore the fixed basis, `nuclei_macenko_per_slide = False`; the estimate is
still computed and reported as a diagnostic, because its drift is the evidence
for this paragraph and it may well go the other way on a marker with a stronger
counterstain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.common import stains
from app.common.imaging import optical_density
from app.core.config import settings

#: Index of each stain in a three-column H-DAB basis, per `stains.RUIFROK_CHANNELS`.
HAEMATOXYLIN, DAB, RESIDUAL = 0, 1, 2

#: The direction a haematoxylin amount is *drawn* in, always. See
#: `haematoxylin_only_rgb` for why this is fixed even when the basis is not.
_H_DIRECTION = np.asarray(stains.REFERENCE_BY_NAME["haematoxylin"], dtype=np.float64)


@dataclass(frozen=True)
class StainBasis:
    """The 3x3 matrix a slide's tiles are un-mixed with, and where it came from."""

    matrix: np.ndarray
    inverse: np.ndarray
    #: "macenko" or "ruifrok". Reported, because it changes what the H channel is.
    source: str
    #: How planar the density cloud really was. Macenko assumes two absorbers, and
    #: this is the honesty term on that assumption. None for the fixed basis.
    explained: float | None = None
    #: Angle between the estimated haematoxylin direction and Ruifrok's published
    #: one, in degrees. A large value is not necessarily wrong - it is how far this
    #: slide's counterstain sits from the textbook - but it is worth seeing.
    haematoxylin_drift_deg: float | None = None


def ruifrok_basis() -> StainBasis:
    """Ruifrok's published H-DAB matrix, closed to three columns."""
    return StainBasis(
        matrix=stains.RUIFROK_HDAB,
        inverse=stains.RUIFROK_INVERSE,
        source="ruifrok",
    )


def estimate_basis(
    od_flat: np.ndarray,
    *,
    percentile: float | None = None,
    alpha: float = 1.0,
) -> StainBasis:
    """Macenko on this slide's own pixels, closed to an invertible 3x3.

    `od_flat` is an Nx3 sample of optical density from anywhere on the slide -
    the caller decides what to sample, because "which pixels carry enough stain
    to have a direction" is a judgement about the acquisition and belongs with
    whoever knows how the sample was drawn.

    Falls back to the published basis, saying so, when the sample cannot support
    an estimate. That is not a silent repair: `source` comes back as "ruifrok"
    and the report carries it, so a viewer can see that the per-slide estimate
    was not used rather than being told a number that was not computed.
    """
    sample = np.asarray(od_flat, dtype=np.float64)
    if sample.ndim != 2 or sample.shape[1] != 3 or len(sample) < 512:
        return ruifrok_basis()

    cut = settings.nuclei_macenko_percentile if percentile is None else percentile
    strength = sample.mean(axis=1)
    admitted = sample[strength >= np.percentile(strength, cut)]
    if len(admitted) < 512:
        return ruifrok_basis()

    try:
        pair, explained = stains.estimate_stain_matrix(admitted, alpha=alpha)
        matrix = stains.complete_basis(pair)
        inverse = np.linalg.inv(matrix)
    except (ValueError, np.linalg.LinAlgError):
        return ruifrok_basis()

    drift = stains.degrees_between(
        matrix[:, HAEMATOXYLIN],
        np.asarray(stains.REFERENCE_BY_NAME["haematoxylin"], dtype=np.float64),
    )
    return StainBasis(
        matrix=matrix,
        inverse=inverse,
        source="macenko",
        explained=float(explained),
        haematoxylin_drift_deg=float(drift),
    )


def concentrations(rgb: np.ndarray, white: np.ndarray, basis: StainBasis) -> np.ndarray:
    """Per-pixel stain amounts for one `HxWx3` uint8 tile. Returns `HxWx3` float32.

    **The floor is passed explicitly, and that is a fix rather than a tidy-up.**
    `optical_density` carries a default of 1.0, and this call used to take it,
    while steps 4 and 5 passed `settings.calibration_od_floor`. The two agreed only
    because the setting happens to be 1.0: change it, and the H&E arm of the
    pipeline would have moved while the arm that produces the actual score - step
    14 measures its DAB through this function - silently stayed where it was. A
    number that is right by coincidence is not right, and the coincidence is
    invisible at every call site.
    """
    density = optical_density(
        rgb.astype(np.float32), white, floor=settings.calibration_od_floor
    )
    flat = density.reshape(-1, 3).astype(np.float64)
    return stains.unmix(flat, basis.inverse).reshape(density.shape).astype(np.float32)


def haematoxylin_only_rgb(
    rgb: np.ndarray,
    white: np.ndarray,
    basis: StainBasis,
    *,
    gain: float | None = None,
) -> np.ndarray:
    """The same field with the DAB removed, as an `HxWx3` uint8 brightfield image.

    Un-mix with `basis` to get how much counterstain each pixel holds, then draw
    that amount back **in Ruifrok's published haematoxylin direction** against a
    white background. What comes out is a canonical blue-on-white section of the
    same tissue: what the "detect on H, never on RGB" rule asks for, and - unlike
    a replicated density map - the kind of picture the segmenter was trained on.

    **Estimating and rendering use different bases on purpose.** The amount comes
    from `basis`, which may be this slide's own; the *direction* it is drawn in is
    always the published one, because the model has only ever seen images where
    haematoxylin is blue. Rendering along an estimated arm would hand it a colour
    cast that varies by slide, which is the one thing a fixed rendering basis
    costs nothing to prevent. It was not a hypothetical: on the CD44 slide the
    Macenko arm comes out achromatic - (0.577, 0.577, 0.577), pure grey - and
    drawing along it produced an image with no blue in it at all.

    **The concentration is clipped at zero here, and only here.** `unmix` must not
    clip - it solves a square system and the coefficients *are* the pixel - but a
    negative amount of stain has no rendering, and leaving it signed would draw a
    pixel brighter than the glass behind it.

    `gain` multiplies the density before the exponential. It is a contrast term,
    not a correction, and it exists because this panel's counterstain is weak
    under heavy DAB; see `settings.nuclei_haematoxylin_gain` for the sweep that
    chose its default and why it sits on a plateau rather than at a peak.
    """
    amount = np.clip(concentrations(rgb, white, basis)[..., HAEMATOXYLIN], 0.0, None)
    strength = settings.nuclei_haematoxylin_gain if gain is None else gain

    density = amount.astype(np.float64)[..., None] * _H_DIRECTION[None, None, :] * strength
    # Drawn against a clean white rather than the slide's own white point: the
    # output is an image for a model that expects brightfield, not a
    # photometrically faithful copy of this scanner's illumination. The slide's
    # white point has already done its job, one line above, in the density.
    return np.clip(255.0 * np.power(10.0, -density), 0.0, 255.0).astype(np.uint8)


def haematoxylin_density(rgb: np.ndarray, white: np.ndarray, basis: StainBasis) -> np.ndarray:
    """Just the haematoxylin coefficient, `HxW` float32.

    Not what the segmenter sees - that is `haematoxylin_only_rgb` - but what a
    nucleus is *measured* on once it has been found: how much counterstain it
    holds is one of the three features that separates a lymphocyte from a tumour
    nucleus in step 12.
    """
    return concentrations(rgb, white, basis)[..., HAEMATOXYLIN]


__all__ = [
    "DAB",
    "HAEMATOXYLIN",
    "RESIDUAL",
    "StainBasis",
    "concentrations",
    "estimate_basis",
    "haematoxylin_density",
    "haematoxylin_only_rgb",
    "ruifrok_basis",
]
