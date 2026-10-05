"""A fourth answer for step 8: "cannot determine", derived after the pass.

The model has three outputs and no way to decline. Every window it is shown gets one
of them, so the in-situ class - the least well supervised of the three - absorbs
whatever does not look like the other two, and there is nowhere for "I do not know" to
go. `step8-insitu-class-collapse-oncostem` ends by asking for exactly this: *a
two-stage design that emits "cannot determine" instead of forcing a coin flip into the
denominator*. This module is that second stage.

**It runs once, over the finished class map.** Not during the pass: the live paint feed
shows what the model said, window by window, and a window's uncertainty is not a
property of the window alone - two of the three terms below need its neighbours, and one
needs the connected component it turns out to belong to. Neither exists until the last
window has been classified. So `derive` is called after `classify` returns, and the
progress screen keeps painting three classes while a finished report shows four.

**It can only ever qualify a class-1 call, and it can never change the score.** The
candidates are the windows the model labelled `non_invasive_epithelium`; invasive is
left alone deliberately, because `SCORED` is what step 9's ROI mask and the whole final
percentage are gated on, and a post-hoc layer that moved that boundary would move a
validated number without any new evidence. In-situ is excluded from the score by Rule 5
already, so `Unknown` is a strict subset of tissue that was already out of the
denominator: `tumour_content`, `scored_mm2` and step 9's input are bit-identical with
this layer on or off. That is a property worth having a test for, and there is one.

--------------------------------------------------------------------------------------
Why the obvious formulation does not work
--------------------------------------------------------------------------------------

The natural design is: flag an in-situ window when the model was unsure *and* its
neighbourhood disagrees. Both halves are reasonable and both are measured to be blind
to this project's two actual class-1 failures.

  **The failures are confident.** On `CAN_00251_26_H&E` the v2 head called 45.9% of the
  section in-situ with a median top-to-runner-up margin of **0.756** - only 5.2% of
  those windows were near-ties. On OncoStem's `CAN_00270` the v3 head *missed* DCIS at
  P(in-situ) = 0.043 with a top-1 confidence of 0.902. Neither is near a decision
  boundary, so no function of one window's softmax reaches either.

  **The failures are spatially coherent.** The 45.9% field was one enormous region, so
  a window inside it has neighbours that all confidently agree with it; local
  disagreement is near zero exactly where the error is worst. And the correct output is
  coherent too - class 1's real components measured z = -40.4 for segregation against a
  shuffled null, and the largest was duct-shaped. Coherence therefore separates nothing
  on its own.

  **Non-epithelium in the neighbourhood is not evidence of error.** A duct is a small
  object embedded in stroma, so *every* true DCIS component is surrounded by class 0.
  Scoring on the local non-epithelium fraction punishes precisely the morphology the
  layer exists to preserve.

So the two statistical terms are kept - they catch a real population, the near-ties and
the windows sitting on the invasive threshold - but they cannot be the only route, and
they must be combined with the third term by **or** rather than by **and**. An `and`
requires every route to fire at once and is what makes a flagging rule silent on the
cases that motivated it.

--------------------------------------------------------------------------------------
The three terms
--------------------------------------------------------------------------------------

**1. Margin, per window.** `M = 1 - (p[c] - max p[k != c])`, where `c` is the *assigned*
label. Assigned, not the argmax, because a checkpoint carrying `tau` decides invasive by
a threshold rather than by a vote - so for a window that just missed `tau` the runner-up
is invasive and `M` is near 1, which is the right reading of a window sitting on the one
boundary the score depends on. It is 1 exactly when the top two tie and 0 when the
assigned class takes everything.

This is not `1 - |p_IC - p_DCIS|`, which was the first formulation tried. That quantity
is 1.0 for a window at `(NE 0.98, DCIS 0.01, IC 0.01)` - a confident non-epithelium call
reported as maximally ambiguous - because it reads the *difference* between two
probabilities without reference to their size. `2 * min(p_IC, p_DCIS)` fixes that but
then reads 0.10 on `(NE 0.40, DCIS 0.55, IC 0.05)`, which is the shape of the measured
stroma-called-in-situ error. The margin against the runner-up is the quantity that is
right in both directions, because the rival that matters is whichever one is actually
close.

**2. Spatial disagreement, per window.** The neighbourhood's own posterior, and then the
share of the local *epithelial* evidence that goes to the rival reading of the same
tissue:

    Pbar[k] = sum_j w_ij p_j[k] / sum_j w_ij        (j != i, over classified windows)
    D       = Pbar[rival] / (Pbar[c] + Pbar[rival])

where `rival` is invasive - the one class that is a competing interpretation of a window
of epithelium, and the boundary Rule 5 actually turns on.

**Non-epithelium is deliberately not in that denominator, and getting this wrong is the
single easiest way to build a layer that flags every real duct.** The obvious
formulation is `D = 1 - Pbar[c]`: the mass the neighbourhood puts on anything other than
this window's claim. It reads well and it is wrong, because a duct is a small object
embedded in stroma - at a 250 um neighbourhood most of a real duct's surroundings are
class 0 no matter how correct the call is. Measured on a synthetic reconstruction of the
duct component from `CAN_00270` - 29 windows, hollow, filling 0.28 of a 13x8 box, at the
measured P(in-situ) of 0.789 - `1 - Pbar[c]` is **0.820**, and the layer flags a
textbook duct with complete confidence. The same reconstruction under the formula above
scores 0.17.

So the rule is: surrounding stroma is a duct's *context*, not evidence against it.
Surrounding invasive carcinoma is evidence against it. `D` is near 1 only for an in-situ
window sitting in a field the model otherwise reads as invasive, which is the one
spatial configuration that genuinely calls a class-1 label into question.

Three details, each of which is a bug if got wrong:

  *sigma is in microns, not in windows.* The grid's stride halves when the overlap goes
  from 0 to 0.5, so a neighbourhood fixed at "two cells" would be 224 um on one run and
  112 um on the next, and the flagged area would move with a parameter that is only
  meant to change how densely the tissue is sampled. `sigma_cells = sigma_um /
  stride_um` keeps the neighbourhood a physical distance.

  *The convolution is normalised.* Numerator and denominator are both smoothed, so a
  window at the edge of the section is compared against the neighbours it has rather
  than against zeros. Without this every window on the tissue rim reads as maximally
  unsupported, which is the shape of the artefact `step8-glass-windows-not-a-retrain`
  records and would be an easy one to reintroduce here.

  *The window is excluded from its own neighbourhood.* The question is whether the
  surroundings corroborate the claim, and a term that includes the claim itself is
  partly self-answering. The centre weight is a few percent of the kernel at the default
  sigma, so this changes little - but "little" is not "nothing" and the excluded version
  is the one that means what the name says.

A window with no classified neighbour inside the kernel gets `D = 0` rather than a
division by zero. Zero and not one: an isolated in-situ window has no *evidence against*
it, and the measurements say those windows are mostly real - on `CAN_00270`, 72% of the
class-1 components outside the DCIS contour were singletons sitting a median 6.3 cells
from the nearest invasive window, and the reading of that population was "almost
certainly normal ducts and lobules, which class 1 legitimately contains". Scoring
isolation as uncertainty would paint every normal lobule on the slide purple.

**3. Morphological implausibility, per component.** This is the term the measurements
support and the one the other two are missing. In-situ carcinoma is disease *inside a
duct*, so its extent is the extent of a duct system - bounded, and shaped like ducts. The
measured real component on `CAN_00270` was 29 windows, elongated and hollow, filling
0.28 of its bounding box. The measured false field on `CAN_00251_26` was 11,257 windows
and 141 mm2. Those are not the same kind of object and no per-window statistic tells them
apart, but their geometry does.

Over each 8-connected component R of in-situ windows:

    E(R) = clip( log(area(R) / area_ref) / log(area_max / area_ref), 0, 1 )
    S(R) = clip( (fill(R) - fill_lo) / (fill_hi - fill_lo), 0, 1 )   if R is big enough
         = 0                                                         otherwise
    G(R) = max(E(R), S(R))

`E` is on a log scale because component areas span orders of magnitude and a linear ramp
between 2 and 20 mm2 would put a 4 mm2 duct system a ninth of the way up and a 141 mm2
field at the same 1.0 as a 21 mm2 one. They combine with `max` because either is
independently sufficient: a component can be implausible by being far too large, or by
being a solid sheet at a size no single duct reaches.

**`S` is gated on the component's area, not on its window count, and that gate is the
whole content of the term.** Solidity is not evidence of anything on a small object: a
duct traced along its length is hollow and elongated, but the *same duct cut across* is
a solid disc, and a solid 4x4 block of windows is 0.2 mm2 - a small comedo-type DCIS
focus, exactly the morphology this layer must not touch. The first version of this gated
on `|R| >= 12` windows and flagged that block at 1.0. Solidity only starts to mean
something above the size a single duct can reach, which is an area and not a count -
`min_shape_mm2` is that size, and below it the term is silent.

**`area_ref` and `area_max` are the least grounded numbers in this module.** They say how
big a duct system may plausibly get, and extensive DCIS is genuinely a multi-millimetre
thing. They are settings for that reason, and the defaults are deliberately permissive -
the 141 mm2 field clears the ceiling seven times over, so nothing subtle rests on where
between 2 and 20 the ramp sits.

--------------------------------------------------------------------------------------
Combining, and the threshold
--------------------------------------------------------------------------------------

    W = sqrt(M * D)                       the statistical route
    U = 1 - (1 - W) * (1 - G)             either route, as a noisy-or

`W` is a geometric mean rather than a product. Both are soft `and`s and both are zero
when either factor is, but a product of two numbers in [0, 1] lives on a different scale
from its own factors - `0.95 * 0.20` is 0.19, and a threshold on that has to be reasoned
about backwards. The geometric mean of the same pair is 0.44, so a threshold reads as
"about this much of both", on the same scale as the terms it combines.

`U` is a noisy-or so that the morphology term can flag a component on its own. That is
the whole point of the third term: on the one failure this project has actually measured,
`W` is near zero everywhere inside the error.

A window is `Unknown` when `U >= threshold`. One number, tuned on held-out slides, and
the layer is a pure function of arrays already on disk - so a different threshold is a
re-derivation and a redraw, not another pass through the model.

--------------------------------------------------------------------------------------
What this cannot do
--------------------------------------------------------------------------------------

It qualifies what the model *called* in-situ. It cannot surface what the model missed:
on `CAN_00270` the DCIS the head failed to find is labelled class 0 or class 2, is never
a candidate here, and stays exactly as wrong as it was. A slide can be entirely free of
purple and still be entirely wrong about its in-situ disease, and the report says so
rather than letting an absence of flags read as a clean bill.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

#: Label for a window the layer declined to stand behind. A **display** class: it is
#: appended to the three the checkpoint emits and never written into `ClassMap.labels`,
#: so `CLASS_NAMES`, `verify_order` and every count the score is built from are
#: untouched by it.
UNCERTAIN: int = 3


@dataclass(frozen=True)
class UncertaintyParams:
    """The layer's settings, carried with the result so a report can state them.

    Every one of them is a tuning knob rather than a measured constant, which is why
    they travel with the output instead of being read from the settings at display time.
    """

    #: Neighbourhood scale in microns - a physical distance, not a count of windows.
    #: About one duct system: the question the term asks is whether the surrounding
    #: tissue corroborates a duct-sized claim.
    sigma_um: float = 250.0

    #: Component area, in mm2, at which the extent term starts to rise, and the area at
    #: which it saturates. See the module docstring on how soft these are.
    area_ref_mm2: float = 2.0
    area_max_mm2: float = 20.0

    #: Bounding-box fill at which the solidity term starts and saturates. The measured
    #: duct component filled 0.28 of its box.
    fill_lo: float = 0.35
    fill_hi: float = 0.75
    #: Area, in mm2, a component needs before its shape is read at all. A duct cut
    #: across is a solid disc, so solidity says nothing until the object is bigger than
    #: a duct - about a millimetre square. See the module docstring.
    min_shape_mm2: float = 1.0
    #: And a floor in windows, so a bounding-box fill is computed only where there is a
    #: box to speak of. Binds instead of the area at the coarsest geometries, where one
    #: window is already a fifth of a square millimetre.
    min_shape_windows: int = 12

    #: `U` at or above which a window is called Unknown.
    threshold: float = 0.5

    def as_dict(self) -> dict[str, Any]:
        return {
            "sigma_um": float(self.sigma_um),
            "area_ref_mm2": float(self.area_ref_mm2),
            "area_max_mm2": float(self.area_max_mm2),
            "fill_lo": float(self.fill_lo),
            "fill_hi": float(self.fill_hi),
            "min_shape_mm2": float(self.min_shape_mm2),
            "min_shape_windows": int(self.min_shape_windows),
            "threshold": float(self.threshold),
        }


@dataclass(frozen=True)
class UncertaintyLayer:
    """What `derive` produced: three terms, their combination, and the verdict.

    The terms are kept separately rather than only their combination, because a viewer
    who disagrees with a purple region needs to know *which* route flagged it - a window
    the model was unsure about and a component that is the wrong shape are different
    findings with different fixes.
    """

    #: (rows, cols) float32, all zero outside the in-situ candidates.
    margin: np.ndarray
    disagreement: np.ndarray
    implausibility: np.ndarray
    #: The combined score. Zero on every window that is not an in-situ candidate, so
    #: `score > 0` is (almost exactly) the candidate set - a candidate scoring a true
    #: zero on all three terms is possible and is not a candidate worth drawing.
    score: np.ndarray
    #: (rows, cols) bool. `score >= threshold`, and a subset of the in-situ windows.
    unknown: np.ndarray

    params: UncertaintyParams

    #: The neighbourhood in grid cells, after the conversion from microns. Reported
    #: because it is the number that would silently change if the stride did.
    sigma_cells: float

    #: Connected components of in-situ windows, and how many of them the morphology
    #: term flagged outright.
    components: int
    flagged_components: int
    #: Largest in-situ component, in mm2. The figure that decides the extent term, and
    #: the one to look at first when a whole field goes purple.
    largest_component_mm2: float

    @property
    def unknown_windows(self) -> int:
        return int(self.unknown.sum())

    @property
    def mean_score(self) -> float:
        """Mean `U` over the candidates - the in-situ windows, flagged or not."""
        candidates = self.score > 0.0
        if not bool(candidates.any()):
            return 0.0
        return float(self.score[candidates].mean())


def _kernel(sigma_cells: float) -> np.ndarray:
    """A 1D Gaussian truncated at 3 sigma and normalised to sum 1.

    Built here rather than taken from `ndimage.gaussian_filter` because the centre
    weight is needed by name: the 2D separable kernel's centre is `k[c] ** 2`, and that
    is exactly what has to come back out to exclude a window from its own neighbourhood.
    """
    radius = max(1, int(math.ceil(3.0 * sigma_cells)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-(offsets**2) / (2.0 * sigma_cells**2))
    return weights / weights.sum()


def _neighbourhood_posterior(
    probabilities: np.ndarray, inside: np.ndarray, sigma_cells: float
) -> tuple[np.ndarray, np.ndarray]:
    """`Pbar` per window, and the weight of neighbourhood each one actually had.

    A **normalised** convolution - numerator and denominator both smoothed over the same
    mask - so a window is compared against the neighbours it has rather than against the
    zeros beyond the edge of the section. And a **self-excluded** one: the centre weight
    is subtracted from both, so the returned distribution is over other windows only.

    Returns `(posterior, support)`, where `support` is the summed weight of the
    classified neighbours. Where that is zero there was nothing to corroborate against,
    and the caller reads it as no support rather than dividing by it.
    """
    from scipy.ndimage import convolve1d

    kernel = _kernel(sigma_cells)
    centre = float(kernel[len(kernel) // 2]) ** 2

    mask = inside.astype(np.float64)

    def smooth(field: np.ndarray) -> np.ndarray:
        # Zero padding, which is what the normalisation is built around: an absent
        # neighbour must contribute nothing to *both* sums, not a reflected copy of a
        # present one.
        rows = convolve1d(field, kernel, axis=0, mode="constant", cval=0.0)
        return convolve1d(rows, kernel, axis=1, mode="constant", cval=0.0)

    support = smooth(mask) - centre * mask
    numerator = np.stack(
        [
            smooth(probabilities[..., k] * mask) - centre * probabilities[..., k] * mask
            for k in range(probabilities.shape[-1])
        ],
        axis=-1,
    )

    # Guarded rather than clipped: `support` is a sum of non-negative weights, so the
    # only value it can take that this must not divide by is exactly zero, and that is a
    # real state - an isolated window - rather than a numerical accident.
    safe = np.where(support > 0.0, support, 1.0)
    posterior = numerator / safe[..., None]
    return posterior, support


def _component_geometry(
    candidates: np.ndarray, *, cell_mm2: float, params: UncertaintyParams
) -> tuple[np.ndarray, int, int, float]:
    """The morphology term, painted back onto the grid one component at a time.

    8-connected, because a duct traced diagonally across the window grid is one object
    and 4-connectivity would cut it into a string of singletons - which would then read
    as unsupported under the spatial term as well, flagging real ducts twice for a
    reason that is an artefact of the connectivity rule.
    """
    from scipy.ndimage import find_objects, label

    field = np.zeros(candidates.shape, dtype=np.float32)
    if not bool(candidates.any()):
        return field, 0, 0, 0.0

    labelled, count = label(candidates, structure=np.ones((3, 3), dtype=bool))
    boxes = find_objects(labelled)

    ceiling = max(params.area_max_mm2, params.area_ref_mm2 * 1.0001)
    span = math.log(ceiling / params.area_ref_mm2)
    fill_span = max(params.fill_hi - params.fill_lo, 1e-6)

    flagged = 0
    largest_mm2 = 0.0

    for index, box in enumerate(boxes, start=1):
        if box is None:
            continue
        here = labelled[box] == index
        size = int(here.sum())
        area_mm2 = size * cell_mm2
        largest_mm2 = max(largest_mm2, area_mm2)

        extent = 0.0
        if area_mm2 > params.area_ref_mm2:
            extent = min(1.0, math.log(area_mm2 / params.area_ref_mm2) / span)

        solidity = 0.0
        if size >= params.min_shape_windows and area_mm2 >= params.min_shape_mm2:
            fill = size / float(here.size)
            solidity = min(1.0, max(0.0, (fill - params.fill_lo) / fill_span))

        implausible = max(extent, solidity)
        if implausible >= params.threshold:
            flagged += 1

        # Written only where the component actually is, not across its bounding box -
        # `here` is the component's own mask inside the box, and a second component
        # crossing the same box must not inherit this one's score.
        window = field[box]
        window[here] = implausible

    return field, int(count), flagged, float(round(largest_mm2, 4))


def derive(
    labels: np.ndarray,
    probabilities: np.ndarray,
    inside: np.ndarray,
    *,
    candidate: int,
    rival: int,
    stride_um: float,
    cell_mm2: float,
    params: UncertaintyParams | None = None,
) -> UncertaintyLayer:
    """Score every in-situ window for whether the pass should stand behind it.

    `candidate` is the label the layer may qualify - class 1, in-situ - and `rival` is
    the class whose presence in the neighbourhood counts as evidence against it - class
    2, invasive. Both are passed in rather than imported so this module stays a function
    of arrays and can be tested without a slide, a checkpoint or a grid.

    `stride_um` is what turns `sigma_um` into a number of cells; `cell_mm2` is the grid's
    stride-sized core, which is the only area figure that does not move when the overlap
    does.
    """
    settings = params or UncertaintyParams()

    inside = inside.astype(bool)
    classified = inside & (labels >= 0)
    candidates = classified & (labels == candidate)

    sigma_cells = max(0.5, float(settings.sigma_um) / max(float(stride_um), 1e-6))

    # --- 1. margin ------------------------------------------------------------
    #
    # Against the runner-up to the *assigned* label, which under a checkpoint carrying
    # `tau` is not always the argmax. The assigned class is masked out of the vector
    # before the max is taken, which is what makes "runner-up" mean the best of the
    # rivals rather than the best overall.
    assigned = np.clip(labels, 0, probabilities.shape[-1] - 1).astype(np.intp)
    own = np.take_along_axis(probabilities, assigned[..., None], axis=-1)[..., 0]
    others = probabilities.astype(np.float64, copy=True)
    np.put_along_axis(others, assigned[..., None], -np.inf, axis=-1)
    runner_up = others.max(axis=-1)
    margin = np.where(classified, 1.0 - (own - runner_up), 0.0)
    margin = np.clip(margin, 0.0, 1.0).astype(np.float32)

    # --- 2. spatial disagreement ---------------------------------------------
    #
    # The rival's share of the local *epithelial* evidence, not of all of it. See the
    # module docstring: putting non-epithelium in this denominator flags every real duct,
    # because a duct is a small object and its surroundings are stroma by definition.
    posterior, support = _neighbourhood_posterior(probabilities, classified, sigma_cells)
    for_candidate = posterior[..., candidate]
    for_rival = posterior[..., rival]
    epithelial = for_candidate + for_rival
    disagreement = np.divide(
        for_rival,
        epithelial,
        out=np.zeros_like(for_rival),
        where=epithelial > 1e-9,
    )
    # No classified neighbour means no evidence against, not maximal evidence against.
    disagreement = np.where(support > 0.0, disagreement, 0.0)
    disagreement = np.where(classified, np.clip(disagreement, 0.0, 1.0), 0.0).astype(np.float32)

    # --- 3. morphological implausibility -------------------------------------
    implausibility, components, flagged, largest = _component_geometry(
        candidates, cell_mm2=cell_mm2, params=settings
    )

    # --- combine --------------------------------------------------------------
    statistical = np.sqrt(margin.astype(np.float64) * disagreement.astype(np.float64))
    combined = 1.0 - (1.0 - statistical) * (1.0 - implausibility.astype(np.float64))
    score = np.where(candidates, combined, 0.0).astype(np.float32)

    unknown = candidates & (score >= float(settings.threshold))

    return UncertaintyLayer(
        margin=np.where(candidates, margin, 0.0).astype(np.float32),
        disagreement=np.where(candidates, disagreement, 0.0).astype(np.float32),
        implausibility=implausibility.astype(np.float32),
        score=score,
        unknown=unknown,
        params=settings,
        sigma_cells=round(float(sigma_cells), 4),
        components=components,
        flagged_components=flagged,
        largest_component_mm2=largest,
    )


__all__ = [
    "UNCERTAIN",
    "UncertaintyLayer",
    "UncertaintyParams",
    "derive",
]
