"""Turning a continuous optical density into levels - two of them, kept apart.

**Two things happen at this step, and conflating them is the common bug.**

    per-cell binning     each cell gets 0 / 1+ / 2+ / 3+. Internal machinery:
                         it counts positives and, if you want it, computes an
                         H-score for a demo screen.
    slide-level banding  the case's reported intensity, on OncoStem's 0-2 scale
                         with permitted values 0, 0.5, 1, 1.5, 1.75, 2. This is
                         the deliverable, and it happens at step 16.

They run on different scales, they are read by different people, and they answer
different questions. A 3+ cell is not a 2.0 slide. `bin_cells` does the first;
the second is `MarkerCuts.band`, called once at step 16 over the positive cells'
mean. Nothing here maps a per-cell bin onto a reported band, because that
mapping does not exist.

**The scale we report is 0-2, not 0/1+/2+/3+.** All 120 intensity readings in
OncoStem's sheet sit on the 0-2 bands, and 119 of them land exactly on a band
value - which is what you see when a reader picks one band per slide rather than
averaging a continuum. So banding is not optional polish, it is how the output
matches the target. Their deck also describes a 1-3 scale in places; the data
says 0-2, so 0-2 is implemented and the conflict is carried as an open question
rather than buried.

**Where the thresholds come from is the entire question**, and `compare()` is
here to show it rather than assert it:

    per-slide percentiles   derived from this slide's own 1st/99th percentile.
                            Not comparable across slides - 37 % here is not
                            37 % there, because the scale moved underneath it.
    absolute calibrated OD  fixed cutoffs on the I0-calibrated scale. Comparable.
    control-slide anchored  cutoffs set once on known 0/1+/2+/3+ controls.
                            Comparable, and clinically strongest.

The pipeline uses absolute cutoffs. `compare()` computes the per-slide
percentile answer alongside, so a screen can put three slides side by side and
show the per-slide scores coming out wrongly similar while the absolute ones
come out correctly different. The same function compares this antibody's own
cuts against a shared panel-wide set, which is the other half of the argument:
one table across five antibodies cuts at least three of them in the wrong place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.scoring.cuts import MarkerCuts

#: The four per-cell levels. Internal - see the module docstring.
BIN_LABELS: tuple[str, ...] = ("0", "1+", "2+", "3+")


@dataclass(frozen=True)
class Binned:
    """Per-cell bins and the counts that follow from them."""

    #: One of 0, 1, 2, 3 per input cell, in the order given.
    bins: np.ndarray
    #: Cells in each bin, indexed by the bin.
    counts: tuple[int, int, int, int]
    #: Share of all cells in each bin.
    shares: tuple[float, float, float, float]
    #: The three cuts these came from, carried so a count and its thresholds
    #: cannot be shown apart.
    od_cuts: tuple[float, float, float]


def bin_cells(intensity_od: np.ndarray, cuts: MarkerCuts) -> Binned:
    """Sort each cell into 0 / 1+ / 2+ / 3+ by this antibody's own cut points.

    `np.searchsorted` rather than three comparisons: it is the same arithmetic
    written once, and it cannot get the boundary conditions inconsistent between
    the three cuts the way a hand-written chain can.
    """
    values = np.asarray(intensity_od, dtype=np.float64)
    bins = np.searchsorted(np.asarray(cuts.od, dtype=np.float64), values, side="right")
    bins = bins.astype(np.int8)

    counts = tuple(int(np.count_nonzero(bins == level)) for level in range(4))
    total = max(1, values.size)
    shares = tuple(round(count / total, 6) for count in counts)

    return Binned(
        bins=bins,
        counts=counts,  # type: ignore[arg-type]
        shares=shares,  # type: ignore[arg-type]
        od_cuts=cuts.od,
    )


def percentile_cuts(
    intensity_od: np.ndarray, *, low: float = 1.0, high: float = 99.0
) -> tuple[float, float, float]:
    """The three cuts a per-slide percentile scheme would use on this slide.

    Computed **only** so the comparison can be drawn. The span between this
    slide's 1st and 99th percentile is cut into three equal parts, which is what
    a per-slide scheme does: it asks which of this slide's own cells are the
    brownest, so a weak slide and a strong one produce the same answer and the
    diagnosis is normalised away. Never used to score.
    """
    values = np.asarray(intensity_od, dtype=np.float64)
    if values.size == 0:
        return (0.0, 0.0, 0.0)
    floor = float(np.percentile(values, low))
    ceiling = float(np.percentile(values, high))
    if ceiling <= floor:
        return (floor, floor, floor)
    step = (ceiling - floor) / 3.0
    return (floor + step * 0.5, floor + step * 1.5, floor + step * 2.5)


def positive_mask(
    intensity_od: np.ndarray,
    second: np.ndarray,
    cuts: MarkerCuts,
    *,
    rule: str = "count",
) -> tuple[np.ndarray, np.ndarray]:
    """Which cells are positive, and with what weight.

    **Positivity is two conditions, for every marker in this panel**: the optical
    density has to clear the marker's cut, and the second number has to clear
    its own. Which second number that is comes from the compartment - ring
    completeness for a membrane marker, stained fraction of the band for a
    cytoplasmic one - so this function never needs to know which it was handed.

    `rule` is Q1, the open question with the largest effect on our numbers: how
    partial staining enters the percentage.

        "count"     a cell over both cuts is one positive cell. Weight 1.
        "half"      a cell over the optical-density cut whose second number sits
                    between half the cut and the cut counts as half a cell.
                    Weight 0.5.
        "exclude"   both cuts, strictly, and nothing partial. Weight 1 or 0.

    All three are implemented because none of them is known to be right, and the
    two readings give materially different results on intermediate cases. Whatever
    is decided for CD44, ABCC4 and ABCC11 does not automatically transfer to
    N-cadherin and pan-cadherin - the two decisions are recorded separately, which
    is why the rule arrives as an argument rather than being read from a global.

    Returns `(positive, weight)`: a boolean mask and a per-cell weight that the
    percentage sums rather than counts.
    """
    od = np.asarray(intensity_od, dtype=np.float64)
    other = np.asarray(second, dtype=np.float64)
    minimum = cuts.second_min

    dark_enough = od >= cuts.positivity_od

    if rule == "half":
        full = dark_enough & (other >= minimum)
        partial = dark_enough & (other >= minimum * 0.5) & (other < minimum)
        weight = np.where(full, 1.0, np.where(partial, 0.5, 0.0))
        return full | partial, weight

    # "count" and "exclude" agree on which cells are positive; they differ in
    # intent rather than in arithmetic, and keeping both names is what lets the
    # choice be recorded per marker group. "exclude" is the strict reading -
    # nothing partial contributes - and "count" is the same rule stated as
    # admitting any cell that clears both cuts.
    positive = dark_enough & (other >= minimum)
    return positive, positive.astype(np.float64)


@dataclass(frozen=True)
class Comparison:
    """One alternative threshold scheme, and what it would have reported."""

    scheme: str
    label: str
    od_cuts: tuple[float, float, float]
    counts: tuple[int, int, int, int]
    positive_share: float
    #: Points of percent-positive this scheme differs from the pipeline's answer.
    delta_points: float
    note: str


def compare(
    intensity_od: np.ndarray,
    second: np.ndarray,
    cuts: MarkerCuts,
    *,
    shared: MarkerCuts | None = None,
    rule: str = "count",
) -> list[Comparison]:
    """What the two rejected schemes would have reported on these same cells.

    Computed on every run rather than described in a caption, because the claim
    "per-slide percentiles normalise away the diagnosis" is only worth anything
    if the number it would have produced is on the screen beside the one we
    report.
    """
    od = np.asarray(intensity_od, dtype=np.float64)
    if od.size == 0:
        return []

    baseline_positive, baseline_weight = positive_mask(od, second, cuts, rule=rule)
    baseline_share = float(baseline_weight.sum() / od.size)

    out: list[Comparison] = []

    scheme = percentile_cuts(od)
    percentile_view = MarkerCuts(
        letter=cuts.letter,
        name=cuts.name,
        compartment=cuts.compartment,
        od=scheme,
        completeness_min=cuts.completeness_min,
        stained_fraction_min=cuts.stained_fraction_min,
        od_to_band=cuts.od_to_band,
    )
    _, weight = positive_mask(od, second, percentile_view, rule=rule)
    share = float(weight.sum() / od.size)
    out.append(
        Comparison(
            scheme="per_slide_percentile",
            label="Per-slide percentiles",
            od_cuts=scheme,
            counts=bin_cells(od, percentile_view).counts,
            positive_share=round(share, 4),
            delta_points=round((share - baseline_share) * 100.0, 1),
            note=(
                "Cuts taken from this slide's own 1st and 99th percentile. Not "
                "comparable across slides: the scale moves underneath the number, so a "
                "weak slide and a strong one score alike."
            ),
        )
    )

    if shared is not None and shared.od != cuts.od:
        _, shared_weight = positive_mask(od, second, shared, rule=rule)
        shared_share = float(shared_weight.sum() / od.size)
        out.append(
            Comparison(
                scheme="shared_cuts",
                label=f"One shared cut table ({shared.name}'s)",
                od_cuts=shared.od,
                counts=bin_cells(od, shared).counts,
                positive_share=round(shared_share, 4),
                delta_points=round((shared_share - baseline_share) * 100.0, 1),
                note=(
                    "The same cuts applied to every antibody in the panel. Each has its "
                    "own concentration, incubation time and detection chemistry, so the "
                    "optical density that means 'moderate' for one is not the one that "
                    "means 'moderate' for another."
                ),
            )
        )

    for alternative in ("count", "half", "exclude"):
        if alternative == rule:
            continue
        _, alt_weight = positive_mask(od, second, cuts, rule=alternative)
        alt_share = float(alt_weight.sum() / od.size)
        out.append(
            Comparison(
                scheme=f"partial_{alternative}",
                label=f"Partial staining: {alternative}",
                od_cuts=cuts.od,
                counts=bin_cells(od, cuts).counts,
                positive_share=round(alt_share, 4),
                delta_points=round((alt_share - baseline_share) * 100.0, 1),
                note=(
                    "Q1, unanswered: how a cell whose compartment is only partly stained "
                    "enters the percentage. This is what the same cells would report "
                    f"under the '{alternative}' reading."
                ),
            )
        )

    return out


__all__ = [
    "BIN_LABELS",
    "Binned",
    "Comparison",
    "bin_cells",
    "compare",
    "percentile_cuts",
    "positive_mask",
]
