"""The rulebook. No image processing at all - arithmetic and if statements.

By the time anything reaches this module every hard decision has been made:
which cells count, which part of each one, at what scale, against which cuts.
So this is small, heavily tested and separately reviewable, and **nothing here
reaches back into pixels**. That is the property that makes the deliverable
auditable: a disputed number can be re-derived from the stored rows with a
calculator.

**The contract is two numbers per marker, ten per case.**

    percent positive = 100 x (positive tumour cells) / (total tumour cells)
                       over the invasive ROI, then rounded to the nearest 5

    intensity        = mean DAB optical density over the POSITIVE tumour cells
                       then mapped through that antibody's band table to the
                       nearest permitted value: 0 / 0.5 / 1 / 1.5 / 1.75 / 2

**Both are averaged over the ROI by area, and that is not the same as pooling
the cells.** The contract's formula is a ratio over the whole invasive region,
and it would be exactly the pooled ratio if the cells had been sampled evenly
across it. They were not: step 11 segments a fixed number of fields in each
region regardless of how large the region is, so on CAN_00270's CD44 slide the
largest region holds 84 % of the invasive area and contributes 42 % of the
measured cells. Pooling those cells does not give the ROI's percentage - it
gives an average that weights a 2 mm2 region as heavily as a 22 mm2 one, and on
that slide the two answers are 60 % and 81 %.

So each region is measured on its own sample and the regions are combined by
area, which is what "the percentage over the invasive ROI" means when the sample
is not proportional. The pooled figure and the plain mean of the regions are
both computed and reported beside it - this is open question Q3, and the honest
form of an unanswered question is all three numbers with the one in use
labelled, not a single number with a convention hidden inside it.

Nothing else crosses the boundary to OncoStem. H-score, Allred and the ASCO/CAP
HER2 call are computed here and returned, because they are the standard
vocabulary of the field and the demo shows them - but they are labelled
reference, never the deliverable. Their sheet has ten columns: five
percent-and-intensity pairs and no combined score anywhere. **Collapsing the two
numbers into one throws away information their model is using.**

**Both roundings are load-bearing, not cosmetic.** Real reported percentages are
multiples of 5, and 119 of 120 real intensities land exactly on a band value.
Emitting 61.7 % and 1.34 does not look more precise to the reader receiving it -
it looks like a different measurement from the one they asked for, and it cannot
be compared against their sheet. Both the raw and the rounded value are returned
so a screen can show the step happening rather than hide it.

**The marker letter is a parameter, not a global.** Every per-marker rule -
which compartment, which second number, which cut points, which band table -
resolves from it in one place, so a sixth antibody is a config entry rather than
a code change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from app.pipeline.step17_intensity_binning.binning import bin_cells, positive_mask
from app.scoring.cuts import MarkerCuts

#: Reported percentages are multiples of this. Observed in every one of the 120
#: real readings, so it is the shape of the answer rather than a presentation
#: choice.
PERCENT_STEP = 5


def round_to_step(value: float, step: int = PERCENT_STEP) -> int:
    """Round half **up** to the nearest `step`.

    Explicitly, rather than through `round()`, which rounds halves to even: at
    step 5 that sends 62.5 to 60 and 67.5 to 70, so two cases either side of a
    boundary would round in opposite directions for a reason nobody reading the
    output could reconstruct.
    """
    return int(math.floor(value / step + 0.5) * step)


@dataclass(frozen=True)
class RegionScore:
    """One invasive region's own answer, before the regions are combined."""

    rank: int
    area_mm2: float
    cells: int
    positive_cells: float
    percent_raw: float
    intensity_raw: float


@dataclass(frozen=True)
class MarkerScore:
    """Everything step 16 produces for one antibody.

    The first two fields are the deliverable. Everything after them is context,
    a check, or the standard vocabulary - useful, and not what is reported.
    """

    marker: str
    marker_name: str
    compartment: str
    second_measure: str

    # --- the deliverable ----------------------------------------------------
    percent: int
    intensity: float

    # --- the same two numbers before rounding, so the step can be shown -----
    percent_raw: float
    intensity_raw: float
    intensity_band_label: str

    #: The pooled ratio over every measured cell, ignoring which region each
    #: came from. Reported because it is what the contract's formula reads as
    #: literally, and because the gap between it and the reported figure is the
    #: size of the sampling imbalance - see the module docstring.
    percent_pooled: float
    intensity_pooled: float

    # --- the counting behind them -------------------------------------------
    cells: int
    positive_cells: float
    #: Cells in each of 0 / 1+ / 2+ / 3+. Internal machinery, not the 0-2 scale.
    bin_counts: tuple[int, int, int, int]
    bin_shares: tuple[float, float, float, float]

    # --- reference vocabulary, never the deliverable ------------------------
    h_score: float
    allred_proportion: int
    allred_intensity: int
    allred_total: int
    her2_call: str | None
    her2_note: str

    # --- the open questions, answered both ways -----------------------------
    #: Q3: the regions combined by area, and combined as a plain mean. Ours is
    #: naturally area-weighted; which one OncoStem uses is unanswered.
    percent_area_weighted: float
    percent_plain_mean: float
    averaging_gap_points: float
    regions: list[RegionScore] = field(default_factory=list)

    #: Q1: what the percentage would have been under each reading of partial
    #: staining. The rule actually used is `partial_rule`.
    partial_rule: str = "count"
    percent_by_partial_rule: dict[str, int] = field(default_factory=dict)

    #: The cut points these numbers came from, and whether they have been fitted.
    od_cuts: tuple[float, float, float] = (0.0, 0.0, 0.0)
    second_min: float = 0.0
    cuts_provisional: bool = True


def _weighted(values: np.ndarray, weights: np.ndarray, *, fallback: float) -> float:
    """Area-weighted mean, falling back when no area is known.

    The fallback is the pooled figure rather than a plain mean. A run with no
    region areas at all is a run where the regions are indistinguishable, and
    the pooled ratio is then the correct reading of the contract rather than an
    approximation of something better.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.size == 0 or weights.size != values.size or weights.sum() <= 0:
        return float(fallback)
    return float((values * weights).sum() / weights.sum())


def _allred_proportion(share: float) -> int:
    """Allred's proportion score, 0-5, from the share of positive tumour cells."""
    if share <= 0.0:
        return 0
    if share < 0.01:
        return 1
    if share <= 0.10:
        return 2
    if share <= 0.33:
        return 3
    if share <= 0.66:
        return 4
    return 5


def _her2_call(
    cuts: MarkerCuts,
    *,
    compartment: str,
    complete_strong_share: float,
    complete_any_share: float,
    incomplete_faint_share: float,
) -> tuple[str | None, str]:
    """The ASCO/CAP HER2 categories, applied to a membrane marker's cells.

    **This is not a HER2 test and must never be read as one.** HER2 is a
    different antibody with its own validated scoring rules, cut points and
    clinical meaning; none of the five markers in this panel is HER2. What the
    guideline supplies, and the reason it is computed at all, is the vocabulary:
    "complete, intense membrane staining in more than 10 % of tumour cells" is
    its definition of 3+, and that is precisely the distinction between complete
    and partial membrane staining that this pipeline measures. Showing the call
    on a demo screen is how an audience recognises what the completeness axis is
    for.

    Returns `(call, note)`. For a cytoplasmic marker the call is `None`: the
    rules are written about a membrane, and applying them to a cytoplasm band
    would produce a confident category for a measurement the guideline does not
    describe.
    """
    if compartment != "membrane":
        return None, (
            "The ASCO/CAP categories are defined on complete membrane staining. This is "
            "a cytoplasmic marker, so they are not computed - a category here would be "
            "a confident answer to a question the guideline does not ask."
        )

    note = (
        "ASCO/CAP's HER2 vocabulary applied to this marker's completeness and intensity. "
        "Reference only: none of the five antibodies in this panel is HER2, and this is "
        "not a HER2 result. It is here because 'complete, intense membrane staining' is "
        "the guideline's own definition of 3+, which is the distinction this step "
        "measures."
    )

    if complete_strong_share > 0.10:
        return "3+", note
    if complete_any_share > 0.10 or complete_strong_share > 0.0:
        return "2+", note
    if incomplete_faint_share > 0.10:
        return "1+", note
    return "0", note


def score(
    marker_letter: str,
    cells: list[dict],
    cuts: MarkerCuts,
    *,
    marker_name: str,
    compartment: str,
    second_measure: str,
    region_areas: dict[int, float] | None = None,
    partial_rule: str = "count",
) -> MarkerScore:
    """The whole deliverable for one antibody, from stored rows and cut points.

    `cells` are step 14's rows exactly as they were written: dicts with
    `intensityOd`, `second` and `regionRank`. Taking them in that form rather
    than as a fitted object is deliberate - it means this function can be run
    over a JSON file by anybody wanting to check the arithmetic, without
    importing the half of the pipeline that produced it.
    """
    if not cells:
        raise ValueError(
            f"no measured cells for {marker_name} ({marker_letter}). A percentage over "
            "an empty denominator is not zero, it is undefined, and reporting 0 % would "
            "be indistinguishable from a slide with no staining on it."
        )

    od = np.array([float(cell["intensityOd"]) for cell in cells], dtype=np.float64)
    second = np.array([float(cell["second"]) for cell in cells], dtype=np.float64)
    ranks = np.array([int(cell.get("regionRank", 1)) for cell in cells], dtype=np.int64)

    positive, weight = positive_mask(od, second, cuts, rule=partial_rule)
    binned = bin_cells(od, cuts)

    total = int(od.size)
    positive_cells = float(weight.sum())

    # The pooled ratio: every measured cell counted once, wherever it came from.
    # Kept and reported, but not the answer - see the module docstring on why
    # pooling an unevenly sampled ROI is not the ROI's percentage.
    percent_pooled = 100.0 * positive_cells / total
    intensity_pooled = float(od[positive].mean()) if positive.any() else 0.0

    # --- each region on its own sample, then combined by area (Q3) -----------
    areas = region_areas or {}
    region_scores: list[RegionScore] = []
    for rank in sorted({int(value) for value in ranks}):
        mask = ranks == rank
        region_positive = float(weight[mask].sum())
        region_total = int(mask.sum())
        region_od = od[mask][positive[mask]]
        region_scores.append(
            RegionScore(
                rank=rank,
                area_mm2=float(areas.get(rank, 0.0)),
                cells=region_total,
                positive_cells=round(region_positive, 2),
                percent_raw=round(100.0 * region_positive / max(1, region_total), 2),
                intensity_raw=round(float(region_od.mean()) if region_od.size else 0.0, 4),
            )
        )

    weights = np.array([region.area_mm2 for region in region_scores], dtype=np.float64)
    percents = np.array([region.percent_raw for region in region_scores], dtype=np.float64)
    area_weighted = _weighted(percents, weights, fallback=percent_pooled)
    plain_mean = float(percents.mean()) if percents.size else percent_pooled

    # Intensity is combined the same way, over the regions that HAVE a positive
    # cell. A region where nothing stained has no mean optical density of stained
    # cells, and folding a zero in for it would be counting its emptiness twice -
    # once in the percentage, which is where it belongs, and again here.
    stained = np.array(
        [region.intensity_raw > 0.0 for region in region_scores], dtype=bool
    )
    intensities = np.array(
        [region.intensity_raw for region in region_scores], dtype=np.float64
    )
    intensity_raw = (
        _weighted(intensities[stained], weights[stained], fallback=intensity_pooled)
        if stained.any()
        else 0.0
    )

    # Intensity is the mean over the POSITIVE cells, not over all of them.
    # Including the negatives would drag every slide's intensity towards zero in
    # proportion to how few cells stained - which is the percentage's job, and
    # would be counted twice if the intensity did it as well.
    percent_raw = area_weighted

    # --- Q1: what each reading of partial staining would have reported -------
    by_rule: dict[str, int] = {}
    for rule in ("count", "half", "exclude"):
        _, rule_weight = positive_mask(od, second, cuts, rule=rule)
        rule_percents = np.array(
            [
                100.0 * float(rule_weight[ranks == region.rank].sum())
                / max(1, int((ranks == region.rank).sum()))
                for region in region_scores
            ]
        )
        by_rule[rule] = round_to_step(
            _weighted(
                rule_percents,
                weights,
                fallback=100.0 * float(rule_weight.sum()) / total,
            )
        )

    # --- the reference scores ------------------------------------------------
    # H-score over the percentages of ALL tumour cells, which is its definition:
    # 1 x %1+ + 2 x %2+ + 3 x %3+, range 0-300. Pooled rather than area-weighted,
    # deliberately: the published formula is a ratio over the cells counted, and
    # re-weighting it by region area would produce a number that is not the
    # H-score anybody else computes. It is reference vocabulary, so it is left in
    # the form the reference defines.
    h_score = (
        1.0 * binned.shares[1] + 2.0 * binned.shares[2] + 3.0 * binned.shares[3]
    ) * 100.0

    # Allred reads the reported percentage, so it uses the area-weighted one -
    # unlike the H-score above, whose definition is over the cells counted.
    allred_proportion = _allred_proportion(percent_raw / 100.0)
    # Allred's intensity term is the average intensity of the positive cells,
    # binned - the same quantity the reported intensity comes from, on the 0-3
    # scale the Allred system uses rather than the 0-2 one OncoStem reports.
    allred_intensity = cuts.bin_of(intensity_raw) if positive.any() else 0

    strong = positive & (od >= cuts.od[2])
    complete = second >= cuts.second_min
    her2_call, her2_note = _her2_call(
        cuts,
        compartment=compartment,
        complete_strong_share=float((strong & complete).sum()) / total,
        complete_any_share=float((positive & complete).sum()) / total,
        incomplete_faint_share=float(((od >= cuts.positivity_od) & ~complete).sum()) / total,
    )

    return MarkerScore(
        marker=marker_letter.upper(),
        marker_name=marker_name,
        compartment=compartment,
        second_measure=second_measure,
        percent=round_to_step(percent_raw),
        intensity=cuts.band(intensity_raw),
        percent_raw=round(percent_raw, 2),
        intensity_raw=round(intensity_raw, 4),
        intensity_band_label="",  # filled by the caller, which owns the label table
        percent_pooled=round(percent_pooled, 2),
        intensity_pooled=round(intensity_pooled, 4),
        cells=total,
        positive_cells=round(positive_cells, 2),
        bin_counts=binned.counts,
        bin_shares=binned.shares,
        h_score=round(h_score, 1),
        allred_proportion=allred_proportion,
        allred_intensity=allred_intensity,
        allred_total=allred_proportion + allred_intensity,
        her2_call=her2_call,
        her2_note=her2_note,
        percent_area_weighted=round(area_weighted, 2),
        percent_plain_mean=round(plain_mean, 2),
        averaging_gap_points=round(abs(area_weighted - plain_mean), 2),
        regions=region_scores,
        partial_rule=partial_rule,
        percent_by_partial_rule=by_rule,
        od_cuts=cuts.od,
        second_min=cuts.second_min,
        cuts_provisional=cuts.provisional,
    )


__all__ = [
    "PERCENT_STEP",
    "MarkerScore",
    "RegionScore",
    "round_to_step",
    "score",
]
