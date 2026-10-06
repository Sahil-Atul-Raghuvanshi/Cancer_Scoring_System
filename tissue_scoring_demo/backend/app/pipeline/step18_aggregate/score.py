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

**Both are estimated over the ROI from a sample, region by region (P-06).** The
contract's formula is a ratio of cell counts over the whole invasive region.
Step 13 does not count every cell; it reads a sample of fields in each region.
So each region's cells are scaled up by how much of it was sampled - a region of
9 mm2 read through 0.5 mm2 of fields stands for eighteen times the cells that
were counted in it - and the ratio is taken over those estimated totals:

    percent = sum_r (area_r / sampled_r) x positive_r
              ----------------------------------------
              sum_r (area_r / sampled_r) x cells_r

That is the contract's own ratio, estimated from a stratified sample. Where the
fields were spread in proportion to area it is exactly the pooled ratio; where
they were not, it corrects for it.

**Why not area weighting, which this module used to report.** Area weighting
gives each region a say proportional to its area *whatever was found in it*. A
9.73 mm2 region on CAN_00865 whose one field held 3 cells carried most of the
slide's weight on those 3 cells, and on CAN_00303 Pan-cadherin one 9.3 mm2 region
read at 0 % from a single field dragged the slide to 33 % against 92 % pooled and
80 % from the readers. Weighting by estimated cells keeps the large region's
say where it is full of cells, and takes it away where the sample shows it is
mostly not: three cells in a field is a measurement of how few cells are there.
Area weighting, the pooled ratio and the plain mean of the regions are all still
computed and reported beside it - open question Q3 (how OncoStem weights
sub-areas) is unanswered, and the honest form of an unanswered question is every
reading on the record with the one in use labelled.

**And every percentage carries a 95 % interval.** A two-stage bootstrap:
resample the fields inside each region, then the cells inside each field. A
figure resting on a few fields or a few hundred cells has a wide interval, and
step 18 refuses to call it a measurement when the interval is wider than a
pathologist's reading could tolerate.

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
    #: Fields read in this region and the area they covered. Zero fields means the
    #: region was carried but not sampled (P-06).
    fields: int = 0
    sampled_mm2: float = 0.0
    #: Cells the whole region is estimated to hold: `cells x area / sampled`.
    estimated_cells: float = 0.0
    #: This region's share of the reported percentage's weight.
    weight_share: float = 0.0


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

    # --- P-06: how the regions were combined, and how sure the answer is ------
    #: "estimated_cells" when every region's sampled area was known, "pooled" when
    #: it was not and the sample had to be taken as evenly spread.
    averaging_used: str = "estimated_cells"
    #: 95 % interval on `percent_raw` from the two-stage bootstrap.
    percent_ci_low: float = 0.0
    percent_ci_high: float = 0.0
    #: Share of the weight resting on regions read through a single field, whose
    #: field-to-field spread the interval cannot see.
    single_field_weight: float = 0.0
    #: Carried regions no field landed in, and their area. Not imputed: they are
    #: tumour this sample did not reach.
    unsampled_regions: int = 0
    unsampled_area_mm2: float = 0.0
    total_area_mm2: float = 0.0

    @property
    def percent_ci_width(self) -> float:
        return self.percent_ci_high - self.percent_ci_low


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


def _ratio(weights: np.ndarray, numerators: np.ndarray, denominators: np.ndarray) -> float:
    """sum(w x num) / sum(w x den), or 0 when the weighted denominator is empty."""
    below = float((weights * denominators).sum())
    return float((weights * numerators).sum()) / below if below > 0 else 0.0


#: The bootstrap's seed. Fixed, so the same rows give the same interval on every
#: run - a number that changes when nothing it rests on has changed is one a
#: reader stops trusting.
BOOTSTRAP_SEED = 20261006


def bootstrap_interval(
    fields_by_region: list[tuple[float, np.ndarray, np.ndarray]],
    *,
    reps: int,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """95 % interval on the estimated percentage, by a two-stage bootstrap.

    `fields_by_region` holds, per sampled region, its expansion weight and two
    arrays over its fields: cells, and positive cells. Each replicate resamples
    the fields inside every region with replacement - the field is the sampling
    unit, and cells in one field are not independent of each other - and then the
    cells inside each drawn field: how many there are, as a Poisson draw on the
    count, and how many of them are positive, as a binomial on its positive share.
    The count draw matters because the count is what scales a region up: a region
    read from one field of 20 cells has a density known to about +/-22 %, and that
    is uncertainty in its weight, not only in its percentage.

    Regions are strata, not resampled themselves: every carried region is part
    of the tumour by construction, so which regions there are is not a source of
    sampling error; what was read inside each one is. A region read through one
    field has no between-field spread to resample, so its interval understates how
    much a second field could have differed; the caller reports how much of the
    weight rests on such regions rather than pretending the interval covers it.
    """
    if reps <= 0 or not fields_by_region:
        return 0.0, 0.0

    rng = np.random.default_rng(seed)
    positive = np.zeros(reps, dtype=np.float64)
    total = np.zeros(reps, dtype=np.float64)
    for weight, cells, pos in fields_by_region:
        k = cells.size
        if k == 0 or weight <= 0:
            continue
        share = np.divide(pos, cells, out=np.zeros_like(pos), where=cells > 0).clip(0, 1)
        drawn = rng.integers(0, k, size=(reps, k))
        drawn_cells = rng.poisson(cells[drawn])
        drawn_pos = rng.binomial(drawn_cells, share[drawn])
        positive += weight * drawn_pos.sum(axis=1)
        total += weight * drawn_cells.sum(axis=1)

    valid = total > 0
    if not valid.any():
        return 0.0, 0.0
    percents = 100.0 * positive[valid] / total[valid]
    low, high = np.percentile(percents, [2.5, 97.5])
    return float(low), float(high)


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
    field_areas: dict[tuple[int, int], float] | None = None,
    partial_rule: str = "count",
    bootstrap_reps: int = 1000,
) -> MarkerScore:
    """The whole deliverable for one antibody, from stored rows and cut points.

    `cells` are step 14's rows exactly as they were written: dicts with
    `intensityOd`, `second`, `regionRank` and `fieldIndex`. Taking them in that
    form rather than as a fitted object is deliberate - it means this function can
    be run over a JSON file by anybody wanting to check the arithmetic, without
    importing the half of the pipeline that produced it.

    `region_areas` is every carried region's area, sampled or not. `field_areas`
    is the area each segmented field was counted over, keyed `(rank, field)` -
    including fields that yielded no cells, because an empty field is evidence of
    how few cells a region holds. Without `field_areas` the sample cannot be
    scaled up and the pooled ratio is reported, labelled as such.
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
    field_ids = np.array([int(cell.get("fieldIndex", 0)) for cell in cells], dtype=np.int64)

    positive, weight = positive_mask(od, second, cuts, rule=partial_rule)
    binned = bin_cells(od, cuts)

    total = int(od.size)
    positive_cells = float(weight.sum())

    # The pooled ratio: every measured cell counted once, wherever it came from.
    # Kept and reported, but not the answer - see the module docstring on why
    # pooling an unevenly sampled ROI is not the ROI's percentage.
    percent_pooled = 100.0 * positive_cells / total
    intensity_pooled = float(od[positive].mean()) if positive.any() else 0.0

    # --- each region on its own sample --------------------------------------
    areas = region_areas or {}
    sampled_fields = field_areas or {}
    sampled_by_rank: dict[int, float] = {}
    fields_by_rank: dict[int, list[int]] = {}
    for (rank, index), mm2 in sampled_fields.items():
        sampled_by_rank[int(rank)] = sampled_by_rank.get(int(rank), 0.0) + float(mm2)
        fields_by_rank.setdefault(int(rank), []).append(int(index))

    # Every region that was read: it has cells, or fields that found none.
    read = sorted({int(value) for value in ranks} | set(fields_by_rank))
    # Expansion only when every read region's area and sampled area are known;
    # otherwise the sample is taken as evenly spread, which is the pooled ratio.
    expandable = bool(sampled_fields) and all(
        areas.get(rank, 0.0) > 0 and sampled_by_rank.get(rank, 0.0) > 0 for rank in read
    )
    averaging_used = "estimated_cells" if expandable else "pooled"

    def expansion(rank: int) -> float:
        return areas[rank] / sampled_by_rank[rank] if expandable else 1.0

    region_weight = np.array([expansion(rank) for rank in read], dtype=np.float64)
    region_cells = np.array([int((ranks == rank).sum()) for rank in read], dtype=np.float64)
    estimated = region_weight * region_cells
    estimated_total = float(estimated.sum())

    region_scores: list[RegionScore] = []
    for at, rank in enumerate(read):
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
                fields=len(fields_by_rank.get(rank, [])),
                sampled_mm2=round(sampled_by_rank.get(rank, 0.0), 4),
                estimated_cells=round(float(estimated[at]), 1),
                weight_share=round(
                    float(estimated[at]) / estimated_total if estimated_total > 0 else 0.0, 4
                ),
            )
        )

    # Every cell carries its region's expansion weight from here on.
    cell_weight = np.array([expansion(int(rank)) for rank in ranks], dtype=np.float64)
    ones = np.ones_like(od)

    # The reported percentage: the contract's ratio over the ROI, estimated.
    percent_raw = 100.0 * _ratio(cell_weight, weight, ones)

    # Intensity is the mean over the POSITIVE cells, not over all of them.
    # Including the negatives would drag every slide's intensity towards zero in
    # proportion to how few cells stained - which is the percentage's job, and
    # would be counted twice if the intensity did it as well. Weighted the same way
    # as the percentage, so both numbers describe the same estimated population.
    intensity_raw = (
        _ratio(cell_weight[positive], od[positive], ones[positive]) if positive.any() else 0.0
    )

    # --- the other readings of Q3, reported beside it --------------------------
    area_weights = np.array([region.area_mm2 for region in region_scores], dtype=np.float64)
    percents = np.array([region.percent_raw for region in region_scores], dtype=np.float64)
    area_weighted = _weighted(percents, area_weights, fallback=percent_pooled)
    plain_mean = float(percents.mean()) if percents.size else percent_pooled

    # --- the interval -----------------------------------------------------------
    per_field: list[tuple[float, np.ndarray, np.ndarray]] = []
    for at, rank in enumerate(read):
        in_region = ranks == rank
        indices = fields_by_rank.get(rank) or sorted({int(i) for i in field_ids[in_region]})
        field_cells = np.array(
            [int((in_region & (field_ids == i)).sum()) for i in indices], dtype=np.float64
        )
        field_pos = np.array(
            [float(weight[in_region & (field_ids == i)].sum()) for i in indices],
            dtype=np.float64,
        )
        per_field.append((float(region_weight[at]), field_cells, field_pos))
    ci_low, ci_high = bootstrap_interval(per_field, reps=bootstrap_reps)

    # --- what was carried but never read ---------------------------------------
    unsampled = [rank for rank in areas if int(rank) not in set(read)]
    unsampled_area = float(sum(areas[rank] for rank in unsampled))
    total_area = float(sum(areas.values()))

    # --- Q1: what each reading of partial staining would have reported -------
    by_rule: dict[str, int] = {}
    for rule in ("count", "half", "exclude"):
        _, rule_weight = positive_mask(od, second, cuts, rule=rule)
        by_rule[rule] = round_to_step(100.0 * _ratio(cell_weight, rule_weight, ones))

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
        averaging_used=averaging_used,
        percent_ci_low=round(ci_low, 2),
        percent_ci_high=round(ci_high, 2),
        single_field_weight=round(
            sum(region.weight_share for region in region_scores if region.fields <= 1), 4
        )
        if expandable
        else 0.0,
        unsampled_regions=len(unsampled),
        unsampled_area_mm2=round(unsampled_area, 4),
        total_area_mm2=round(total_area, 4),
    )


__all__ = [
    "BOOTSTRAP_SEED",
    "PERCENT_STEP",
    "MarkerScore",
    "RegionScore",
    "bootstrap_interval",
    "round_to_step",
    "score",
]
