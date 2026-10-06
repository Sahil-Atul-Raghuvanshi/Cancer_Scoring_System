"""Steps 15 and 16: the binning and the rulebook, on cells with a known answer.

Everything here is arithmetic over dicts. That is the point of step 16 being a
separate module with no image processing in it - a disputed number can be
re-derived with a calculator, and so can a test.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.pipeline.step17_intensity_binning.binning import (
    bin_cells,
    compare,
    percentile_cuts,
    positive_mask,
)
from app.pipeline.step18_aggregate.score import round_to_step, score
from app.scoring import cuts as cut_points


@pytest.fixture
def membrane_cuts():
    return cut_points.for_marker("A")


@pytest.fixture
def cytoplasm_cuts():
    return cut_points.for_marker("U")


def _cell(od: float, second: float, rank: int = 1) -> dict:
    return {"intensityOd": od, "second": second, "regionRank": rank}


# --- step 15 ---------------------------------------------------------------


def test_bins_follow_the_markers_own_cuts(membrane_cuts):
    od = np.array([0.0, 0.14, 0.15, 0.34, 0.35, 0.69, 0.70, 1.2])
    binned = bin_cells(od, membrane_cuts)
    assert list(binned.bins) == [0, 0, 1, 1, 2, 2, 3, 3]
    assert binned.counts == (2, 2, 2, 2)


def test_the_two_marker_groups_are_cut_in_different_places(
    membrane_cuts, cytoplasm_cuts
):
    """The reason one shared table cannot serve five antibodies."""
    od = np.array([0.13])
    assert bin_cells(od, membrane_cuts).counts[0] == 1  # negative for CD44
    assert bin_cells(od, cytoplasm_cuts).counts[1] == 1  # 1+ for N-cadherin


def test_positivity_needs_both_numbers_not_just_the_density(membrane_cuts):
    """A dark speckle is not a positive cell."""
    od = np.array([0.8, 0.8])
    second = np.array([0.9, 0.05])
    positive, _ = positive_mask(od, second, membrane_cuts)
    assert list(positive) == [True, False]


def test_the_partial_rule_changes_the_answer_and_all_three_are_available(
    membrane_cuts,
):
    """Q1 - a cell just under the completeness cut."""
    od = np.array([0.5])
    second = np.array([membrane_cuts.second_min * 0.75])

    _, counted = positive_mask(od, second, membrane_cuts, rule="count")
    _, halved = positive_mask(od, second, membrane_cuts, rule="half")
    _, excluded = positive_mask(od, second, membrane_cuts, rule="exclude")

    assert counted.sum() == 0.0  # below the cut, so not positive under "count"
    assert halved.sum() == 0.5  # admitted at half weight
    assert excluded.sum() == 0.0


def test_percentile_cuts_move_with_the_slide_which_is_the_failure(membrane_cuts):
    """A weak slide and a strong one produce the same per-slide cuts, scaled."""
    weak = np.linspace(0.02, 0.20, 500)
    strong = np.linspace(0.20, 2.00, 500)
    assert percentile_cuts(weak) != percentile_cuts(strong)

    ones = np.ones(500)
    weak_view = _with_od(membrane_cuts, percentile_cuts(weak))
    strong_view = _with_od(membrane_cuts, percentile_cuts(strong))
    weak_share = positive_mask(weak, ones, weak_view)[0].mean()
    strong_share = positive_mask(strong, ones, strong_view)[0].mean()
    # Wrongly similar: two completely different slides, near-identical answers.
    assert abs(weak_share - strong_share) < 0.05

    # Absolute cuts correctly separate them.
    absolute_weak = positive_mask(weak, np.ones(500), membrane_cuts)[0].mean()
    absolute_strong = positive_mask(strong, np.ones(500), membrane_cuts)[0].mean()
    assert absolute_strong - absolute_weak > 0.5


def _with_od(base, od):
    from dataclasses import replace

    return replace(base, od=od)


def test_compare_reports_every_rejected_scheme(membrane_cuts, cytoplasm_cuts):
    od = np.random.default_rng(0).uniform(0.0, 1.0, 400)
    second = np.random.default_rng(1).uniform(0.0, 1.0, 400)
    schemes = {entry.scheme for entry in compare(od, second, membrane_cuts, shared=cytoplasm_cuts)}
    assert "per_slide_percentile" in schemes
    assert "shared_cuts" in schemes
    assert {"partial_half", "partial_exclude"} <= schemes


# --- step 16 ---------------------------------------------------------------


def test_rounding_is_half_up_not_half_to_even():
    assert round_to_step(62.5) == 65
    assert round_to_step(67.5) == 70
    assert round_to_step(62.4) == 60
    assert round_to_step(0.1) == 0


def test_the_contract_is_two_numbers_and_both_are_rounded(membrane_cuts):
    # 61 of 100 cells positive: 61 % raw -> 60 reported.
    cells = [_cell(0.5, 0.9) for _ in range(61)] + [_cell(0.02, 0.0) for _ in range(39)]
    result = score(
        "A", cells, membrane_cuts,
        marker_name="CD44", compartment="membrane", second_measure="ring_completeness",
    )

    assert result.percent_raw == 61.0
    assert result.percent == 60
    assert result.intensity in cut_points.PERMITTED_BANDS
    assert result.intensity_raw == pytest.approx(0.5)


def test_intensity_is_the_mean_over_positive_cells_only(membrane_cuts):
    """Negatives would drag it down in proportion to the percentage - counted twice."""
    cells = [_cell(0.8, 0.9), _cell(0.8, 0.9), _cell(0.01, 0.0), _cell(0.01, 0.0)]
    result = score(
        "A", cells, membrane_cuts,
        marker_name="CD44", compartment="membrane", second_measure="ring_completeness",
    )
    assert result.intensity_raw == pytest.approx(0.8)
    assert result.percent == 50


def test_h_score_is_the_published_formula(membrane_cuts):
    """1x%1+ + 2x%2+ + 3x%3+, over all tumour cells, range 0-300."""
    cells = (
        [_cell(0.20, 0.9) for _ in range(25)]   # 1+
        + [_cell(0.50, 0.9) for _ in range(25)]  # 2+
        + [_cell(0.90, 0.9) for _ in range(25)]  # 3+
        + [_cell(0.01, 0.0) for _ in range(25)]  # 0
    )
    result = score(
        "A", cells, membrane_cuts,
        marker_name="CD44", compartment="membrane", second_measure="ring_completeness",
    )
    assert result.h_score == pytest.approx(1 * 25 + 2 * 25 + 3 * 25)
    assert 0 <= result.h_score <= 300


def test_allred_is_proportion_plus_intensity_bounded_at_eight(membrane_cuts):
    cells = [_cell(0.9, 0.9) for _ in range(90)] + [_cell(0.0, 0.0) for _ in range(10)]
    result = score(
        "A", cells, membrane_cuts,
        marker_name="CD44", compartment="membrane", second_measure="ring_completeness",
    )
    assert result.allred_proportion == 5
    assert result.allred_intensity == 3
    assert result.allred_total == 8


def test_her2_call_is_not_computed_for_a_cytoplasmic_marker(cytoplasm_cuts):
    cells = [_cell(0.8, 0.9) for _ in range(50)]
    result = score(
        "U", cells, cytoplasm_cuts,
        marker_name="N-cadherin", compartment="cytoplasm", second_measure="stained_fraction",
    )
    assert result.her2_call is None
    assert "cytoplasmic" in result.her2_note


def test_her2_three_plus_needs_complete_intense_staining(membrane_cuts):
    cells = [_cell(0.9, 0.95) for _ in range(50)] + [_cell(0.0, 0.0) for _ in range(50)]
    result = score(
        "A", cells, membrane_cuts,
        marker_name="CD44", compartment="membrane", second_measure="ring_completeness",
    )
    assert result.her2_call == "3+"


def test_area_weighted_and_plain_mean_differ_on_unequal_regions(membrane_cuts):
    """Q3 - and the gap is reported rather than resolved."""
    cells = (
        [_cell(0.9, 0.9, rank=1) for _ in range(90)]
        + [_cell(0.0, 0.0, rank=1) for _ in range(10)]
        + [_cell(0.0, 0.0, rank=2) for _ in range(10)]
    )
    result = score(
        "A", cells, membrane_cuts,
        marker_name="CD44", compartment="membrane", second_measure="ring_completeness",
        region_areas={1: 20.0, 2: 1.0},
    )
    assert result.percent_area_weighted > result.percent_plain_mean
    assert result.averaging_gap_points > 10


def test_an_empty_denominator_is_refused_not_reported_as_zero(membrane_cuts):
    with pytest.raises(ValueError, match="undefined"):
        score(
            "A", [], membrane_cuts,
            marker_name="CD44", compartment="membrane",
            second_measure="ring_completeness",
        )


def test_every_partial_rule_is_reported_beside_the_one_used(membrane_cuts):
    cells = [_cell(0.5, membrane_cuts.second_min * 0.75) for _ in range(100)]
    result = score(
        "A", cells, membrane_cuts,
        marker_name="CD44", compartment="membrane", second_measure="ring_completeness",
    )
    assert result.percent_by_partial_rule["count"] == 0
    assert result.percent_by_partial_rule["half"] == 50
    assert result.percent_by_partial_rule["exclude"] == 0


def test_a_slide_at_the_edge_of_the_band_table_says_so(membrane_cuts):
    """The reported intensity has six values and, on this pipeline, about two.

    Every positive-cell density this pipeline produces on the real case lands
    between 0.30 and 0.44 OD, which the band table puts inside two of its six
    bands. That is a property of the table against these slides, measurable with
    no reference to what the right answer is - and somebody comparing our
    intensity against a pathologist's needs it before reading anything into the
    gap.
    """
    from app.services.score_service import score_service

    # Densities far below the table's first breakpoint: every cell lands in the
    # bottom band however the slide really stained.
    faint = [_cell(0.16, 0.9) for _ in range(50)]
    result = score(
        "A", faint, membrane_cuts,
        marker_name="CD44", compartment="membrane", second_measure="ring_completeness",
    )
    caveats = score_service._caveats("he", "ihc", result, _measured())
    assert any("EDGE OF THE BAND TABLE" in caveat for caveat in caveats)


def _measured():
    """The minimum step 14 report `_caveats` reads."""
    from app.schemas.per_cell import MeasurementParams, PerCellReport

    return PerCellReport(
        he_upload_id="he",
        ihc_upload_id="ihc",
        marker="A",
        marker_name="CD44",
        second_measure="ring_completeness",
        generated_at="2026-09-15T00:00:00Z",
        params=MeasurementParams(
            intensity_statistic="mean",
            compartment="membrane",
            expansion_um=4.0,
            positivity_od=0.15,
            second_min=0.35,
            cuts_version=1,
        ),
        cells=50,
    )


# --- P-06: estimated over the ROI, with an interval ---------------------------


def _field_cells(positive: int, negative: int, *, rank: int, field: int, cuts) -> list[dict]:
    cell = lambda od, second: {  # noqa: E731
        "intensityOd": od, "second": second, "regionRank": rank, "fieldIndex": field,
    }
    strong = max(cuts.od) + 0.2
    return [cell(strong, 1.0) for _ in range(positive)] + [cell(0.0, 0.0) for _ in range(negative)]


def _score(cells, cuts, **kwargs):
    return score(
        "A", cells, cuts,
        marker_name="CD44", compartment="membrane", second_measure="ring_completeness",
        **kwargs,
    )


def test_a_large_region_read_from_a_few_cells_no_longer_carries_the_slide(membrane_cuts):
    """CAN_00303 Pan-cadherin, in miniature.

    Three dense regions read 90 % positive over many fields; one 9.3 mm2 region was
    read from a single field holding 20 unstained cells. Area weighting let those 20
    cells speak for 9.3 mm2 and pulled the slide down to about a third. Weighting by
    estimated cells takes the sparse region at what its field showed: few cells.
    """
    cells: list[dict] = []
    fields: dict[tuple[int, int], float] = {}
    for rank in (1, 2, 3):
        for field in range(8):
            cells += _field_cells(90, 10, rank=rank, field=field, cuts=membrane_cuts)
            fields[(rank, field)] = 0.054
    cells += _field_cells(0, 20, rank=4, field=0, cuts=membrane_cuts)
    fields[(4, 0)] = 0.054
    areas = {1: 1.5, 2: 1.5, 3: 1.5, 4: 9.3}

    result = _score(cells, membrane_cuts, region_areas=areas, field_areas=fields)

    assert result.averaging_used == "estimated_cells"
    assert result.percent_area_weighted < 40  # the old answer
    assert result.percent_raw > result.percent_area_weighted + 25  # the new one
    # The sparse region still counts - for the cells it is estimated to hold (about
    # 29 %), not for its area (67 %).
    sparse = next(r for r in result.regions if r.rank == 4)
    assert 0.2 < sparse.weight_share < 0.35


def test_proportional_sampling_reduces_the_estimate_to_the_pooled_ratio(membrane_cuts):
    """Where fields follow area, scaling up changes nothing: same answer as pooling."""
    cells = _field_cells(30, 70, rank=1, field=0, cuts=membrane_cuts)
    cells += _field_cells(80, 20, rank=2, field=0, cuts=membrane_cuts)
    cells += _field_cells(80, 20, rank=2, field=1, cuts=membrane_cuts)
    fields = {(1, 0): 0.05, (2, 0): 0.05, (2, 1): 0.05}

    result = _score(cells, membrane_cuts, region_areas={1: 1.0, 2: 2.0}, field_areas=fields)

    assert result.percent_raw == pytest.approx(result.percent_pooled, abs=0.01)


def test_an_empty_field_lowers_its_regions_weight(membrane_cuts):
    """A field that found no cells is evidence the region holds few; it is not dropped."""
    cells = _field_cells(50, 50, rank=1, field=0, cuts=membrane_cuts)
    cells += _field_cells(100, 0, rank=2, field=0, cuts=membrane_cuts)
    base = {(1, 0): 0.05, (2, 0): 0.05}
    with_empty = {**base, (2, 1): 0.05}
    areas = {1: 1.0, 2: 1.0}

    before = _score(cells, membrane_cuts, region_areas=areas, field_areas=base)
    after = _score(cells, membrane_cuts, region_areas=areas, field_areas=with_empty)

    assert after.percent_raw < before.percent_raw


def test_without_field_areas_the_pooled_ratio_is_reported_and_labelled(membrane_cuts):
    cells = _field_cells(30, 70, rank=1, field=0, cuts=membrane_cuts)
    cells += _field_cells(90, 10, rank=2, field=0, cuts=membrane_cuts)
    result = _score(cells, membrane_cuts, region_areas={1: 9.0, 2: 1.0})
    assert result.averaging_used == "pooled"
    assert result.percent_raw == pytest.approx(result.percent_pooled)


def test_unsampled_regions_are_reported_not_imputed(membrane_cuts):
    cells = _field_cells(50, 50, rank=1, field=0, cuts=membrane_cuts)
    result = _score(
        cells, membrane_cuts,
        region_areas={1: 2.0, 2: 0.03, 3: 0.02},
        field_areas={(1, 0): 0.05},
    )
    assert result.unsampled_regions == 2
    assert result.unsampled_area_mm2 == pytest.approx(0.05)
    assert result.percent_raw == pytest.approx(50.0)


def test_the_interval_narrows_with_more_fields_and_is_reproducible(membrane_cuts):
    def slide(n_fields: int):
        cells: list[dict] = []
        fields = {}
        for field in range(n_fields):
            positive = 70 if field % 2 else 30  # fields disagree, as real ones do
            cells += _field_cells(positive, 100 - positive, rank=1, field=field, cuts=membrane_cuts)
            fields[(1, field)] = 0.05
        return _score(cells, membrane_cuts, region_areas={1: 5.0}, field_areas=fields)

    few, many = slide(2), slide(40)
    assert few.percent_ci_low <= few.percent_raw <= few.percent_ci_high
    assert many.percent_ci_width < few.percent_ci_width / 2
    again = slide(40)
    assert (again.percent_ci_low, again.percent_ci_high) == (
        many.percent_ci_low, many.percent_ci_high,
    )


def test_a_handful_of_cells_gets_a_wide_interval(membrane_cuts):
    cells = _field_cells(5, 5, rank=1, field=0, cuts=membrane_cuts)
    result = _score(cells, membrane_cuts, region_areas={1: 1.0}, field_areas={(1, 0): 0.05})
    assert result.percent_ci_width > 40
