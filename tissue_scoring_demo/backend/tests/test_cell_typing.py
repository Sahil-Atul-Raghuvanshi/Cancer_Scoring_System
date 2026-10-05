"""Step 12 - cell typing.

The rules are the output here, so the tests are mostly statements about what the
rules mean: a small round dark nucleus is an immune cell, an elongated one is a
support cell, and everything else inside an invasive region is tumour. Plus the
two properties that matter more than the classes themselves - that the answer
says how much it depends on its own thresholds, and that it refuses to present a
mix when the nuclei underneath are too small to be cells.
"""

from __future__ import annotations

import numpy as np

from app.pipeline.step14_cell_typing.classify import (
    CellType,
    Rules,
    classify,
    sensitivity,
)
from app.services.cell_typing_service import CellTypingService


def one(area: float, circ: float, ecc: float, haem: float) -> tuple[np.ndarray, ...]:
    return (
        np.array([area]),
        np.array([circ]),
        np.array([ecc]),
        np.array([haem]),
    )


def test_small_round_and_dark_is_an_immune_cell() -> None:
    # Darkness is relative to the field median, so a second, paler nucleus is
    # needed for "dark" to mean anything at all.
    areas = np.array([25.0, 60.0])
    circ = np.array([0.85, 0.60])
    ecc = np.array([0.30, 0.50])
    haem = np.array([0.9, 0.5])

    labels = classify(areas, circ, ecc, haem).labels

    assert labels[0] == CellType.LYMPHOCYTE
    assert labels[1] == CellType.TUMOUR


def test_elongated_is_a_support_cell() -> None:
    labels = classify(*one(60.0, 0.35, 0.93, 1.0)).labels
    assert labels[0] == CellType.SPINDLE


def test_large_and_irregular_is_tumour() -> None:
    labels = classify(*one(120.0, 0.55, 0.45, 1.0)).labels
    assert labels[0] == CellType.TUMOUR


def test_tumour_is_the_default_inside_an_invasive_region() -> None:
    """Anything that fails both specific rules counts, rather than being dropped.

    Making tumour a positive test would shrink the denominator every time a
    nucleus was merely unusual, which is a bias in the direction that inflates
    the score.
    """
    rng = np.random.default_rng(0)
    count = 500
    labels = classify(
        rng.uniform(40, 150, count),
        rng.uniform(0.2, 0.7, count),
        rng.uniform(0.1, 0.8, count),
        rng.uniform(0.4, 1.2, count),
    ).labels
    assert (labels == CellType.TUMOUR).all()


def test_a_big_elongated_object_is_not_called_a_fibroblast() -> None:
    """More likely two tumour nuclei segmented as one than one giant spindle cell."""
    labels = classify(*one(400.0, 0.3, 0.95, 1.0)).labels
    assert labels[0] == CellType.TUMOUR


def test_darkness_is_relative_so_a_pale_slide_is_not_all_lymphocytes() -> None:
    """The same shapes at half the stain uptake must give the same classes."""
    areas = np.array([25.0, 25.0, 60.0, 60.0])
    circ = np.array([0.85, 0.85, 0.5, 0.5])
    ecc = np.array([0.2, 0.2, 0.5, 0.5])
    strong = np.array([1.4, 0.8, 0.9, 0.7])

    assert np.array_equal(
        classify(areas, circ, ecc, strong).labels,
        classify(areas, circ, ecc, strong * 0.4).labels,
    )


def test_a_field_with_no_counterstain_does_not_call_everything_dark() -> None:
    """Dividing by a near-zero median would make every nucleus infinitely dark."""
    labels = classify(
        np.array([20.0, 20.0]),
        np.array([0.9, 0.9]),
        np.array([0.2, 0.2]),
        np.zeros(2),
    ).labels
    assert (labels == CellType.TUMOUR).all()


def test_no_nuclei_is_an_empty_answer_not_a_crash() -> None:
    typed = classify(np.empty(0), np.empty(0), np.empty(0), np.empty(0))
    assert typed.labels.size == 0
    assert typed.counts[int(CellType.TUMOUR)] == 0


def test_the_sensitivity_sweep_reports_how_much_each_threshold_decides() -> None:
    """The number that says whether the threshold or the data is doing the work."""
    rng = np.random.default_rng(1)
    count = 400
    areas = rng.uniform(15, 120, count)
    circ = rng.uniform(0.3, 0.95, count)
    ecc = rng.uniform(0.1, 0.95, count)
    haem = rng.uniform(0.5, 1.5, count)

    sweeps = sensitivity(areas, circ, ecc, haem)

    names = {entry["parameter"] for entry in sweeps}
    assert "lymphocyte_max_area_um2" in names
    for entry in sweeps:
        assert len(entry["points"]) >= 3
        assert entry["swing"] >= 0.0
        # The sweep has to actually move the answer on data this spread out, or
        # it is not telling anyone anything.
        if entry["parameter"] == "lymphocyte_max_area_um2":
            assert entry["swing"] > 0.0


def test_moving_a_threshold_moves_the_mix_in_the_direction_it_should() -> None:
    areas = np.array([20.0, 30.0, 40.0, 50.0])
    circ = np.full(4, 0.9)
    ecc = np.full(4, 0.2)
    # Darkness is relative to the field median, so the two small nuclei have to
    # be dark *against the other two* for the rule to fire at all.
    haem = np.array([1.6, 1.6, 0.5, 0.5])

    strict = classify(areas, circ, ecc, haem, rules=Rules(lymphocyte_max_area_um2=25.0))
    loose = classify(areas, circ, ecc, haem, rules=Rules(lymphocyte_max_area_um2=45.0))

    assert loose.counts[int(CellType.LYMPHOCYTE)] > strict.counts[int(CellType.LYMPHOCYTE)]


# --- the honesty term -------------------------------------------------------


def test_nuclei_too_small_to_be_cells_make_the_answer_untrustworthy() -> None:
    """The CD44 case: a 15 um2 median means these rules are sorting fragments."""
    trustworthy, reason = CellTypingService._trust(15.0, 0.72)

    assert trustworthy is False
    assert reason is not None
    assert "15" in reason
    # The two independent measurements of the same failure are reported together.
    assert "72%" in reason


def test_a_healthy_median_area_is_trusted() -> None:
    """43 um2 is what this project's own H&E measures."""
    trustworthy, reason = CellTypingService._trust(43.0, 0.0)
    assert trustworthy is True
    assert reason is None


def test_the_floor_is_below_a_real_nucleus_and_above_debris() -> None:
    """Pins the constant: a 7 um nucleus is ~38 um2, so the floor must sit under it."""
    assert 10.0 < CellTypingService.MIN_CREDIBLE_MEDIAN_AREA_UM2 < 38.0


def test_the_shipped_thresholds_are_the_documented_ones() -> None:
    rules = Rules()
    assert rules.lymphocyte_max_area_um2 == 35.0
    assert rules.lymphocyte_min_circularity == 0.72
    assert rules.spindle_min_eccentricity == 0.85
