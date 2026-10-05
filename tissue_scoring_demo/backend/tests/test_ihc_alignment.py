"""Step 10 - the parts that can be tested without registering two real slides.

The registration itself is fitted beforehand by `slide_registration/` on two
gigabyte scans, and `transform_warp.py --selftest` checks it. What is pinned here is
everything around it: the geometry that prepares a border for warping, and the
gate that decides whether a warped border is allowed out at all.
"""

from __future__ import annotations

import math

import pytest

from app.registration.exceptions import RegistrationRefused
from app.registration.gate import check, evaluate
from app.registration.transform import densify_ring, simplify_ring, um_to_px


def _passing() -> dict:
    return {
        "matched_keypoints": 400,
        "residual_error_um": 20.0,
        "tissue_area_ratio": 1.0,
        "round_trip_median_um": 1.0,
    }


# --- densifying, which is what makes a non-rigid warp follow the tissue -------


def test_densify_adds_points_so_no_edge_outruns_the_spacing():
    ring = ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0))
    dense = densify_ring(ring, max_spacing_px=10.0)

    for index in range(len(dense)):
        x0, y0 = dense[index]
        x1, y1 = dense[(index + 1) % len(dense)]
        assert math.hypot(x1 - x0, y1 - y0) <= 10.0 + 1e-9


def test_densify_closes_the_ring():
    """The last-to-first edge is a border too, and a warp bends it just the same."""
    ring = ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0))
    dense = densify_ring(ring, max_spacing_px=10.0)

    closing_edge = math.hypot(dense[-1][0] - dense[0][0], dense[-1][1] - dense[0][1])
    assert closing_edge <= 10.0 + 1e-9


def test_densify_keeps_every_original_vertex():
    ring = ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0))
    dense = [tuple(point) for point in densify_ring(ring, max_spacing_px=7.0)]
    for vertex in ring:
        assert vertex in dense


def test_densify_leaves_a_ring_alone_when_it_is_already_fine_enough():
    ring = ((0.0, 0.0), (5.0, 0.0), (5.0, 5.0))
    assert densify_ring(ring, max_spacing_px=100.0) == [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]]


# --- simplifying, which must not move a border a viewer is judging -----------


def test_simplify_drops_collinear_points_but_keeps_the_shape():
    straight = [[float(x), 0.0] for x in range(0, 101, 5)]
    assert simplify_ring(straight, tolerance_px=0.5) == [[0.0, 0.0], [100.0, 0.0]]


def test_simplify_keeps_a_corner_that_matters():
    corner = [[0.0, 0.0], [50.0, 0.0], [50.0, 50.0]]
    assert simplify_ring(corner, tolerance_px=0.5) == corner


def test_simplify_never_moves_a_kept_point():
    ring = [[0.0, 0.0], [10.0, 1.0], [20.0, 0.0], [30.0, 40.0]]
    simplified = simplify_ring(ring, tolerance_px=2.0)
    for point in simplified:
        assert point in ring


# --- microns to pixels --------------------------------------------------------


def test_um_to_px_uses_the_slide_scale():
    assert um_to_px(40.0, 0.25) == pytest.approx(160.0)


def test_um_to_px_falls_back_to_pixels_when_a_slide_records_no_scale():
    """A file with no mpp still has to run; refusing would fail a whole slide."""
    assert um_to_px(40.0, None) == 40.0


# --- the gate -----------------------------------------------------------------


def test_a_good_registration_passes():
    assert evaluate(_passing()) == []
    check(_passing())  # does not raise


def test_too_few_keypoints_is_refused():
    diagnostics = _passing() | {"matched_keypoints": 3}
    reasons = evaluate(diagnostics)
    assert len(reasons) == 1
    assert "matched features" in reasons[0]


def test_a_near_blank_section_is_refused_on_tissue_area():
    """The docs' own worked example: 17.5 mm2 of IHC against 84.8 of H&E."""
    diagnostics = _passing() | {
        "tissue_area_ratio": round(17.5 / 84.8, 3),
        "he_tissue_mm2": 84.8,
        "ihc_tissue_mm2": 17.5,
    }
    reasons = evaluate(diagnostics)
    assert any("tissue-area ratio" in reason for reason in reasons)


def test_a_large_residual_is_refused():
    reasons = evaluate(_passing() | {"residual_error_um": 5000.0})
    assert any("apart after warping" in reason for reason in reasons)


def test_a_transform_that_does_not_agree_with_itself_is_refused():
    reasons = evaluate(_passing() | {"round_trip_median_um": 900.0})
    assert any("round-trip" in reason for reason in reasons)


def test_every_failing_measure_is_reported_not_just_the_first():
    """A refusal has to explain itself; one reason out of three is not an explanation."""
    diagnostics = {
        "matched_keypoints": 2,
        "residual_error_um": 5000.0,
        "tissue_area_ratio": 0.1,
        "round_trip_median_um": 900.0,
    }
    assert len(evaluate(diagnostics)) == 4


def test_check_raises_with_the_numbers_attached():
    diagnostics = _passing() | {"matched_keypoints": 1}
    with pytest.raises(RegistrationRefused) as raised:
        check(diagnostics)
    assert raised.value.reasons
    assert raised.value.diagnostics["matched_keypoints"] == 1


def test_missing_measurements_do_not_silently_pass_the_keypoint_gate():
    """An absent keypoint count is not evidence of a good registration."""
    assert evaluate({}) != []


def test_an_unmeasured_residual_is_refused_rather_than_skipped():
    """A check that never ran must not read as a check that passed.

    This one has been silently disabled for real: a `dst_dir` deep enough to
    pass Windows' 260-character path limit made VALIS write no summary, so the
    residual arrived as None and three gates quietly did the work of four.
    """
    diagnostics = _passing()
    del diagnostics["residual_error_um"]
    reasons = evaluate(diagnostics)
    assert any("no residual error was recorded" in reason for reason in reasons)


def test_the_reason_carries_the_worker_s_own_explanation_when_it_has_one():
    diagnostics = _passing() | {
        "residual_error_um": None,
        "measure_error": "VALIS recorded no registration summary.",
    }
    reasons = evaluate(diagnostics)
    assert any("VALIS recorded no registration summary." in reason for reason in reasons)
