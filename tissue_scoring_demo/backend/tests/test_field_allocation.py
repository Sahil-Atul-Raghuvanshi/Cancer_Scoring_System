"""How the sampling budget is split between regions.

The allocation IS the sample, so a wrong split is not a loss of precision - it
is a measurement of a different population. Giving every region the same number
of fields makes the measured cells a sample of the regions rather than of the
tumour, and on CAN_00270 that put 42 per cent of the cells inside a region
holding 84 per cent of the invasive area.
"""

from __future__ import annotations

from app.core.config import settings
from app.pipeline.step13_nuclei_segmentation.sampling import allocate


def test_fields_follow_area_not_region_count():
    """The failure this replaces: one region ten times the size, same fields."""
    big, small = allocate([20.0, 2.0], budget=44, minimum=2)
    assert big > small * 5


def test_the_budget_is_spent_exactly():
    shares = allocate([5.0, 3.0, 2.0], budget=50, minimum=2)
    assert sum(shares) == 50


def test_every_region_clears_the_floor():
    """A region with one field has a percentage of 0 or 100 and no way to tell."""
    shares = allocate([100.0, 0.5, 0.5], budget=40, minimum=4)
    assert min(shares) >= 4


def test_a_large_floor_would_swamp_a_proportional_split():
    """Why the shipped floor is one.

    Covering a real tumour takes about twenty regions. At a floor of four that
    is eighty of a ninety-six field budget gone before proportionality is
    considered, and the region that holds most of the tumour ends up with a
    minority of the sample - which is the failure proportional allocation was
    added to fix.
    """
    areas = [22.07, 2.11, 2.01, 1.3, 1.25, 0.8] + [0.4] * 13

    swamped = allocate(areas, budget=96, minimum=4)
    proper = allocate(areas, budget=96, minimum=1)

    share = areas[0] / sum(areas)
    assert swamped[0] / sum(swamped) < share / 2  # the big region loses most of its say
    assert proper[0] / sum(proper) > swamped[0] / sum(swamped) * 2


def test_the_floor_wins_when_the_budget_cannot_cover_it():
    """The floor is a validity condition; the budget is a cost control.

    Sampling a region too thinly to read, in order to respect a cost ceiling,
    buys nothing - the fields spent on it are wasted either way.
    """
    shares = allocate([1.0] * 10, budget=12, minimum=4)
    assert shares == [4] * 10


def test_proportional_sampling_makes_the_two_averagings_agree():
    """Why this is a correctness fix rather than a tuning knob.

    With fields allocated in proportion to area, a region's share of the sample
    equals its share of the tumour - so the plain mean over all sampled fields
    and the area-weighted mean of the regions are the same number. Open question
    Q3 stops changing the answer instead of being resolved by fiat.
    """
    areas = [22.07, 2.11, 2.01, 1.30]
    shares = allocate(areas, budget=400, minimum=1)

    total_area = sum(areas)
    total_fields = sum(shares)
    for area, fields in zip(areas, shares, strict=True):
        assert abs(fields / total_fields - area / total_area) < 0.02


def test_no_regions_allocates_nothing():
    assert allocate([]) == []


def test_a_zero_budget_is_honoured_rather_than_floored():
    assert allocate([1.0, 2.0], budget=0) == [0, 0]


def test_the_defaults_come_from_settings():
    shares = allocate([1.0, 1.0])
    assert sum(shares) == settings.nuclei_field_budget
    assert min(shares) >= settings.nuclei_min_tiles_per_region
