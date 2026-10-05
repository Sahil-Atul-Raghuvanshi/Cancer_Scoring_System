"""The bounded memo steps 3 and 4 hold their last few answers in.

It replaced a single slot. The slot was right while the pipeline looked at one
slide; it became wrong when steps 2-6 became a per-slide prefix, because then the
two things on screen are the H&E and the immunostained slide, and a viewer flipping
between them evicted the one they were coming back to every single time.
"""

import pytest

from app.common.memo import DEFAULT_CAPACITY, BoundedMemo


def test_returns_what_was_put_in() -> None:
    memo: BoundedMemo[int] = BoundedMemo()
    memo.put(("he", 200), 1)
    assert memo.get(("he", 200)) == 1


def test_a_miss_is_none_rather_than_an_error() -> None:
    """Callers branch on None; raising would make every call site a try block."""
    assert BoundedMemo().get("nothing stored") is None


def test_both_slides_of_a_case_stay_resident() -> None:
    """The whole reason this type exists: two slides, no eviction between them."""
    memo: BoundedMemo[str] = BoundedMemo()
    memo.put("he", "H&E")
    memo.put("ihc", "CD44")
    assert memo.get("he") == "H&E"
    assert memo.get("ihc") == "CD44"


def test_eviction_is_least_recently_used_not_oldest() -> None:
    """A viewer alternating between two slides must not lose the one in use.

    With insertion-ordered eviction the entry thrown away is the one that arrived
    first, which - when someone is flipping back and forth - is exactly the one
    they are about to ask for again.
    """
    memo: BoundedMemo[int] = BoundedMemo(capacity=2)
    memo.put("he", 1)
    memo.put("ihc", 2)

    assert memo.get("he") == 1  # touched, so "ihc" is now the stale one
    memo.put("third", 3)

    assert memo.get("he") == 1
    assert memo.get("third") == 3
    assert memo.get("ihc") is None


def test_capacity_is_honoured() -> None:
    memo: BoundedMemo[int] = BoundedMemo(capacity=2)
    for index in range(5):
        memo.put(index, index)
    assert [memo.get(index) for index in range(5)] == [None, None, None, 3, 4]


def test_a_capacity_below_one_is_refused() -> None:
    """Zero would be a memo that never hits - a silent performance cliff."""
    with pytest.raises(ValueError, match="at least 1"):
        BoundedMemo(capacity=0)


def test_clear_forgets_everything() -> None:
    memo: BoundedMemo[int] = BoundedMemo()
    memo.put("he", 1)
    memo.clear()
    assert memo.get("he") is None


def test_the_default_holds_a_case_not_a_slide() -> None:
    """Two slides, two recent settings of the one control that matters."""
    assert DEFAULT_CAPACITY >= 2
