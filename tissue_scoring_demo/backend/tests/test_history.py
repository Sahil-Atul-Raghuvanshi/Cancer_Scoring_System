"""Guards on the tables that decide what moves when a run is filed or re-opened.

Filing a run is a *rename* rather than a copy - a marker's artefacts are about half a
gigabyte and a case's five are two and a half - so the tables in `history_service` are
not bookkeeping. A name in the wrong one, or in none, means somebody's work is moved
somewhere nothing looks for it, or not moved at all and read as an orphan by the
start-up sweep.
"""

from __future__ import annotations

from app.services import history_service


def test_every_named_tree_has_a_directory() -> None:
    """A tree named in one of the three tuples but absent from `_trees` is a KeyError.

    And it is a `KeyError` raised *inside* `open_marker` or `archive_marker` - halfway
    through renaming half a gigabyte of somebody's work out of the working tree. This
    caught exactly that, on the day steps 10 and 11 were added to `SHARED_TREES` and
    not to the table beside it.
    """
    mapped = set(history_service._trees())
    named = set(
        history_service.SHARED_TREES
        + history_service.IHC_TREES
        + history_service.PAIR_TREES
    )
    assert named <= mapped, sorted(named - mapped)


def test_the_he_keyed_trees_are_shared_and_not_per_marker() -> None:
    """Steps 7 to 11 are computed once per case and read by all five antibodies.

    Step 11's is the one that matters most: it is minutes of BEETLE per region *and*
    the tumour mask every marker is scored inside. Filed as a marker's own, it would be
    re-computed per antibody and the five would disagree about where the tumour is.
    """
    for name in ("tiling", "tissue_type", "roi", "roi_selection", "roi_refinement"):
        assert name in history_service.SHARED_TREES
        assert name not in history_service.PAIR_TREES
        assert name not in history_service.IHC_TREES


def test_the_score_is_the_last_pair_step() -> None:
    """`_state_from` calls a marker complete on this step, so it has to be the score."""
    assert history_service.PAIR_STEPS[-1][1] == "scores"
    assert str(history_service.PAIR_STEPS[-1][0]) == history_service.LAST_PAIR_STEP


def test_a_manifest_written_before_the_renumber_still_reads_as_complete() -> None:
    """Filed cases on disk say `"16": true`, and the score is step 18 now.

    They cannot be rewritten by reading them, so the old key is accepted alongside the
    current one - otherwise every previously scored case would list as "partial" with a
    score sitting in it.
    """
    assert history_service._state_from({"16": True}) == "complete"
    assert history_service._state_from({history_service.LAST_PAIR_STEP: True}) == "complete"
    assert history_service._state_from({"13": True}) == "partial"
    assert history_service._state_from({}) == "not-started"
