"""Step 7's committed choice, and what happens to step 8's answer when it moves.

Two behaviours, and they are the same behaviour seen from two ends.

**The record.** Step 7's choice used to live nowhere. Step 8's worker asked
`tiling_service` for the index with no arguments, so it got the *configured default*
field of view - which meant a viewer could choose 448 um, watch step 7 lay and price
that grid, and have step 8 classify a 224 um grid underneath them. The two grids then
disagreed and step 8's window gate silently stopped taking its identity fast path.

**The erase.** Once the choice can move, the previous answer has to go. Step 8's
directory holds a report, a class map and four PNGs, and `restart` used to unlink only
the report - leaving `asset()` serving the old pictures and `class_map()` handing step 9
the old labels. A stale answer that looks fresh is worse than no answer.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.pipeline.step08_tissue_type_segmentation.branches import ModelBranch


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Point every directory this test writes into at a temporary one."""
    from app.core.config import settings

    for name in ("tiling_dir", "tissue_type_dir", "roi_dir"):
        target = tmp_path / name
        target.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(settings, name, target, raising=False)
    return tmp_path


# --- the record ---------------------------------------------------------------


def test_the_default_selection_is_the_configured_one(isolated) -> None:
    """Not an error and not a sentinel: "nobody has chosen" and "somebody chose the
    default" produce the same grid, and every caller wants an answer.

    **Read from the settings, not written out here.** This used to assert
    `ModelBranch.H_CHANNEL` literally, which made it a test of one particular
    configuration rather than of the sentence in its own name - so committing the
    pipeline to the H&E branch failed it for the one reason it should not have.
    `committed_branch()` and `DEFAULT_BRANCH` are two different questions now, and the
    selection is built from the first: what this pipeline runs, rather than what a
    manifest with no declared channel means.
    """
    from app.core.config import settings
    from app.services.tiling_service import committed_branch, tiling_service

    selection = tiling_service.selection("never-chosen")
    assert selection.branch == committed_branch().value
    assert selection.branch == settings.tiling_branch
    assert selection.field_of_view_um == settings.tiling_field_of_view_um
    assert selection.overlap == settings.tiling_overlap


def test_a_committed_choice_is_read_back_with_the_geometry_it_resolves_to(isolated) -> None:
    from app.services.tiling_service import tiling_service

    committed = tiling_service.commit_selection(
        "slide-1", branch="h_channel", fov=448.0
    )
    assert committed.field_of_view_um == 448.0

    # 224 px at 2.0 um/px. The suite repoints `data_dir`, so no checkpoint is
    # published here and this is the *planned* geometry - which is the point worth
    # pinning: the record resolves a real square either way, so the grid step 8 will
    # rebuild is the grid step 7 priced even before a head exists.
    assert committed.tile_px == 224
    assert committed.mpp == 2.0

    again = tiling_service.selection("slide-1")
    assert again.field_of_view_um == 448.0
    assert again.branch == "h_channel"
    assert again.tile_px == 224 and again.mpp == 2.0


def test_the_selection_outlives_the_memo(isolated) -> None:
    """The record is on disk precisely so it survives eviction. The memo holds four
    entries now, and a fifth choice pushes the first out - which must not lose the
    committed answer."""
    from app.services.tiling_service import tiling_service

    tiling_service.commit_selection("slide-2", branch="h_channel", fov=672.0)
    tiling_service._memo.clear()
    assert tiling_service.selection("slide-2").field_of_view_um == 672.0


def test_beetle_commits_and_records_its_own_geometry(isolated) -> None:
    """`beetle` is a served branch now, and committing it records BEETLE's grid.

    This test used to assert the opposite - that committing `beetle` was refused as not
    developed - and it is inverted rather than deleted because the *geometry* is the
    part worth pinning. BEETLE holds its resolution at the 0.5 um/px it was fitted at
    and varies the pixel side, so 224 um is a 448 px window; the two ResNet branches
    hold the pixel side at 224 and vary the resolution instead. A commit that recorded
    224 px here would be recording our geometry under BEETLE's name, and the pass would
    then read tissue at four times the scale the network was trained on - which does not
    fail, it returns a confident wrong segmentation.

    Deliberately independent of whether the 1.9 GB archive is on disk: `window_px` is
    arithmetic on a published constant, so the grid can be laid and priced on a machine
    that has never downloaded the weights. Step 8 refuses separately, naming the file it
    wants.
    """
    from app.services.tiling_service import tiling_service

    record = tiling_service.commit_selection("slide-3", branch="beetle", fov=224.0)
    assert record.branch == "beetle"
    assert record.mpp == 0.5
    assert record.tile_px == 448
    assert tiling_service.selection("slide-3").branch == "beetle"


def test_a_branch_off_the_served_list_still_cannot_be_committed(isolated, monkeypatch) -> None:
    """The refusal itself, with the served list narrowed to make it reachable.

    Nothing is currently off `SERVED_BRANCHES` - all three run - so this is the only way
    left to exercise the guard, and it is worth exercising: it is what stands between a
    fourth name added to the wire enum and a selection record naming a branch no grid can
    be laid for. Narrowing the list rather than inventing a branch, because an invented
    name would be refused one line earlier by `parse_branch` and would test the typo path
    instead of this one.
    """
    from app.services import tiling_service as service

    monkeypatch.setattr(service, "SERVED_BRANCHES", (ModelBranch.H_CHANNEL,))
    with pytest.raises(service.TilingError, match="not developed yet"):
        service.tiling_service.commit_selection("slide-3b", branch="beetle", fov=224.0)


def test_a_field_of_view_nobody_offers_is_refused_at_commit(isolated) -> None:
    from app.services.tiling_service import TilingError, tiling_service

    with pytest.raises(TilingError, match="not one of the fields of view"):
        tiling_service.commit_selection("slide-4", branch="h_channel", fov=300.0)


def test_forgetting_a_slide_drops_its_record_and_its_memo_entries(isolated) -> None:
    from app.services.tiling_service import tiling_service

    tiling_service.commit_selection("slide-5", branch="h_channel", fov=112.0)
    tiling_service.forget_selection("slide-5")
    assert tiling_service.selection("slide-5").field_of_view_um != 112.0


def test_an_unreadable_record_falls_back_rather_than_crashing(isolated) -> None:
    """A corrupt selection file must not make the slide unopenable. The default grid
    is a worse answer than the viewer's choice and a far better one than a 500."""
    from app.services.tiling_service import committed_branch, tiling_service

    path = tiling_service._selection_path("slide-6")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ this is not json", encoding="utf-8")

    selection = tiling_service.selection("slide-6")
    assert selection.branch == committed_branch().value


# --- the erase ----------------------------------------------------------------


def _plant_cached_pass(upload_id: str, *, key: dict) -> "object":
    """A finished step 8 pass on disk: the report, the class map and all four PNGs.

    Planted rather than computed, because what is under test is what happens to these
    files - not how they were made.
    """
    from app.services.tissue_type_service import tissue_type_service

    directory = tissue_type_service._dir(upload_id)
    directory.mkdir(parents=True, exist_ok=True)

    (directory / "report.json").write_text(
        json.dumps({
            "_key": key,
            "run": {"upload_id": upload_id, "state": "done"},
            "params": key,
        }),
        encoding="utf-8",
    )
    np.savez_compressed(
        directory / "classmap.npz",
        labels=np.zeros((4, 4), dtype=np.uint8),
        probabilities=np.zeros((3, 4, 4), dtype=np.float32),
        inside=np.ones((4, 4), dtype=bool),
        geometry=np.zeros(11, dtype=np.float64),
        counters=np.zeros(3, dtype=np.int64),
    )
    for name in ("map.png", "flat.png", "scored.png", "confidence.png"):
        (directory / name).write_bytes(b"\x89PNG\r\n\x1a\n")
    return directory


def test_changing_the_field_of_view_erases_the_cached_class_map(isolated) -> None:
    """The reported symptom: the field of view moved and step 8's previous answer
    stayed on screen beside numbers that no longer described it."""
    from app.services.tissue_type_service import tissue_type_service

    directory = _plant_cached_pass(
        "erase-1",
        key={"model": "invasive_tile_fov224_concat", "field_of_view_um": 224.0,
             "branch": "h_channel", "input_channel": "haematoxylin"},
    )
    assert (directory / "classmap.npz").exists()

    erased = tissue_type_service._erase_stale(
        "erase-1",
        {"model": "invasive_tile_fov448_concat", "field_of_view_um": 448.0,
         "branch": "h_channel", "input_channel": "haematoxylin"},
    )
    assert erased
    assert not directory.exists(), "the whole directory goes, not only the report"


def test_changing_the_branch_erases_the_cached_class_map(isolated) -> None:
    """Same field of view, different input contract. The class map is a statement
    about what the model was shown, so it does not survive the branch moving."""
    from app.services.tissue_type_service import tissue_type_service

    directory = _plant_cached_pass(
        "erase-2",
        key={"model": "invasive_tile_fov224_concat", "field_of_view_um": 224.0,
             "branch": "h_channel", "input_channel": "haematoxylin"},
    )
    erased = tissue_type_service._erase_stale(
        "erase-2",
        {"model": "invasive_tile_fov224_he_concat", "field_of_view_um": 224.0,
         "branch": "he", "input_channel": "rgb_he"},
    )
    assert erased
    assert not directory.exists()


def test_a_matching_choice_still_returns_the_cached_pass(isolated) -> None:
    """The negative control, and it matters as much as the positive one: that cache is
    half an hour of somebody's CPU, and revisiting the screen has to stay instant."""
    from app.services.tissue_type_service import tissue_type_service

    key = {"model": "invasive_tile_fov224_concat", "field_of_view_um": 224.0,
           "branch": "h_channel", "input_channel": "haematoxylin"}
    directory = _plant_cached_pass("erase-3", key=key)

    assert not tissue_type_service._erase_stale("erase-3", dict(key))
    assert (directory / "classmap.npz").exists()
    assert (directory / "map.png").exists()


def test_discarding_takes_the_region_built_from_it_too(isolated) -> None:
    """Step 9 caches on its *own* parameters, so it cannot notice that the labels
    underneath it changed. A region traced from a class map that no longer exists
    looks current, and nothing downstream can tell."""
    from app.services.roi_service import roi_service
    from app.services.tissue_type_service import tissue_type_service

    directory = _plant_cached_pass(
        "erase-4",
        key={"model": "m", "field_of_view_um": 224.0, "branch": "h_channel"},
    )
    roi_dir = roi_service._dir("erase-4")
    roi_dir.mkdir(parents=True, exist_ok=True)
    (roi_dir / "report.json").write_text("{}", encoding="utf-8")

    tissue_type_service.discard("erase-4")
    assert not directory.exists()
    assert not roi_dir.exists(), "step 9 reads step 8's class map, so it goes too"


def test_discarding_refuses_while_a_pass_is_running(isolated) -> None:
    """The worker writes into this directory when it finishes. Deleting underneath it
    would leave exactly the half-state the discard exists to prevent."""
    from app.services.tissue_type_service import Job, TissueTypeError, tissue_type_service

    _plant_cached_pass(
        "erase-5", key={"model": "m", "field_of_view_um": 224.0, "branch": "h_channel"}
    )
    job = Job(upload_id="erase-5", params={})
    job.state = "running"
    tissue_type_service._jobs["erase-5"] = job
    try:
        with pytest.raises(TissueTypeError, match="cancel it"):
            tissue_type_service.discard("erase-5")
    finally:
        tissue_type_service._jobs.pop("erase-5", None)


def test_the_cache_key_carries_the_branch_and_the_field_of_view() -> None:
    """A changed choice must invalidate even when the resolved checkpoint happens to
    be the same file - and a cache written before these keys existed must fail the
    comparison rather than raise, which is what `get` is for."""
    from app.services.tissue_type_service import tissue_type_service

    key = tissue_type_service._cache_key(
        {"model": "m", "branch": "he", "field_of_view_um": 448.0,
         "input_channel": "rgb_he"}
    )
    assert key["branch"] == "he"
    assert key["field_of_view_um"] == 448.0
    assert key["input_channel"] == "rgb_he"

    legacy = tissue_type_service._cache_key({"model": "m"})
    assert legacy["branch"] is None
    assert legacy != key
