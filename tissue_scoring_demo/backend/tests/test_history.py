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


# --- P-19: moving a marker, with real directories ------------------------------

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from app.core.config import settings  # noqa: E402

_TREE_SETTINGS = (
    "qc_dir", "tissue_dir", "calibration_dir", "tiling_dir", "tissue_type_dir", "roi_dir",
    "roi_selection_dir", "roi_refinement_dir", "ihc_alignment_dir", "nuclei_dir",
    "cell_typing_dir", "compartments_dir", "per_cell_dir", "binning_dir", "scores_dir",
)


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A live tree and a history tree of their own, so no real run can be moved."""
    for attribute in ("slides_dir", *_TREE_SETTINGS):
        directory = tmp_path / "demo" / attribute.removesuffix("_dir")
        directory.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(settings, attribute, directory)
    monkeypatch.setattr(settings, "data_dir", tmp_path / "demo")
    monkeypatch.setattr(history_service, "HISTORY_ROOT", tmp_path / "history")
    monkeypatch.setattr(history_service, "_ensure_thumbnail", lambda *_: None)
    return tmp_path


def _run(case_id: str, marker: str, he: str, ihc: str, *, scored: bool = True) -> None:
    """Lay down what a finished marker leaves in the live tree, and register it."""
    pair = f"{he}__{ihc}"
    (settings.slides_dir / f"{he}.json").write_text("{}", encoding="utf-8")
    (settings.slides_dir / f"{ihc}.json").write_text("{}", encoding="utf-8")
    (settings.tissue_type_dir / he).mkdir(exist_ok=True)
    (settings.tissue_type_dir / he / "report.json").write_text("{}", encoding="utf-8")
    steps = ("ihc_alignment_dir", "scores_dir") if scored else ("ihc_alignment_dir",)
    for attribute in steps:
        directory = getattr(settings, attribute) / pair
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "report.json").write_text(json.dumps({"score": {}}), encoding="utf-8")
    cases = settings.data_dir / "cases"
    cases.mkdir(exist_ok=True)
    (cases / f"{case_id}_{marker}.json").write_text(
        json.dumps({"case_id": case_id, "marker": marker, "he_upload_id": he, "ihc_upload_id": ihc}),
        encoding="utf-8",
    )


def test_a_marker_records_the_he_it_was_run_against(tree: Path):
    _run("CASE", "A", "heOne", "ihcA")
    history_service.refresh("CASE")
    manifest = history_service.load_manifest("CASE")
    assert manifest["markers"]["A"]["heUploadId"] == "heOne"


def test_two_hes_in_one_case_each_file_and_open_their_own_work(tree: Path):
    """The case-level id used to be assumed for every marker; the second H&E's pairs
    were looked for under the first one's key, nothing moved, and the marker was still
    recorded as filed."""
    _run("CASE", "A", "heOne", "ihcA")
    _run("CASE", "F", "heTwo", "ihcF")
    history_service.refresh("CASE")

    history_service.archive_marker("CASE", "A")
    history_service.archive_marker("CASE", "F")

    for he, ihc in (("heOne", "ihcA"), ("heTwo", "ihcF")):
        assert not (settings.scores_dir / f"{he}__{ihc}").exists(), "left live"
        assert not (settings.tissue_type_dir / he).exists(), "shared work left live"
    filed = tree / "history" / "CASE"
    assert (filed / "markers" / "F" / "scores" / "heTwo__ihcF").is_dir()
    assert (filed / "shared" / "tissue_type" / "heTwo").is_dir()

    history_service.open_marker("CASE", "F")
    assert (settings.scores_dir / "heTwo__ihcF").is_dir()
    assert (settings.tissue_type_dir / "heTwo").is_dir()
    assert not (settings.tissue_type_dir / "heOne").exists(), "only F's H&E comes back"


def test_shared_work_stays_while_another_marker_of_that_he_is_open(tree: Path):
    _run("CASE", "A", "heOne", "ihcA")
    _run("CASE", "F", "heOne", "ihcF")
    history_service.refresh("CASE")

    history_service.archive_marker("CASE", "A")
    assert (settings.tissue_type_dir / "heOne").is_dir()
    history_service.archive_marker("CASE", "F")
    assert not (settings.tissue_type_dir / "heOne").exists()


def test_filing_refuses_when_a_finished_step_is_missing_and_moves_nothing(tree: Path):
    """`_move` used to return False on a missing source and the marker was still
    recorded as `history` - a record pointing at nothing."""
    _run("CASE", "A", "heOne", "ihcA")
    history_service.refresh("CASE")
    import shutil

    shutil.rmtree(settings.scores_dir / "heOne__ihcA")  # the score the manifest says exists

    with pytest.raises(history_service.HistoryError, match="Nothing was moved"):
        history_service.archive_marker("CASE", "A")

    assert history_service.load_manifest("CASE")["markers"]["A"]["location"] == "demo"
    assert (settings.ihc_alignment_dir / "heOne__ihcA").is_dir(), "a partial move happened"


def test_a_partial_run_still_files(tree: Path):
    """A marker that never reached the score has no score directory, and that is fine."""
    _run("CASE", "A", "heOne", "ihcA", scored=False)
    history_service.refresh("CASE")
    history_service.archive_marker("CASE", "A")
    assert history_service.load_manifest("CASE")["markers"]["A"]["location"] == "history"
