"""The borrowed classes: do they stay separable from the real ones?

BCSS supplies nine human-drawn `dcis` tiles in the whole release. Class 1 is therefore
borrowed from BRACS, labelled by BEETLE's released nnU-Net, and exported by the same
exporter into the same directories. Once written, a borrowed tile and a real one are
byte-identical in kind: 224 px, uint8, mode `L`, in the same class folder.

Every test here guards the one thing that keeps that honest - the `source` column - and
the split rules built on it. All four failures below produce a model that trains, scores
well, and is reporting something other than what it claims:

  * a borrowed tile in the training half of a patient whose other tiles are in test
  * the nine real tiles left in training, so class-1 recall is again measured on nothing
  * the nine real tiles averaged in with thousands of teacher-labelled ones, so the
    headline reports agreement with BEETLE as though it were accuracy
  * a tile from one lesion tree voting on a class its region's consensus never covered

**Three trees, not one.** `dcis`, `ic` and `normal` are separate `source` values because
they are separate claims: a DCIS consensus corroborates BEETLE's in-situ and contradicts
its invasive, an IC consensus does the opposite, and a normal consensus licenses both
non-carcinoma answers while never adjudicating invasive against in-situ. The rules below
that used to be about "the borrowed half" are about all three, and the tests say so.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import bcss  # noqa: E402
import datasets  # noqa: E402
import report  # noqa: E402


def row(
    tile_id: str,
    label: int,
    *,
    slide: str,
    institution: str,
    source: str = "bcss",
) -> dict[str, str]:
    return {
        "tile_id": tile_id,
        "slide_id": slide,
        "institution": institution,
        "label": str(label),
        "label_name": bcss.CLASS_NAMES[label],
        "source": source,
        "roi_id": tile_id.split("__")[0],
        "non_invasive_frac": "0.0",
    }


def bcss_rows() -> list[dict[str, str]]:
    """A miniature BCSS: two training institutions, one held-out, nine real class 1."""
    rows = []
    # A training institution with no in-situ at all.
    for i in range(20):
        rows.append(row(f"t{i}", i % 2 * 2, slide="TCGA-A1-0001", institution="A1"))
    # The three slides that really do carry human-drawn in-situ tiles.
    for slide, n_one in (("TCGA-AR-A2LH", 7), ("TCGA-A2-A0D0", 1), ("TCGA-A2-A1G6", 1)):
        institution = slide.split("-")[1]
        for i in range(n_one):
            rows.append(row(f"{slide}_one{i}", bcss.NON_INVASIVE,
                            slide=slide, institution=institution))
        for i in range(5):
            rows.append(row(f"{slide}_zero{i}", bcss.NON_EPITHELIUM,
                            slide=slide, institution=institution))
    # A published held-out institution.
    for i in range(20):
        rows.append(row(f"h{i}", i % 2 * 2, slide="TCGA-OL-0002", institution="OL"))
    return rows


def dcis_rows(cases: int = 12, per_case: int = 6) -> list[dict[str, str]]:
    return borrowed_rows("dcis", bcss.NON_INVASIVE, cases=cases, per_case=per_case)


def borrowed_rows(tree_name: str, label: int, *, cases: int = 12,
                  per_case: int = 6, case_offset: int = 0) -> list[dict[str, str]]:
    """Tiles from one BRACS lesion tree, stamped with that tree's own `source`.

    `case_offset` exists so two trees can be given disjoint patients or overlapping
    ones deliberately - the overlap is what
    `test_a_patient_in_two_trees_lands_on_one_side` is about.
    """
    tree = bcss.BORROWED_TREES[tree_name]
    suffix = {"dcis": "DCIS", "ic": "IC", "normal": "N"}[tree_name]
    return [
        row(f"BRACS_{100 + case_offset + case}_{suffix}_1__r{i:03d}", label,
            slide=f"BRACS_{100 + case_offset + case}",
            institution=bcss.BRACS_INSTITUTION, source=tree.source)
        for case in range(cases)
        for i in range(per_case)
    ]


# --- the split ----------------------------------------------------------------


def test_bcss_only_manifest_keeps_the_published_split():
    """The auto default must not change what the pipeline did before BRACS existed.

    Holding the three real-in-situ slides out of a BCSS-only manifest would leave zero
    class-1 tiles to fit on, and `class_weights` refuses that - correctly. So the nine
    tiles can only be spent as a test set once there is a replacement to train on.
    """
    train, test = datasets.institution_split(bcss_rows())
    labels = {int(r["label"]) for r in train}
    assert bcss.NON_INVASIVE in labels, "class 1 must stay in training with no BRACS data"
    assert {r["institution"] for r in test} == {"OL"}


def test_real_in_situ_slides_move_to_test_once_bracs_is_present():
    rows = bcss_rows() + dcis_rows()
    train, test = datasets.institution_split(rows)

    real_in_test = [
        r for r in test
        if r["source"] == "bcss" and int(r["label"]) == bcss.NON_INVASIVE
    ]
    assert len(real_in_test) == 9, "all nine human-drawn in-situ tiles belong in test"
    assert not [
        r for r in train
        if r["source"] == "bcss" and int(r["label"]) == bcss.NON_INVASIVE
    ]


def test_the_whole_slide_moves_not_just_its_class_one_tiles():
    """Scoring a model on one tile of a slide it was fitted on scores its memory."""
    rows = bcss_rows() + dcis_rows()
    train, test = datasets.institution_split(rows)
    moved = datasets.real_dcis_slides(rows)

    assert moved == {"TCGA-AR-A2LH", "TCGA-A2-A0D0", "TCGA-A2-A1G6"}
    for slide in moved:
        assert not [r for r in train if r["slide_id"] == slide]
        # ...including the class-0 tiles of those slides, which is the point.
        assert [r for r in test
                if r["slide_id"] == slide and int(r["label"]) == bcss.NON_EPITHELIUM]


def test_no_slide_or_patient_lands_on_both_sides():
    rows = bcss_rows() + dcis_rows()
    train, test = datasets.institution_split(rows)
    datasets.assert_no_leak(train, test)
    assert not ({r["slide_id"] for r in train} & {r["slide_id"] for r in test})


def test_borrowed_tiles_are_split_by_patient_and_both_sides_are_populated():
    rows = bcss_rows() + dcis_rows(cases=40)
    train, test = datasets.institution_split(rows)

    train_cases = {r["slide_id"] for r in train if bcss.is_borrowed(r["source"])}
    test_cases = {r["slide_id"] for r in test if bcss.is_borrowed(r["source"])}
    assert train_cases and test_cases
    assert not (train_cases & test_cases)
    # Roughly a quarter held out; loose bounds because it is a hash, not a quota.
    share = len(test_cases) / (len(train_cases) + len(test_cases))
    assert 0.10 < share < 0.45, f"held-out share {share:.2f} is nowhere near the target"


def test_a_patients_side_of_the_split_does_not_move_when_others_are_added():
    """A hash, not a shuffle: today's split must survive tomorrow's bigger export.

    With `random.sample(seed=0)` every assignment changes the moment a case is added,
    so two models both claiming seed 0 would be scored on different tiles.
    """
    small = [f"BRACS_{i}" for i in range(30)]
    large = small + [f"BRACS_{i}" for i in range(30, 200)]
    before = bcss.bracs_test_cases(small)
    after = bcss.bracs_test_cases(large)
    assert before == {case for case in after if case in set(small)}


def test_borrowed_institution_is_not_a_tcga_source_site():
    """A two-letter code here would put BRACS tiles in a held-out TCGA institution."""
    assert bcss.BRACS_INSTITUTION.upper() not in bcss.TEST_INSTITUTIONS
    assert len(bcss.BRACS_INSTITUTION) > 2


# --- the three trees ----------------------------------------------------------


def test_each_tree_teaches_only_what_its_consensus_covers():
    """`SOURCE_CLASSES` is the authority rule, and it is built from the tree table.

    A drift here is the failure v1 shipped: BRACS's 270 DCIS-region "invasive" tiles
    were BEETLE's guess inside regions selected for their in-situ disease, and the model
    trained on them scored 0.19 Dice on invasive. The rule is not "BRACS is untrusted",
    it is "each region is trusted for what three pathologists said about it".
    """
    assert datasets.SOURCE_CLASSES["bcss"] == frozenset({0, 1, 2})
    assert datasets.SOURCE_CLASSES["bracs_dcis"] == frozenset({bcss.NON_INVASIVE})
    assert datasets.SOURCE_CLASSES["bracs_ic"] == frozenset({bcss.INVASIVE})
    assert datasets.SOURCE_CLASSES["bracs_normal"] == frozenset(
        {bcss.NON_EPITHELIUM, bcss.NON_INVASIVE}
    )
    # The table and the filter must not be two independent statements of one fact.
    for name, tree in bcss.BORROWED_TREES.items():
        assert datasets.SOURCE_CLASSES[tree.source] == tree.teaches, name


def test_a_tile_voting_outside_its_trees_remit_is_dropped():
    """An IC region's in-situ is probably adjacent DCIS the ROI label does not cover."""
    rows = (
        borrowed_rows("ic", bcss.INVASIVE, cases=2, per_case=3)
        + borrowed_rows("ic", bcss.NON_INVASIVE, cases=2, per_case=3)
        + borrowed_rows("normal", bcss.INVASIVE, cases=2, per_case=3, case_offset=50)
        + borrowed_rows("normal", bcss.NON_EPITHELIUM, cases=2, per_case=3,
                        case_offset=50)
    )
    kept = datasets.by_source_authority(rows)
    surviving = {(r["source"], int(r["label"])) for r in kept}

    assert ("bracs_ic", bcss.INVASIVE) in surviving
    assert ("bracs_ic", bcss.NON_INVASIVE) not in surviving
    assert ("bracs_normal", bcss.NON_EPITHELIUM) in surviving
    # A region three pathologists called free of carcinoma cannot teach invasive.
    assert ("bracs_normal", bcss.INVASIVE) not in surviving


def test_every_borrowed_source_is_recognised_as_borrowed():
    """`is_borrowed` is what every report and split asks; missing a tree is silent.

    The failure it prevents: a `bracs_ic` row surviving into a report that says its
    labels were drawn by people, because the check was spelled `== DCIS_SOURCE`.
    """
    assert not bcss.is_borrowed("bcss")
    assert not bcss.is_borrowed(None)
    for tree in bcss.BORROWED_TREES.values():
        assert bcss.is_borrowed(tree.source), tree.name
    assert bcss.BORROWED_SOURCES == {t.source for t in bcss.BORROWED_TREES.values()}


def test_a_patient_in_two_trees_lands_on_one_side():
    """BRACS patients contribute regions of several lesions; the split is per patient.

    Hashing each tree's cases separately would give the same draw - the hash reads only
    the id - but pooling is what makes that guaranteed rather than incidental, and it is
    the failure that would be invisible: one patient's DCIS region training the model
    that is then scored on their IC region.
    """
    shared = borrowed_rows("dcis", bcss.NON_INVASIVE, cases=40)
    also = borrowed_rows("ic", bcss.INVASIVE, cases=40)  # the same 40 patients
    rows = bcss_rows() + shared + also
    train, test = datasets.institution_split(rows)

    train_cases = {r["slide_id"] for r in train if bcss.is_borrowed(r["source"])}
    test_cases = {r["slide_id"] for r in test if bcss.is_borrowed(r["source"])}
    assert train_cases and test_cases
    assert not (train_cases & test_cases)


def test_normal_ducts_are_enough_to_spend_the_nine_real_tiles():
    """The `holdout_real_dcis` default asks for class 1, not for the DCIS tree.

    `bracs_normal` supplies class 1 as normal acinus/duct under a no-carcinoma
    consensus. Keying the default on tree identity would refuse to move the three real
    slides out of training even with thousands of class-1 tiles to fit on - and that
    refusal is invisible, because the pipeline still trains.
    """
    rows = bcss_rows() + borrowed_rows("normal", bcss.NON_INVASIVE, cases=12)
    train, _ = datasets.institution_split(rows)
    assert not [
        r for r in train
        if r["source"] == "bcss" and int(r["label"]) == bcss.NON_INVASIVE
    ]

    # ...and with no borrowed class 1 anywhere, the nine stay in training, because
    # moving them would leave `class_weights` an empty class.
    only_invasive = bcss_rows() + borrowed_rows("ic", bcss.INVASIVE, cases=12)
    train_ic, _ = datasets.institution_split(only_invasive)
    assert [
        r for r in train_ic
        if r["source"] == "bcss" and int(r["label"]) == bcss.NON_INVASIVE
    ]


# --- the report ---------------------------------------------------------------


def test_the_two_class_one_populations_are_reported_separately():
    """Nine real tiles must not be averaged into thousands of teacher-labelled ones."""
    truth = np.array([bcss.NON_INVASIVE] * 10)
    # Every real tile right, every borrowed tile wrong. A pooled recall would read
    # 0.1 and describe neither population.
    predicted = np.array([bcss.NON_INVASIVE] * 2 + [bcss.INVASIVE] * 8)
    sources = np.array(["bcss"] * 2 + [bcss.DCIS_SOURCE] * 8)

    scored = report.evaluate(truth, predicted, sources=sources)
    assert scored.by_source["bcss"]["non_invasive_recall"] == pytest.approx(1.0)
    assert scored.by_source[bcss.DCIS_SOURCE]["non_invasive_recall"] == pytest.approx(0.0)
    assert scored.non_invasive_recall == pytest.approx(0.2)


def test_the_report_says_who_drew_each_sources_labels():
    truth = np.array([bcss.NON_INVASIVE, bcss.NON_INVASIVE])
    sources = np.array(["bcss", bcss.BORROWED_TREES["ic"].source])
    scored = report.evaluate(truth, truth, sources=sources)

    assert scored.by_source["bcss"]["labels_drawn_by"] == "people"
    borrowed = bcss.BORROWED_TREES["ic"].source
    assert "BEETLE" in scored.by_source[borrowed]["labels_drawn_by"]
    assert "unreviewed" in scored.by_source[borrowed]["labels_drawn_by"]


def test_a_source_with_no_class_one_reports_none_not_zero():
    """`None` means "not measured"; 0.0 means "measured and got them all wrong"."""
    truth = np.array([bcss.INVASIVE, bcss.NON_EPITHELIUM])
    scored = report.evaluate(truth, truth, sources=np.array(["bcss", "bcss"]))
    assert scored.by_source["bcss"]["non_invasive_recall"] is None


def test_by_source_is_absent_unless_asked_for():
    scored = report.evaluate(np.array([0, 2]), np.array([0, 2]))
    assert scored.by_source == {}


# --- the exporter's provenance column -----------------------------------------


def test_the_exporter_stamps_the_source_it_is_told():
    import export

    region = bcss.Region(
        roi_id="BRACS_1_DCIS_1", slide_id="BRACS_1",
        institution=bcss.BRACS_INSTITUTION,
        image=Path("nonexistent.png"), mask=Path("nonexistent.png"),
    )
    # Only the signature is under test here; the pixels are covered by the app's own
    # end-to-end test. A default of "bcss" on a borrowed region is the failure that
    # would make every split above silently inoperative.
    import inspect

    signature = inspect.signature(export.export_region)
    assert signature.parameters["source"].default == "bcss"
    assert "source" in inspect.signature(export.write_region).parameters
    assert region.institution == bcss.BRACS_INSTITUTION


def test_find_borrowed_regions_groups_by_patient(tmp_path):
    from PIL import Image

    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    for name in ("BRACS_1247_DCIS_1", "BRACS_1247_DCIS_9", "BRACS_311_DCIS_2"):
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(images / f"{name}.png")
        Image.fromarray(np.zeros((8, 8), np.uint16)).save(masks / f"{name}.png")

    regions = bcss.find_borrowed_regions(tmp_path)
    assert {r.slide_id for r in regions} == {"BRACS_1247", "BRACS_311"}
    assert all(r.institution == bcss.BRACS_INSTITUTION for r in regions)
    assert all(not r.is_test for r in regions), "BRACS is not a TCGA held-out site"


def test_an_image_without_a_mask_is_refused(tmp_path):
    from PIL import Image

    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(
        tmp_path / "images" / "BRACS_1_DCIS_1.png"
    )
    with pytest.raises(FileNotFoundError, match="no mask"):
        bcss.find_borrowed_regions(tmp_path)
