"""The campaign driver's resume predicates: `10_he_campaign.py`.

These decide whether a stage is skipped. Getting one wrong costs either forty minutes of
recomputation or - far worse - a run that trusts a half-written artefact and trains on
it. Neither failure announces itself: the first looks like a slow night, the second like
a model that will not learn.

So each predicate is checked to be **positive only on a complete, self-consistent
artefact**. Presence is deliberately not enough, and the tests below are mostly about
the ways a file can exist and still be wrong:

  * an export summary recording a different resolution than the arm being resumed
  * an export summary recording the wrong channel
  * a feature cache whose fingerprint belongs to a different manifest
  * a truncated `.npz` that `np.load` opens far enough to look plausible
  * a checkpoint whose bytes no longer hash to what its manifest records
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_driver():
    """Import `10_he_campaign.py`, whose name is not an identifier.

    Registered in `sys.modules` *before* it is executed, and that order matters:
    `@dataclass` resolves its annotations through `sys.modules[cls.__module__]`, so a
    module built by `spec_from_file_location` and never registered gives dataclasses a
    `None` to look in - which surfaces as `'NoneType' object has no attribute
    '__dict__'` from inside the standard library, several frames away from the cause.
    """
    path = ROOT / "scripts" / "10_he_campaign.py"
    spec = importlib.util.spec_from_file_location("he_campaign", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver():
    return _load_driver()


@pytest.fixture
def store(driver, tmp_path, monkeypatch):
    """A `<data>/he/<fov>um` tree of its own for the predicates to look at.

    Patches `DATA`, the driver's own handle on this stage's export tree, rather than
    `REPO`. The two stopped being the same thing when storage moved out of the project
    folder into `<workspace>/v<N>_data/data/tissue_type_model_training/` - and patching the wrong
    one does not fail loudly, it silently points every predicate at the real store,
    where a complete export makes `export_done` answer True for the wrong reason.
    """
    data = tmp_path / "data" / "tissue_type_model_training"
    monkeypatch.setattr(driver, "DATA", data)
    return data / "he"


def _publish(driver, workspace: Path, fov: int) -> Path:
    """Put a self-consistent published head for `fov` under a patched `MODELS`.

    A checkpoint and a manifest whose recorded sha256 is genuinely that of the bytes
    beside it, which is what `published_done` checks. The contents are not a real
    checkpoint and do not need to be - nothing here loads it.
    """
    import hashlib

    models = workspace / "models" / "tissue_type"
    models.mkdir(parents=True, exist_ok=True)
    assert driver.MODELS == models, "patch driver.MODELS to this workspace first"
    name = driver.head_name(fov)
    payload = f"checkpoint for {name}".encode()
    (models / f"{name}.pt").write_bytes(payload)
    (models / f"{name}.manifest.json").write_text(
        json.dumps({"sha256": hashlib.sha256(payload).hexdigest()}), encoding="utf-8"
    )
    return models


def _write_export(store: Path, fov: int, *, mpp: float, channel: str = "rgb_he",
                  groups=("bcss", "beetle")) -> Path:
    directory = store / f"{fov}um"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "tiles_manifest.csv").write_text(
        "tile_id,label,tile_path\nt0,2,bcss/invasive_epithelium/t0.png\n",
        encoding="utf-8",
    )
    (directory / "export_summary.json").write_text(
        json.dumps({"spec": {"mpp": mpp, "tile_px": 224}, "input": {"channel": channel}}),
        encoding="utf-8",
    )
    for group in groups:
        (directory / group).mkdir(exist_ok=True)
    return directory


# --- the export ---------------------------------------------------------------


def test_a_finished_export_is_recognised(driver, store) -> None:
    _write_export(store, 224, mpp=1.0)
    assert driver.export_done(224, 1.0)


def test_an_export_at_another_resolution_is_not_this_arm(driver, store) -> None:
    """The one that matters most on resume: the staging tree is shared between arms,
    so an export left from the previous field of view is exactly what is on disk when
    the next one starts."""
    _write_export(store, 224, mpp=2.0)
    assert not driver.export_done(224, 1.0)


def test_an_export_in_the_other_channel_is_not_this_arm(driver, store) -> None:
    """A density map and a photograph of the same square are both 224x224 uint8, so
    nothing downstream would notice the mix. The summary records which, and this is
    where that record earns its keep."""
    _write_export(store, 224, mpp=1.0, channel="haematoxylin")
    assert not driver.export_done(224, 1.0)


def test_an_export_that_was_never_split_is_not_done(driver, store) -> None:
    """`07_split_export_by_source.py` is what creates the per-source directories. An
    export whose manifest exists but whose tiles are still in the staging tree would
    fail at feature extraction, forty minutes later."""
    _write_export(store, 224, mpp=1.0, groups=())
    assert not driver.export_done(224, 1.0)


def test_an_unreadable_summary_is_not_done(driver, store) -> None:
    directory = _write_export(store, 224, mpp=1.0)
    (directory / "export_summary.json").write_text("{ truncated", encoding="utf-8")
    assert not driver.export_done(224, 1.0)


def test_a_missing_export_is_not_done(driver, store) -> None:
    assert not driver.export_done(672, 3.0)


# --- the pairing assertion ----------------------------------------------------


def test_the_pairing_stage_is_done_only_when_it_passed(driver, store) -> None:
    directory = _write_export(store, 448, mpp=2.0)
    reports = directory / "reports"
    reports.mkdir()

    (reports / "paired.json").write_text(json.dumps({"ok": False}), encoding="utf-8")
    assert not driver.paired_done(448, 2.0)

    (reports / "paired.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    assert driver.paired_done(448, 2.0)


# --- the feature caches -------------------------------------------------------


def test_a_feature_cache_is_done_only_when_its_fingerprint_matches(
    driver, store, monkeypatch
) -> None:
    """The fingerprint is what stops a feature array being paired with a different
    manifest. That failure trains to about chance and looks exactly like a model that
    will not learn."""
    import datasets

    directory = _write_export(store, 224, mpp=1.0)
    features = directory / "features"
    features.mkdir()

    rows = datasets.read_manifest(directory / "tiles_manifest.csv")
    monkeypatch.setattr(datasets, "by_source_authority", lambda given: given)
    want = datasets.manifest_fingerprint(rows)

    np.savez_compressed(features / "imagenet.npz", fingerprint=np.array(want))
    assert driver.features_done(224, "imagenet")

    np.savez_compressed(
        features / "simclr.npz", fingerprint=np.array("0" * 64)
    )
    assert not driver.features_done(224, "simclr")


def test_a_truncated_feature_cache_is_not_done(driver, store) -> None:
    """**Presence is not enough, and this is the case that proves it.** A run killed
    inside `savez_compressed` leaves a file of plausible size. Treating it as done is
    the difference between skipping a stage and forty minutes of silently wrong
    features - which is why `03_features.py` writes to `.tmp.npz` and replaces, and why
    this predicate opens the file rather than stat-ing it."""
    directory = _write_export(store, 224, mpp=1.0)
    features = directory / "features"
    features.mkdir()
    (features / "imagenet.npz").write_bytes(b"PK\x03\x04truncated-and-not-a-zip")
    assert not driver.features_done(224, "imagenet")


def test_a_missing_feature_cache_is_not_done(driver, store) -> None:
    _write_export(store, 224, mpp=1.0)
    assert not driver.features_done(224, "imagenet")


# --- the published head -------------------------------------------------------


def test_a_published_head_is_done_only_when_its_bytes_match_its_manifest(
    driver, tmp_path, monkeypatch
) -> None:
    """The same rule `model.load_pinned` enforces at serve time. A kill between
    `torch.save` and the hash leaves a checkpoint the manifest does not describe, and
    re-publishing it is cheap where serving it is not."""
    import hashlib

    models = tmp_path / "models" / "tissue_type"
    models.mkdir(parents=True)
    monkeypatch.setattr(driver, "WORKSPACE", tmp_path)
    monkeypatch.setattr(driver, "MODELS", tmp_path / "models" / "tissue_type")

    name = driver.head_name(448)
    payload = b"not really a checkpoint"
    (models / f"{name}.pt").write_bytes(payload)

    (models / f"{name}.manifest.json").write_text(
        json.dumps({"sha256": hashlib.sha256(payload).hexdigest()}), encoding="utf-8"
    )
    assert driver.published_done(448)

    (models / f"{name}.manifest.json").write_text(
        json.dumps({"sha256": "0" * 64}), encoding="utf-8"
    )
    assert not driver.published_done(448)


def test_a_checkpoint_without_a_manifest_is_not_published(
    driver, tmp_path, monkeypatch
) -> None:
    models = tmp_path / "models" / "tissue_type"
    models.mkdir(parents=True)
    monkeypatch.setattr(driver, "WORKSPACE", tmp_path)
    monkeypatch.setattr(driver, "MODELS", tmp_path / "models" / "tissue_type")
    (models / f"{driver.head_name(672)}.pt").write_bytes(b"orphan")
    assert not driver.published_done(672)


# --- the comparison -----------------------------------------------------------


def test_the_comparison_is_done_only_once_it_names_an_he_head(
    driver, tmp_path, monkeypatch
) -> None:
    """A dry run of `10b` against the haematoxylin arms alone writes a perfectly valid
    document with four empty cells. Treating that as done would skip the stage whose
    entire purpose is to fill them."""
    # The document is written beside the campaign's own logs now, not at the
    # workspace root, so `LOG_DIR` is what decides where `compare_done` looks.
    monkeypatch.setattr(driver, "WORKSPACE", tmp_path)
    monkeypatch.setattr(driver, "MODELS", tmp_path / "models" / "tissue_type")
    monkeypatch.setattr(driver, "LOG_DIR", tmp_path)
    document = tmp_path / "HE_VS_HCHANNEL.md"

    # `compare_done` weighs the document against the heads that are actually
    # published, and short-circuits to False when none are. Without this the
    # positive case below can never hold however the document is written - the
    # predicate never reaches it.
    _publish(driver, tmp_path, 224)

    assert not driver.compare_done()

    document.write_text("| 224 um | he | not published |", encoding="utf-8")
    assert not driver.compare_done()

    document.write_text(
        f"| 224 um | he | `{driver.head_name(224)}` |", encoding="utf-8"
    )
    assert driver.compare_done()


# --- the arms themselves ------------------------------------------------------


def test_the_arms_run_cheapest_first(driver) -> None:
    """The ordering is what makes a blown budget graceful: it lands inside the 112 um
    arm - two thirds of the compute on its own - with the other three banked, rather
    than inside the first arm with none."""
    assert [fov for fov, _ in driver.ARMS] == [672, 448, 224, 112]


def test_every_arm_is_224_pixels_and_only_the_resolution_moves(driver) -> None:
    """Holding the pixel side constant is what leaves the backbone untouched, and it is
    also what makes the two branches comparable at each scale."""
    for fov, mpp in driver.ARMS:
        assert round(224 * mpp) == fov


def test_every_tree_is_named_rather_than_defaulted(driver) -> None:
    """The tree list *is* the dataset, so it belongs in the log where somebody can read
    what was actually cut."""
    assert set(driver.TREES) == {
        "dcis",
        "ic",
        "normal",
        "bach_insitu",
        "bach_invasive",
        "bach_normal",
    }
