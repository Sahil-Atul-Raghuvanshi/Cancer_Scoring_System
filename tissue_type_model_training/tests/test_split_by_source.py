"""The provenance split: `07_split_export_by_source.py`.

One export becomes `<fov>/{bcss,beetle,bach}/<class>/`, and the two failures worth
guarding are a tile filed under the wrong laboratory and a manifest whose `tile_path`
no longer finds its file. Both would survive to a training run and be discovered as a
number that made no sense.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "split_by_source", ROOT / "scripts" / "07_split_export_by_source.py"
)
split_by_source = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(split_by_source)


def test_every_source_this_project_exports_has_a_group():
    """A source with no group raises, and every real one must not."""
    assert split_by_source.group_for("bcss") == "bcss"
    for source in ("bracs_dcis", "bracs_ic", "bracs_normal"):
        assert split_by_source.group_for(source) == "beetle"
    for source in ("bach_insitu", "bach_invasive", "bach_normal"):
        assert split_by_source.group_for(source) == "bach"


def test_an_unknown_source_raises_rather_than_defaulting():
    """The whole point of the group table: a tile nobody can place must stop the run,
    not be filed under whichever group is first."""
    with pytest.raises(ValueError, match="no known group"):
        split_by_source.group_for("tcga_something_new")


def _tile(tmp: Path, group_source: str, label_name: str, tile_id: str) -> dict:
    (tmp / label_name).mkdir(parents=True, exist_ok=True)
    (tmp / label_name / f"{tile_id}.png").write_bytes(b"not-a-real-png")
    return {
        "tile_id": tile_id,
        "roi_id": tile_id.split("__")[0],
        "slide_id": "slide1",
        "institution": "X",
        "label": "1",
        "label_name": label_name,
        "source": group_source,
        "tile_path": f"{label_name}/{tile_id}.png",
    }


def test_split_moves_each_tile_and_repaths_both_manifests(tmp_path):
    """The end-to-end shape, on three tiles from three laboratories.

    Asserts the two things a layout change can silently break: the group manifest's
    `tile_path` resolves against its own directory, and the field-of-view manifest's
    resolves against the level above - so `03_features.py` pointed at either one finds
    real files.
    """
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    rows = [
        _tile(tiles, "bcss", "invasive_epithelium", "bcss_r000c000"),
        _tile(tiles, "bracs_dcis", "non_invasive_epithelium", "BRACS_1_DCIS_1__r000c000"),
        _tile(tiles, "bach_insitu", "non_invasive_epithelium", "is001__r000c000"),
    ]
    with (tiles / "tiles_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (tiles / "export_summary.json").write_text(
        json.dumps({"tiles_kept": 3, "spec": {"tile_px": 224, "mpp": 1.0,
                                              "tile_um": 224.0}}),
        encoding="utf-8",
    )

    dest = tmp_path / "224_um"
    out = split_by_source.split(tiles, dest)

    assert sorted(out["by_group"]) == ["bach", "bcss", "beetle"]
    assert {g: b["tiles"] for g, b in out["by_group"].items()} == {
        "bach": 1, "bcss": 1, "beetle": 1
    }

    # Every path in the combined manifest resolves from the field-of-view directory.
    with (dest / "tiles_manifest.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            assert (dest / row["tile_path"]).exists(), row["tile_path"]
            # `parts`, not a split on "/": the exporter writes native separators and
            # this manifest must stay consistent with the ones 02_export.py produces.
            assert Path(row["tile_path"]).parts[0] in ("bcss", "beetle", "bach")

    # And every path in a group manifest resolves from that group's directory.
    for group in ("bcss", "beetle", "bach"):
        with (dest / group / "tiles_manifest.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            for row in csv.DictReader(handle):
                assert (dest / group / row["tile_path"]).exists(), row["tile_path"]

    # The tiles moved rather than being copied: nothing is left behind.
    assert not list(tiles.rglob("*.png"))


def test_each_group_carries_the_geometry_that_produced_it(tmp_path):
    """A group directory read on its own must still say what field of view it is, or it
    will eventually be pooled with tiles of another."""
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    rows = [_tile(tiles, "bcss", "non_epithelium", "bcss_r000c000")]
    with (tiles / "tiles_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (tiles / "export_summary.json").write_text(
        json.dumps({"tiles_kept": 1,
                    "spec": {"tile_px": 224, "mpp": 2.0, "tile_um": 448.0}}),
        encoding="utf-8",
    )

    dest = tmp_path / "448um"
    split_by_source.split(tiles, dest)
    block = json.loads((dest / "bcss" / "export_summary.json").read_text(encoding="utf-8"))
    assert block["spec"]["tile_um"] == 448.0
    assert "human pixel annotation" in block["labels_are"]
