"""Nucleus ids repeat between fields, and the class map has to survive that.

Step 11 segments each sampled field separately, so every field's instance map
starts again at 1 and one region holds a dozen nucleus 14s. A class map keyed on
the bare id therefore loses most of its entries to collisions, and steps 13 and
14 - which ask it "which ids are tumour" and apply the answer to every field -
keep the *union* of the tumour ids rather than a subset of the tumour cells.

Measured on CAN_00270's CD44 slide before the fix: step 12 reported 32.7 %
tumour and step 13 built compartments for 965 of 1,064 cells. After it, 348.
"""

from __future__ import annotations

import json

import pytest

from app.pipeline.step14_cell_typing import types_map


def _write(tmp_path, payload: dict):
    path = tmp_path / "types.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_the_same_id_in_two_fields_is_two_different_cells(tmp_path):
    """The whole bug, in four entries."""
    path = _write(
        tmp_path,
        {
            "rank": 1,
            "format": 2,
            "types": {
                "0:14": 0,  # field 0, nucleus 14 - tumour
                "0:15": 1,  # field 0, nucleus 15 - lymphocyte
                "3:14": 1,  # field 3, nucleus 14 - lymphocyte
                "3:15": 0,  # field 3, nucleus 15 - tumour
            },
        },
    )

    assert types_map.tumour_ids(path, 0) == {14}
    assert types_map.tumour_ids(path, 3) == {15}


def test_a_field_with_no_tumour_cells_returns_an_empty_set(tmp_path):
    path = _write(tmp_path, {"format": 2, "types": {"0:1": 1, "0:2": 2}})
    assert types_map.tumour_ids(path, 0) == set()


def test_an_unknown_field_returns_an_empty_set_rather_than_everything(tmp_path):
    """The failure mode that made the bug invisible: a miss that looks like a hit."""
    path = _write(tmp_path, {"format": 2, "types": {"0:1": 0, "0:2": 0}})
    assert types_map.tumour_ids(path, 7) == set()


def test_the_old_format_is_refused_rather_than_read_as_current(tmp_path):
    """A version-1 map cannot say which cells in a given field are tumour.

    Reading one anyway is what produced the 965-of-1,064 result, and it produced
    it silently - so the reader raises instead of guessing.
    """
    path = _write(tmp_path, {"rank": 1, "types": {"14": 0, "15": 1}})
    with pytest.raises(types_map.TypesUnavailableError, match="old format"):
        types_map.tumour_ids(path, 0)


def test_a_missing_map_is_refused_rather_than_read_as_no_tumour(tmp_path):
    """An empty set and "step 12 has not run" look identical downstream.

    Both produce no compartments and therefore no score. One of them is a slide
    with no tumour on it; the other is a step that was skipped.
    """
    with pytest.raises(types_map.TypesUnavailableError, match="no cell types"):
        types_map.tumour_ids(tmp_path / "absent.json", 0)


def test_only_tumour_cells_come_back(tmp_path):
    path = _write(
        tmp_path,
        {"format": 2, "types": {"2:1": 0, "2:2": 1, "2:3": 2, "2:4": 0}},
    )
    assert types_map.tumour_ids(path, 2) == {1, 4}


def test_the_tumour_share_of_a_field_matches_the_map(tmp_path):
    """The invariant the bug broke: what is kept is a subset of what was typed."""
    types = {f"0:{index}": (0 if index % 3 == 0 else 1) for index in range(1, 31)}
    path = _write(tmp_path, {"format": 2, "types": types})

    kept = types_map.tumour_ids(path, 0)
    assert len(kept) == 10
    assert len(kept) <= types_map.typed_count(path)
