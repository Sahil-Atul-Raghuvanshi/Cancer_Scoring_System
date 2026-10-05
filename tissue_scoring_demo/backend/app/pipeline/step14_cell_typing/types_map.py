"""Reading step 12's class map, once, for the two steps that filter on it.

Steps 13 and 14 both ask the same question - which of this field's nuclei are
tumour - and they must get the same answer, because one builds the compartments
the other measures. Two copies of this lookup is two chances for the numerator
and the denominator to be drawn from different sets of cells.

**A nucleus is identified by its field and its id, never by its id alone.** Step
11 segments each sampled field separately, so every field's instance map starts
again at 1 and one region holds a dozen nucleus 14s. Asking a map keyed on the
bare id "which ids are tumour" and applying the answer to every field keeps the
*union* of the tumour ids across the whole region - which on CAN_00270's CD44
slide turned a 33 % tumour share into 91 % of the cells being measured.
"""

from __future__ import annotations

import json
from pathlib import Path

#: 0 is TUMOUR in `step14_cell_typing.classify.CellType`.
TUMOUR = 0


class TypesUnavailableError(ValueError):
    """Step 12 has not run for this region, or ran before the current format."""


def tumour_ids(path: Path, field_index: int) -> set[int]:
    """Nucleus ids in `field_index` that step 12 called tumour.

    Raises rather than returning an empty set when the map is missing or stale.
    An empty set and "no map" look identical downstream - both produce no
    compartments - and one of them is a slide with no tumour on it while the
    other is a step that has not run.
    """
    if not path.is_file():
        raise TypesUnavailableError(
            f"step 12 has stored no cell types at {path}. Only tumour cells are "
            "measured, so without it there is no defensible denominator."
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("format", 1)) < 2:
        raise TypesUnavailableError(
            f"the cell types at {path} were written in the old format, which keyed a "
            "nucleus by its id alone. Ids repeat between fields, so that map cannot say "
            "which cells in THIS field are tumour. Re-run step 12 for this pair."
        )

    out: set[int] = set()
    for key, value in payload.get("types", {}).items():
        if value != TUMOUR:
            continue
        field, _, identifier = key.partition(":")
        if field == str(field_index):
            out.add(int(identifier))
    return out


def typed_count(path: Path) -> int:
    """How many nuclei step 12 sorted in this region, for the report."""
    if not path.is_file():
        return 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    return len(payload.get("types", {}))


__all__ = ["TUMOUR", "TypesUnavailableError", "tumour_ids", "typed_count"]
