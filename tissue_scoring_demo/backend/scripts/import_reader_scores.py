"""Write the pathologist readings into the sheet step 17 reads.

    python scripts/import_reader_scores.py

**This is a transcription, not the client's file.** `6Slide Reports2.xlsx` itself
is not in this repository. What is encoded here was read off a screenshot of it
supplied on 2026-09-15, and that screenshot showed **six of the eleven columns**:
the five percent columns and one intensity column, `WI-C` (pan-cadherin). The
other four intensity columns - `AI-C`, `UI-C`, `RI-C`, `FI-C` - were not visible
and are therefore absent here rather than guessed.

So this file supports a full comparison on **percent positive** and a comparison
on intensity for **pan-cadherin only**. Step 17 reports the rest as unavailable,
which is the honest state: a missing column is not a zero.

Two checks that the transcription is right, both of which it passes:

* The per-marker case averages match the ones recorded independently in
  `docs/guides/images-to-scores-mapping.md` Part 6 - CD44 at 11.25, 20, 50, 60, 71.25,
  80 across the six cases, and so on for the other four.
* `docs/guides/images-to-scores-mapping.md` notes one anomaly in the whole dataset:
  reader SA on `CAN_00270_26` recorded pan-cadherin intensity as **1.25**, a
  value no reporting band permits. That is exactly what the screenshot shows.

Replace this script's output with the real workbook the moment it is available;
nothing downstream needs to change, because step 17 reads a path.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402

#: Column headers, in the sheet's own order. The order is A, W, U, R, F - NOT
#: panel order - which is the trap `images-to-scores-mapping.md` warns about:
#: joining positionally swaps ABCC4 with ABCC11, and both are plausible pump
#: markers with overlapping ranges, so the error would not announce itself.
#: Nothing here or in `validation_service` joins by position; the letter in the
#: header is what is read.
HEADERS = ["ID", "A%-M", "W%-C", "U%-C", "R%-M", "F%-M", "WI-C"]

#: `(row id, CD44 %, pan-cadherin %, N-cadherin %, ABCC11 %, ABCC4 %,
#: pan-cadherin intensity)`. `AVG_` rows are deliberately included exactly as
#: they appear; `validation_service` skips them, because they are a plain mean of
#: the four readers above them and including them would weight each case's
#: consensus into its own comparison twice.
ROWS: list[tuple] = [
    ("PS_CAN/00251/26_H26-364-B1", 60, 80, 80, 60, 50, 2),
    ("SA_CAN/00251/26_H26-364-B1", 50, 80, 80, 50, 45, 2),
    ("DP_CAN/00251/26_H26-364-B1", 50, 80, 80, 60, 45, 2),
    ("NK_CAN00251/26_H26-364-B1", 40, 80, 80, 45, 40, 2),
    ("AVG_CAN00251/26_H26-364-B1", 50, 80, 80, 53.75, 45, 2),

    ("NK_CAN00259/26_H2600058-A1", 60, 80, 80, 45, 45, 1.75),
    ("SA_CAN00259/26_H2600058-A1", 60, 80, 80, 60, 50, 1.75),
    ("DP_CAN00259/26_H2600058-A1", 60, 80, 80, 45, 50, 1.5),
    ("PS_CAN00259/26_H2600058-A1", 60, 80, 80, 50, 50, 1.5),
    ("AVG_CAN00259/26_H2600058-A1", 60, 80, 80, 50, 48.75, 1.625),

    ("SA_CAN00267/26_25H-19421-A2", 25, 75, 80, 35, 45, 1.5),
    ("NK_CAN00267/26_25H-19421-A2", 10, 80, 80, 35, 35, 1.75),
    ("DP_CAN00267/26_25H-19421-A2", 25, 75, 80, 35, 40, 1.75),
    ("PS_CAN00267/26_25H-19421-A2", 20, 80, 80, 35, 35, 2),
    ("AVG_CAN00267/26_25H-19421-A2", 20, 77.5, 80, 35, 38.75, 1.75),

    ("PS_CAN00270/26_H26-154-C6", 80, 80, 80, 50, 65, 1.5),
    ("NK_CAN00270/26_H26-154-C6", 75, 80, 80, 60, 60, 1.5),
    ("DP_CAN00270/26_H26-154-C6", 80, 80, 80, 50, 60, 1.5),
    # The one reading in all 120 that lands on no permitted band. Kept as it is.
    ("SA_CAN00270/26_H26-154-C6", 85, 80, 80, 60, 65, 1.25),
    ("AVG_CAN00270/26_H26-154-C6", 80, 80, 80, 55, 62.5, 1.4375),

    ("PS_CAN00303/26_B2011-T", 70, 80, 80, 40, 50, 2),
    ("DP_CAN00303/26_B2011-T", 70, 80, 80, 40, 50, 2),
    ("SA_CAN00303/26_B2011-T", 65, 80, 80, 35, 35, 2),
    ("NK_CAN00303/26_B2011-T", 80, 80, 80, 35, 35, 2),
    ("AVG_CAN00303/26_B2011-T", 71.25, 80, 80, 37.5, 42.5, 2),

    ("NK_CAN00865/26_2460/GL-A8", 10, 80, 80, 60, 65, 2),
    ("DP_CAN00865/26_2460/GL-A8", 5, 80, 80, 65, 55, 2),
    ("SA_CAN00865/26_2460/GL-A8", 15, 80, 80, 60, 60, 2),
    ("PS_CAN00865/26_2460/GL-A8", 15, 80, 80, 60, 60, 2),
    ("AVG_CAN00865/26_2460/GL-A8", 11.25, 80, 80, 61.25, 60, 2),
]


def main() -> int:
    from openpyxl import Workbook

    book = Workbook()
    sheet = book.active
    sheet.title = "readings"
    sheet.append(HEADERS)
    for row in ROWS:
        sheet.append(list(row))

    destination = settings.data_dir.parent / "original" / "reader_scores_transcribed.xlsx"
    destination.parent.mkdir(parents=True, exist_ok=True)
    book.save(destination)

    readers = [row for row in ROWS if not row[0].startswith("AVG")]
    print(f"wrote {destination}")
    print(f"  {len(readers)} reader rows over {len(ROWS) - len(readers)} cases")
    print("  percent: all five markers.  intensity: pan-cadherin only (WI-C).")
    print()
    print("  Transcribed from a screenshot, not the client's own file. The four")
    print("  missing intensity columns are absent rather than guessed, so step 17")
    print("  reports them as unavailable.")
    print()
    print(f"  Point step 17 at it with:  REGISTRY_READER_SCORES_PATH={destination}")
    print("  or leave it - the service checks this location by convention.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
