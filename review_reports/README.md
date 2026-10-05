# review_reports

Builds the two review documents in `v<N>_data/results/` (the active data version, see `data_versions.py`) from the stored pipeline runs:

| Document | What it is |
| --- | --- |
| `results/HE_IHC_Registration_Review.docx` | One page per case and marker: the H&E invasive outline and the same outline registered onto the IHC slide, pixel-wise (BEETLE) and tile-based, with areas. For a pathologist to confirm the registrations. |
| `results/AI_vs_Pathologist_Scores.docx` | Both AI score files against the four pathologists in `6Slide Reports2.xlsx`: how the scores are calculated, whether they are reliable, why some are 0, held-out cut-point refit and ML recalibration, visual evidence, one page per case and marker, and a "things that need attention" report. |

## Running it

```bash
python review_reports/run.py              # both documents
python review_reports/run.py registration # just the registration review
python review_reports/run.py scores       # just the score comparison
```

Run it after a scoring pass has refreshed `results/oncostem_ai_scores*.csv` and
`v<N>_data/data/history/`. Needs the demo backend's virtual environment (the Python steps reuse
its slide reader, cut points and `scripts/calibrate_marker_cuts.py`) and Node 18+;
`run.py` installs the `docx` npm package here on first use. The score comparison
takes several minutes, most of it re-reading every stored per-cell row.

Close the documents in Word first: an open `.docx` is locked and the final write fails.

## Where things go

Code only lives here. Everything the steps produce on the way — joined tables,
compressed slide images, charts, `analysis.json`, `attention.json` — is written to
`v<N>_data/data/review_reports/<document>/`, and only the finished `.docx` goes to `v<N>_data/results/`.

## The steps

**Registration review** (`registration_review/`)
1. `prepare.py` — picks the latest confirmed step-12 run per case, marker and method from
   `v<N>_data/data/history`, compresses its `he_borders.png` / `ihc_borders.png`, and measures the
   outlined area on both slides from the stored rings.
2. `build.js` — lays out the document.

**Score comparison** (`score_comparison/`)
1. `prepare.py` — joins the reader sheet with both CSVs, one row per case and marker.
2. `analysis.py` — reproduces every score from the stored per-cell rows (a check that the
   right data is being read), then the leave-one-case-out cut-point refit, the ridge
   recalibration against a no-AI baseline, agreement statistics and warning tallies.
3. `prepare_slides.py` — images and reports from the exact run behind each CSV row.
4. `charts.py` — figures 1–4 and the per-marker agreement table.
5. `attention.py` — evidence for the attention section: measurement funnel, region
   selection, weighting, pixel-vs-tile, sensitivity sweeps, full agreement set, and the
   data-lineage check (region files on disk must match each report).
6. `build.js` — lays out the document.

## Caveat

The figures, tables and per-slide pages are regenerated from the data on every run. Some
of the explanatory prose names specific cases as examples (for instance the failed
CD44 registrations on CAN_00267 and CAN_00865, as of the 2026-09-29 runs); after a re-score,
read those passages against the regenerated numbers.
