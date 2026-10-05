# Steps 14–16 (+17): scoring implementation — COMPLETE

> **Superseded in part on 2026-09-15 by [SCORING_ALIGNMENT_ANALYSIS.md](SCORING_ALIGNMENT_ANALYSIS.md).**
> The requirement docs (`data/oncostem_requiremnet_docs/`) turned out to contain the real
> `6Slide Reports2.xlsx` and OncoStem's scoring SOP. Reading them changed the sampling —
> region selection is now by area covered rather than a fixed top-3, and fields are
> allocated in proportion to region area — so the numbers below predate those changes.
> The analysis document has the current numbers and the reasoning.


Run unattended 2026-09-15, 02:40 → 10:25 local.
Progress log: `SCORING_RUN_LOG.md` · raw stdout: `scoring_run_stdout.txt` · workbook: `CAN_00270_scores.xlsx` (7 sheets; **Agreement** holds the comparison against the readers)

The deliverable, as implemented:

> **Two numbers per marker, ten per case.**
> `percent positive` = positive tumour cells / total tumour cells over the invasive ROI, rounded to the nearest 5.
> `intensity` = mean DAB OD over the positive tumour cells, banded to 0 / 0.5 / 1 / 1.5 / 1.75 / 2.
> H-score, Allred and the ASCO/CAP call are computed and shown beside it, never instead of it.

---

## The result — CAN_00270, five IHC slides

| | Marker | Antibody | Compartment | **Percent positive** | **Intensity** | Tumour cells |
|---|---|---|---|---|---|---|
| A | CD44 | membrane | ring | **75 %** | **1.5** Moderate | 348 |
| F | ABCC4 | membrane | ring | **100 %** | **1.5** Moderate | 398 |
| R | ABCC11 | membrane | ring | **70 %** | **1** Weak | 281 |
| U | N-cadherin | cytoplasm | band | **100 %** | **1.5** Moderate | 292 |
| W | Pan-cadherin | cytoplasm | band | **85 %** | **1** Weak | 296 |

Every pair was re-derived from the stored per-cell rows by a separate script that imports
none of the scoring code (`scripts/verify_case_scores.py`) — **all five agree exactly.**

## Against the pathologists — CAN_00270 (added 2026-09-15)

The four readers' numbers for this case were supplied and are now transcribed into
`data/original/reader_scores_transcribed.xlsx`, so step 17 has real data.

| Marker | Ours | Readers | Diff | Within ±10? | The four reads | Their spread |
|---|---|---|---|---|---|---|
| A CD44 | 75 % | 80 % | **−5** | **yes** | 75, 80, 80, 85 | 10 |
| F ABCC4 | 100 % | 62.5 % | **+37.5** | no | 60, 60, 65, 65 | 5 |
| R ABCC11 | 70 % | 55 % | **+15** | no | 50, 50, 60, 60 | 10 |
| U N-cadherin | 100 % | 80 % | **+20** | no | 80, 80, 80, 80 | 0 |
| W Pan-cadherin | 85 % | 80 % | **+5** | **yes** | 80, 80, 80, 80 | 0 |

Mean absolute error **16.5 points**; mean signed bias **+14.5 points**; **2 of 5** inside
the human spread. Intensity can only be checked for pan-cadherin — the supplied sheet
carries one of the five intensity columns — where ours is 1 against a consensus of 1.4375.

**The bias is one-directional and that is the finding.** Four of five markers are
over-called, and the over-call is largest where the readers' percentage is lowest (ABCC4,
62.5 % → we say 100 %). That is the signature of a positivity cut set too low plus a
denominator missing its hardest-to-segment cells — not of arithmetic that is wrong. CD44,
the marker that actually separates cases, is the one we get right.

**Two of the five land outside the range OncoStem has reported**, which the pipeline
flagged on its own before the ground truth arrived: ABCC4 at 100 % against 35–70 %,
N-cadherin at 100 % against 70–85 %. See "What is provisional" below.

---

## A. Configuration — five cut-point sets, one per antibody

- [x] A1 `backend/config/marker_cuts.v1.json` — one versioned file, five independent sets
- [x] A2 `app/scoring/cuts.py` — loader, validation, provenance; refuses a broken file rather than defaulting
- [x] A3 Per-marker second-measure minimum: `completeness_min` for A/F/R, `stained_fraction_min` for U/W
- [x] A4 Band table 0 / 0.5 / 1 / 1.5 / 1.75 / 2, with the 1–3 conflict carried as Q2

## B. Step 14 — per-cell measurement

- [x] B1 Mean DAB OD over the compartment; mean fixed for every marker, max reported but never scored
- [x] B2 Membrane: ring completeness over 36 bins of 10°
- [x] B3 Cytoplasm: stained fraction of the band — completeness never computed, not even defaulted
- [x] B4 `per_cell_service.py`: same fields, same mpp, same stain basis as step 11
- [x] B5 Per-cell rows persisted; steps 15/16 never reopen a slide
- [x] B6 Schema, endpoints, scatter payload with clickable crops
- [x] B7 Tests: complete ring → 1.0; 2/36 speckle → 0.056; crowded cell reports its missing bins

## C. Step 15 — intensity binning

- [x] C1 Per-cell 0 / 1+ / 2+ / 3+ from that antibody's absolute cuts
- [x] C2 Slide-level banding kept separate — different types, different scales, separate screen blocks
- [x] C3 Absolute vs per-slide-percentile computed side by side on the same cells
- [x] C4 Shared-cuts vs per-antibody-cuts computed side by side
- [x] C5 Histogram + cut lines + the valley between the two humps
- [x] C6 Schema, endpoint, tests

## D. Step 16 — aggregate

- [x] D1 `score(marker_letter, cells, cuts)` — letter is a parameter, no globals; imports nothing from the service layer
- [x] D2 Percent rounded to nearest 5, half **up** (not banker's rounding), raw shown beside it
- [x] D3 Intensity = mean OD over positive cells only → banded, raw shown beside it
- [x] D4 H-score, Allred, ASCO/CAP call — computed, labelled reference-only, on their own sheet
- [x] D5 Area-weighted / pooled / plain-mean all computed; area-weighted reported (see "What changed" below)
- [x] D6 Q1 switch: count / half / exclude, all three reported per marker
- [x] D7 Heterogeneity map — the ROI tiled by local percent positive
- [x] D8 Full cascade payload: counts → formula → rounding → pair
- [x] D9 Tests on the arithmetic, including both roundings

## E. Step 17 — validation

- [x] E1 Reports honestly when there is no reader sheet, rather than inventing a comparison
- [x] E2 ±10 percentage points as the target; Bland–Altman and linearly weighted kappa
- [x] E3 Reader data transcribed and joined (`scripts/import_reader_scores.py`); the join tolerates the `/26` year the sheet carries and the case folder does not, and a marker with no intensity column validates on percent alone rather than being dropped

## F. Wiring

- [x] F1 All 17 steps `implemented=True` with rewritten catalogue entries
- [x] F2 `runner.py` runs 1→17 with no stub raise; a test pins runner order against the catalogue
- [x] F3 Router: `per-cell`, `binning`, `scores`, `validation` registered
- [x] F4 The marker chosen before step 1 reaches steps 14–16 — verified end to end, five markers
- [x] F5 `sync_frontend_catalogue.py` re-run; bundled catalogue matches
- [x] F6 Frontend: step 14 scatter, step 15 histogram, step 16 cascade, step 17 panel — wired, `tsc` clean, `vite build` clean

## G. The unattended run

- [x] G1 `scripts/run_case_scores.py` — case folder → five markers → steps 1-16 → one workbook
- [x] G2 H&E registered once and reused; step 8 ran **zero** extra times
- [x] G3–G7 All five markers scored
- [x] G8 `CAN_00270_scores.xlsx` — Scores, Workings, Caveats, Reference, Regions, Run
- [x] G9 `scripts/verify_case_scores.py` re-derives all five independently — exact agreement

---

## What changed in the existing pipeline, and why

**A real bug in step 12, found by the first run's numbers not adding up.** Nucleus ids are
unique within a *field*, not within a region — step 11 segments each sampled field
separately, so every field's instance map restarts at 1. Step 12 flattened a region's
fields and wrote `{id: class}`, so field 3's nucleus 14 overwrote field 0's. On region 1
of this case that collapsed 406 counted nuclei into 175 entries. Steps 13 and 14 then
asked that map which ids were tumour and applied the answer to *every* field, keeping the
union rather than a subset: step 12 said 32.7 % tumour, step 13 built compartments for
965 of 1,064 cells.

Fixed by keying on `field:id`, with a shared reader (`step12_cell_typing/types_map.py`)
that refuses a version-1 map rather than mis-reading one. Step 13 now builds exactly 348
compartments for 348 tumour cells. Regression tests in `tests/test_types_map.py`.

**Region weighting.** Step 11 samples a fixed number of fields per region regardless of
region size, so on this case the largest region holds 84 % of the invasive area and
contributes 42 % of the measured cells. Pooling those cells is not the ROI's percentage —
it weights a 2 mm² region as heavily as a 22 mm² one. Each region is now measured on its
own sample and combined by area, with the pooled and plain-mean figures reported beside
it. On CD44 the three readings are 77 %, 45 % and 55 %.

**Alignment confirmation is now stamped** `person` or `machine`. The batch run has to get
past step 10's human gate to work unattended; what it must not do is leave a record that
looks like a human sign-off. Every score from a machine-confirmed pair carries that caveat.

---

## What is provisional, and what to do next

1. **The cut points are not fitted, and the comparison above now says by how much.**
   The shipped cuts come from the DAB optical-density scale itself; the measured bias is
   **+14.5 points, one-directional**. Raising each marker's positivity cut until its
   percentage lands on the reader consensus is now a solvable one-parameter fit per
   marker — but on **one case**, which is not enough to fit against without overfitting.
   Get the other five cases through the pipeline first, then fit on held-out cases.
   The config is a data file, so nothing but that file changes.
   Note the transcribed sheet carries six of eleven columns: the four missing intensity
   columns mean intensity can only be fitted for pan-cadherin today.
2. **The denominator is incomplete on every slide**, by 68–79 % against the case's own H&E
   inside the same regions. That is step 11's known shortfall under heavy DAB, and missed
   nuclei are disproportionately the strongly stained ones — so it inflates every
   percentage here. This is the largest single threat to these numbers.
3. **No person has checked the registrations** for four of the five pairs.
4. **Q1 does not move this case.** Counting, halving or excluding partially stained cells
   gives the same answer for all five markers — the positive cells are well clear of the
   completeness cut. That is a fact about this case, not a general result.

## Known unrelated failure

`tests/test_roi_borders.py::test_region_bbox_matches_the_outer_ring_extent` fails with a
one-pixel discrepancy (expects `(96, 64)`, gets `(97, 65)`). It exercises step 9's ROI
ring geometry — `step09_roi_mask/`, `roi_service.py` — none of which this work modifies;
no file in this changeset is on its import path. I have not established when it started
failing: `tissue_scoring_demo/` is untracked in git, so there is no baseline to diff
against. Worth a look on its own, but it is not downstream of steps 14–17. 655 other
tests pass.
