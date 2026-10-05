# Getting step 12 registration right

**Status:** implemented and running. Written and built 2026-09-23; overnight pass launched 22:56. See §8 for what was validated before launch.
**Goal:** every case produces a correct H&E→IHC transform for all five markers. Correctness
first; run time is negotiable.
**Scope:** step 12 only. No scoring runs until registration passes its own acceptance test.

---

## 1. What is actually wrong

Measured on 2026-09-21/23 over `data/demo/ihc_alignment/*/valis/pair/`, the two overnight
run logs, and `data/oncostem_docs/`.

### 1.1 The tissue corresponds. The matcher fails.

Threshold each of VALIS's own processed images to a tissue mask, normalise centroid and
area, search rotation for best mask IoU — stain texture ignored entirely:

| case | marker | matched kp | outline IoU | best rotation |
|---|---|---|---|---|
| CAN_00270 ✅ | F | 2996 | 0.909 | 0.0° |
| CAN_00270 ✅ | A | 67 | 0.890 | 7.4° |
| CAN_00251 ✅ | W | 2578 | 0.856 | 0.0° |
| CAN_00251 ✅ | F | 2510 | 0.830 | 9.9° |
| CAN_00267 ❌ | F | 5 | 0.877 | **350.1°** |
| CAN_00267 ❌ | U | 5 | 0.872 | **352.6°** |
| CAN_00267 ❌ | R | 5 | 0.791 | **352.6°** |
| CAN_00267 ❌ | A / W | — | 0.813 / 0.802 | **166.3° / 178.8°** |
| CAN_00303 ❌ | A | 0 | 0.817 | **347.6°** |
| CAN_00303 ❌ | F | 5 | 0.749 | **342.6°** |

**The failing pairs correspond as well as the passing ones.** A rigid transform that lines
them up exists and is findable in about a second from the outline alone.

This supersedes the earlier call that CAN_00267's H&E was "torn into fragments, needs a
re-cut slide, not code". That was read off the stain rendering. The geometry says the block
is intact.

### 1.2 Two CAN_00267 IHC sections are mounted ~180° round

VALIS's default matcher is `DiskFD` + `LightGlueMatcher` (`valis/registration.py:72`). Both
are learned descriptors trained on upright imagery and **not rotation invariant**. A
166°/179° mount produces 0 usable matches by construction. Classical descriptors (SIFT,
BRISK, KAZE) are rotation invariant.

### 1.3 The resolution setting has never taken effect

`registration_max_dim_px = 2048` is documented at `backend/app/core/config.py:1073` as "the
single biggest accuracy lever in step 10". Every processed image on disk is **512 px**.

`Valis.check_img_max_dims()` (`valis/registration.py:~2600`) silently clamps
`max_image_dim_px` to the long edge of the *smallest image it could read*, then clamps
`max_processed_image_dim_px` down to match. It warns through `valtils.print_warning` and
carries on; nothing in our pipeline logs that warning.

The cause is in the files: `PEER-REVIEW-BRIEF.md:71` records the pyramid ladder as
**1, 4, 8, 16, 32, 64, 128, 256 with no 2x step**, depth varying 6–8 across slides. The
smallest readable level for some slides lands near 496 px, so every slide in the pair is
dragged down to it.

This is not what breaks CAN_00267/CAN_00303 — they fail at both 512 and 2048 — but it means
no resolution claim in the codebase has been tested.

### 1.4 Cropping to tissue is already done — this is a dead end

Worth recording so nobody spends a day on it. Measured tissue fill of VALIS's processed
frames: **0.72–0.97**, with the tissue bounding box spanning **0.99 of the frame in every
single pair**, and the H&E:IHC tissue-scale ratio already **1.00**.

`cohort_characterisation.md:35` is right that the raw canvas is a fixed scanner grid with
tissue in a quadrant — but VALIS crops to tissue before processing. There is no padding left
to remove and no scale mismatch left to fix. **Cropping buys 1.01x.**

The resolution problem is entirely §1.3's clamp, not framing.

### 1.5 Slide A is pale across the whole cohort

`cohort_characterisation.md:98` — slide A (CD44) has the lowest DAB density on every case and
is **near-blank on 00267 and 00865**. It is also the slide that failed with 0 matched
keypoints. A near-unstained section gives an optical-density rendering with little signal, so
adaptive equalisation amplifies scanner noise into the dominant texture.

Any tissue mask used here must use the **optical-density rule, not saturation**. The same
document measures saturation capturing only **4% of tissue on 00865/A and 13% on 00267/A**.

---

## 2. What the OncoStem documents do and do not say about the cut sequence

Searched `data/oncostem_docs/` for sectioning order.

**Confirmed:**

- All six slides of a case are serial sections of **one block**, sharing a block ID on the
  label — e.g. `B2011-T` for CAN_00303 (`AlgoDev1-case-00303-analysis.md:35`).
- Labels read `Ab-A`, `Ab-U` — the letter code only. **No section number on the label**, so
  the glass cannot be read for cut order.
- The letter codes are markers, not positions:
  `A = CD44, F = ABCC4, R = ABCC11, U = N-cadherin, W = pan-cadherin`.
- **The H&E is cut first.** SOP clause 1.5: *"A first slide and an H&E review after IHC
  sectioning (7th/9th slide) confirms invasive tumour is still adequate."* So the H&E sits at
  one **end** of the ribbon, with the five IHC sections following it, and a second H&E at
  position 7 or 9.

**Not documented anywhere:** the order of A, F, R, U, W within the ribbon, or the section
thickness.

**Consequence for the plan.** The H&E being an endpoint is the worst case for direct
registration: the last IHC section is 5–9 sections of block depth away from it. It is the
best case for chained registration, because every intermediate section is available. Since
the order is not on record, **the plan measures it** (Phase 2) and reports it — which both
solves the registration and answers the question.

**Context worth knowing, not a blocker:** OncoStem's own answer to G26
(`archive/OncoStem-Data-Requirements-and-Validation-Plan.md:137`) is that *"the pathologist
re-identifies the region by eye on each slide, using normal ducts as landmarks. No
registration is used."* Registration is our addition. The alternative — running the region
model directly on each IHC appearance — has been measured and fails: the H-channel invasive
class returns 0% invasive on all five stains at 0.96+ confidence. So registration is the
path, and it has to work.

---

## 3. The approach

**Register each case once, as a six-slide stack, at a controlled resolution, in an order we
measured, with orientation normalised first and a rotation-invariant fallback.**

Five changes, each independently testable. Phases 1–3 are the correctness fix; 4–5 make it
runnable overnight.

### Phase 1 — Render all six slides to a common scale (defeats the clamp)

The root cause of §1.3 is that VALIS reads slides through their own pyramids and normalises
to the weakest one. Take that decision away from it.

For each of the six slides in a case:

1. Read at the pyramid level nearest the target µm/px, selecting **by measured downsample,
   never by index** (`cohort_characterisation.md:37` — the ladder has no 2x step).
2. Build a tissue mask with the **optical-density rule** (`app/ingestion/tissue.py`, the one
   fixed for padding-vs-glass by flatness, not brightness). Not saturation — see §1.5.
3. Crop to the tissue bounding box and resample to **exactly the same µm/px for all six**.
4. Pad all six to one common canvas, centred on the tissue centroid.
5. Write flat pyramidal TIFFs to `data/registration/<case>/render/`.

**Target resolution: 8 µm/px**, giving roughly 2000–4000 px on the long edge for these
specimens (35–315 mm²). That is **3.5x finer than the 27.6 µm/px everything has run at so
far**, and it is a real number rather than one VALIS will override. Make it a parameter;
4 µm/px is the stretch target if run time allows.

This phase alone guarantees identical physical scale across all six, which pairwise VALIS has
never guaranteed.

### Phase 2 — Measure the cut order

On the six masks from Phase 1, for all 15 unordered pairs:

- best-rotation outline IoU (the measurement in §1.1 — about a second per pair),
- normalised mutual information on absorbance (the arbiter `_tune.py` already uses).

Then: **the cut order is the path through all six slides maximising consecutive similarity**,
with the H&E pinned to one end per SOP 1.5. Six nodes with one end fixed is 120 permutations —
brute-force it.

Write `data/registration/<case>/order.json` with the similarity matrices, the chosen order,
and the margin over the runner-up. If the margin is thin, say so rather than asserting an
order.

Run this across all six cases before anything else. If the order is consistent case to case,
that is a fact about the lab's protocol worth sending to OncoStem.

### Phase 3 — Normalise orientation, then register the stack

1. **Rotate before matching.** Apply each slide's Phase-2 rotation to bring all six into a
   common frame. Record every angle. This removes the 180° problem (§1.2) before any matcher
   sees an image, and it is stain-blind so it cannot fail the way these are failing.

2. **One VALIS run per case**, not five:

   ```
   img_list            = the six renders, in Phase-2 order
   imgs_ordered        = True          # we measured it; do not let VALIS re-sort
   reference_img_f     = the H&E render
   align_to_reference  = False         # register neighbour-to-neighbour, compose the chain
   ```

   `align_to_reference=False` is the point of the whole exercise. Consecutive sections of one
   block resemble each other far more than an H&E resembles a DAB slide, so H&E→A routes
   through F/R/U as stepping stones and VALIS composes the chain. It is also **one
   registration per case instead of five**.

3. **Detector ladder, inside one JVM start.** Try in order, stop at the first rung whose
   weakest consecutive link clears 50 matched keypoints:

   | rung | configuration | why |
   |---|---|---|
   | 1 | `OD` + `adaptive_eq=True`, default DISK/LightGlue | current setting, works on 4/6 cases |
   | 2 | classical rotation-invariant detector (SIFT or BRISK) + `Matcher()` | §1.2; `_tune.py` already has the harness |
   | 3 | rung 2 at 4 µm/px | more signal for the pale slides (§1.5) |

   A rung costs minutes, not an hour, because the JVM is already up and Phase 1's renders are
   already on disk. Note `_tune.py` records `MicroRigidRegistrar` as tried and rejected —
   47 minutes and 2.3 CPU-hours without finishing. Do not put it on the ladder.

### Phase 4 — Gate the composed transform, and fail fast

Keep all four existing checks in `app/registration/gate.py`. Add two things:

- **Chain-weakest keypoints.** For a composed H&E→marker transform, the trustworthy number is
  the **minimum** matched-keypoint count over the links in that chain, not the endpoint pair's.
  A chain is as strong as its weakest join. Report both.
- **Fail fast, once per case.** Phases 1–2 are minutes and are entirely stain-blind. If the
  measured outline IoU for a marker is below ~0.6, or Phase 3's stack registration fails after
  rung 3, mark the **case** blocked and skip its remaining markers. Last night that would have
  turned 4.5 hours of identical repeats into about 20 minutes.

Also log `registrar.max_processed_image_dim_px` **after** `register()` returns, so §1.3's
silent clamp can never hide again.

### Phase 5 — Cache by case, not by pair

Phase 3 produces one registrar for six slides, so the current
`data/demo/ihc_alignment/<he>__<ihc>/` layout no longer fits. Move to
`data/registration/<case>/`, with per-marker reports underneath.

Two known traps to carry across:

- The start-up orphan sweep read pair-keyed directories as orphans and deleted 483 MB on one
  restart. The case-keyed layout must be registered with the sweep **before** the first
  overnight run.
- `registrar.summary_df` is empty on a reused registration; the CSV on disk is the fallback.
  `_summary_from_disk` already handles this — keep it.

---

## 4. The overnight run

Registration only. No nuclei, no scoring — those are cheap to add back once the transforms
are right, and leaving them out keeps the loop short enough to iterate on.

### Files — built 2026-09-23

```
slide_registration/common.py           paths, checkpoint, atomic JSON, case discovery
slide_registration/render.py           Phase 1 - six slides onto one common grid
slide_registration/order.py            Phase 2 - similarity matrices, cut order, rotations
slide_registration/stack.py            Phase 3 - rotate, then one VALIS stack run
slide_registration/run_overnight.py    driver: all cases, checkpointed, summary table
slide_registration/run_overnight.cmd   detached wrapper
slide_registration/watch.py            read-only status, safe to run mid-pass
slide_registration/beetle_coverage.py  §5's three CSVs
slide_registration/archive_scores.py   dated copy of the scores, before anything changes
tissue_scoring_demo/valis_service/register_stack.py   the VALIS-side stack registration
```

`run_overnight.cmd` is a `.cmd` wrapper calling Python directly — the agent shell cannot
spawn a nested `powershell.exe` (EPERM), and this is how the previous overnight runs were
detached.

### Nothing here is specific to these six cases

The pass has to work on slides nobody has seen yet, so every case-specific fact is
discovered rather than configured:

- cases are listed from the slide library on disk, not from a constant;
- slide codes come from the filenames, handling both `CAN_00865_26-A.svs` and
  `CAN_00267_26_A.svs`, and a code nobody recognises is registered anyway rather than
  silently dropped;
- the cut order is measured per case, never assumed, and falls back from exhaustive search
  to a greedy chain above eight sections so a larger panel does not blow up;
- run order uses measured tissue area where it exists and total slide bytes as a proxy
  where it does not, so an unknown case sorts sensibly instead of last;
- the only hard requirement is that a case has one slide identifiable as H&E.

### Order of execution

```
# 1. One small case end to end, to see the numbers before committing a night.
python slide_registration/run_overnight.py --only CAN_00303

# 2. Cut order across all six. Minutes, and it answers the open question.
python slide_registration/run_overnight.py --phase render,order

# 3. Full overnight: render + order + register, every case.
slide_registration\run_overnight.cmd --deadline 6.5 --timeout 5400
```

Check progress at any time, from another window, without touching the pass:

```
python slide_registration/watch.py            # one snapshot, with a pass/fail verdict
python slide_registration/watch.py --follow   # refreshes every 30s
```

**Smallest case first, not hardest first** — changed from the first draft of this plan.
Registration cost scales with tissue and the cohort spans 36 mm² to 315 mm², a ninefold
range. Small-first means a night that gets cut short still answers the question on several
cases, and a mistake in the approach surfaces in twenty minutes rather than three hours.
Resulting order: `CAN_00303, CAN_00865, CAN_00267, CAN_00251, CAN_00270, CAN_00259`.

Budget: Phase 1 is about 10 seconds per case, Phase 2 about 2 minutes. Phase 3 is the
whole cost. Per-case cap 90 minutes covering the entire ladder, deadline 6.5 hours for
starting new cases — which bounds the night at roughly 8 hours.

### What was found while building it

Three things measured during the build that were not in the original plan:

1. **The clamp is genuinely defeated.** VALIS's processed images for CAN_00303 came out at
   2048 px, against 512 for every registration this project has ever run. Since the render
   is also cropped to tissue rather than being the full scanner grid, the effective gain is
   larger than 4×.
2. **`torch` here is `2.14.0+cpu`, `cuda.is_available() == False`.** VALIS's default
   detector and matcher — `DiskFD` and `LightGlueMatcher` — are both neural networks, so
   both run on the CPU. At 2048 px the first test was still inside feature matching after
   28 minutes on the *smallest* case in the cohort, at 115% of one core. This is very
   likely where most of the old pairwise step 12's 16-to-60 minute registrations went, and
   it is the reason the ladder now leads with BRISK: OpenCV, fast on a CPU, and rotation
   invariant by construction.
3. **The cut order is measurable but the margin is thin.** CAN_00303 comes out as
   `HE → U → F → A → R → W`, but the winning chain beats the runner-up by 0.021 of total
   IoU on 4.63 — under half a percent. All 15 pairs sit between 0.84 and 0.94, which is
   the real finding: every section of this block resembles every other closely enough to
   register against. The order is therefore recorded and used, but it is not load-bearing
   — the ladder's direct-to-reference rung owes nothing to it.

### If a case still refuses

The ladder is the automated answer and it has five rungs, but if a case comes out of the
night with nothing, these are the next things to try, roughly in order of how likely they
are to be the problem:

1. **Look at `data/registration/<case>/render/*.png`.** Every diagnosis in this document
   started by looking at what the matcher was actually shown. If the tissue mask is wrong,
   nothing downstream can be right.
2. **Check `order.json`'s pairwise IoU table.** If some section sits below ~0.6 against
   every other, that section really is different — torn, folded, or cut from somewhere
   else in the block — and no registration setting will fix it.
3. **Raise `--mpp`,** i.e. render finer. 8 µm/px is the default and 4 is the stretch
   target; a small specimen like CAN_00303 can afford it.
4. **Install a CUDA torch in the VALIS environment.** The learned detector and matcher are
   the strongest option available and are currently unusable on cost alone. This is the
   single change that would most widen what the ladder can attempt.

### Acceptance criteria — check these in the morning

| # | criterion | source of truth |
|---|---|---|
| 1 | All 6 cases produce a composed H&E→marker transform for all 5 markers | `data/registration/<case>/report.json` |
| 2 | Weakest chain link ≥ 50 matched keypoints on every marker | same |
| 3 | Residual ≤ 250 µm and round-trip ≤ 100 µm, the existing gate | same |
| 4 | Absolute alignment NMI ≥ 1.008, and any gain over identity above +0.002 | old successes landed 1.0096–1.0173; failures gained ~0.0000 |
| 5 | CAN_00251 and CAN_00270 do not regress | keypoints still in the thousands |
| 6 | Logged `max_processed_image_dim_px` equals what was requested | the run log |
| 7 | Cut order reported with its margin for all six cases | `order.json` |

Criterion 4 is the one that matters most. Keypoint counts flatter a fit made on the features
it was fitted to; NMI over the whole section does not.

It is stated on the **absolute** NMI rather than on the gain, and that is a correction to
this plan's first draft. Phase 1 already centres every section on its tissue centroid and
Phase 3 already rotates it into the H&E's frame, so "unregistered" here is a far better
starting point than the old raw-canvas overlay: it measures 1.0064 where the old baseline
measured 1.0028. The gain is smaller because the baseline is better, not because the
registration is worse, and comparing gains across the two would be comparing different
things.

**Then, and only then**, put two eyes on `ihc_borders.png` for one marker per case. The gate
can refuse but it cannot approve — a mask on plausible-looking but wrong tissue is the failure
numbers have never reliably caught, and nothing above changes that.

---

## 5. Is the tile-based ROI actually finding tumour?

Added 2026-09-23 at the user's request, and it turned out to be cheap: everything needed
is already on disk from previous step-11 runs, so this is a read-only analyser
(`slide_registration/beetle_coverage.py`) rather than a pipeline change.

**The question.** Step 9 draws candidate regions on step 8's tile grid — squares of a
fixed field of view, a staircase around whatever the tile model called tumour. Step 11
then runs BEETLE inside each box at the pixel level and traces the real boundary. The
ratio between the two says whether the tile approach is finding tumour or finding
tumour-shaped neighbourhoods.

Reported at three grains, into three CSVs at the repo root:

| file | one row per | the number that matters |
|---|---|---|
| `beetle_coverage_by_slide.csv` | slide | `contourSharePct` — of all area the tiles claimed, how much BEETLE kept |
| `beetle_coverage_by_region.csv` | ROI | the same per candidate, plus the per-tile spread inside it |
| `beetle_coverage_by_tile.csv` | tile | `invasiveSharePct` for every individual tile |

**Why the per-tile grain is the point.** A slide at 15% kept could be every tile 15%
invasive — a systematic over-reach the grid could be shrunk to fix — or 15% of tiles at
100% and the rest empty, which is a different problem with a different fix. The averages
cannot tell those apart. So the region rows also carry `tilesMostlyInvasive` (share ≥ 50%)
and `tilesNearlyEmpty` (share < 5%).

**Two area rules, both reported.** `maskShare` is BEETLE's raw per-pixel verdict.
`contourShare` is what survives tracing, which drops components below
`roi_refinement_min_component_mm2` and is what actually gets carried onto the IHC slide.
The second is what the score depends on; the difference between them is how much of the
gap is tracing rather than BEETLE.

**First measurement, over the six cases' existing step-11 output:**

| case | tiles claimed | BEETLE kept | tiles mostly invasive | tiles nearly empty |
|---|---|---|---|---|
| CAN_00303 | 25.74 mm² | **56.9%** | 49.8% | 15.0% |
| CAN_00270 | 36.17 mm² | **48.2%** | 35.3% | 33.3% |
| CAN_00865 | 239.69 mm² | 15.0% | 13.4% | 77.7% |
| CAN_00251 | 8.38 mm² | 4.7% | 1.3% | 58.8% |
| CAN_00267 | 1.96 mm² | 0.7% | 0.0% | 89.2% |
| CAN_00259 | 199.11 mm² | **0.0%** | 0.0% | 100.0% |

Read it as two populations rather than a gradient. On CAN_00303 and CAN_00270 the tile
grid is working: roughly half the claimed area survives and roughly half the tiles are
mostly invasive, which is what a slightly over-reaching box around real tumour looks like.
On the other four it is not: CAN_00259 claimed 199 mm² and BEETLE kept none of it, and
CAN_00251 has only 1.3% of tiles mostly invasive against 58.8% nearly empty — the grid is
claiming empty ground, not over-reaching around tumour.

Note the analyser reports one row per **H&E** slide, because step 11 only ever runs on the
H&E; the IHC slides inherit the region through registration rather than having one drawn
on them. That is the pipeline's design, not a gap in the measurement.

These tables are rebuilt at the end of every registration pass, so they stay in step with
whatever step 11 last produced.

## 6. Re-scoring the two cases that already have numbers

Asked for, and deliberately **not** run unattended. The reasoning is worth writing down
rather than leaving as a gap.

CAN_00251 (5/5) and CAN_00270 (4/5) are the only scored rows that exist, and they took two
overnight passes to earn. Re-scoring them under the new registration is the right thing to
do — two sets of numbers produced by different registrations are not comparable, so the
cohort should end up all one method. But it needs the new stack registration wired into
the demo backend's step 12, replacing `ihc_alignment_service`'s pairwise call, and that
wiring did not exist when this run was launched. Running an untested integration
unattended, against the only good results in the project, risks ending the night with
neither the old numbers nor new ones.

So the safe half was done and the risky half was not:

- **Done:** `slide_registration/archive_scores.py` took a dated copy of
  `oncostem_ai_scores.csv`, `score_all_slides/results/`, `state/` and `logs/` into
  `data/score_archive/<stamp>_before-stack-registration/`. It copies rather than moves, so
  the live pass still reads the originals and nothing has been made worse.
- **Not done:** the backend integration and the re-score.

**The hard part is already built and tested.** The coordinate round trip — the one place
where a mistake produces a mask on the wrong tissue and no error message — lives in
`slide_registration/warp.py` and `valis_service/warp_stack.py`, written while the
overnight pass was running. Carrying a ring from H&E level-0 pixels to IHC level-0 pixels
is three hops:

```
H&E level-0  --affine A_he-->      H&E aligned render
H&E aligned  --VALIS------->       IHC aligned render
IHC aligned  --affine A_ihc⁻¹-->   IHC level-0
```

`A_he` and `A_ihc` are the 2×3 matrices Phase 3 stores in `aligned.json`. They are
inverted numerically rather than derived by hand at the call site, which is how a
scale-factor bug gets in.

`warp.py --selftest` takes a lattice over the H&E's own tissue mask, carries it across and
back through both affines, and reports how far the points came home. That exercises the
whole composition — an inverted sign or a transposed matrix shows up immediately — and
unlike a keypoint residual it is measured on the region actually being moved rather than
on the features the fit was made from. It exits non-zero if the median round trip exceeds
50 µm.

**What remains**, for whoever picks it up:

1. `ihc_alignment_service.execute()` currently calls `valis_align.warp_rings()` per pair.
   Point it at `StackWarper` instead. The interface is the same shape: rings in, rings
   out, both in level-0 pixels.
2. Keep all four existing gate checks, and add the chain's weakest link — a composed
   transform is only as good as the worst join along it.
3. Only then re-score, with `--redo CAN_00251,CAN_00270`.

## 7. What this does not fix

- **CAN_00259's region selection.** The unattended default took the two slide-spanning
  step-9 components and spent the whole coverage budget on them, so the four real tumour
  regions were never selected. Separate problem, separate decision — it changes the
  denominator the scores are measured inside, so it is a call to make deliberately.
- **Slide A's staining.** If CD44 is genuinely near-blank on 00267 and 00865, registration can
  succeed and the score still be meaningless. `cohort_characterisation.md:98` says only
  OncoStem's own readings settle whether that is biology or a failed stain.
- **Machine confirmation.** The batch runner stamps `confirmed by machine` when the gate
  passes. That caveat rides on every score produced this way and should stay visible.

---

## 8. Run log — 2026-09-23

Launched 22:56, detached, `--deadline 6.0 --timeout 5400`.

**Validated before launch:**

- **Registration works on the case that used to return zero.** CAN_00303 previously got
  0 matched features after 938 seconds. Under the new pipeline it registered in
  **118 seconds** on the first and cheapest rung, with **151 matched keypoints on the
  weakest link of the chain** and all five markers between 151 and 1842. Non-rigid errors
  2.1–5.4 µm. VALIS's own six-way overlay comes out near-neutral grey across the whole
  section, with colour fringing only at a duct — where serial sections genuinely differ
  with depth through the block.
- **The resolution clamp is defeated.** A 2048-px run produced 2048-px processed images,
  against 512 for every registration this project has previously run.
- **The tissue mask was wrong and was fixed.** An optical-density floor of 0.28, set to
  clear scanner padding when padding had already been removed by flatness, clamped the
  threshold up past the tissue on pale sections: CAN_00865's near-blank CD44 slide
  measured 0.05 mm² against a published 17.5, and CAN_00267's pale H&E 4.6 against 163.5 —
  while every number on CAN_00303 looked perfect. `validate_masks.py` now checks all 36
  slides against the cohort characterisation's published table; 28 of 36 are within 35%.

**Known-imperfect at launch, deliberately not chased further in the dark:**

- CAN_00259 measures 0.43–0.64 of its published tissue area on **all six** slides. The
  ratio being uniform across the case is what makes it tolerable: every section is
  measured the same way, so they remain comparable to each other, which is what
  registration needs. It is still worth understanding.
- CAN_00267/F (0.33) and CAN_00270/A (0.64) are outliers *within* their cases, so those
  two slides are the most likely to register badly. Check them first.

### The coordinate chain is verified

`warp.py --selftest` on CAN_00303, 401 points sampled over the H&E's own tissue mask,
carried H&E → CD44 → H&E through both affines and VALIS in between:

```
roundTripMedianUm      0.007
roundTripP95Um         0.217
roundTripMaxUm       151.207
displacementMedianUm  1115.8
```

A median round trip of **7 nanometres** over points spanning the whole section means the
composition is exact: both affines invert properly and VALIS's forward and reverse warps
agree. The 151 µm maximum is a handful of points near the border, where a non-rigid warp
pushes past the canvas edge and is clamped.

The median displacement of **1.1 mm** is the other half of the result: the registration is
moving tissue a real distance, so this is a substantial correction rather than a transform
that quietly does nothing.

### Measured on the first case through the new pipeline

CAN_00303, `brisk_chained_512`, 108 seconds:

| marker | stain | keypoints | NMI | gain | non-rigid error |
|---|---|---|---|---|---|
| A | CD44 | 870 | 1.0303 | +0.0157 | 20 µm |
| F | ABCC4 | 309 | 1.0305 | +0.0136 | 28 µm |
| R | ABCC11 | 163 | 1.0267 | +0.0120 | 42 µm |
| U | N-cadherin | 1353 | 1.0246 | +0.0085 | 21 µm |
| W | pan-cadherin | 1959 | 1.0255 | +0.0132 | 16 µm |

**Every one of these is better than any registration this project has previously
produced.** The old successes landed at NMI 1.0096–1.0173; these are 1.0246–1.0305, on a
case that used to return zero matched features.

One reporting note: VALIS records its errors in the units of the images it registered, and
those are rendered PNGs with no physical scale, so the CSV says `pixel`. Those are render
pixels at 8 µm each — the table above is already converted, and `watch.py` converts too.

### A second pass is chained behind the first

`slide_registration/rerun_weak.py` starts automatically when the first pass exits. It
re-renders and re-registers **only** the cases whose first-pass result was weak — failed
outright, weakest chain link under 50 keypoints, or any marker's NMI under 1.008 — and
moves their first-pass output aside to `*.first-pass.json` rather than deleting it, so
both attempts are on disk and comparable.

It exists because of a fix made while the first pass was already running. `render.py` now
builds **two** masks per slide, and the distinction matters:

| mask | built from | used for |
|---|---|---|
| strict | Otsu on optical density, within the non-padding region | placing the section, and every tissue-area number |
| `visible` | anything that is not padding and not bare glass | deciding what the feature matcher may see |

The strict mask alone was whitening near-negative sections out of existence. CAN_00865's
CD44 slide measures 14 mm² against a published 17.5 while its siblings measure 85–96 —
that measurement is *correct*, the section really is almost unstained, and the cohort
characterisation says so in as many words: "tissue plainly present but almost unstained".
Whitening everything outside that mask left the matcher a handful of specks, and the first
pass duly returned **6 keypoints** on CAN_00865's weakest chain link. The permissive mask
keeps the faint tissue, which is what the registration actually needs.

Restarting a working pass to pick that up would have risked the results it had already
earned. Running it afterwards, only where it matters, costs a little CPU and risks
nothing.

### Course correction, 23:11 — two masks, and what each is for

The first pass was stopped ten minutes in and restarted over all six cases. The reason is
worth recording because the bug was subtle and the symptom pointed somewhere else.

`render.py` originally built one mask and used it for everything. That is wrong, and it
only shows up on a near-negative section:

| the mask is used to… | and the right error to make is… |
|---|---|
| place the section, and report tissue area | be **strict** — a pale fringe barely moves a centroid, and an area number must not count glass |
| decide what the feature matcher may see | be **permissive** — whitening faint tissue destroys exactly what the registration needs, silently |

CAN_00865's CD44 slide measures 14.25 mm² against a published 17.5 while its siblings
measure 85–96. That measurement is *correct*: the cohort characterisation says the slide
has "tissue plainly present but almost unstained". But using it to whiten the render left
the matcher a handful of specks, and the first pass returned **6 matched keypoints** — on
the chained rung *and* on the direct rung, which is what proved it was the input rather
than the detector or the chain.

There was a second, worse consequence I nearly missed. `order.py` also read the strict
mask, so it measured CAN_00865/A as 0.36 IoU against every sibling and fitted it a
**253° rotation** from those specks — a rotation `stack.py` would then have *applied* to
the real image before registering it. A bad number quietly becoming a bad transform is
precisely the failure mode this whole document exists to avoid.

Both now read `<code>_visible.png`, the permissive mask. After the fix, slide A renders as
a complete section matching its siblings' outline.

**What is preserved.** Every first-pass output was renamed to `*.first-pass.json` rather
than deleted, including CAN_00303's successful registration, so the two attempts can be
compared rather than taken on trust.

**What is chained.** `rerun_weak.py --mpp 4.0` fires when this pass exits and retries only
the cases that are still weak, at double the resolution. That is a genuinely different
attempt rather than a repeat — re-running the same inputs through the same ladder would
produce the same answer.

### One bad slide was failing four good markers

Found at 23:27 while watching CAN_00865, and it is the difference between a case scoring
0 of 5 and 4 of 5.

`register_stack.py` reports `chain_weakest_matched`: the fewest matched keypoints over
**every** link in the chain. As a single number for "did this registration work" that is
right. As the criterion for whether a *particular marker* can be trusted it is wrong, and
wrong in the expensive direction.

A chained registration composes the transform along the cut order, so the transform for a
marker depends only on the joins **before** it. CAN_00865 registers as:

```
HE → U → F → W → R → A
                      ↑ near-negative CD44 section, 5 matched keypoints
```

A is last. U, F, W and R compose through joins that never touch it — yet all four were
being judged by A's 5 keypoints and marked failed.

`watch.py` now reports `toHere` per marker: the minimum over the sub-chain from the H&E to
that marker only, falling back to the slide's own count for a direct-to-reference rung and
to the whole-chain figure when the order is unknown. `run_overnight.py` imports the same
function for the CSV's `chainWeakestToMarker`, so the screen and the spreadsheet cannot
disagree.

The verdict column reads `weak chain to here (N)`, naming the number rather than saying
"weak chain", because the next question is always "how weak".

**What this does not change.** The rung-selection logic inside `register_stack.py` still
requires *every* slide to clear the bar before it stops climbing the ladder — which is
right, because a rung that fixes the hard slide is worth finding. What changed is only how
a finished result is judged, and that partial success is now visible instead of being
rounded down to failure.

### How a marker is judged — and a correction I had to make to it

Three things are reported, but they do **not** carry equal weight, and getting that wrong
nearly cost a good registration.

| signal | threshold | what it is |
|---|---|---|
| **alignment NMI** | ≥ 1.008, gain > 0.002 | **the decider.** Measured over a lattice covering the whole tissue, and owes nothing to the fit |
| keypoints to here | ≥ 50 | context: how much the sub-chain had to work with |
| non-rigid residual | ≤ 250 µm | context: only meaningful when there were enough keypoints to measure it on |

**The correction.** I added the residual as a hard check after seeing CAN_00865's R slide
report a **2374 µm non-rigid residual**, and wrote here that its chain "breaks at W→R".
Then I looked:

```
rigid-registered overlap   W vs R   0.908
                           W vs HE  0.677
                           R vs HE  0.672
```

**R and W overlap at 0.908 IoU after registration.** They are well aligned. The 2374 µm
figure is not a displacement at all.

Both of VALIS's error columns are distances between the keypoints the transform was
*fitted to*. On a slide with five matches they are computed from five points and describe
that sample, not the transform. They mislead in both directions: they flatter a bad fit
made from many matches, and they malign a good fit made from few.

So the verdict now leads on NMI, and reports a large residual as a **note** rather than a
failure when the keypoint count is too low to support it — because two unreliable signals
do not combine into a reliable one.

This is the same trap this document warned about in §3 and I walked into it anyway, which
is worth recording. The lesson that actually generalises: *look at the overlay*. Every
question tonight was settled faster by an image than by a column of numbers.

What remains true about CAN_00865: **the H&E is the outlier**, overlapping the IHC sections
at only ~0.67 after registration while they overlap each other at 0.908. That is consistent
with its shape IoU (0.744–0.806 against the others, which sit at 0.80–0.94 among
themselves) and with SOP clause 1.5 putting it first in the ribbon, furthest from
everything else in the block. Whether ~0.67 is good enough to carry a mask is the morning's
real question, and it is a question about the H&E, not about R.

### CAN_00865, as far as it is understood

Superseded reading, kept because the mistake is instructive: I first concluded its chain
"breaks at W→R" from the residual column. It does not — see the correction above. R and W
overlap at 0.908 after registration.

What the evidence actually supports:

*Not a rotation error.* Every section's measured rotation against the H&E is small and
plausible, between −5° and −22°. No spurious half-turns.

*Not a shape mismatch between the IHC sections.* Every link in the chain is strong:

```
HE→U 0.804    U→F 0.914    F→W 0.938    W→R 0.916    R→A 0.813
```

*The H&E is the odd one out, before and after registration.* Its shape agrees with the IHC
sections at 0.744–0.806 where they agree with each other at 0.80–0.94; after rigid
registration it overlaps them at ~0.67 where they overlap each other at 0.908. Both
measurements say the same thing, and SOP clause 1.5 explains it: the H&E is cut first and
is the section furthest from every other in the block.

*Slide A is separately hard*, and for a different reason — texture, not shape. Its outline
IoU rose from 0.364 to 0.804 once the permissive mask stopped whitening it away, and the
matcher still found five keypoints, because a near-unstained section has tissue with almost
no absorbance contrast to key on. Fixing the shape comparison did not fix the matching.
Those are two separate problems and the second one is not obviously solvable by
registration settings.

**The morning question for this case** is therefore about the H&E rather than about R:
is ~0.67 overlap between the H&E and the sections it is carrying a mask onto good enough
for what the score measures? At region scale it may well be. At the 1.5 µm membrane shell
the scores are computed in, it may not be. That is a denominator decision.

First thing to look at: `data/registration/CAN_00865/valis/*/stack/overlaps/`.

---

## 9. The biggest finding: the outline alignment is doing most of the work

CAN_00865 finished having exhausted all five rungs, and its numbers say something that
reframes the whole approach:

| marker | keypoints | outline-only NMI | after registration | gain |
|---|---|---|---|---|
| U | 1705 | **1.0386** | 1.0243 | **−0.0143** |
| W | 1392 | **1.0326** | 1.0211 | **−0.0114** |
| R | 6 | **1.0320** | 1.0241 | **−0.0079** |
| A | 6 | **1.0170** | 1.0042 | **−0.0128** |
| F | 41 | 1.0347 | **1.0501** | +0.0154 |

**Four of five markers were better before VALIS touched them.** And the outline-only
figures are not poor — U at 1.0386 is higher than anything CAN_00303 reached *after* a
successful registration.

"Outline-only" here means what Phase 1 and Phase 2 already do without any feature matching
at all: render every section to one physical scale, crop to tissue, centre on the tissue
centroid, and rotate by the angle measured from the tissue outline. That is a complete
rigid registration derived entirely from shape, and on a block of serial sections it is
evidently a strong one.

Compare CAN_00303, where feature registration clearly earns its place:

| marker | outline-only | after registration | gain |
|---|---|---|---|
| A | 1.0095 | 1.0344 | +0.0250 |
| W | 1.0102 | 1.0318 | +0.0216 |
| R | 1.0126 | 1.0304 | +0.0179 |

So the honest summary is: **outline alignment is a strong baseline that feature
registration improves on good cases and degrades on hard ones.** Which means the pipeline
has to be able to choose, and until tonight it could not.

### Two things fixed as a result

**Rungs are now compared on NMI, not on keypoint count.** Selecting on keypoints chose
`vgg_chained_512` for CAN_00865 — six matches and an NMI *below* the outline alignment it
started from. A rung that actively made things worse was preferred because it had
marginally more matches than one that did not. `_quality_of()` now takes the median
per-marker NMI gain, median rather than mean so one unregisterable section does not
condemn a rung that handled the other four.

**`watch.py` shows the outline baseline beside the registered figure**, and says
`WORSE than outline alone` in as many words when the gain is negative. That comparison was
computable from the first run and simply was not on screen.

### What is deliberately NOT done

**Nothing automatically falls back to the outline transform.** The machinery is all there —
the affines are stored, `warp.py` composes them, and skipping the VALIS hop would give the
outline-only transform exactly — but which transform carries the mask decides which tissue
the scores are measured inside. That is the same class of decision as the region-selection
question, and it belongs to you, not to a heuristic that fires at two in the morning.

What is worth knowing before deciding: the outline transform is **rigid**. It cannot follow
the local deformation a section picks up being floated onto glass, which is what the
non-rigid pass exists for. On CAN_00865 that pass is making things worse, but "rigid beats
a bad non-rigid fit" is not the same claim as "rigid is sufficient", and the 1.5 µm
membrane shell the scores are measured in is a demanding target for a rigid fit.

### The order phase did not scale, and now does

CAN_00267 is the largest canvas in the cohort at 3742 px, and its Phase 2 was taking
**about 2.5 minutes per pair** — 37 minutes for the 15 pairs of one case. The cost grows
with the square of the render resolution, so the chained 4 µm/px retry would have had
sixteen times as many points and taken hours per case.

The rotation search rasterises each slide's tissue cloud 145 times per pair, and
CAN_00267/A holds **6.2 million** tissue pixels. The destination is a 160×160 grid — 25,600
cells — so the overwhelming majority of those points were being discarded by the
rasteriser anyway, just slowly. Capping at 100,000 points saturates the grid four times
over.

**The sample has to be random, not every Nth pixel**, and that is not a nicety.
`np.nonzero` returns raster order, so striding walks the tissue row by row and lays down a
regular lattice that aliases against the comparison grid. Measured on CAN_00267's A–F pair:

| sampling | answer | overlap | time per pair |
|---|---|---|---|
| every Nth pixel | 198.6° | 0.855 | 3.2 s |
| **random, seeded** | **198.6°** | **0.859** | **3.0 s** |
| every pixel | 198.6° | 0.859 | 124.4 s |

The random sample reproduces the full-resolution answer exactly, at 41× the speed. The
stride did not. The seed is fixed so the same slides always give the same number.

Phase 2 now costs under a minute per case rather than 37, which is what makes a 4 µm/px
pass affordable at all.

### CAN_00267's A slide really is upside down

Worth recording as confirmation rather than inference. Phase 2 measures it at **184–199°**
against its siblings, which matches what VALIS's own processed images showed back in §1.1,
and explains the 0 matched features the old pairwise step 12 returned: its default
DISK + LightGlue matcher is not rotation invariant.

The tie-break did **not** suppress it, which is the behaviour that was wanted — leaving
that section unturned scores far worse than `TIE_TOLERANCE`, so the half-turn wins on
merit. A tie-break that could hide a real flip would be worse than no tie-break.

### Correction to §1.1: one CAN_00267 section is flipped, not two

The table in §1.1 was measured on VALIS's own `processed/` images, which VALIS had already
cropped and oriented for its own purposes, and it reported CAN_00267's **A and W** at
166.3° and 178.8°. Measured from the raw slides through Phase 1 and 2, the rotations
against the H&E are:

```
A 196.1°    F 12.4°    R 9.9°    U 7.5°    W 5.0°
```

**Only A is mounted the other way up.** W is within five degrees of the H&E.

The Phase 1–2 measurement is the authoritative one: it works from the slides themselves at
a known physical scale, where the §1.1 figures were taken from a downstream artefact of a
run that had already failed. The conclusion that a flip was present and was breaking the
learned matcher stands — it was just one slide rather than two.

After rotation, A's shape agrees with the H&E at 0.805 IoU, squarely among its siblings
(0.79–0.86), which is what a correctly detected flip should look like.

---

## 10. The method comparison: what is tested, in what order, and why

Added 2026-09-24 01:30, after establishing that VALIS's failure is specifically in feature
*detection* and that methods which do not use features can succeed where it cannot.

### The decisive first result

On CAN_00865, working at 768 px:

| marker | outline | mask | **mattes (MI)** | winner |
|---|---|---|---|---|
| **A** | 1.0249 | 1.0104 | **1.0488** | **mattes** |
| F | **1.0449** | 1.0082 | 1.0365 | outline |
| R | **1.0435** | 1.0086 | 1.0386 | outline |
| U | **1.0493** | 1.0068 | 1.0321 | outline |
| W | **1.0423** | 1.0065 | 1.0356 | outline |

**Slide A is the whole point.** It is the near-negative CD44 section that VALIS scored
1.0042 on with five matched features, after five ladder rungs and fifty-two minutes.
Mutual information scores it **1.0488** in about thirty seconds, because it uses every
pixel instead of a few hundred keypoints and a nearly unstained section still absorbs
light.

And on the other four markers the **outline alone still wins**, which is the second half
of the finding: no single method is right for every pair, so the pipeline has to compare
and choose rather than commit.

### The methods, in the order they are tried

Ordered by likelihood of solving the problem, not by sophistication, so that a run cut
short has still answered the important question.

| # | method | what it fits | needs features? | why at this priority |
|---|---|---|---|---|
| 1 | `outline` | nothing — the identity after phases 1–2 | no | free, and already beats VALIS on 4 of 5 CAN_00865 markers |
| 2 | `mattes` | similarity (shift, rotation, uniform scale) by Mattes MI | **no** | **the one that solves the actual problem** — see above |
| 3 | `affine` | affine (adds shear, anisotropic scale) by Mattes MI | no | more freedom, more chance to overfit a weak signal; judged not assumed |
| 4 | `bspline` | free-form non-rigid on top of the similarity fit | no | the **only** method that can follow local deformation from floating a section onto glass |
| 5 | `mask` | similarity fitted to tissue masks by distance transform | no | stain-blind geometry; kept for diagnosis, has measured worst |
| — | `valis` | similarity + non-rigid from matched features | **yes** | the incumbent, measured by `run_overnight.py` and joined into the same table |

### How the winner is chosen

Every method is scored by **normalised mutual information between the two sections'
absorbance, over a lattice covering the H&E's tissue** — the same probe `register_stack.py`
uses, so VALIS's numbers sit in the same table.

The critical property: the score is computed on the **result**, not on whatever the method
optimised. `mattes` optimises mutual information, so grading it on mutual information
would flatter it — but it is graded on a *different sampling* of the same quantity, over
the tissue mask, after the transform is fixed. `mask` optimises a distance transform and is
graded on MI. `outline` optimises nothing. No method is graded on its own objective.

Every row also carries `gainOverOutline`, because the only question that matters for a
given pair is whether a method beat doing nothing.

### What is fitted on what

**All transforms are fitted on whole tissue, never on the BEETLE invasive mask.** Two
tumour regions do not constrain a transform — a roughly round region matches many
positions about equally well — while a section outline does. The invasive mask is only
ever *carried* by the chosen transform, never used to find it. This was already the
architecture and is restated because it is easy to get backwards.

### Running it

```
python slide_registration/methods_run.py                    # all 6 cases, all methods
python slide_registration/methods_run.py --only CAN_00865
python slide_registration/methods_run.py --methods outline,mattes
```

Self-sufficient: it renders, measures the cut order and writes the aligned images itself if
they are not already there, so it does not depend on the VALIS pass having run first.

Checkpointed per case — a completed case is skipped, so it is restartable. Results land in
`data/registration/<case>/methods.json` and are collated into
**`registration_methods.csv`**, one row per pair per method, 30 pairs across 36 slides.

---

## 11. The unattended run

One orchestrator, `slide_registration/night.py`, runs five stages in sequence. One
process rather than several launched independently, because when the registration pass and
the method sweep were started separately they each halved the other's speed competing for
the same cores.

| stage | what | why here in the order |
|---|---|---|
| `methods` | every registration method on all 30 pairs | minutes, and it decides which transform everything downstream uses — no point scoring a cohort with a registration that loses |
| `score_tiles` | score the cohort on step 9's **coarse tile** regions, skipping step 11 | **the fast one, deliberately first.** Step 11 is the most expensive stage in the pipeline; if the scores barely move without it, most of the cohort's compute is buying very little |
| `score_beetle` | the same cohort on BEETLE's per-pixel boundaries | slow, and what the pipeline normally does |
| `coverage` | tile-versus-BEETLE area tables | seconds |
| `decide` | rewrite `DECISIONS.md` | seconds |

### Scoring both ways, without either contaminating the other

The two scoring runs keep **entirely separate state, per-case records and CSV**:

```
oncostem_ai_scores_tiles.csv   state_tiles/   results_tiles/
oncostem_ai_scores.csv         state/         results/
```

That separation is not tidiness. Interleaving them in one checkpoint would let a case
scored one way satisfy the other's "already done" check; interleaving them in one CSV
would produce a file whose rows mean two different things with nothing on the row to say
which. They are measured inside different tissue and are **not comparable as scores** —
measuring the difference is the entire point.

Step 12 normally *refuses* to carry tile regions, and that refusal is deliberate and
documented. It is now opt-in by name (`--roi-source tiles`), never a fallback: the source
travels on every region and is recorded on the report, so a number produced one way can
always be told from one produced the other.

### Surviving things nobody is awake for

**The stages do not depend on the session that started them.** They are detached Windows
processes; an assistant session ending, a usage limit being reached or a terminal closing
has no effect on them at all.

What they do not survive alone is a reboot or a hard crash, so:

- **`supervise.cmd`** loops `night.py` until nothing is pending. Safe to run any number of
  times — every stage is checkpointed, a completed stage is skipped, a half-finished case
  resumes from its own record. A lock file stops two supervisors running at once.
- **`night_pending.py`** answers "is anything left", separately from `night.py`'s exit
  code, because those are different questions: `night.py` exits 0 when *this attempt* is
  over, including attempts where a stage crashed. A stage is given three attempts and then
  treated as finished rather than spinning the supervisor forever.
- **A Startup-folder shortcut** restarts the supervisor at logon, so a reboot resumes.
  A scheduled task would have been tidier but `schtasks` needs elevation this session did
  not have; the Startup entry needs none and fires on the same event. **It should be
  deleted when the run is finished.**

`RESUME.md`, beside this file, is the operator's copy of all of this.

### The watchdog gap, and closing it

Found by the user asking a direct question: *is the watcher active?* Checkpoints were, and
had been writing correctly throughout. The watching was only half there, and the missing
half mattered.

`supervise.cmd` restarts `night.py` **when it exits**. But `night.py` ran each stage with
`subprocess.run(..., timeout=None)` — no cap at all. A stage that *hangs* never exits, so
the supervisor would have waited for it all night: nine unattended hours producing nothing
and signalling nothing.

Three things now cover it, and they cover different failures:

| piece | detects | acts by |
|---|---|---|
| `sentry.py` | a genuinely hung stage | kills the worker, so the stage fails and the supervisor restarts it |
| `night.py` hard cap | a stage exceeding 1.8× its own budget | `taskkill /T` on that stage's process tree |
| `supervise.cmd` | `night.py` exiting with work outstanding | re-runs it; three attempts per stage, then that stage is left alone |

**The sentry's whole difficulty is that slow is not hung**, and this project has already
paid for that lesson once: `score_all_slides`' log-silence watchdog was sized against step
8 and would have killed step 11, which held one region of CAN_00259 for forty-seven
minutes while burning four cores and printing nothing. A log line is a proxy for liveness;
CPU time *is* liveness. So the sentry requires **both** signals to agree — the log silent
for 45 minutes **and** the whole process tree accruing under 2 CPU-seconds per poll — and
three consecutive polls to agree before it kills anything. A slow stage keeps burning CPU
and is left strictly alone.

It demonstrated the distinction within a minute of starting: log nine minutes stale, CPU
climbing 3033 → 3321, verdict `alive`.

A second real defect surfaced while fixing the first: on timeout `subprocess.run` kills
only the direct child, never the grandchild. Every stage here spawns a worker in the other
virtual environment, so an orphaned VALIS or SimpleITK process would have held four cores
for the rest of the night while the supervisor restarted the stage beside it. The stage
runner now uses `Popen` and kills the tree by pid.

### The `score_tiles` path was untested, and is now tested

It was scheduled to run unattended for three and a half hours having never once executed.
Rather than find out at four in the morning, `_crossable` was called directly with
`roi_source="tiles"` against the three cases that have step 10 data:

| case | tiles branch | the coverage table's own figure | regions |
|---|---|---|---|
| CAN_00267 | 1.96 mm² | 1.9563 mm² | 11 |
| CAN_00303 | 25.74 mm² | 25.7351 mm² | 1 |
| CAN_00270 | 36.17 mm² | 36.1697 mm² | 26 |

The areas agree with a table computed by a completely separate script from different files,
which is the useful kind of confirmation. `source=step9_tile` is stamped on every region
and the "not comparable with refined ones" warning fires on every call.

---

## 12. The method comparison, measured over 25 pairs

Five of six cases complete. Every method scored by the same probe, computed on the result
rather than on what the method optimised.

| method | mean gain over outline | median | worst | wins |
|---|---|---|---|---|
| **bspline** | **+0.0400** | +0.0399 | −0.0067 | 13 |
| mattes | +0.0292 | +0.0228 | −0.0045 | 4 |
| affine | +0.0231 | +0.0199 | −0.0030 | 2 |
| outline | 0 (the baseline) | — | — | 1 |
| valis | −0.0322 | −0.0371 | −0.0554 | 0 |
| mask | −0.0351 | −0.0253 | −0.0911 | 0 |

### What this settles

**Two methods belong off the ladder**, on the same evidence: `mask` and `valis` both
average *worse than the identity transform*. `mask` has never won a pair on any case;
`valis` has a comparable number on only five pairs from one case, so that verdict rests on
a thin sample and should be read as one measurement rather than a cohort result — but it
is the measurement there is.

**Non-rigid deformation is real and material.** `bspline` winning 13 of 25 with a +0.040
mean answers a question this document had been leaving open: a rigid fit — however well the
outline is placed — does leave accuracy on the table, by a margin comparable to the entire
gain from registering at all. Sections are floated onto glass and they stretch.

**But the scoreboard is the product, not any single method on it.** CAN_00270 is the case
that proves it:

| marker | outline | winner | gain |
|---|---|---|---|
| A | 1.0173 | mattes 1.0729 | +0.056 |
| F | 1.0416 | affine 1.1096 | +0.068 |
| R | 1.0152 | affine 1.1171 | **+0.102** |
| U | 1.0431 | bspline 1.1040 | +0.061 |
| W | 1.0147 | mattes 1.1226 | **+0.108** |

**Three different methods win across five markers of one case.** Committing to `bspline`
on its aggregate lead would have left real accuracy on three of those five. Running all of
them costs minutes; choosing between them costs nothing.

### The two extremes, for calibration

CAN_00259 and CAN_00270 gain +0.04 to +0.11 from registration. CAN_00267 gains +0.0003 to
+0.026, and on two of its markers nothing beats doing nothing. Absolute NMI is not
comparable *between* cases — different tissue has different entropy — so the gain over
outline within a case is the figure to read.

CAN_00267 is also the case whose tissue masks sit furthest from the published table (H&E
217 mm² against 163, F 46 against 160). Before concluding those slides are simply hard, it
is worth checking whether the mask is capping what any registration there can achieve.

### An error worth recording

An earlier tally in this session reported "VALIS has won nothing" from a script that read
only `methods.json` — which does not contain VALIS. It scored zero **by construction**, not
by measurement. The table above joins both sources. The conclusion happened to survive; the
reasoning behind it did not, and a number that cannot lose is not evidence.

---

## 13. The sentry killed a healthy case. My bug, and the exact one I had warned about.

At 03:44 the sentry killed `run_case.py` while CAN_00251's N-cadherin marker was
registering. The case was declared done with **3 of 5 markers scored**; U and W were never
attempted.

Its own log records the whole thing:

```
03:34  SLOW not hung: log 47 min old but +290 CPU-s
03:38  SLOW not hung: log 51 min old but +280 CPU-s
03:40  STALLED: log 53 min old and only +0.0 CPU-s   strike 1/3
03:42  STALLED: log 55 min old and only +0.0 CPU-s   strike 2/3
03:44  STALLED: log 57 min old and only +0.0 CPU-s   strike 3/3
03:44  ACTING - killed pid 6772 (run_case.py CAN_00251 tiles)
```

**Two independent defects, both mine, which had to coincide:**

1. **It watched the wrong log.** `common.RUN_LOG` is written by `night.py`, which during a
   scoring stage writes nothing for hours by design. "Silent" was therefore guaranteed
   within 45 minutes of the stage starting, on every stage, always.
2. **It could not see the process doing the work.** The match listed script names and
   included `register_stack` but not `valis_service/register.py` — which is what burns the
   CPU during step 12 while `run_case.py` blocks waiting on it. So the sentry watched an
   idle parent, read zero CPU, and concluded the case had hung.

Either alone was survivable. The CPU check had been correctly reporting `SLOW not hung` for
six minutes on exactly this basis. It only failed when step 12 handed all the work to a
process the sentry was blind to.

**This is the failure the sentry's own docstring warns about**, in these words: *"a
log-silence watchdog sized against one stage killed a different stage that was working
perfectly"*. I wrote that paragraph about `score_all_slides`' watchdog, then shipped the
same defect. Writing the lesson down is not the same as applying it.

### The fix, and how it was checked

- Liveness is now the newest mtime across **every** log the run writes, including the
  per-case scoring logs — not `run.log` alone.
- The process match is now **any python process whose command line contains the repository
  path**, excluding only the sentry itself. Naming the scripts you expect is what created
  the blind spot; the repository path cannot develop one.
- Silence limit raised from 45 to 90 minutes now that the signal is trustworthy.

Verified before restarting it, rather than assumed: two samples twenty seconds apart showed
**+156 CPU-seconds across 10 processes** where the old version had reported zero, and the
verdict was `alive`.

### What it cost, and the recovery

CAN_00251's A, F and R are scored and on disk. U and W need a retry, and the checkpoint
marks the case `done`, so the pass will not revisit it on its own. `run_all.py` also holds
the checkpoint in memory from before the round loop, so editing the file mid-pass does not
survive. The retry therefore has to run after the tiles pass finishes:

```
tissue_scoring_demo\backend\.venv\Scripts\python.exe score_all_slides\run_all.py ^
    --roi-source tiles --redo CAN_00251
```

Steps 2–9 are cached for that H&E, so it should only cost U and W.

---

## 14. B-spline folds tissue. `mattes` is the method. (2026-09-24)

Prompted by one question from the user: *does B-spline produce biologically sensible
alignment?* I had chosen it by ranking on mutual information across 30 pairs and had not
checked. The answer is no, and the check that shows it takes seconds per pair.

### What mutual information cannot see

A free-form B-spline can fold tissue through itself, tear it, or stretch a region
sixfold — and **each of those can raise MI** by lining up intensity statistics. The
transform then scores best and is anatomically impossible. Here that is not academic: the
transform carries an invasive-tumour boundary and the reported percentages are measured
*inside* it, so an impossible warp puts the mask on cells that correspond to nothing, and
every number downstream still looks healthy.

The test is the **Jacobian determinant** of the transform, sampled over the H&E's tissue. A
negative determinant means the map has folded — topologically impossible, not merely
improbable. Values far from 1 mean implausible local stretch or compression: a section
4–5 µm from its neighbour, floated onto glass, changes area by a few per cent.

### Measured over 20 pairs

| method | pairs | folded | Jacobian range |
|---|---|---|---|
| **bspline** | 11 | **5 (45%)** | −0.48 → 6.13 |
| **mattes / affine** | 9 | **0** | 0.9395 → 1.0077, uniform |

The worst are not marginal:

```
CAN_00267 F  bspline  jac -0.477 → 4.67   FOLDED at 421 points (10.59%)
CAN_00267 R  bspline  jac -0.233 → 6.13   FOLDED at 232 points (5.84%)
CAN_00259 A  bspline  jac -0.008 → 1.81   FOLDED at  68 points (1.24%)
```

CAN_00259/A scored **the highest NMI of its whole case** (1.0688). It won the scoreboard by
folding.

**CAN_00267 is the clearest case, and explains the mechanism.** It is the case where
registration gains almost nothing (+0.0003 to +0.026 NMI) because there is little real
correspondence to find — and *every one* of its B-splines folds. With weak signal, the
free-form warp fits noise, and the more freedom it has the more elaborate the nonsense. The
NMI gain was real as a number and meaningless as an alignment.

### The decision: `mattes` for all 30 pairs

1. **B-spline is unsafe on this data.** Not a tuning problem — with an 8×8 control grid on
   weakly-corresponding images, the freedom *is* the failure.
2. **`mattes` cannot fail this way.** A similarity transform has one Jacobian for the whole
   section: positive once means positive everywhere. The failure mode is unreachable by
   construction rather than merely unobserved.
3. **It costs 0.0065 mean NMI** against B-spline — and part of that lead was bought by
   deformation the tissue cannot have undergone, so the true gap is smaller.
4. **It beats the alternatives.** As a single method for all 30 pairs, mean loss against
   per-pair best: `mattes` 0.0117, `affine` 0.0185, outline alone 0.0436.

### What this reverses

§12 of this document concluded that `bspline` "leads decisively" and that "non-rigid
deformation is real and material… a rigid fit does leave accuracy on the table". That was
inferred from NMI alone. The NMI numbers were right; the inference from them was wrong, and
no amount of additional NMI would have revealed it. Ranking on a metric the failure mode is
invisible to is not evidence, however many pairs it is computed over.

---

## 15. Three checks that could not fail (2026-09-24)

Worth collecting in one place, because the same mistake happened three times in a day and
the pattern is more useful than any of the individual bugs.

| the check | what it was supposed to catch | why it could not |
|---|---|---|
| "which method wins a pair" | VALIS losing to the new methods | the tally read `methods.json`, which does not contain VALIS. It scored zero **by construction** |
| "did this case store transforms" | a case where every pair threw | it reported the worker's `ok` — which means *it ran*, not *it produced anything*. Printed "0 cases failed" while all 5 pairs had thrown |
| "is this cached report from the current method" | stale VALIS reports being served after the switch | it read `registration_kind` from JSON that stores `registrationKind`, and from a model that dropped the field entirely. Discarded **every** report, including the ones it was written to accept |

Each was tested before shipping. Each passed. The tests were written against what I assumed
the artefact looked like - a hand-made dict, a return value read from the code rather than
from disk - instead of against the artefact the pipeline actually produces.

**The rule that would have caught all three: assert against the file on disk, after a full
round trip, not against the object you just constructed.** The third one is the clearest -
`AlignmentDiagnostics` silently drops undeclared fields, so a diagnostic added at the call
site is absent from the stored JSON and no error is raised anywhere. The fix was to declare
the fields in the schema *and* read both spellings, and the test is now an end-to-end
`start -> execute -> report` that reads the file back.

### The one that actually cost results

The stale-cache failure is the one to remember, because it was silent in the dangerous
direction. `ihc_alignment_service.start()` returns a stored report if one exists, so after
step 12 was switched from feature matching to stored intensity transforms, twelve reports
from the old method were served straight back. **Four of them said `ready`** - so four pairs
were scored against a VALIS transform while the run believed it was using the approved
`mattes` one, and nothing in the CSV said which method any row came from.

A failure that produces an error is cheap. A failure that produces a plausible number is
what this whole document has been about.

---

## 16. BEETLE case order, from what previous runs measured

Requested: run the cases cheapest first. Taken from observed runs rather than estimated.

**Step 11 dominates a BEETLE case, and it is cached per case.** BEETLE runs once on the
shared H&E and every later marker reuses the result - CAN_00865's first pass through step 11
took **164 minutes** (15:00:15 to 17:44:35 on 18 Sep) and its retry took **three seconds**.
So ordering the cases is really ordering their step-11 cost.

| case | measured | what was measured | tiles |
|---|---|---|---|
| CAN_00270 | **62 min** | whole case, 5/5 scored, 18 Sep | 1081 |
| CAN_00251 | **77 min** | whole case, 5/5 scored, 17 Sep | 306 |
| CAN_00267 | 274 min | crashed - on *alignment*, since fixed | **74** |
| CAN_00303 | — | never completed under BEETLE | 640 |
| CAN_00259 | ~142 min | step 11 only (58 min for ROI-1, plus a larger ROI-2) | 3969 |
| CAN_00865 | **164 min** | step 11 only | 5201 |

Order: `CAN_00270 → CAN_00251 → CAN_00267 → CAN_00303 → CAN_00259 → CAN_00865`

**Two measured times invert against tile count**, which is why the proxy alone was not
used: CAN_00270 has three times CAN_00251's tiles and finished fifteen minutes sooner.
Step 11 scales with region *area*, but nuclei segmentation, cell typing and per-cell
measurement scale with *cell count*, and those are most of a case once step 11 is cached.
So a measured whole-case time is trusted over the proxy, and the proxy only orders the
cases that have never finished.

CAN_00267 sits third despite its 274-minute crash because that was an alignment failure -
now fixed - and its step 11 is the smallest in the cohort.

An unseen case sorts **last**, not first: nothing is known about its cost, and the point of
the ordering is to bank cheap certain work before spending hours on expensive uncertain
work. `score_all_slides/run_all.py::CASE_ORDER`.

---

## 17. The gate was measuring tissue with the rule its own documentation calls broken

CAN_00865 lost all five markers in 0.3 minutes. Not to registration - `mattes` fitted
cleanly with no folding - but to the gate's **tissue-area ratio** check, which read 0.17
for CD44 and 0.31 for ABCC4 against a permitted band of 0.5 to 2.

The measurement was wrong, and three independent sources say so:

| rule | H&E | F | ratio |
|---|---|---|---|
| the gate's (step 3, **colour saturation**) | **357.2 mm²** | 112.1 | **0.314** |
| optical density, from the registration renders | 96.5 | 89.2 | **0.925** |
| `cohort_characterisation.md` section 3, published | 74.3 | 91.1 | **1.23** |

The saturation rule put CAN_00865's H&E at **357 mm² where the published figure is 74.3** -
five times too much - because it measures how *stained* a slide is rather than whether
tissue is present. A strongly-stained H&E scores generously, a paler IHC section does not,
and the ratio between them collapses.

**This is documented, in this repository, since August.** `cohort_characterisation.md`
section 2 is titled "A measurement defect found and fixed before it did damage" and records
the saturation rule capturing **4% of the tissue on CAN_00865/A** and 30% on CAN_00251/A.
`render.py` was written to use optical density *because of that document*, and it quotes it
in its own docstring. The gate was simply never looked at.

### The fix

`_tissue_areas` now takes its figures from the registration render, where both sections are
measured the same way at the same resolution by the rule that was validated against the
published table. The threshold is untouched at 0.5 to 2. Step 3's numbers are still
recorded beside them as `he_tissue_saturation_mm2` so the two can be compared and this
change argued with, and a case without a render falls back to the old rule rather than
losing the check entirely.

Under the corrected measurement CAN_00865 reads:

```
A  14.3 mm²  ratio 0.15  refuse   <- genuinely near-blank; refusing is right
F  89.2 mm²  ratio 0.93  pass
R  87.0 mm²  ratio 0.90  pass
U  94.4 mm²  ratio 0.98  pass
W  87.5 mm²  ratio 0.91  pass
```

Four markers recovered; the one honest refusal kept.

### Why this keeps happening

This is the same shape as section 15's three checks that could not fail: **a check whose
own measurement is not trustworthy is not a check**, and it fails in the expensive
direction - it produced a confident, specific, quantified refusal that was simply wrong.
The pattern across all of them is that I verified the *logic* of each check and never the
*input* it was reading.

### Correcting the measurement was not enough: a refusal outlives the rule that made it

The measurement fix above had no effect on the first re-run. CAN_00865 failed again in
0.2 minutes with the *same* 0.17 and 0.31 ratios, while the corrected code, called directly,
returned 0.925.

The refusals were cached. `start()` serves a stored report if one exists, and the guard
added earlier only checks which **registration method** produced it - both refusals were
`intensity`, so both were served straight back. Changing a threshold or a measurement had
no effect on any pair already refused, and nothing said so.

Deleting the two files would have worked once. Instead the gate now carries a version:

```
GATE_VERSION = 3     1: feature-based gate
                     2: intensity gate - NMI, folding, tissue ratio, round trip
                     3: tissue ratio measured by optical density, not saturation
```

A stored **refusal** made by an older version is re-decided. A stored **success** is kept,
because a gate change does not alter the transform it carries - only whether a refusal was
justified. Verified: old refusal re-decided, current refusal reused, old success reused.

The general form, which is worth more than this instance: **a cached negative result is
only valid while the rule that produced it stands.** A cache keyed on the input alone will
happily outlive three separate corrections to the logic, and every one of those corrections
will appear to have failed.

### The ratio needed one cut per case, not one rule

Correcting the *rule* was not enough either. `render.py` chooses an Otsu threshold **per
slide**, and Otsu reliably picks a lower cut for an H&E than for a pale IHC section:

```
CAN_00270  H&E  cut 0.190 -> 192.4 mm2   (published 189.8)
CAN_00270  A    cut 0.251 ->  84.1 mm2   (published 101.2)
CAN_00251  H&E  cut 0.166 -> 202.6 mm2   (published 166.3)
CAN_00251  A    cut 0.251 ->  95.6 mm2   (published  95.0)
```

The H&E is measured generously, the pale section conservatively, and the ratio between them
is systematically depressed. So the density fix, which recovered four markers on CAN_00865,
would have **refused CD44 on CAN_00251 and CAN_00270** - two markers that had scored fine.
Net +2 rather than the clean win claimed at the time.

That is the same flaw as the saturation rule, one level down: **comparing two sections
measured on different terms.** Changing the rule did not fix the comparison.

`gate_areas.py` measures every section of a block at the threshold the **H&E** chose - the
reference slide every ROI is drawn on. Whatever bias that cut carries now falls on both
sides of the ratio equally, which is the only property a ratio needs.

Validated against the cohort's published figures on every pair:

```
30 of 30 pairs fall on the same side of the 0.5 gate as the published measurement
0 disagreements
```

The only refusals are CAN_00865/A (0.19 against a published 0.24) and CAN_00267/A (0.36
against 0.24) - the two near-blank CD44 sections, which refuse under the published figures
too.

`render.py` keeps its per-slide Otsu for **placement**, where a tight mask per slide is
exactly right and no comparison is being made. The shared cut is only for the gate.

`GATE_VERSION` is now 4, so every refusal made under versions 1-3 is re-decided rather than
served from cache.

## 18. A refusal that was stale code, and two silent Windows traps (2026-09-24)

`GATE_VERSION` 4 was in place and `gate_areas.json` was on disk, and CAN_00270's marker A
was still refused: *"tissue-area ratio 0.33 is outside 0.5-2"*. Under the shared cut that
ratio is **138.58 / 192.39 = 0.72**, comfortably inside the gate. The refusal was wrong.

**It was not a stale cache — it was a stale process.** `run_all.py` spawns one
`run_case.py` per case, and that process imports the backend modules once and holds them
for every marker of the case. The timeline settles it:

| time | event |
|---|---|
| 15:55:24 | CAN_00270's `run_case.py` starts and imports its modules |
| 15:58:11 | the schema gains `gate_version` and `tissue_area_source` |
| 17:24–17:25 | `gate_areas.py` written; `CAN_00270/gate_areas.json` created |
| 17:27:01 | the service learns to read the shared-cut areas |
| **16:53:36** | **marker A refused** |

A was decided by code that predates the file it was supposed to read. Editing the backend
during a pass does not reach the case already running, and nothing says so: there is no
`--reload` here and no version in the log.

**Two tells distinguish this from a real refusal**, and both are worth keeping, because in
the CSV a stale-code refusal is indistinguishable from a biological finding:

1. **It arrives in ~7 seconds.** The cheap tissue-ratio check runs before registration, and
   a genuine registration takes about five minutes - marker U ran 17:54:57 to 17:59:45.
2. **The stored report carries fewer diagnostics than the current model emits** - 21 keys
   against 25, with `gateVersion` absent entirely rather than `null`. Comparing
   `AlignmentDiagnostics(...).model_dump(by_alias=True)` against the file on disk shows it
   immediately.

No repair was needed. `run_all.py` retries a `partial` case while `attempts < 3`, and a
refusal carrying no `gateVersion` is re-decided rather than reused, so the follow-on pass
recomputes A against the current gate on its own. The markers that *scored* in that process
are unaffected: the shared-cut areas feed only the ratio check, not any measurement.

### The follow-on, and two Windows traps found building it

`night.py` passes each stage its deadline as an argument when it launches it, so raising
`budgets.json` mid-stage does nothing for the stage already running. BEETLE had been given
4.5 hours and measurement during the run put it nearer **3.4 hours per case** - about 30
minutes a marker, because step 13 allocates a minimum field per region and BEETLE's tighter
regions end up sampled almost in full. Restarting the stage would have discarded the case in
flight, so `finish_beetle.cmd` runs *after* it instead, resumes from the same checkpoint and
gives the remaining cases the rest of the night.

It waits first, through `wait_for_scoring.py`: `run_all.py` takes no lock, and two passes
over one mode would interleave their writes to a single checkpoint - each reloads it, edits
its own case's slot and writes the whole file back, so the last writer silently reverts the
other's completed cases to `pending`.

Building that guard turned up two failures that are invisible rather than loud:

*   **A detached, console-less process that spawns `powershell.exe` hangs**, and
    `subprocess.run(timeout=...)` does not break it out - killing the child does not close
    the pipe the grandchild still holds. Five minutes of total silence with a stuck child.
    `sentry.py`'s CIM probe is fine because it runs with a console; this one reads the
    process table in-process with `psutil` instead.
*   **`DETACHED_PROCESS` loses the child python's stdout even through the `.cmd`'s own
    `>> log 2>&1`.** `echo` from the same script lands normally, so the log looks alive
    while every python line is dropped. A four-way probe: `DETACHED_PROCESS` loses output;
    `CREATE_NO_WINDOW`, `CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP` and
    `CREATE_NEW_CONSOLE` all keep it. The launcher now uses `CREATE_NO_WINDOW |
    CREATE_NEW_PROCESS_GROUP`.

Both were caught only by checking that a known line actually reached the log, which is the
same discipline §15 is about: verify the state the check reads, not just its logic.

## 19. The two CD44 pairs are allowed past the gate, with the failure attached (2026-09-24)

The tiles CSV holds 28 rows, not 30. The two missing pairs are exactly the two near-blank
CD44 sections, and both were **refused outright** rather than scored:

| pair | refused on | measured |
|---|---|---|
| CAN_00267 / A | alignment NMI below the 1.008 floor | **1.0060** |
| CAN_00865 / A | tissue-area ratio outside 0.5–2 | **0.15** |

The instruction for these two was that registration should proceed and their near-blank
staining should remain a caveat on the resulting scores. A refusal does not do that - it
drops the marker from the results entirely, which is a different outcome from a flagged one.

**Implemented as a named per-pair exemption, not a looser threshold.** `GATE_OVERRIDES` in
`ihc_alignment_service` lists the two pairs; `_check_gate` evaluates the gate normally, and
only for a listed pair converts the refusal into a recorded override. Lowering
`MIN_ALIGNMENT_NMI` or widening the ratio bounds would have bought these two at the cost of
every other pair's protection, and the gate exists precisely because a mask carried through
a bad transform lands on unrelated cells while every downstream number still looks healthy.
A named pair is auditable; a loosened threshold is invisible.

The reasons are never discarded. They are logged as a warning, stored in the report's
diagnostics as `gate_overridden`, and become a caveat on every score the pair produces:

> ALIGNMENT FAILED ITS OWN CHECK AND WAS ALLOWED THROUGH. […] Treat every figure below as
> indicative only: if the registration is wrong, these cells are not the cells the regions
> were drawn around, and no measurement here can detect that.

**Verified against real upload ids, not an assumed pair.** Given identical failing
diagnostics, CAN_00267/A and CAN_00865/A proceed with both reasons recorded, while
CAN_00251/A - a control that is not on the list - is still refused. That is the test that
matters: it confirms the exemption is scoped rather than a gate that no longer fails.

`GATE_VERSION` is now **5**, because the override changes the outcome for these pairs and so
has to invalidate the refusals version 4 cached for them.

The tiles top-up for the two pairs runs from `finish_beetle.cmd`, sequenced **after** the
BEETLE pass rather than beside it - separate modes mean no checkpoint conflict, but both are
heavy and would only slow each other down.

### A note on editing a running `.cmd`

`cmd.exe` reads a batch file incrementally by byte offset, so editing one while it runs can
make it resume mid-line. The chain was restarted after this change rather than edited in
place. It was only waiting, so the restart cost nothing - but the same edit made during the
scoring run would not have been free.

## 20. Neither region source is right for every case (2026-09-24)

The two-pass exercise was set up to answer "does BEETLE's refinement change the scores, and
can the tile grid substitute for it". With three cases scored both ways the answer is not
the one either pass was designed to give: **the right region source depends on the case, and
which one is right is knowable before scoring.**

### Where there is ample invasive tissue, BEETLE refines a real measurement

CAN_00270 keeps 17.44 mm2 of 36.17 mm2 of tile squares (48%). Nine pairs scored both ways
move by -5, 0, 0, 0, +5, +5, +15, +15 and +30 points - six of nine by at least five points,
a 35-point spread, and a sign change. There is no constant that converts one to the other,
so the tile grid cannot stand in for the refinement by calibration.

### Where there is little, BEETLE removes what was left to measure

CAN_00267 keeps 0.01 mm2 of 1.96 mm2 (1%), and the denominators collapse:

| marker | tiles | BEETLE |
|---|---|---|
| A | refused | 0 % on **1 cell** |
| F | 35 % on 58 cells | 100 % on **5 cells** |
| R | 5 % on 458 cells | 0 % on **32 cells** |
| U | 95 % on 643 cells | 100 % on **7 cells** |

R's 5% -> 0% and U's 95% -> 100% are not refinements. A seven-cell denominator can only
return multiples of 14 points; a one-cell denominator can only return 0 or 100. On this case
the **tiles** run holds the only usable denominator.

### The deciding variable is measured before any scoring happens

`beetle_coverage.py` already reports the share of tile area BEETLE keeps, and it ranges from
0.73% to 48.21% across the cases measured - a 66x spread, with the "nearly empty" tile
fraction tracking it. That figure, not the marker and not the stain, predicts whether
refinement sharpens a measurement or erases it.

### What this means for the scores on disk

A percentage counted over a handful of cells is the most misleading thing this pipeline can
print, because `0 %` and `Negative` read exactly like a measured absence of staining. Two
caveats now fire from the score itself:

* **NOT A MEASUREMENT**, under 50 cells - states how many points one cell is worth and that
  the row means "no usable tissue", never a negative or positive result.
* **THIN DENOMINATOR**, under 400 cells - the same threshold `compare_scores.py` uses for a
  figure too imprecise to quote to the nearest percent.

Rows produced before those caveats existed do not carry them, because `run_case.py` imports
the backend once per case (§18). CAN_00267 is forced to re-run from `finish_beetle.cmd` for
exactly that reason.

## 21. CAN_00259's refinement does not terminate, and no watchdog could see it (2026-09-28)

Step 11's refinement of **CAN_00259 ROI-002 does not complete**, and the reason it went
unnoticed for two runs is worth more than the fault itself.

| region | area | windows | time | kept |
|---|---|---|---|---|
| ROI-001 | 199.1 mm2 | 3,969 | **3,327 s** (55 min) | 0.000 mm2 |
| ROI-002 | 156.4 mm2 | ~3,120 expected | **114 min, no output** | - |

ROI-002 is the *smaller* region, so at ROI-001's measured 0.84 s/window it should finish in
about 45 minutes. It had already been given 41 minutes on 24 September before the machine
was switched off, which hid the problem as an interrupted run rather than a stuck one.

### The signature

The first ~55 minutes burn **~4 cores**, matching ROI-001's windowed inference. CPU then
drops to **~1 core** and stays there, with **zero disk IO**, 29 threads, and RSS cycling
**18-24.5 GB** - rising 2.6 GB inside a 20-second sample - on a 33.7 GB machine, taking free
memory down to 7.6 GB. The windowed phase finishes; something single-threaded and very
memory-hungry runs after it and never returns.

ROI-001's equivalent phase was trivial, which fits: its mask is empty, so whatever component
analysis follows had nothing to work on. That makes the post-window stage the place to look,
though the root cause is **not yet diagnosed**.

### Why every watchdog in this project is blind to it

`sentry.py`, `finish_watch.py` and `run_all.py`'s own silence limit all treat **CPU time as
liveness**, deliberately and for good reason - a log-silence watchdog once killed a healthy
case that was using four cores and printing nothing for 47 minutes (section 13). A stall
that *burns* CPU satisfies every one of those checks indefinitely.

It was found by measuring **RSS and disk IO**, not CPU: 24 GB of resident memory and zero
bytes written is not a process making progress. That is the check worth adding, and the
general lesson is that "is it alive" and "is it progressing" are different questions.

### What was done, and why killing it was the right call

The `run_case.py` worker was killed. `run_all.py` recorded CAN_00259 as crashed and moved
straight to **CAN_00865**, retrying CAN_00259 on a later round - so the kill **reorders the
queue rather than losing work**, and the case with real tissue is no longer queued behind one
that consumes the machine.

That trade was one-sided. CAN_00865 keeps 35.82 mm2 of invasive tissue and is the last case
with a usable denominator; CAN_00259's dominant region keeps **0.000 mm2**, so its markers
were heading for CAN_00267's unmeasurable rows whatever happened. Leaving it running risked
page-file thrash at 7.6 GB free, which would have starved CAN_00865 of the hours it needed.

### 21a. The fault is reproducible, and the watcher now catches it (2026-09-28, closing)

The retry settled two questions this section left open.

**It is not memory pressure.** The first occurrence ran with free RAM already down at 7.6 GB,
which made a pressure spiral a reasonable alternative explanation. The retry began with
**24 GB free** and failed identically - healthy 4-core inference for ~82 minutes, then a
collapse to one core with memory climbing 6.9 -> 9.9 -> 12.8 -> 15.1 -> 17.0 -> 17.3 GB and
**not one byte written**. So something in the post-window stage for ROI-002 genuinely does
not terminate. The root cause is still **not diagnosed**.

**The transition is later than first measured**, at 82-84 minutes rather than the 55-60 the
first failure suggested. Anything keyed to the earlier figure would fire too soon.

**`finish_watch.py` killed it unaided** at 96 minutes, against the 114 that had to be caught
by hand - and `run_all.py` then declined to spend the third attempt, reporting *"round 2
scored nothing new; the cases still outstanding fail the same way each time"*. That guard is
what turned a reproducible defect into a bounded cost rather than an open-ended one.

### Two flaws in my own detection, both found by watching it run

1.  **CPU cannot see this fault.** Every watchdog here treats CPU as liveness, deliberately,
    after a log-silence watchdog once killed a healthy case (section 13). This fault burns a
    core throughout. It was found by measuring **RSS and disk IO** - 24 GB resident and zero
    bytes written is not progress - and that is the check that was added.
2.  **A single memory threshold defeats itself on an oscillating fault.** The limit was first
    set at 10 GB, six times the healthy 1.7 GB. But the fault's memory *cycles*: a sample at
    9.9 GB read as healthy and **reset the strike counter**, so it alternated stalled/working
    and would never have reached three strikes. Two changes fixed it - the limit dropped to
    **4 GB**, which clears the healthy peak of 2.4 GB and sits below the fault's troughs, and
    the counter now **decays rather than resets**, so an oscillating fault still accumulates.
    The decay is the more general fix: any fault that flickers across a threshold would have
    defeated the reset.

Both are the same lesson as section 15, in a new place: a check is only worth what the state
it reads is worth, and the way to find out is to watch it meet the thing it was built for.

### Final state

| | rows | cases |
|---|---|---|
| `oncostem_ai_scores_tiles.csv` | **30** | 6/6 |
| `oncostem_ai_scores.csv` (BEETLE) | **25** | 5/6 - CAN_00259 absent |

25 pairs scored both ways. Both near-blank CD44 pairs are present in **both** files, carrying
their gate-override caveats, where before they appeared in neither.

## 22. CAN_00259 diagnosed and fixed: `clean()` was O(components x mask) (2026-09-28)

**The fault is in `step11_roi_refinement/contours.py`, not in the inference.** `clean()`
allocated one boolean array **the size of the entire mask** for every surviving component,
and accumulated them all:

```python
kept.append(_fill_small_holes(labelled == position + 1, min_hole_px))
```

ROI-002's padded canvas is **18,142 x 20,606 = 374 MP**, so each entry is **374 MB**. Its
refined mask has **690 components** (`focusCount`), because `roi_refinement_min_component_mm2`
of 0.005 is only 5,000 px at 1 um/px and a slide with scattered invasive foci produces
hundreds above that.

    690 components x 374 MB = about 258 GB

The time went the same way: `_fill_small_holes` runs `binary_fill_holes`, a second `label`
and an `np.isin` over the **full canvas** for every component regardless of its size - four
more passes over 374 MP each, single-threaded, which is the one-core signature.

### Why the larger region completed and the smaller one did not

ROI-001 is **199 mm2** against ROI-002's **156** and finished in 55 minutes, which is what
made this look inexplicable. Its refined mask is **empty** - `keptShare 0.0` - so
`ndimage.label` returns `count == 0`, `clean()` returns `[]` immediately and the loop never
runs. An O(components x mask) cost is invisible when there are no components. The inversion
was the clue, not the mystery.

### The fix

Crop each component to its bounding box with `ndimage.find_objects` and carry the offset,
re-applying it in `regions()` before the mask-to-slide conversion. A 0.005 mm2 speck now
costs its own bounding box instead of the whole canvas.

The offset is bound as **default arguments, not captured in a closure**. A closure over the
loop variable would hand every component the last box's offset, and that failure is silent:
rings of the right shape on the wrong tissue, which is the exact failure this module's own
header warns about.

| | before | after |
|---|---|---|
| held in `kept` at ROI-002's scale | 17.2 GB and climbing | **5.1 MB** |
| `clean()` | never returned | **2.3 s** |
| `regions()` end to end | never returned | **10.4 s** |
| ROI-002 on the real slide | killed at 96 and 114 min | **completed, 4,215 s** |

Verified for equivalence as well as speed: same component count, pixel-identical after
re-placing the crops, same traced geometry and pixel totals, holes preserved.

### A correction to section 21's arithmetic

Section 21 inferred ~110 components from the observed 17.3 GB and called it a clean match at
156 MB each. **Both figures were wrong and the errors cancelled.** The canvas is 374 MP, not
the 156 MP the region's own area implies, and there are 690 components, not 110. The real
requirement is about 258 GB, so 17.3 GB was roughly component 46 of 690 - **7% of the way
in**, not near a ceiling. It would not have finished on any machine; it was bounded only by
how long it was allowed to run.

The mechanism in section 21 was right. The magnitude was understated by a factor of fifteen,
because a number was inferred from the symptom instead of being read from the metadata the
completed run now provides.
