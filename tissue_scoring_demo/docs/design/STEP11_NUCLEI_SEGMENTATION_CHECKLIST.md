# Steps 11, 12 and 13 — unattended overnight build

Written as the work happened, 2026-09-15 01:05 → 06:10. **All three steps are
built, tested, and running on your real `CAN_00270` CD44 data.**

Legend: `[x]` done · `[!]` needs a decision from you

---

## Read this first: the one finding that matters

**The code is right. Your IHC slides are a hard case, and the pipeline now says
so out loud instead of quietly under-counting.**

Same code, same regions, same model — only the slide changes:

| | nuclei / mm² | median nuclear area |
| --- | --- | --- |
| **H&E** (`CAN_00270_26_H&E`) | **1,979** | **43 µm²** |
| **CD44 IHC** (`_A`, same block) | **549** | 15 µm² |

Those are consecutive slices of one block, so they hold essentially the same
cells. A **72% shortfall** is not biology — it is nuclei being missed, because
under dense DAB the counterstain is too faint for nuclear boundaries to survive
deconvolution. 43 µm² is the right size for a breast epithelial nucleus; 15 is
debris-sized.

This is exactly the failure the guide predicts for heavy staining, and exactly
the check it prescribes for catching it. It fired on the first run. **Every
nucleus missed is a cell removed from the denominator, which inflates the
positive percentage** — so this number now travels with the score:
`heDensityPerMm2`, `densityShortfall`, a warning note, and a coloured panel on
the screen. Step 12 independently refuses to trust its own cell mix for the same
reason, from the other direction.

**It is measured on every run**, not when a second marker appears — otherwise the
check only starts working after a failure has already gone unnoticed once.

---

## What you can look at

Run the demo, pick **A / CD44**, point at
`data/original/oncostem_slides/CAN_00270`, and walk to steps 11–13. Steps 2–10
are cached. Or headless:

```
cd tissue_scoring_demo/backend
.venv\Scripts\python scripts\run_case_nuclei.py <he-upload-id> <ihc-upload-id>
```

⚠ **Step 11 will refuse until you confirm step 10's alignment in the UI.** That
is deliberate and I did not work around it: whether two sections line up is a
person's judgement. My end-to-end runs bypassed the gate inside a test script
only — **the stored alignment is still `confirmed: false`**.

---

## Step 11 — Nuclei segmentation ✅

### The model, proved before anything was built on it

- [x] **InstanSeg**, Apache-2.0 — **and so is its training data**: tnbc_2018,
      lynsec, nuinsseg and ihc_tma are CC BY 4.0, consep is Apache-2.0. **Two of
      the five are IHC**, which is the real evidence behind "it handles IHC".
- [x] **`instanseg-torch` cannot be installed here and is not needed.** PyPI
      declares 3.9–3.11 and ships **no 3.13 wheel**; this backend is 3.13.3 on
      numpy 2.5.2 — the same collision that forced VALIS into its own venv.
      Avoided entirely: the published model is TorchScript with post-processing
      compiled in, so `torch.jit.load` runs it on the torch 2.8.0+cpu already
      installed for step 2. **No new dependency, no second venv.**
- [x] **Bit-for-bit parity with the upstream's own test pair** under this
      interpreter — 336 instances, `np.array_equal` true. The checkpoint also
      hashes to the sha256 its own `rdf.yaml` declares.
- [x] **A silent-zero failure found and pinned.** Fed the raw 0–255 test tensor,
      the model returns **zero** nuclei — no error. The `rdf.yaml` percentile
      stretch is what makes it work. For a step whose job is a denominator,
      "this tissue has no cells" is the worst way to fail, so the loader now
      *runs* the parity pair and refuses to serve anything that does not
      reproduce it.
- [x] Cost measured (4 threads, 0.5 µm/px): 512 px is the knee at **60 s/mm²**.

### Three decisions that reverse the guide, each on a measurement

**1. Per-slide Macenko is *worse* here.**

| basis | nuclei found |
| --- | --- |
| **Ruifrok, fixed** | **695** |
| Macenko, per slide | 581 |
| hybrid (DAB from Macenko, H from Ruifrok) | 574 |

Macenko recovers **DAB** well — 3.3° from the published direction, because DAB
dominates — and returns an **achromatic** second arm, (0.577, 0.577, 0.577),
18.7° off haematoxylin. Under heavy DAB with a weak counterstain there is no
second *colour* in the cloud, so the method finds the darkness axis. **The
condition that is supposed to justify estimating is the condition that breaks the
estimate of the stain we need.** Default is now the fixed basis; the estimate is
still computed and reported, because its drift is the evidence.

**2. The rendering gain sits on a plateau, not a peak.** Nuclei found at gain 1.0
→ 325, 1.5 → 523, 2.0 → 640, **2.5 → 695**, 3.0 → 699, 4.0 → 692. From 2.5 the
answer stops depending on the setting. That is what makes 2.5 defensible rather
than tuned.

**3. 0.5 µm/px, not 40×.** The guide says zoom to 40×; the *viewer* does. The
model's metadata declares 0.5 µm/px and 0.25 would be out of distribution.

### A bug found and fixed on the way

The first H-channel reconstruction produced **6 nuclei where raw RGB found 54** —
a grey, texture-inverted image with no blue in it. Cause: it recomposed along the
*estimated* stain arm, which on this slide is achromatic. Fixed by separating the
two jobs — the **amount** comes from the chosen basis, the **direction** it is
drawn in is always Ruifrok's published one. Same field afterwards: **74**.

### Built

- [x] Weights vendored to `models/nuclei/` + committed manifest + 4
      `models.lock.json` entries. **A new `zip` source kind** added to `setup.py`,
      since InstanSeg publishes one archive rather than separate assets.
      `python setup.py --check` verifies all four.
- [x] `app/nuclei/` (loader, sha256 pin, parity gate, capability),
      `app/pipeline/step13_nuclei_segmentation/` (sampling, stain_input, segment,
      instances, watershed, overlay), service, schemas, endpoints, router, config
- [x] **The step samples rather than exhausts**, as you asked. 12 fields of
      256 µm per region ≈ 90 s, against ~33 min for the regions whole. Fields are
      taken at a **uniform stride**, never ranked by stain — ranking would pick
      the densest fields and destroy the cross-slide density check.
- [x] Border band: a nucleus whose centre falls within 24 px of a field edge is
      drawn but not counted, **and its area is removed from the denominator too**
- [x] **UI: the three regions separately, each with zoom.** Region tabs, per-field
      chips, and a viewer that zooms to 8× (≈40× equivalent) with a **live count
      of the nuclei currently in view**, in three views: the tissue / what the
      model saw / what it found
- [x] Both comparison panels the guide asks for, on the page with their counts:
      **naive watershed vs the model**, and **brown removed vs brown left in**
      (region 2 field 11: **50** against **16** — the argument, measured)
- [x] 22 tests

**Real run:** 91 s · 1,258 detected · **1,064 counted** over 1.937 mm² · 549/mm²

---

## Step 12 — Cell typing ✅

- [x] Sorts step 11's nuclei into **tumour / immune / support** on size,
      roundness, elongation and stain darkness — every threshold in microns
- [x] **A rule, not a fitted model, and it says so everywhere.** Nobody has
      labelled any nuclei here, and a model fitted on invented labels would carry
      the same judgements with an accuracy figure that implied they were
      measured. A reader cannot argue with 0.91; they can argue with "an immune
      cell nucleus is under 35 µm²".
- [x] **The thresholds are live on screen as sliders** and the mix moves with
      them — the only honest way to present parameters nobody has fitted
- [x] **The sensitivity sweep is part of the output**, and it immediately earned
      its place: `spindle_min_eccentricity` swings the tumour share by **50.7
      points**. That threshold, not the data, is deciding the answer — which is
      itself a symptom of the under-segmentation above.
- [x] **It refuses to trust itself** when the median nuclear area falls below
      25 µm². On CD44 it comes out at 15.75, so `trustworthy: false` with the
      reason — and the mix is still shown, because hiding the evidence would be
      worse.
- [x] Request-shaped, not job-shaped: no slide opened, no model run, answers in
      well under a second. 14 tests.

**Real run:** 1,064 cells · tumour **32.7%** · trustworthy **false**

---

## Step 13 — Build compartments ✅

- [x] **The first step that reads the antibody letter** — and the fork is read
      from `app/panel.py`, which already carried `compartment`,
      `compartment_width_um` and `second_measure` for all five markers. No second
      copy.
- [x] **The kind is not a request parameter.** You can ask for a different width;
      you cannot ask for a ring on a cadherin. The guide names that regression
      explicitly, and the cleanest way to prevent it is to make it unreachable.
- [x] **Voronoi-constrained expansion**: every cell grows only to the midline
      between it and its neighbours, so two cells can never claim the same pixel.
      Verified visually — the rings stop cleanly where cells meet.
- [x] **`contestedPx` is measured, not asserted**: it runs plain dilation as well
      and counts the actual overlap. *(My first version derived it from the
      constrained result alone, and those two sets are identical by construction,
      so it reported zero on every field including obvious collisions. The tests
      caught it.)*
- [x] **Non-tumour cells are removed *before* the expansion**, not after —
      otherwise the tumour cell's compartment comes out dented by a cell nobody
      is measuring
- [x] **The width sweep is part of the output**, as the guide demands. On CD44:
      2 µm → 38 µm², 4 µm → 97, 6 µm → 169, 8 µm → 252. A **6.5× swing** across
      defensible widths, with the crowding rising from 3% to 21% alongside it.
- [x] UI: the fork stated as a fact with the wrong alternative beside it, the
      compartment drawn, a live width slider, and the sweep. 13 tests.

**Real run:** 3.3 s · CD44 · membrane · 4.0 µm · 941 tumour cells · measured
region 91.8 µm² per cell

---

## State of the tree

- **631 backend tests pass.** One failure remains and it is the
  **pre-existing** `test_region_bbox_matches_the_outer_ring_extent`, documented
  as such in the step 10 checklist and untouched.
- Frontend **typecheck, lint and production build all clean**.
- Steps **1–13 implemented**; the stage catalogue and `stages.ts` agree.
- New: `data/demo/nuclei/`, `data/demo/cell_typing/`, `data/demo/compartments/`,
  all keyed `<he>__<ihc>` and all added to the maintenance inventory — **along
  with `ihc_alignment_dir`, which was missing from it** (steps 10+ are keyed on a
  *pair*, so a single-upload sweep never matched them).

### One thing worth knowing

Partway through, the maintenance tests **deleted the real `data/demo/nuclei/`**,
because they run against every directory in the inventory and the isolation
fixture did not yet know about the new one. That fixture has a guard test
written precisely to catch this, and it did — the artefacts were regenerable in
90 s and the fixture is now correct. Nothing else was lost.

---

## `[!]` Decisions waiting for you

1. **The 72% shortfall is the real result of the night.** Options, in the order I
   would try them:
   - raise `nuclei_haematoxylin_gain` further for the heavily-stained markers;
   - **segment nuclei on the H&E and warp them onto the IHC** — step 10's
     registrar pickle makes re-warping cheap, and the inverse warp already exists
     inside the VALIS worker, just unexposed;
   - accept the IHC count and carry the shortfall into the score as a stated
     uncertainty.

   **I have not chosen.** It changes what the denominator *means*, which is your
   call.

2. **Step 10 is still `confirmed: false`.** Confirm it in the UI before step 11
   will run there.

3. **`nuclei_tiles_per_region = 12`** gives a 75% field-to-field density spread in
   region 1 (22 mm²), which is thin. Raising it is linear in time.

4. **Step 12's 50-point sensitivity on elongation** should be revisited once the
   segmentation improves — on properly-sized nuclei it will almost certainly
   shrink, and if it does not, that threshold needs labelled cells rather than
   reasoning.

---

## Log

- 01:05 — Model proved before a line of step 11 was written.
- 02:10 — Step 11 backend complete, first end-to-end run green.
- 02:40 — H-channel reconstruction bug found and fixed (6 → 74 nuclei on a field).
- 03:05 — H&E control run. The 72% shortfall found and promoted to a
  first-class, always-measured output.
- 03:30 — Step 11 frontend complete; typecheck, lint, build clean.
- 04:15 — Step 12 built, and its sensitivity table immediately exposed a
  50-point dependence on one threshold.
- 05:20 — Step 13 built; `contestedPx` bug caught by its own tests and fixed.
- 06:10 — Full API smoke test of steps 11–13 green. 631 tests pass.
