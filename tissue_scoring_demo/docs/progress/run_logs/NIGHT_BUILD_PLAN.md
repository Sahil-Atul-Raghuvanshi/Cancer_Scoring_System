# Night build — steps 11–15 visual rework + previous cases / history

Started 2026-09-16 ~02:15. Runs unattended. Target marker for the end-to-end
check: **CD44 (letter `A`) on CAN_00270**.

Progress lives in `NIGHT_BUILD_CHECKPOINT.json`. Every phase writes its state
there before moving on, so a restart resumes rather than redoes.

---

## What was asked

**A. Steps 11–15 must be seen on the real slide, not on cropped field PNGs.**

| Step | Ask |
|------|-----|
| 11 Nuclei | The full slide as in step 1 — same pan/zoom/fullscreen — with nuclei drawn inside the invasive regions. All light blue. |
| 12 Cell typing | Same viewer, nuclei now coloured: **red** tumour, **yellow** lymphocyte, **green** other. |
| 13 Compartments | Same viewer. **Nucleus red, cytoplasm yellow, membrane green.** The current screen is unreadable. |
| 14 Per-cell | Click a cell → its numbers (intensity, membrane continuity, …). |
| 15 Binning | Click a cell → its bin; plus the totals. |

Everything colourful, labelled, correct, and interactive.

**B. Previous cases.**

- Home page gets a *Previous cases* entry.
- It lists cases (e.g. `data/original/oncostem_slides/CAN_00270`) with a small H&E thumbnail.
- Opening a case shows the five biomarkers and which are done.
- Finishing a marker registers it there.
- Its data is saved under `data/history/` — complete enough to re-render the whole
  walkthrough without re-running anything.
- **The same run must never sit in both `data/demo` and `data/history`.**
- A completed marker replays without running.
- A half-finished marker resumes.
- Each marker can be deleted and re-scored.

---

## The two design decisions everything else follows from

### 1. One copy, moved — never copied

`data/demo/` and `data/history/` are on one volume, so `os.replace` of a
directory is a rename: instant, atomic, and it cannot leave two copies behind.

- **Archive** = rename the run's directories out of `data/demo/**` into `data/history/<CASE>/…`.
- **Open** = rename them back.
- The manifest records `location: "history" | "demo"` for every piece, so the list
  screen can tell "archived" from "open right now" without stat-ing 14 trees.

A copy-based archive would double 555 MB per marker, which is exactly what the
brief forbids.

### 2. Shared H&E work is archived at *case* level, marker work at *marker* level

Measured on the CD44 run already on disk:

| keyed on | dirs | size |
|---|---|---|
| H&E upload id (shared by all five markers) | `tissue`, `calibration`, `tiling`, `tissue_type`, `roi`, `slides/<he>.json` | ~72 MB |
| the H&E+IHC pair (one marker only) | `ihc_alignment`, `nuclei`, `cell_typing`, `compartments`, `per_cell`, `binning`, `scores` | ~483 MB |
| IHC upload id | `tissue/<ihc>`, `calibration/<ihc>`, `slides/<ihc>.json` | ~13 MB |

Archiving marker A must not destroy the H&E tissue mask / ROI that marker F
needs. So the layout is:

```
data/history/<CASE_ID>/
  manifest.json          case id, folder, H&E upload id, per-marker state
  he_thumb.png           the small H&E thumbnail the list screen shows
  shared/                the six H&E-keyed trees
  markers/<LETTER>/      that marker's pair-keyed trees plus its IHC-keyed ones
```

---

## Phases

### P0 — baseline
Confirm `npm run build` and a backend import both pass *before* anything changes,
so a later failure is attributable. Memory says `npx tsc --noEmit` is not enough —
the build is the only real check.

### P1 — `SlideOverlayViewer`, the one component steps 11–15 share
`frontend/src/features/slideOverlay/`. OpenSeadragon over
`/slides/{ihcUploadId}.dzi` — same tile source, controls, scale bar, readout,
navigator and real-Fullscreen-API behaviour as step 1's `SlideViewer`, because the
ask is explicitly "same functionality".

On top of it, a `<canvas>` overlay drawn in slide level-0 coordinates:

- region outlines and sampled-field boxes, so a viewer zoomed out can see *where*
  there is anything to look at;
- a shape layer per cell, culled to the viewport and to a zoom threshold — drawing
  40 000 polygons into the DOM would stall the frame, a canvas redrawn on
  `animation` will not;
- hit-testing for click-to-select, which steps 14 and 15 need.

Nuclei ring coordinates are already IHC level-0 pixels (`NucleusOut.rings`), which
is exactly OpenSeadragon's image coordinate space — no new transform.

### P2 — step 11
Swap the field viewer for the slide viewer. All nuclei light blue. Keep the
watershed and H-channel-vs-RGB comparisons and the `<details>` disclosure: they are
the argument of the step, and the house rule puts specialist numbers there.

### P3 — step 12
Same viewer, `colourBy="type"`. Red tumour / yellow lymphocyte / green other, with
legend and stacked bar driven by the same palette so the picture and the key cannot
disagree. Thresholds stay live on their sliders.

### P4 — compartment geometry on the server
Step 13 ships only a rendered PNG per field today, which is why it is unreadable.
Store a per-cell geometry file alongside the label maps: `nucleus`, `cytoplasm` and
`membrane` rings for every measured cell, plus which of the three this antibody
actually measures. Served per region, like step 11's nuclei, for the same reason —
it is megabytes and only one region is on screen.

### P5 — step 13
Three-colour compartments on the slide viewer, a single blown-up cell beside it,
and the width slider in microns with the measured area updating live. The measured
compartment is marked as such, so the membrane/cytoplasm fork is visible rather
than asserted.

### P6 — step 14
Viewer + click a cell → a card with its mean DAB OD, its second number (ring
completeness over 36 bins for A/F/R, stained fraction for U/W), area, occupied
bins, region and field. Selection shared with the scatter in both directions.

### P7 — step 15
Same selection; the selected cell's bin, where it falls against the cut lines on
the histogram, and the totals table.

### P8 — history service (backend)

```
GET    /history                          cases, per-marker state
GET    /history/{case}/thumbnail.png     the H&E overview
POST   /history/{case}/{marker}/open     rename back into demo; return upload ids + resume point
POST   /history/{case}/{marker}/archive  rename out of demo into history
DELETE /history/{case}/{marker}          delete one marker's run
DELETE /history/{case}                   delete the case's stored data
```

Plus a start-up sweep that archives anything a crash left in `demo`, so the buffer
does not silently accumulate.

### P9 — previous cases (frontend)
Home page section; case cards with the H&E thumbnail; a marker grid per case
showing done / part-way / not started; open, resume, replay and delete.

### P10 — replay mode
Opening a finished marker walks the same screens with every run suppressed — the
panels already prefer a cached report, so replay is "restore, then do not
auto-start". Finishing a marker archives it.

### P11 — CD44 end to end
Run A on CAN_00270 through all 17 steps, archive it, replay it from history, delete
it, confirm the disk goes back.

### P12 — close out
`npm run build`, backend tests, docs, memory.

---

## Things already known that this must not trip over

- A backend edit restarts uvicorn under `--reload` and destroys an in-flight run.
  So: finish a phase's backend edits, *then* run anything long.
- Nucleus ids restart at 1 in every field. A cell is `field:id`, never `id`.
- `npx tsc --noEmit` passes things `npm run build` catches.
- The agent shell cannot spawn a nested `powershell.exe`; long jobs detach through a
  `.cmd` wrapper.

---

# Outcome — 2026-09-16, 06:15

Every phase done. `npm run build` green, `npm run lint` clean at
`--max-warnings 0`, backend imports with 102 routes.

## The five markers

Scored unattended in 78 minutes, then filed:

| | marker | % positive | intensity |
|---|---|---|---|
| A | CD44 | 65 | 1 Weak |
| F | ABCC4 (MRP4) | 95 | 1.5 Moderate |
| R | ABCC11 (MRP8) | 60 | 1 Weak |
| U | N-cadherin (CDH2) | 100 | 1.5 Moderate |
| W | Pan-cadherin | 85 | 1 Weak |

Workbook: `CAN_00270_scores.xlsx`. Log: `NIGHT_RUN_LOG.md`.

Every one carries two caveats worth reading before the numbers: the cut points
are **provisional** (never fitted against the 120 reader scores, that sheet is not
on disk), and the **denominator is 76–83% short** against the case's own H&E,
which inflates every percentage. U is outside its expected 70–85% range, which
with unfitted cuts is more likely the cuts than the tissue.

## State left on disk

`data/demo` 48 KB (empty directories), `data/history` 2.6 GB, all five filed.
22 GB free on C:.

## Four bugs the browser check found

The build was green through all of them.

1. The start-up sweep deleted steps 10–16 on **every** backend restart — 483.2 MB
   in one go. Pair-keyed `<he>__<ihc>` directories are never in the set of known
   upload ids. Marker A was the casualty and was re-run.
2. Step 8 never showed its cached result, so every already-processed slide was
   offered a 10–30 minute recompute.
3. A failed QC probe was a dead end — no run, no skip, no continue.
4. Timeouts reported as "signal is aborted without reason".

## Evidence

`walkthrough_shots/` (CD44) and `walkthrough_shots_F/` (ABCC4): each of steps
11–15 as it arrives and zoomed in on tissue. `morning_home.png` is the previous
cases screen as left.
