# Previous cases, and steps 11–15 on the real slide

Built 2026-09-16. Two changes that turned out to be one: the five per-cell steps
now draw on the slide itself, and a finished run is kept somewhere it can be
opened again instead of being recomputed.

---

## 1. Steps 11–15 pan the slide

Every one of those steps used to show a 512 px PNG of one sampled field. That
picture is honest about the cells and silent about everything else — where on the
slide they are, how much of the tumour was sampled, how small a cell is against
the tissue around it. A reader who has just watched step 1 pan a 127,000 px slide
and is then handed a thumbnail has no way to connect the two.

`components/viewer/SlideOverlayViewer.tsx` is step 1's viewer — same tile source,
same controls, same real-Fullscreen-API behaviour — with a `<canvas>` on top.

| Step | What is drawn | Colours |
|------|---------------|---------|
| 11 Nuclei | every nucleus | light blue; hollow when it sits in a square's edge band and is not counted |
| 12 Cell typing | every nucleus, by class | **red** tumour · **amber** immune · **green** support |
| 13 Compartments | nucleus and the measured band | **red** nucleus · **green** membrane · **amber** cytoplasm; the fork this antibody does *not* use is drawn dashed |
| 14 Per-cell | every measured cell, by DAB density | grey-blue → amber ramp |
| 15 Binning | every measured cell, by level | grey (0) → amber ramp (1+, 2+, 3+) |

Steps 13, 14 and 15 answer a click with a card: which cell, its numbers, the cuts
they are judged against, and the verdict.

### Three decisions worth keeping

**Everything is in the slide's level-0 pixels.** Step 10 warps the regions into
IHC level-0 pixels, step 11 stores its outlines there, and OpenSeadragon's image
coordinates are the same space. No transform lives in the viewer.

**A canvas, not DOM overlays.** A region holds tens of thousands of nuclei.
Rings are flattened into `Float64Array`s once, culled field-first, and redrawn per
frame.

**Shapes disappear when they would be lies.** Below about five screen pixels a
nucleus outline is a coloured dot and a field of them reads as a heat map of
something. Under that threshold the cells are not drawn and the viewer says to
zoom in; the sampled squares and region borders stay, because at that scale those
are what there is to see.

**Every step has the same "fly to" row** (`features/cells/FlyTo.tsx`). The cells
occupy a fraction of a percent of the slide, so without it these screens are a
hunt. The targets are derived from the overlay being drawn, so a screen cannot
offer to fly somewhere it has nothing to show.

### New on the server

- `geometry.anatomy()` builds both compartment bands from one distance transform.
- `step15_compartments/rings.py` traces them into slide coordinates, dropping
  collinear vertices (lossless; removes about nine in ten).
- `GET /compartments/{he}/regions/{rank}/geometry`
- `GET /binning/{he}/cells` — one bin and one positivity verdict per cell,
  computed server-side so the browser never re-implements the partial-staining
  rule.

---

## 2. Previous cases

`data/demo/` is the working buffer; `data/history/` is where finished runs are
filed. Both are under `data/` on one volume, so filing is a directory rename.

```
data/history/<CASE>/
  manifest.json     what was run, when, and where each piece is now
  he_thumb.png      the overview the list screen shows
  shared/           the H&E-keyed trees — steps 2 to 9
  markers/<L>/      one antibody's own trees
```

Measured on CAN_00270: archive 505 MB in 0.65 s, open in 0.02 s, total bytes
conserved (2685.5 MB either way), `data/demo` empties completely when all five
markers are filed.

**Shared H&E work is filed at case level.** A case's five antibodies share one
tissue mask, white point, tile grid, class map and ROI. Filing CD44 must not take
the ROI ABCC4 still needs, so the H&E-keyed trees move only when no marker of that
case is open, and return the moment one is.

**State is read off the disk** — which `report.json` exist — never from a flag
written when a run finished. A flag is wrong after a crash, and this list is what
offers to resume.

```
GET    /history                          cases, per-marker state
GET    /history/{case}/thumbnail.png     the H&E overview
POST   /history/{case}/{marker}/open     rename back into the working tree
POST   /history/{case}/{marker}/archive  file a finished run
DELETE /history/{case}/{marker}          remove one antibody's run
DELETE /history/{case}                   remove the case's stored work
```

Opening a finished marker replays the same seventeen screens without running
anything: the rename puts every cached report where each step looks for it.

---

## 3. Four bugs the browser check found

`frontend/scripts/check-walkthrough.mjs` drives the demo in Chromium, walks the
walkthrough and photographs steps 11–15 zoomed in. A canvas drawing the wrong
thing type-checks perfectly, so the build proves nothing about these screens.

1. **The start-up sweep deleted steps 10–16 on every backend restart.**
   `maintenance_service` compared directory names against known upload ids, but
   pair-keyed directories are named `<he>__<ihc>`, which is never in that set. One
   restart removed 483.2 MB. Fixed with `is_orphaned()`, which splits on `__`.

2. **Step 8 never showed its cached result.** `useTissueType` cleared the report on
   an `inputsKey` that step 7 changes a moment after load, while the cache lookup
   keyed on `uploadId` alone — so it fetched the report, then cleared it, then never
   looked again. Every already-processed slide was offered a 10–30 minute recompute.

3. **A failed QC probe was a dead end.** That branch offered only "Try again" while
   the branch next to it offered Skip, so the walkthrough stopped at step 2 with no
   way forward.

4. **Timeouts reported as "signal is aborted without reason".** Step 13's fetch gave
   up at 30 s when the real cost is ~18 s cached and minutes uncached.

Run it with both servers up:

```
cd tissue_scoring_demo/frontend
node scripts/check-walkthrough.mjs --marker A
```

The only expected failures are two 409s from `/qc/...` on a case scored without
quality control.

---

# Addendum — a membrane marker now measures a shell (2026-09-16)

Step 13 was showing two colours where the brief asked for three, and the reason
was not the screen: `ring_um` equalled `expansion_um`, so a "membrane ring" was
the **entire 4 µm annulus** — the whole cell body. Membrane and cytoplasm were
the same shape at two widths, nested, so there was no cytoplasm left to draw.

Measured on CAN_00270 before the change, per cell: nucleus radius ~2.0 µm,
membrane edge ~5.8 µm, cytoplasm edge ~7.8 µm — two nested annuli 2 µm apart.

## What changed

`MarkerSpec.membrane_shell_um`, `1.5 µm` for A/F/R and `0` for U/W. A membrane
marker is now measured in the outer 1.5 µm of its 4 µm body — the shell sits from
2.5 to 4.0 µm out, with cytoplasm genuinely between it and the nucleus. A
cytoplasmic marker still measures the whole body. Both numbers come from
`app.panel` and neither is reachable from a request.

Measured after: nucleus ~2.0 µm, cytoplasm out to 4.4 µm, membrane shell 1.4 µm
on top — three disjoint regions summing to the 4 µm body.

`geometry.anatomy()` now returns `nucleus / cytoplasm / membrane / alternate`
with `measured` naming which one the antibody uses, and asserts nothing overlaps.

## What it did to the scores

| marker | before | after |
|---|---|---|
| A CD44 | 65% / 1 | **60% / 1** |
| F ABCC4 | 95% / 1.5 | **90% / 1.5** |
| R ABCC11 | 60% / 1 | **55% / 1** |
| U N-cadherin | 100% / 1.5 | 100% / 1.5 (cytoplasmic — untouched) |
| W Pan-cadherin | 85% / 1 | 85% / 1 (cytoplasmic — untouched) |

All three membrane markers fell about five points; no intensity band moved.
Mean DAB fell too (CD44 0.248 → 0.225), which is worth noticing: if CD44 were
cleanly membranous the rim should be *browner* than the whole band, not paler.
On these heavily and diffusely stained sections it is not.

**F is not a clean comparison.** Its step 10 had to be re-run from scratch (see
below), which redrew the carried regions, so its nuclei count changed from 2,4xx
to 3,890. Its five points are the shell *and* a different sample.

## Two traps this surfaced

**The API server does not reload.** It is started without `--reload`, so backend
edits do not reach it. The batch runner (a fresh process) wrote the new geometry;
the browser then hit the stale server, which recomputed with the old code and
overwrote it. Symptom: `Cannot read properties of undefined (reading 'toFixed')`
in `CompartmentsPanel`, because half the run was in each format. **Restart the
API after any backend change.**

**Re-running a marker has no undo.** Filing moves rather than copies, so when
step 10 refused on re-run it overwrote F's good alignment report with a refusal
and there was no second copy to fall back on. Recovered with
`--restart-alignment` (21 min). The refusal itself is the known VALIS-on-reuse
bug: it reported "only 0 matched features" while `pair_summary.csv` on disk
recorded a good registration (rigid error 42.8 px, rTRE 0.066).
