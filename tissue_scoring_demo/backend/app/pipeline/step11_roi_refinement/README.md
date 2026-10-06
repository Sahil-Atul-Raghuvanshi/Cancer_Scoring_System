# Step 11 - Refine the regions per pixel

Run BEETLE on each region a person ticked on step 10, and trace what it finds. What
comes out is the invasive-tumour mask every later step measures inside.

## Input / output

| | |
| --- | --- |
| Input | Step 10's selection; step 8's window grid and its gates; the H&E slide |
| Output | `refined.json` - one polygon-with-holes per focus, in H&E level-0 pixels - plus a before/after picture per region |

## Where the region comes from (P-10, P-11, October 2026)

`roi_refinement_source` picks it, and **the default is `"step9"`, not BEETLE.**

- `"step9"` - each selected region's share of step 9's scoring mask (in-situ carve-out,
  smoothing, 0.25 mm2 minimum), inside the region's own territory. No network runs; a
  case takes seconds.
- `"beetle"` - the per-pixel refinement this README describes below.

Chosen by measurement against OncoStem's own QuPath outlines (precision / recall against
the pathologist's invasive tumour):

| Case | Step 9 mask | BEETLE as previously scored | BEETLE + 40 um grouping |
| --- | --- | --- | --- |
| CAN_00303 | 97 / 74 % | 99 / 44 % | 99 / 55 % |
| CAN_00270 | 78 / 82 %, 12 % in DCIS | 63 / 49 %, **27 % in DCIS** | 60 / 59 %, 27 % in DCIS |

BEETLE decides in-situ vs invasive per window, and on a solid mass the answer flips
between whole windows (a checkerboard). Wider windows (448 um) and half-overlapping ones
(256 um, 50 %) did not change the split and cost 2-3x, so it is a near-tie in the model,
not missing context.

Two things apply to **both** sources:

- **Territory.** A region traces only its own tile outline grown by the pad, less what
  higher-ranked regions own (`crop.territory`). Padded boxes overlap, and the same pixels
  used to become foci of two regions, sampled and weighted twice; a slide-spanning box
  swept in tumour nobody selected.
- **Step 12 follows.** Its report stores a fingerprint of `refined.json`, so re-running
  this step makes step 12 re-warp instead of handing on the old outlines.

On the BEETLE path, invasive pixels within `roi_refinement_group_um` (40 um) are joined
before specks are dropped: a tumour that infiltrates as single cells (CAN_00267) used to
lose every cell to the speck filter and come back empty.

## BEETLE does not run on the slide

That is the whole step. `pixels.segment` already existed and already ran BEETLE over
every window of a section; this step points it at the windows of a handful of padded
boxes instead. Nothing about the network changes - same weights, same fixed 0.5 um/px,
same preprocessing - and the only thing that differs is which windows are marked.

Measured on `CAN_00270`'s H&E, with step 10's default selection of 26 regions:

| | |
| --- | --- |
| BEETLE windows the selection costs | 2,161 |
| Windows a whole-section pass would cost | every window of the kept tissue - far more |
| Two small test regions (0.80 mm2 of tile) | 76 windows, 121 s on CPU |

`crop.restrict` is where that happens, and it **narrows step 8's grid rather than
building a new one**. That is the safety property: every window it returns is one the
slide-wide pass would also have run, so this step cannot look at tissue step 7's audited
tile list and step 8's own tissue gate had already excluded.

## The padding is an input decision

`roi_refinement_pad_um` (224, one window) pads what the *network is shown*, which is a
different thing from `roi_crop_pad_um` padding a picture. A segmentation network asked
to decide a boundary with nothing beyond it draws the boundary at the edge of what it
was given, and a candidate box is a staircase around tumour that does not stop at the
staircase.

One window is the smallest padding that works, and the reason is arithmetic:
`build_grid` marks a window when its **centre** lands on kept tissue, so a pixel on the
very edge of a candidate is only covered by a window whose centre is up to half a window
outside it.

The padding is recorded on every region (`cropX/Y/Width/Height`, `padUm`) because the
mask is placed back into the slide from that origin. A result that lost it would be a
shape with no position.

## Coordinate integrity

    candidate box (H&E level-0)
      -> padded box, the canvas BEETLE paints into      crop.padded_box
      -> mask pixels at 1 um/px, origin at the box      pixels.segment(mask_bounds=...)
      -> polygons in H&E level-0 pixels                 contours.regions
      -> VALIS warp                                      step 12
      -> polygons in IHC level-0 pixels                  step 12

Every arrow is a multiplication and an addition. `contours.to_level0` owns the one where
a *picture* becomes *geometry*, and it is the one where a mistake does not fail - a mask
appears, in the right shape, on the wrong tissue.

`pixels.mask_shape` gained a `bounds` argument for this. Without it the canvas is the
whole slide, which at 1 um/px is 784 megapixels for a 28 mm section - which is why the
slide-wide pass reduces to 4 um/px. A region is not a section, so a pass restricted to
one can afford four times the linear precision.

## Foci stay grouped

A single coarse box routinely refines into several separate foci - ROI-007 above became
**nine**. Each is stored as its own outer ring followed by its own holes, and each
carries **its own area**.

Both of those are load-bearing:

* Flattened into one list of rings, the first focus would be read as the outer boundary
  and every other focus as a hole in it, because that is the convention
  `sampling.rasterise` fills with and `step12/regions.py` warps with. The tumour would
  cross as its first focus with the rest punched out of it - wrong, and entirely
  plausible-looking.
* Given an equal share of the region's area instead of its own, step 13 would mis-sample:
  it splits its field budget between regions in proportion to area, and on ROI-007 the
  nine foci run from 0.052 mm2 down to 0.005 - a tenfold spread an equal split erases.
  That is the exact failure `sampling.allocate` was written to prevent, reintroduced one
  level down.

## Three clean-ups before tracing

A per-pixel network at 1 um/px produces things a per-window one cannot.

| | |
| --- | --- |
| **Specks dropped** below `roi_refinement_min_component_mm2` | otherwise one boundary becomes hundreds of rings, each densified, warped, simplified, stored and drawn |
| **Pinholes filled** below `roi_refinement_min_hole_mm2` | a lumen is a real hole and a three-pixel gap is not; `binary_fill_holes` alone would remove both |
| **Staircase simplified** at `roi_refinement_simplify_um` | the trace runs along mask-pixel edges, so a millimetre of boundary is a thousand vertices. Two mask pixels of tolerance can remove the staircase and nothing larger |

## A region is the unit of work, storage, failure and retry

Each region writes its own directory the moment it finishes, so a screen shows it while
the next one is still going; a region that fails writes its error there and the pass
continues; a restart re-runs only what is missing. None of that needs special cases,
because none of it is one - it is what storing per region means.

`refined.json` is **rewritten from the region directories** each time a pass ends, never
appended to as regions land. An accumulator would have to be correct under cancellation,
partial failure and retry; a recomputation from what is on disk is correct by
construction.

A stored region is reused only when it is complete, produced at this pass's geometry,
*and* belongs to the class map currently on disk. Without the last condition, a region
refined before step 8 was re-run would be resumed into a report describing a different
slide's labels.

## What this step hands downstream

`roi_refinement_service.regions()` is the authoritative invasive mask, and step 12
**refuses** rather than falling back to step 9's tile squares when it is absent. That
refusal is the point of the step: a square ROI and a pixel ROI give coherent-looking
scores on different denominators, and nothing after step 12 could tell which it had.

## Files

| | |
| --- | --- |
| `grid.py` | BEETLE's own window grid for the slide, shared with step 10 so the price and the run agree |
| `crop.py` | the padded box, and narrowing the grid to the windows whose cores meet it |
| `contours.py` | raster answer to polygons in slide coordinates - the coordinate integrity |
| `overlay.py` | the comparison: both panels from one read and one drawing path |
| `pipeline.py` | the `run(context)` adapter |

## Known behaviour worth reading before tuning

**BEETLE and the tile head disagree about invasive versus in-situ.** On the two regions
measured above, the tile head called the patches invasive and BEETLE called most of the
same epithelium *non-invasive* - `classShare` came back around 0.77-0.90 `other`,
0.07-0.17 `non_invasive_epithelium` and 0.02-0.05 `invasive_epithelium`, so `keptShare`
was 6-18 %. The refined mask is therefore much smaller than the tile mask, and that
directly shrinks the denominator every later step measures in. Whether that is BEETLE
being right or BEETLE being conservative on small padded crops is **not settled here**;
`beetle_mask.png` is the panel to look at, and the existing notes on step 8's in-situ
behaviour are the context.

**Window seams are visible in the class map.** BEETLE sees one 224 um window at a time
by design - see `pixels.py` on why the block is a read amortisation and not the unit of
inference - so adjacent windows can disagree along a core boundary. `overlap` is fixed at
zero here because it changes no pixel (cores partition the tissue) and would multiply the
cost; the knob that actually reduces seams is a larger `roi_refinement_fov_um`.
