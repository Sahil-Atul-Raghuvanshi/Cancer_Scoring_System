# Step 09 - Build the ROI mask

Turn step 8's per-window class probabilities into the single region every later step
measures inside. Classical: no model, no training, no slide access.

## Input / output

| | |
| --- | --- |
| Input | Step 8's stored class map (`data/tissue_type/<id>/classmap.npz`), read through `tissue_type_service.class_map` |
| Output | A boolean mask on step 8's own grid, plus one rectilinear polygon per focus in level-0 slide pixels |

## The recipe

`mask.build` follows the guide's step 10, with one addition:

1. **Smooth** `P(invasive)` with a Gaussian, in grid cells.
2. **Threshold** to binary.
3. **Close** to merge nearby foci and fill small holes.
4. **Carve out** windows the model calls in-situ with confidence. *Not in the guide.*
5. **Drop** components below a physical area, in mm2.
6. Optionally **keep the largest N**.

## Why step 4 exists

The guide justifies closing with "a blood vessel inside a tumour should not punch a
hole in the region". True - and a **duct of in-situ carcinoma inside a tumour
absolutely should**, because Rule 5 keeps in-situ disease out of the denominator.
Morphology cannot tell those apart; both are a not-invasive island in an invasive sea.
Only the class map can, so the class map decides.

This is measured, not assumed. On `CAN_00270_26_H&E`, against a pathologist's DCIS
contour:

| closing radius | ROI | in-situ swallowed inside the DCIS contour | elsewhere |
| --- | --- | --- | --- |
| 0 | 31.8 mm2 | 6.6 % | 3.3 % |
| 1 (112 um) | 32.7 mm2 | 10.6 % | 3.4 % |
| 3 (338 um) | 37.6 mm2 | **20.3 %** | 3.6 % |
| 6 (675 um) | 45.0 mm2 | **35.0 %** | 4.0 % |

The merge is selective for exactly the in-situ that matters - 82 % of the in-situ
windows inside that contour sit within one window of invasive tissue, against a median
of 6.3 windows for the isolated normal lobules elsewhere. Carving it back costs
1.4 mm2 of 37.6 and returns the swallowed fraction to 1.6 %.

Set `protect_in_situ=0.0` (or `?protectInSitu=0`) for the guide's plain recipe, so the
two can be compared rather than argued about.

## Polygons are rectilinear on purpose

The ROI is a union of whole grid cells, so its true boundary runs along cell edges. A
spline through it would look more anatomical and would claim sub-window precision the
class map does not have - 112 um here. Holes come out of the same trace, which matters,
because a hole is how in-situ disease inside a tumour is represented.

## What this step cannot fix

The carve-out removes in-situ the model **recognises**. It cannot remove in-situ the
model confidently calls invasive, and on an unseen laboratory that is the larger error:
on this same slide roughly 31 % of the scored area falls inside a contour two
pathologists called pure DCIS. That is step 8's recall problem, not step 9's, and no
morphology here touches it. See `tissue_type_model_training/README.md`.

## Status

Implemented. `mask.py` is pure and testable from a hand-built array;
`app/services/roi_service.py` builds, caches and draws. Unlike step 8 there is no job:
the region takes ~0.1 s, and the ~20 s a cold build costs is the thumbnail and the five
1600 px panels. A cached read is immediate.

Measured on `CAN_00270_26_H&E` (252 x 252 grid): load class map 0.014 s, build the mask
0.097 s, thumbnail 4.2 s, panels 1.7-3.9 s each.
