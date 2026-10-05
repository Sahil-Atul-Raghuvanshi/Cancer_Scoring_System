# Step 10 - Review and select candidate regions

Turn step 9's invasive patches into a list a person chooses from. No model, no training,
no measurement: the only thing this step produces that did not exist before it is a set
of ticked ids.

## Input / output

| | |
| --- | --- |
| Input | Step 8's stored class map, through `borders.class_regions`; step 9's report, as a precondition |
| Output | A ranked candidate list with a card image each, and the selection |

## Why a whole step for a checkbox

Because of what is on the other side of it. Step 11 runs a segmentation network per
pixel, and the cost of that is proportional to how much slide it is pointed at. On
`CAN_00270`'s H&E the numbers are:

| | |
| --- | --- |
| Section | ~800 mm2 of glass |
| Tissue step 7 kept | the part step 8 classified |
| Invasive patches step 8 drew | 104, totalling 41.4 mm2 |
| Offered here (>= 0.10 mm2, capped at 40) | 40, totalling 37.9 mm2 |
| Pre-ticked by the coverage rule | 26, totalling 36.2 mm2 - 87 % of the invasive area |
| BEETLE windows that selection costs | **2,161** |

The step is the filter. A coarse detector deciding where a precise one should run is the
standard shape of this; what this step adds is that the decision is *visible and
editable* rather than a constant buried in a service.

## The candidate id is an area rank

`ROI-001` is the largest invasive patch on the slide, `ROI-002` the second, and so on -
`ClassRegion.index + 1`, and nothing else. Two consequences, and both matter:

* the id is a **pure function of the class map**, so rebuilding step 8 with the same
  parameters gives the same ids on the same tissue;
* the id is **not stable across a different class map**. On a map rebuilt at another
  threshold, `ROI-007` is other tissue.

So a selection is stored with `classMapKey` beside it, and a selection made against a
different key is dropped with a note rather than reinterpreted. Carrying a tick across
would be the quietest possible way to segment regions nobody chose.

## What is not offered, and why it is reported

`borders.class_regions` keeps every connected component with no area cutoff, deliberately
- it is showing the model's raw call, speckle included. A review screen is a different
job, so two cutoffs apply here and **both are reported**:

* `roi_selection_min_area_mm2` (0.10) drops patches too small to hold a boundary. A
  single 224 um window padded for context is a couple of forward passes deciding an edge
  from almost no evidence. On CAN_00270 this removed 58 patches holding 2.9 mm2.
* `roi_selection_max_candidates` (40) caps the tail. On CAN_00270 this removed 6 more,
  holding 0.6 mm2.

A screen headed "the invasive regions" that silently showed 40 of 104 would be making a
different claim than it looks like it is making, so the report carries both counts and
both areas and the notes say them in words.

## The default selection

Enough of the largest candidates to cover `roi_selection_default_coverage` (95 %) of the
**offered** area - over what is offered rather than over all invasive tissue, so the
default does not quietly grow to compensate for a speck it cannot reach.

This is step 12's own coverage rule, moved one step earlier. It used to run inside the
alignment step, after the point where anyone could see it; pre-ticking makes it the
visible default a person can disagree with. `chosenByPerson` records which happened, the
same way step 12's `confirmedBy` distinguishes a human sign-off from a machine one.

## Files

| | |
| --- | --- |
| `candidates.py` | the ranking, the cutoffs and the default. Pure - no slide, no pictures |
| `overlay.py` | one card per candidate: the crop with the tile boundary drawn and lightly filled |
| `pipeline.py` | the `run(context)` adapter |

The boundary on a card is drawn as the **staircase it actually is**. Step 8 answered once
per 224 um window, so its border runs along window edges; smoothing it into a curve would
claim a precision the tile model does not have, and would make step 11 look like a
cosmetic change rather than the difference between a square and a tumour.
