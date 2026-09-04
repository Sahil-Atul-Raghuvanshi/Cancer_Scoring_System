# Step 07 — Tiling

Implemented. Catalogue entry: `app/data/pipeline_steps.py`, `id="tiling"`.
Guide: `docs/demo-pipeline-guide.md`, **step 8** there — the guide's step 6 is
stain normalisation, which this pipeline skips (Rule 2), so every step from
deconvolution on is one lower here than in the guide.

## Input

Step 3's tissue mask and step 2's artefact map, via `tissue_service.footprint()`,
plus the slide's own dimensions. Step 6 is required too, for a reason that is not
about data: the index this step produces is *addresses*, and the pixels those
addresses resolve to are the haematoxylin channel — so a tile index built in a
pipeline where nothing has defined what a tile's pixels are would be an index of
nothing.

## Output

A filtered tile index: level-0 origin, extent and the two shares each tile was
judged on. **No pixels.**

## Why it runs here — the funnel

This is the whole step. A whole-slide grid holds hundreds of thousands of tiles;
a section holds a fraction of that; the part QC did not object to holds slightly
less again.

| gate | what it removes | why the step runs after it |
| --- | --- | --- |
| tissue | empty glass — two thirds of an Aperio frame | the big cut, and the reason tiling follows step 3 |
| clean | folds, blur, pen marks | an artefact tile produces a *confident wrong* class at step 8, not no class |

Every surviving tile is one forward pass through the region model, which is the
most expensive thing in the pipeline. The number that leaves this step is the
compute bill for everything after it.

## The three parameters

Only one of the three is this step's to choose.

- **Size** — the region model's, read from its checkpoint's manifest by
  `tiling_service.window`; 224 px for the published model. Not a preference: the
  model's field of view is a property of the checkpoint, and a square of any other
  size would be one the model is never shown.
- **Resolution** — the same manifest's. This is the one place the pipeline's
  working magnification does not rule: steps 4–6 work at `target_mpp`, this grid
  works at the checkpoint's, and where they differ the checkpoint is right about
  what the model sees.
- **Overlap** — 25–50% for segmentation, and the one real choice here. A model has
  no context past a tile's edge, so its predictions there are its worst; overlap
  lets those edges be averaged away rather than stitched into visible seams. It is
  the one setting here that costs compute, and it costs it quadratically.

This step used to lay a 512 px tile at the pipeline's working resolution while step
8 laid its own 224 px windows on top, joined by a centre test. The funnel then
priced 4,952 squares and the model made 98,262 passes over the same slide, gated at
0.10 tissue here and 0.50 there. One grid removes both discrepancies: the count
below is the number of forward passes, and `min_tissue_share` is now the pipeline's
only tissue gate on a square, carrying step 8's 0.50.

## Two things this step is careful about

**The shares are measured on step 3's coarse mask, on purpose.** One mask pixel
covers about sixteen tile pixels, so each share is quantised to a few percent.
Measuring properly would mean reading the whole slide at working magnification —
the exact cost this step exists to avoid — for a number only ever compared
against a threshold. The quantisation is reported rather than hidden.

**Covered area counts overlap once.** `count × tile area` is four times the real
covered area at 50% overlap, which reads as though overlapping tiles see more of
the slide. They do not — they see the same tissue more often.

## Layout

| File | What it holds |
| --- | --- |
| `index.py` | The grid, the two gates, the funnel, the coverage arithmetic |
| `overlay.py` | Two panels — the grid on the slide, and one tile as the model gets it |
| `pipeline.py` | `run(context)` for the end-to-end runner |

Service, schemas and endpoints sit where every other step's do:
`app/services/tiling_service.py`, `app/schemas/tiling.py`,
`app/api/v1/endpoints/tiling.py`.

The sample panel draws step 6's haematoxylin channel through step 6's own ramp
and scale, obtained from `deconvolution_service`. That is deliberate: this is the
first screen downstream of the fork, so it is the first place the guide's rule —
train and inference call *the same* deconvolution function — can be broken. A
local `rgb2hed` here would break it invisibly, because the picture would still
look right.

## References

Dolezal JM et al. *Slideflow.* BMC Bioinformatics 25:134 (2024), arXiv:2304.04142
— the tile-index shape and the tissue-fraction filter.
Tellez D et al. *Quantifying the effects of data augmentation and stain color
normalization…* Medical Image Analysis 58:101544 (2019) — why the tiles carry the
haematoxylin channel rather than RGB.
