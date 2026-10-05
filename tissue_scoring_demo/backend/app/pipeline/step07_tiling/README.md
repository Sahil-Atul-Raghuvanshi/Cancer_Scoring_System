# Step 07 — Tiling

Implemented. Catalogue entry: `app/data/pipeline_steps.py`, `id="tiling"`.
Guide: `docs/guides/demo-pipeline-guide.md`, **step 8** there — the guide's step 6 is
stain normalisation, which this pipeline skips (Rule 2), so every step from
deconvolution on is one lower here than in the guide.

## The two choices — settled, not removed

**Step 7 no longer asks these questions. It states their answer.** The pipeline
commits to the `he` branch at **224 µm**, which resolves to
`invasive_tile_fov224_he_concat`; the branch comes from `settings.tiling_branch` and
the scale from `settings.tiling_field_of_view_um`. The screen reports what ran, and
the two pickers (`BranchPicker`, `FieldOfViewPicker` in the frontend) are still on
disk, unreferenced, for whoever re-opens the question.

Why those two values, and not the eight-way choice below:

- **`he` rather than `h_channel`.** The h_channel branch's whole justification was
  serving all six slides of a case from one model. Measured, it does not: **0%
  invasive on all five immunostained markers**, at 0.96–0.98 confidence, against
  12.7% on the same case's H&E. It fails plausibly, not loudly. Since step 10 finds
  the region on the H&E and registers it across, that generality is generality
  nothing uses — and on the H&E, eosin is half the evidence, because a duct wall and
  a collagen band are the same shade of nothing in the H channel.
- **224 µm** is the widest field that still keeps a large tile count. The trade is
  measured in `docs/segmentation_research/RESULTS_224_VS_448.md`: a wide window reads architecture but
  inflates area — 448 µm reported 77.5 mm² of invasive against 112 µm's 42.0 mm² over
  identical tissue.

One consequence worth stating, because nothing upstream catches it any more: the
branch payload still marks the colour option unavailable on a section step 5 says is
not H&E, but that was advice to a picker that no longer exists. `commit_selection`
validates that a branch is *served*, not that it suits the slide. Step 7's report
therefore carries an explicit warning note when the colour model meets a non-H&E
section — see `TilingService._notes`.

The rest of this section describes the choice as it was offered, and remains accurate
about how `model_for` resolves a branch and a scale to a checkpoint.

Step 7 asked two questions, in order, and committed both before it ran.

**What should the model be shown?** Three options, in `ModelBranch`:

| option | what the model gets | where it works |
| --- | --- | --- |
| `h_channel` | Ruifrok's haematoxylin channel, as optical density | every slide in the panel — the H&E and all five IHC |
| `he` | the sRGB photograph, ImageNet-normalised and nothing else | H&E sections only, and only where step 5 found eosin |
| `beetle` | — | not developed; shown disabled so the scope is visible |

**How much slide fits in one square?** 112, 224, 448 or 672 µm, all at a 224 px
window — only the mpp moves, which is what leaves the backbone untouched and makes
the four comparable.

The two together name one of **eight** published checkpoints. `model_for(fov,
branch)` matches on the manifest's own `input.channel` and `tile_px * mpp`, never on
a filename: both branches share all four geometries by construction, so geometry
alone stopped being a unique key the moment the H&E heads were published, and
letting `discover()`'s ordering decide would feed a colour model optical density —
which does not raise, it just scores badly and reads as a modelling result.

The `he` option is gated on evidence rather than on a filename.
`step05_optical_density/staining.py` reads step 5's point cloud: two separate dye
directions, one of them within `he_eosin_tolerance_deg` of Ruifrok's eosin vector.
Measured over 18 tiles of six of this project's sections, every H&E tile put an arm
5.0–9.4° from eosin and **no** immunostained tile produced an eosin-nearest arm at
all. The verdict carries every angle it decided on, and the disabled option renders
its `reason` verbatim.

## Where the choice lives

`data/tiling/<upload_id>/selection.json` — the branch, the field of view, the
overlap and the threshold. Kilobytes; not the index, which stays in memory.

**It exists because the choice had nowhere to live.** Step 8's worker asked
`tiling_service` for the index with no arguments, so it rebuilt the grid at
`tiling_field_of_view_um` no matter which head the caller had named. The two grids
then diverged, `inference._mark_inside` silently dropped out of its identity fast
path, and the untargeted call evicted step 7's single-entry memo on the way past.
Threading four parameters through every call site would have fixed the symptom and
left the same failure available to the next caller who forgot one. Every read now
resolves *explicit argument → committed record → settings default*, so the
no-argument call is correct by construction.

Committed by `POST /tiling/{id}/selection` and never as a side effect of a GET: a
GET that wrote it would make "the viewer glanced at 672 µm" indistinguishable from
"the viewer chose 672 µm", and every panel request carries the same query string.

Changing it deletes `data/tissue_type/<id>/` **and** `data/roi/<id>/` — see step 8.

## Input

Step 3's tissue mask and step 2's artefact map, via `tissue_service.footprint()`,
plus the slide's own dimensions. Step 6 is required too, for a reason that is not
about data: the index this step produces is *addresses*, and the pixels those
addresses resolve to are whatever the chosen option shows the model — so a tile
index built in a pipeline where nothing has defined what a tile's pixels are would
be an index of nothing.

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

## The four parameters

Two of the four are this step's to choose, and they are not the same kind of choice.
The overlap moves the price. **The field of view moves which model runs**, because a
field of view is a property of the weights — so choosing the scale *is* choosing the
checkpoint, and step 8 inherits both.

- **Field of view** — **224 µm or 448 µm**, and the one decision on this screen with a
  consequence rather than a cost. See `FIELDS_OF_VIEW` in `tiling_service`. The served
  112 µm window is why it exists: a duct is 300–1500 µm across, so a 112 µm window
  centred inside one holds a sheet of tumour cells with no wall, no periductal stroma
  and no rim — physically indistinguishable from invasive. That is arithmetic, not a
  training deficiency, and it is why the served checkpoint records
  `dcis_called_invasive = 0.224`. See `FIX1_FIX2_PLAN.md`.
- **Size** — 224 px at both fields of view, so only the resolution moves between them
  and the backbone is untouched. Read from the chosen checkpoint's manifest by
  `tiling_service.window`, and from `FIELDS_OF_VIEW` only when no head has been
  published at that field of view yet.
- **Resolution** — the chosen checkpoint's: 1.0 µm/px at 224 µm, 2.0 at 448. This is
  the one place the pipeline's working magnification does not rule: steps 4–6 work at
  `target_mpp`, this grid works at the model's, and where they differ the checkpoint is
  right about what the model sees. It is also the cost of the wider field — the same
  224 pixels spread over four times the ground, so a nucleus is about five pixels across
  instead of ten.
- **Overlap** — **none.** The guide asks for 25–50% for segmentation and the reason
  is real: a model has no context past a tile's edge, so its predictions there are
  its worst, and overlap is what lets those edges be averaged away rather than
  stitched into visible seams. It is also the one setting here that costs compute,
  and it costs it quadratically — 25% is 1.8× the tiles, 50% is 4×, which at step 8
  measured about 10 / 15 / 29 minutes on `CAN_00251_26_A`. The demo takes the ten
  minutes and the seams. Step 7's screen offers this setting alone; the endpoint's
  `overlap` query parameter still runs the other two, and step 8 inherits whichever
  step 7 ran.

### When no head exists at the chosen field of view

The grid is still laid, on the planned geometry, and `params.model` comes back null.
That is deliberate: the square count **is** the compute bill, so a reader deciding
between two fields of view needs it before either head exists. Step 8 refuses
separately and names the file it wants.

What must never happen is the obvious fallback — serving whichever checkpoint is
configured. That one was fitted at a different field of view, so it would score a grid
it never saw, and nothing would show it: the class map would still be three-valued and
the confidences would still look reasonable. `test_step_8_refuses_when_step_7s_field_of_view_has_no_head` pins the refusal.

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
