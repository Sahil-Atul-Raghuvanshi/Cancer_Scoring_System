# Step 08 — Tissue-type segmentation

The one learned step in the pipeline, and **inference only**. It loads the checkpoint
trained in [`tissue_type_model_training`](../../../../../tissue_type_model_training) and answers
one question about every patch of tissue step 7 kept.

Guide: `docs/guides/demo-pipeline-guide.md` **step 9** (the code is one lower from
deconvolution onwards — stain normalisation is deliberately not implemented).
Catalogue: `app/data/pipeline_steps.py`, `id="tissue-type-segmentation"`.

## Eight checkpoints, and which one runs

Four fields of view — 112, 224, 448, 672 µm, all at a 224 px window — times two
input contracts. Which of the eight runs is **step 7's choice, inherited**, not a
setting of this step's own: a field of view is a property of the weights, and so is
what the network is shown, so taking either decision in two places is how the two
steps come apart.

| contract | `input.channel` | the transform, in full |
| --- | --- | --- |
| haematoxylin | `haematoxylin` | optical density → Ruifrok H-DAB → optional per-tile p99 → clip to 1.5 → scale → gamma → polarity → replicate ×3 → ImageNet |
| full colour | `rgb_he` | `/255` → ImageNet. That is all of it. |

The second column's brevity is the point. Every term the first one carries is a
statement about dye concentration, and the RGB branch makes none of them — so there
is nothing on that path that can drift between training and serving.
`descriptor(channel="rgb_he")` **refuses** a manifest claiming `invert`, a `gamma`
other than 1, or per-tile standardisation, because there is no correct behaviour for
those on a photograph and serving one under a transform nobody implemented is worse
than refusing to serve it.

`Pinned.channel` decides which reader the worker composes (`window_reader`) and which
transform `inference.classify` applies. On `rgb_he` there is no white point and no
deconvolution: passing the tile through `field_for` would divide a colour image by a
lamp profile before handing it to a network fitted on undivided ones.

**Do not add a key to the haematoxylin descriptor.** `_verify_input_contract`
compares this code's `descriptor` against the manifest key by key, and a key absent
from an already-published manifest compares `None != value` — so one added field
refuses every checkpoint ever published here. New facts about an input belong in a
new channel's block.

## What changing step 7 does to this step

`discard()` removes the **whole** directory: `report.json`, `classmap.npz` and all
four PNGs, and then `data/roi/<id>/` as well.

Removing only the report — which is what `restart` used to do — left `asset()`
serving the previous choice's `map.png` and `class_map()` handing step 9 the previous
choice's labels. And step 9 caches on *its own* parameters, so it cannot notice that
the labels underneath it changed: a region traced from a class map that no longer
exists looks current, and nothing downstream can tell. A stale answer that looks
fresh is worse than no answer.

`_cache_key` therefore carries `branch`, `field_of_view_um` and `input_channel`
alongside the model and its hash, so a changed choice invalidates even when the
resolved checkpoint happens to be the same file. It reads with `.get`, so a cache
written before those keys existed fails the comparison and re-runs rather than
raising. `discard` refuses while a pass is queued or running: the worker writes into
that directory when it finishes, so deleting underneath it would leave exactly the
half-state this is here to prevent.


## Input / output

| | |
| --- | --- |
| **Input** | step 7's tile index, and step 6's haematoxylin channel |
| **Output** | a class per window, plus the softmax behind it, plus five panels |

```
class 2  invasive epithelium       carcinoma that has broken out of the duct   ← the only thing scored
class 1  non-invasive epithelium   DCIS, LCIS, normal ducts and lobules        ← excluded (Rule 5)
class 0  non-epithelium            stroma, fat, inflammation, necrosis         ← excluded (Rule 1)

class 3  uncertain                 a class-1 call the post-pass will not back  ← DISPLAY ONLY
```

**Class 3 is not a fourth output of the model.** The head has three logits and no way
to decline; `uncertainty.py` runs over the finished map and recolours class-1 windows
whose surroundings or shape do not support the call. It never touches `labels` on disk,
it can only ever land on class 1 — which Rule 5 excludes before it is looked at — so
`tumourContent`, `scoredMm2` and step 9's ROI are bit-identical with it and without it.
`CLASS_NAMES` stays three for the same reason: it is the checkpoint's contract, and
appending to it would make every published head fail `verify_order`.

Two of the guide's ordering rules land here and they are the reason the step exists:
**fat leaves here rather than at the tissue mask**, because fat is a semantic class and
not a brightness; and **scoring is gated on invasive tumour rather than on "tumour"**,
because ductal carcinoma in situ is carcinoma that has not escaped the duct and a
percentage that pools the two answers a question about a different patient.

## What lives where

```
input.py       THE input transform - RGB to the tensor the network sees. Read first.
classes.py     the three classes, which one is scored, and the class-order refusal
model.py       finding, verifying and loading a checkpoint. Four refusals.
inference.py   the window grid, the block reads, the sweep, the batching, the class map
uncertainty.py the post-pass that adds "cannot determine". Read its docstring before
               touching the rule - three formulations that look right are wrong there,
               and each one is wrong against a slide this project measured.
overlay.py     the five panels
pipeline.py    run(context), for the end-to-end runner
```

The service, schemas and endpoints are `app/services/tissue_type_service.py`,
`app/schemas/tissue_type.py` and `app/api/v1/endpoints/tissue_type.py`.

## The three things worth understanding before changing anything here

### 1. `input.py` is shared with the training folder, not copied from it

`tissue_type_model_training/src/hchannel.py` **imports** this module. It used to be the
definition; it is now a re-export plus one training-only helper.

A model fitted on one definition of "haematoxylin channel" and served another scores
nothing like its validation number, and the drop looks like a modelling failure for a
week before anyone finds the plumbing. Two copies plus a test asserting they agree only
helps while somebody keeps running the test — and the copy that goes stale is always the
one nobody is looking at. So there is one function object, and the dependency runs
research → backend, which is the direction it already ran for `optical_density`,
`RUIFROK_HDAB` and `separate`. Nothing in `backend/` imports the research folder, so a
non-commercial training input cannot reach the shipped app through an import.

Backing that up, `model.load_pinned` compares the checkpoint's recorded `input`
descriptor against `input.descriptor(...)` and **refuses** the checkpoint on any
disagreement. And `scripts/check_tissue_model.py` replays the tiles the manifest
recorded logits for, through this process, and compares — gate G6. That is the only
check in the project that can catch a training-to-serving mismatch.

```
$ .venv\Scripts\python scripts\check_tissue_model.py
[  ok  ] G6 PASS - this process reproduces the manifest's logits: 21 probe tiles,
         worst logit difference 2.37e-06
```

### 2. This step's grid is step 7's grid

The model takes **224 px at the checkpoint's mpp** — 112 µm. That is not a preference
anything may override: it is a property of the checkpoint, recorded in its manifest, and
resizing some other tile to 224 px would keep the pixel count and change the
magnification, which is worse than either. So step 7 reads that manifest and lays that
square, rather than this step laying a second grid over a coarser one.

It was two grids until then — 512 px tiles at the pipeline's working resolution here,
224 px windows there — and the cost was visible on screen: step 7 priced **4,952
squares** and this step made **98,262 passes** over the same slide, gated at 0.10 tissue
on one side and 0.50 on the other. Nothing about the model required that. Step 7's tile
had simply been sized for its own display.

| | |
| --- | --- |
| **step 7** | says **where to look**, and now also **what shape**. Its two gates turn a whole-slide grid into a few thousand squares, and it owns the overlap. |
| **step 8** | says **what the squares mean**: loads the checkpoint, runs it forward, writes a class per square. |

`build_grid` is still the join, and `inside` is the only thing it decides. With the two
aligned it reproduces step 7's grid exactly and its tissue gate removes nothing —
`test_the_two_grids_coincide_when_step_7_lays_the_models_window` and
`test_step_8s_window_gate_removes_nothing_once_step_7_gates_the_same_square` pin both.
It is not redundant: a caller naming an overlap on this step's own route still pulls the
grids apart, and the centre test is what keeps that case correct. A window is kept when
its **centre** lands on a kept tile rather than when it overlaps one, because overlap
would spill half a window past the section on every side.

`test_inside_matches_a_brute_force_centre_test` checks the fast marking against the rule
it implements, window by window — the fast version marks outward from a few thousand
kept tiles, the slow one asks the question directly of a few hundred thousand windows,
and they have to agree exactly.

### 3. Areas are counted per stride, never per span

Windows overlap, so their spans do not partition the tissue. `windows × span²` reports
**four times** the ground at 50 % overlap and makes it look as though overlapping
windows see more of the slide. They do not — they see the same tissue more often. The
stride-sized cores do partition it, one per window, so `grid.cell_mm2` uses the stride.
Step 7 draws the same distinction for the same reason.

## Cost, measured

One IHC slide (`CAN_00251_26_A`, 198 mm² of kept tissue) at 0.5 µm/px on four CPU
threads, measured:

| overlap | windows | rate | wall clock |
| --- | --- | --- | --- |
| 0 | 15,713 | ~10 /s | ~26 min *(partial run)* |
| 0.5 *(default — the guide's stride 112)* | 62,962 | 16.4 /s avg | **63.7 min, completed** |

The completed run is the honest headline, with one caveat attached: for roughly 35 of
those 64 minutes an unrelated training job (`pixel_unet_resnet18/scripts/02_train_seg.py`)
was competing for the same cores, and the instantaneous rate fell from ~30 windows/s to
~4 while it ran and recovered to ~18 afterwards. **On an otherwise idle machine expect
somewhere around 32–35 min**, from the 30–33 windows/s the pass held before the
contention started. Nothing here is worth tuning on the contended number.

**The window count is not the cost, and that is worth knowing before tuning the
overlap.** A block holds `block_windows` squared windows at any overlap, so at *no*
overlap it spans four times the area and reads about three times the pixels **per
window**. Four times the windows at 0.5 therefore does *not* cost four times the time —
the extra windows are substantially paid for by cheaper reads. The plain "overlap costs
quadratically" reasoning that holds for step 7's tiling does not hold here.

*(The two rows above are not a clean comparison of that claim — one was contended and
one was a partial run. The mechanism is measured and real; the exact ratio is not, and
should be re-measured on an idle machine before anyone leans on it.)*

Where the time goes, over dense blocks: **86 % the forward passes**, 7 % the
deconvolution, 4 % the reads, 3 % building the tensors. Raising `torch.set_num_threads`
does not help — measured 47.7 windows/s at 4 threads, 47.6 at 8, so 4 is already this
machine's ceiling and the pure-network figure is the floor the pass is approaching.

The reads are still more expensive than they look, because of the slide's own pyramid:
these scans are 0.222 µm/px at level 0 and their next level is a 4× downsample, so a
0.5 µm/px read has no level to come from and is taken from level 0 and area-averaged
down — five times the pixels off disk. That is step 5's `read_tile` rule (resample in
intensity space, before the logarithm, with area averaging) and it is correct; it is
just expensive here. Blocks are what keep it to one read per 64 windows.

**The guide's "roughly 45 minutes for six H&E slides on a CPU" is wrong by about an
order of magnitude** — it is closer to half an hour for one. This is why the step is a
background job whose result is cached on disk: unlike step 2's cache, this one is not an
optimisation, it is what makes the screen openable at all. It is also why the step can
be cancelled: half an hour is long enough that changing your mind has to be possible,
and the stop lands in about 1.4 seconds, measured.

### The pass paints itself while it runs

Half an hour is also long enough that a bar and a percentage are not enough. A bar says
how much is left; it does not say *what the model is doing*, and at this duration the
difference between "working" and "wedged" is the thing a viewer actually wants. So the
run has a second output while it runs: **the class of each window, as it is decided.**

    classify(..., painted=cb)      cb((row, col, class), ...) after each batch
    GET /run?paintedSince=N        the windows decided since N, as flat triples
    TissueTypePainting.tsx         draws them onto step 4's slide thumbnail

The screen accumulates them onto a canvas over a picture of the slide, so the class map
builds up in front of the viewer in the colours the finished `map` panel uses. Four
consequences worth knowing about.

**It is a readout, not an animation.** A square appears when the model has answered for
that window and not before, so the paint stalls when the machine is busy and crosses
glass faster than tumour. Those are properties of the run, and smoothing them away
would make this a decoration rather than a readout.

**`sweep` exists for it.** The blocks are visited left to right along one band, right to
left along the next, top to bottom — a ploughed field — so the painting reads as one
line of work crossing the section rather than jumping the width of the slide at every
band. It cannot change a label, because a block is an independent read and an
independent set of forward passes; `test_the_sweep_order_does_not_change_a_single_label`
pins that.

**The client computes the rectangles.** The run publishes the grid's geometry once —
`cols`, `rows`, `span`, `stride` and the slide's level-0 size — and the feed itself is
three integers per window. So a whole slide's feed is one pass over the grid rather than
a rectangle per window, and it needs no per-checkpoint case: the four published fields
of view differ only in the resolution a 224 px window is read at, and that arrives here
already converted to level-0 pixels. `stride` is what gets painted and never `span` —
`test_the_painted_grid_lands_on_the_windows_at_every_published_scale` checks that at all
four, and carries the copy of the client's own formula that stops the TypeScript and
`WindowGrid.x_of` drifting apart.

**It makes the cancel button useful rather than only available.** An obviously wrong
class map — the false in-situ field of the source confound, say — is visible in the
first minute instead of after thirty.

## Licences — **the default checkpoint is RESEARCH ONLY**

| Checkpoint | Trained on | Licence | Ships |
| --- | --- | --- | --- |
| `invasive_tile_v1_imagenet_std` *(default)* | BCSS + BRACS regions labelled by BEETLE | **research only** | ❌ |
| `invasive_tile_v1_simclr_std` | the same, SimCLR backbone | **research only** | ❌ |
| `invasive_tile_v1_imagenet` | BCSS alone | CC0 + BSD-3 | ✅ |

The default is the model this step was asked to serve, and it is the only one that can
separate in-situ from invasive disease — BCSS supplies **nine** non-invasive epithelium
tiles in its entire 151-region release and **none** in its held-out institutions, so
that class could be neither fitted nor measured from it alone. The in-situ tiles come
from BRACS (non-commercial) labelled by BEETLE's nnU-Net (CC BY-NC-SA), and the model
inherits both terms — as does every number computed from it.

**The licence track is data, not this paragraph.** `model._licence_track` reads it out of
the manifest, restrictive wins, `Pinned.licence_track` carries it, and it reaches the
screen as a blocking caveat. A checkpoint that cannot be told apart from the permissive
one by looking at it is how a non-commercial dependency reaches a client.

## What this model cannot do

Reported on screen as structured caveats rather than left in a docstring, because the
load-bearing one is a *number*:

- **the in-situ boundary is tested against nine pathologist-drawn tiles.** The rest of
  that class's training and test data is another model's opinion, unreviewed. Rule 5
  gates the whole score on exactly this boundary, so it is the number to be most careful
  about. `held_out.by_source` keeps the human-labelled and model-labelled sources apart
  and never averages them.
- **it was trained on H&E and is served on immunostained sections.** The two are made
  comparable by dropping the DAB and keeping the haematoxylin, but the haematoxylin
  itself differs between them in distribution *shape* rather than strength — a 23×
  spread across percentiles. No monotone correction can close that (a scale multiplies
  every quantile equally and cannot change a ratio), which is why the fix is a
  training-time `gamma` term plus per-tile standardisation rather than a per-slide
  normalisation here. There is no in-domain test set: nobody has drawn class boundaries
  on an immunostained breast section for this project yet.

## The fourth colour: "cannot be determined"

`uncertainty.py`. A post-pass over the finished class map that recolours class-1
windows it will not stand behind. It exists because the head has three logits and no
way to decline, so class 1 — the least well supervised of the three — absorbs whatever
looks like neither of the other two. The OncoStem `CAN_00270` measurement (§ below and
the project notes) ends by asking for exactly this: *a two-stage design that emits
"cannot determine" instead of forcing a coin flip into the denominator*.

**It cannot move the score, by construction.** The candidates are class-1 windows,
which Rule 5 excludes before they are looked at. `labels` in `classmap.npz` is written
untouched, step 9 reads that array, and `test_flagging_cannot_move_the_score` pins it.

### Why the obvious rule does not work

Three formulations look right and are each wrong against a slide this project measured.
They are recorded here because all three are the first thing anyone reaches for.

| formulation | fails on | measured |
| --- | --- | --- |
| flag where the model was **unsure** | the v2 false in-situ field on `CAN_00251_26_H&E` | 45.9% of the section called in-situ at a **median margin of 0.756**; only 5.2% near-ties. Confidently wrong. |
| flag where the **neighbourhood disagrees** | the same field | it is one enormous coherent region, so every window's neighbours agree with it. Disagreement is near zero exactly where the error is worst. |
| flag where the neighbourhood is **non-epithelium** | every real duct | a duct is a small object embedded in stroma. The reconstruction of the 29-window duct component from `CAN_00270` scores **0.820** under this and is flagged outright. |

So the rule is an **or** of a statistical route and a morphological one, not an `and` of
two statistical ones. An `and` requires every route to fire at once, which is what makes
a flagging rule silent on the cases that motivated it.

### The rule

```
M = 1 - (p[c] - max p[k != c])              margin against the runner-up to the
                                            ASSIGNED label - which under `tau` is
                                            not the argmax

D = Pbar[invasive] / (Pbar[in-situ] + Pbar[invasive])
                                            the rival's share of the local EPITHELIAL
                                            evidence. Pbar is a Gaussian-weighted,
                                            normalised, self-excluded neighbourhood
                                            posterior. Non-epithelium is deliberately
                                            not in that denominator.

G = max(E, S)   per 8-connected component of in-situ windows
    E = clip(log(area / area_ref) / log(area_max / area_ref), 0, 1)
    S = clip((bbox_fill - fill_lo) / (fill_hi - fill_lo), 0, 1)   if area >= min_shape_mm2

W = sqrt(M * D)                             geometric mean, so a threshold reads on the
                                            same scale as its factors
U = 1 - (1 - W) * (1 - G)                   noisy-or: either route alone can flag

unknown  ⟺  U >= threshold
```

Four details that are bugs if got wrong, each with a test:

- **`sigma` is in microns, not windows.** The stride halves between overlap 0 and 0.5,
  so a neighbourhood fixed in cells would change physical size with a parameter meant
  only to change sampling density. `test_the_neighbourhood_is_a_distance_and_not_a_count_of_windows`.
- **The convolution is normalised.** Numerator and denominator smoothed over the same
  mask, or every window on the tissue rim reads as unsupported — the rim artefact of
  [§ The rim](#the-rim-found-explained-fixed), reintroduced from the other end.
- **An isolated window is `D = 0`, not `D = 1`.** On `CAN_00270`, 72% of the class-1
  components outside the DCIS contour were singletons a median 6.3 cells from invasive
  tissue, read as normal ducts and lobules. Scoring isolation as uncertainty paints
  every normal lobule purple.
- **`S` is gated on an area, not a window count.** A duct traced along its length is
  hollow; the same duct cut across is a solid disc. A solid 4×4 block of windows is
  0.2 mm², a small comedo-type focus — the first version gated on `|R| >= 12` and
  flagged it at 1.0.

### What it cannot do

It qualifies what the model **did** call in-situ. It cannot surface what the model
missed: on `CAN_00270` the DCIS the head failed to find is labelled class 0 or 2, is
never a candidate, and stays exactly as wrong as it was. **A slide with no purple on it
is not a slide that has been checked**, and the report says so rather than letting an
absence of flags read as a clean bill.

### Tuning

`area_ref_mm2` and `area_max_mm2` are the softest numbers in the module — extensive
DCIS is genuinely a multi-millimetre thing. They are deliberately permissive: the 141
mm² false field clears the ceiling seven times over, so nothing subtle rests on where
between 2 and 20 the ramp sits. Tune `threshold` first.

The settings are in the run's cache key, so changing one re-runs the pass. That is
correct — a cached report and its rendered panels genuinely do not describe what the new
settings would produce — but it is also half an hour per tweak. The layer is a pure
function of `classmap.npz`, so re-deriving without re-running the model is a real
optimisation available if tuning becomes routine; **dropping the settings from the cache
key is not the fix**, it would serve panels that disagree with the report beside them.

## Gates

| | |
| --- | --- |
| **G6** | a fresh process reproduces the manifest's own probe logits — `scripts/check_tissue_model.py`. **Passing**, worst difference 2.4e-06 over 21 tiles. |
| **G7a** | a full slide scores end to end. **Passing** — 62,962 windows, 1,241 pyramid reads, 2,247 batches, completed and cached. |
| **G7b** | window (row, col) really covers the pixels it claims to — `scripts/check_tissue_geometry.py`. **Passing**, worst probability difference **2.68e-07** over 32 windows re-read one at a time with the block machinery bypassed, clamped last row and column included. This is the check that would catch an offset-by-one-stride in `plan_block`, which is invisible in every other signal: the map would still look like a map and the tumour share would still be a plausible number, it would just describe the tissue half a window away. |

## ⚠ The class map on our own IHC slide is not believable

The implementation is verified; **the model's answer on this slide is not usable**, and
that has to be said next to the numbers rather than discovered later.

On `CAN_00251_26_A`:

| class | windows | share | area | confidence |
| --- | --- | --- | --- | --- |
| non-epithelium | 57,368 | **91.1 %** | 179.9 mm² | 0.91 |
| non-invasive epithelium | 5,505 | 8.7 % | 17.3 mm² | 0.62 |
| **invasive epithelium** | **89** | **0.1 %** | **0.28 mm²** | **0.54** |

**0.1 % invasive on a breast cancer case is not a plausible reading**, and 91 %
non-epithelium is high for a tumour block. The shape of the error is diagnostic: the
model is *confident* about "not tumour" (0.91) and barely choosing at all between the
two epithelium classes (0.62 and 0.54, against a three-class floor of 0.33). That is the
signature of a domain shift, not of biology.

It is also consistent with what the checkpoint's **own manifest** already records:
`metrics.held_out.by_source.bracs_dcis.invasive_recall` is **0.179**. The model's
invasive recall collapses the moment it leaves BCSS, on a second *H&E* dataset — and
this slide is a third domain again, immunostained. Nothing about this run contradicts
the model; it reproduces a weakness that was measured before it was deployed.

**So: do not quote the tumour share from this step yet.** What is needed is an in-domain
test set — a few hundred windows on an immunostained section with classes clicked by a
pathologist. Until that exists this step is verified plumbing around a model whose
accuracy on the slides it is actually served has never been measured, and the screen says
exactly that in its caveats.
## The rim: found, explained, fixed

The first full run put the in-situ class in a **one-window-wide rim around the entire
section** and around every internal void, rather than on duct-shaped structures. In-situ
epithelium does not outline a specimen, so this was a false positive with a cause worth
chasing.

**Measured, from the cached class map rather than by re-running:** each window's tissue
fraction against step 3's own mask, then the class shares as a function of it.

| class | median tissue in its windows | share under 50 % tissue |
| --- | --- | --- |
| non-epithelium | **100 %** | 16 % |
| **in-situ** | **11 %** | **69 %** |
| invasive | 37 % | 56 % |

**The typical window called in-situ was 89 % empty glass.**

**Why it happened.** Step 7's tissue gate was 0.10 - deliberately generous, because
"step 8 needs all the tissue" - and its tiles were 512 px against this step's 224 px. A
window whose *centre* landed in a kept tile could itself be almost entirely glass. Both
halves of that are now gone: one grid, and one gate at 0.50 held by step 7. The gate
here remains for the case where a caller names its own overlap and the grids come
apart.

**And why the model does something rather than nothing with such a window:** it has never
seen one. Checked directly in the tile manifest - BRACS class-1 training tiles have a
median haematoxylin density of 0.225 (p10 0.143), against 0.291 for BCSS invasive, and
the exporter's `min_usable = 0.7` means every training tile is at least 70 % annotated.
Only 1.7 % are near-empty. So the model never learned "glass implies in-situ"; a
mostly-glass window is simply **out of distribution**, its behaviour there is undefined,
and it happens to land on class 1.

**That is why the fix is a gate and not a retrain.** Retraining cannot teach a model to
handle an input the training set has none of, and there is no class for "glass" - glass
is not a tissue type. The fix is to stop showing the model inputs it was never trained
on. Hence `tissue_type_min_tissue_share`, this step's own window-level gate.

**Result on the reference slide**, everything else held constant:

| | before the gate | after |
| --- | --- | --- |
| windows classified | 62,962 | 48,968 (78 %) |
| non-epithelium | 88.4 % | 95.3 % |
| **in-situ** | **11.4 %** | **4.5 %** |
| invasive | 0.2 % | 0.1 % |
| mean confidence | 88.1 % | **91.2 %** |

The rim is gone from `flat.png` and what in-situ remains sits in interior patches, which
is what the class should look like. Confidence rose because the windows the model was
least sure about were the empty ones.

**What the gate does not do is find the tumour.** Invasive stays at 0.1-0.2 % at every
gate setting. Removing a false positive is not the same as fixing a false negative, and
the invasive class not firing on an immunostained section remains the open problem - see
the caveats above. It is a domain problem, and no amount of gating or relabelling
addresses it.

### And the picture says something the numbers do not

`flat.png` from this run is worth opening. Three things in it:

1. **The map traces the section's shape correctly** — which is independent
   corroboration of G7b. A broken addressing would not produce an outline that follows
   the specimen.
2. **Almost the whole section is one class**, non-epithelium, as the 91 % says.
3. **The in-situ class forms a one-window-wide rim around the entire perimeter** and
   around every internal void, rather than sitting on duct-shaped structures.

**That rim is the actual bug signature, and it is not domain shift in general — it is
specific and it is actionable.** In-situ epithelium does not outline a specimen. What
outlines a specimen is *the tissue boundary*, so class 1 is being triggered by
tissue-meets-glass rather than by duct architecture. Two plausible contributors, both
checkable:

- **The borrowed class-1 training data is made of ROI crops.** The in-situ tiles come
  from BRACS regions of interest — extracted rectangles — so their edge tiles carry an
  artificial image boundary too. A model can learn "boundary ⇒ in-situ" from that and
  never be penalised for it on held-out crops, because they have the same artefact.
- **Per-tile `standardise` amplifies edge windows.** A window that is mostly glass has a
  low p99, so standardising multiplies what little signal it has up to the target —
  inventing texture where the slide has none. `STANDARDISE_FLOOR = 0.10` exists to stop
  exactly this, but a window with 10 % tissue on it clears that floor easily, and step
  7's tissue gate is deliberately set to **10 %** because step 8 "needs all the tissue".

**The cheapest next experiment** is therefore not more training: it is to re-run this
step counting only windows above a much higher tissue fraction, and see whether the rim
disappears and the class shares change. If they do, the fix is a step-8-specific tissue
gate — the current one was chosen for step 7's purposes, and this step turns out to need
a stricter one. That is a settings change and an afternoon, not a data-collection
project.
