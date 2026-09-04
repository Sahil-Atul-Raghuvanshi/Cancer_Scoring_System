# `pixel_unet_resnet18/` — the same three classes, per pixel

A **second, standalone pipeline**. Same H-channel input, same 224 px geometry, same three
classes as [`../bcss_bracs_hchannel_resnet18`](../bcss_bracs_hchannel_resnet18) — but a U-Net decoder on
the ResNet18, so the output is a label **per pixel** instead of one label per tile.

| | tile model (approach 1) | this |
| --- | --- | --- |
| output | one label per 224 px tile | one label per pixel |
| boundary error | up to ±112 px | a few px |
| trained on | tile labels from a majority vote | the pixel masks that vote discards |
| params | 11.2 M | 14.3 M |
| licence | CC0 + BSD-3, **ships** | CC BY-NC-SA, **research only** |

---

## It shares no code with approach 1

Not one import. `moco_init_resnet18` and `beetle_teacher_resnet18` both put approach 1's
`src/` on `sys.path` — correctly, because they are *variations* on it: same tiles, same
splits, same head. This is a different model with a different output, and it must be
runnable, publishable and movable without approach 1 being present or correct.

What is shared is the **data**, read-only:

```
bcss_bracs_hchannel_resnet18/data/bcss/{images,masks}/   150 regions, pathologist-drawn masks
bcss_bracs_hchannel_resnet18/data/dcis/{images,masks}/   120 regions, BEETLE-drawn masks
```

Nothing here writes into those directories, and approach 1's `data/tiles/` is deliberately
**not** read — its tiles carry no masks and were filtered by a vote this model must not
inherit.

### The cost of independence, and how it is paid

Approach 1's exporter warns in its own docstring:

> "A second exporter is exactly where a stray `skimage.color.rgb2hed` gets introduced, and
> the failure looks like a model problem for a week."

That warning is right, and this folder wrote a second transform anyway. Three things make
it safe rather than hopeful:

1. **The stain matrix is a hard-coded constant, not a re-derivation.** Two implementations
   of "complete the basis with a cross product" could differ in sign or normalisation and
   nothing would raise. `hstain.RUIFROK_HDAB` is the resolved 3×3 written to full double
   precision, so there is no arithmetic to get wrong.
2. **`tests/test_parity.py` proves the equivalence.** It imports approach 1 *in the test
   only*, skips when it is absent, and asserts bit-for-bit equality of the white point, the
   haematoxylin channel, quantise/dequantise, `to_model_input` across every
   standardise/gamma/invert combination, the resampling, the 22-code mapping, the barcode
   parsing, and even the augmentation draw order. **All 11 pass.**
3. Every constant that had to be copied is asserted equal, not assumed equal.

---

## Why per-pixel is worth a second pipeline

Three things measured while planning, two of which correct assumptions carried over from
the tile work:

| measured | consequence |
| --- | --- |
| The tile exporter keeps 13,501 of 20,278 positions — dropping **2,671 as "mixed"** and 4,106 as "unusable". | A tile with no majority is a tile containing a **boundary**. The vote exists to reject exactly what a segmentation model needs, so this pipeline keeps every position above a low labelled-share floor. |
| Real human class-1 is **4.75 M pixels across 26 regions**, not "9 tiles". | The 9-tile figure was an artefact of the 50 % vote threshold discarding every DCIS focus too small to fill half a tile. At pixel level none of it is lost. |
| **0.38 M of those pixels, in 7 regions, are already in the published held-out institutions** (E2, EW, GM, LL). | DCIS boundaries are **measurable against a pathologist**, on four hospitals the model never sees — so this pipeline uses BCSS's published split unmodified. The tile model had to move three whole slides across because its held-out set had no class 1 at all. |

---

## Architecture

```
input 3 x 224 x 224          (H-channel replicated, ImageNet-normalised — same as approach 1)
  conv1+bn+relu  ->  64 x 112  ────────────────────────┐   frozen
  maxpool+layer1 ->  64 x  56  ───────────────┐        │   frozen
  layer2         -> 128 x  28  ──────┐        │        │   frozen
  layer3         -> 256 x  14  ─┐    │        │        │   trains
  layer4         -> 512 x   7   │    │        │        │   trains
  up 512->256 + skip ───────────┘    │        │        │
  up 256->128 + skip ────────────────┘        │        │
  up 128-> 64 + skip ─────────────────────────┘        │
  up  64-> 32 + skip ──────────────────────────────────┘
  up  32-> 16
  1x1 conv -> 3 x 224 x 224
```

**Why skips at all.** `layer4`'s 7×7 map knows *what* is in the tile but has thrown away
*where* to within 32 px. The boundary information lives in the early layers at 112 and
56 px, and the skips are the only route by which it reaches the output. Without them you
get the blocky output this model exists to replace, just at 32 px instead of 224.

**Why `conv1`/`layer1`/`layer2` are frozen.** They are edge and texture detectors ImageNet
already fits well and 20,000 tiles will not improve, and they sit at the highest
resolutions — so most of the backward-pass cost is there. Freezing roughly halves it.

**Bilinear upsample plus 3×3, not a transposed convolution.** A transposed convolution with
an even kernel and stride 2 produces checkerboard artefacts, and on a boundary task the
artefact lands exactly where the answer is read.

---

## Everything is idempotent

Re-running any step is the normal case, not the exception.

**Export.** Every region writes one JSON shard as its *last* act, after its tiles are on
disk. A region is skipped only when its shard parses, its `settings_fingerprint` matches
the current geometry and floor, **and** every tile file it names still exists. Tiles are
written to a `.partial` name and `os.replace`d, which is atomic on Windows and POSIX alike,
so a kill mid-write leaves nothing for the next run to trust. The combined manifest is
**rebuilt from the shards** rather than accumulated in memory, so a resumed export produces
byte-identical output to an uninterrupted one.

Change any geometry setting and the fingerprint changes, which invalidates every shard —
correctly, because tiles cut at a different resolution are not the tiles the new spec
describes. Stale shards are *ignored*, never mixed in, and the count is reported.

**Training.** A checkpoint lands after every epoch, atomically, carrying the model, the
optimiser state, the epoch, the history, and a fingerprint of the recipe *and* the exact
tile set. A later run resumes only when that fingerprint matches — a checkpoint from a
different tile set or learning rate is not a checkpoint of this run, and continuing from it
would give the model a recorded history that is fiction. Once the epochs are done, running
again just re-scores and rewrites the report.

**Publish.** Recomputes the SHA-256 from the file each time, and refuses to pin a
checkpoint that has no matching report or that stopped early.

---

## Run order

```bash
PY=../Breast_Cancer_IHC_Tissue_Scoring_Demo/backend/.venv/Scripts/python.exe

$PY -m pytest tests/ -q                        # 31 tests, ~8 s, no tiles needed
$PY scripts/01_export_seg_tiles.py             # ~20 min, ~900 MB. Re-runnable.
$PY scripts/02_train_seg.py --smoke            # 2 epochs on 200 tiles; must drive loss down
$PY scripts/02_train_seg.py                    # the real run. Resumable.
$PY scripts/03_publish_seg.py                  # pin + reload check
```

`--bcss-only` on the export drops the BEETLE-labelled regions: the result is CC0/BSD-3 and
shippable, at the cost of almost all the class-1 signal.

---

## What the numbers mean

The report keeps the two label sources apart and **never averages them**:

- **`bcss`** — pathologist-drawn. This block answers *is the model right*. It is the number
  that leaves this pipeline.
- **`bracs_dcis`** — BEETLE's predictions, reviewed by nobody. This block answers *does it
  agree with BEETLE*, which is a different and weaker question.

Per class: Dice, IoU, recall, pixel count, and **boundary F1** on a 3 px trimap.

**Boundary F1 is the metric area Dice cannot be.** A shape dilated by six pixels keeps a
Dice above 0.6 while its boundary F1 collapses below 0.2 — and a slightly-too-large region
is exactly the error a tile model makes. `tests/test_pipeline.py` asserts that gap, so the
metric is known to have teeth before any real result is read.

---

## Which source may teach which class

**BRACS teaches in-situ epithelium. BCSS teaches the other two.** The rule is
`classes.SOURCE_CLASSES`; the pixels it disallows become `IGNORE`.

BRACS regions were selected *because* they are DCIS-heavy, and it shows — BRACS supplies
**98.8 %** of this export's in-situ pixels, 89.5 M against BCSS's 1.09 M. But the same
selection also drags in 58.0 M non-epithelium and 16.0 M invasive pixels, **all of them
BEETLE's predictions rather than anybody's judgement**, against 385 M and 267 M drawn by
hand in BCSS. A DCIS-selected, model-labelled minority should not get a vote on what
invasive carcinoma looks like: it is a biased sample of the class its labeller is worst
at.

There is no conflict to arbitrate. The two sources are *different images*, so no pixel
carries both a human's label and BEETLE's — the only question is which source is
authoritative for which class, and the table is the answer.

### The pixel form of this rule is better than the tile form

Approach 1 has to drop a whole tile whose label it distrusts, losing everything else on
it. Here the tile stays. A BRACS tile holding a duct in stroma keeps teaching in-situ
epithelium; only its stroma pixels stop being a statement about stroma. They become
`IGNORE`, which the loss already skips (`ignore_index`), `segtrain`'s Dice already zeroes,
and `segreport` already excludes from every count. **Nothing new had to be built to absorb
them**, which is the sign the rule is being expressed in the right place.

Applied at **load**, not at export — `--all-sources` turns it off without a re-export, and
it is in the `Recipe`, so it is part of the resume fingerprint: a run started under one
setting cannot silently resume under the other.

`splits.pixel_class_weights` and `segdata.pixel_census` count only the pixels that will
actually supervise. Weighting against counts the loss never sees would balance the model
against supervision that is not there, and the gap is 74 M pixels.

### It makes the confound worse, on purpose — so run the leakage check

Once BRACS may only teach class 1, nothing is left teaching the model that a BRACS image
can be anything *else*. `source` becomes very nearly a synonym for the class this project
exists to find. `scripts/04_leakage_check.py` measures it, and the two must be reported
together:

```
source is predictable from pooled encoder features at 0.821   (0.5 is chance)
in-situ separability, features as they are : 0.793
in-situ separability, source direction gone: 0.708
source separability after removal          : 0.684
INCONCLUSIVE - the projection did not remove the source signal
```

Read that as an upper bound on how far the in-situ number can be trusted, not as a
verdict. The projection removes one direction; the dataset fingerprint is spread across
many, so "in-situ survived it" proves nothing either way.

## What did *not* carry over from the tile model, and why

Approach 1 grew a **window-level tissue gate** (`tissue_type_min_tissue_share`) after its
in-situ class turned out to be firing on a rim of mostly-glass windows around the section
— the median window it called in-situ was 89 % glass. That fix does **not** belong here,
and the reason is worth writing down because the analogy is tempting.

A tile model sees one *window* and answers once, so a window that is nine tenths glass is
an input its training set contains none of — out of distribution, behaviour undefined. A
pixel model answers per pixel, and **a pale pixel is not necessarily glass**. Fat is pale.
Lumen is pale. Both are inside the section and both are things Rule 1 requires this step
to classify rather than skip.

Measured on this export, per-class haematoxylin density of the pixels each class is taught
on:

| class | median | share under 0.05 OD |
| --- | --- | --- |
| non-epithelium | 0.159 | 20.0 % |
| in-situ | 0.212 | 20.5 % |
| invasive | 0.259 | 8.8 % |
| `IGNORE`, incl. blanked | 0.118 | 32.4 % |
| *BRACS non-epithelium (blanked by the rule)* | *0.112* | *32.9 %* |

Two things follow. First, **the source-authority rule already removes the glassiest
supervision** without being asked to: BRACS's non-epithelium was the palest thing in the
set. Second, of the human-labelled non-epithelium that remains, the median tile has only
**6.8 %** of that class near-glass and just 2.4 % of tiles are more than half so — those
are fat and lumina, correctly labelled by a pathologist inside an annotated region.

**A density gate on training would therefore delete fat**, which is precisely the tissue
Rule 1 says this step must recognise in order to remove it from the denominator. So there
is no training-side gate here.

The tile model's lesson still applies at **inference**, where it has not been needed yet
because this model has no serving path: a pixel outside the tissue mask should not be
given a class. When step 8 grows a pixel option, that guard belongs there — on the output,
against step 3's mask — and not in the loss.

## Honest limits

- **0.38 M px of held-out human class-1 is a thin test set.** Seven regions from four
  hospitals — a measurement rather than an anecdote, but the confidence interval on that
  Dice is wide, and any report must say so.
- DCIS boundaries are **trained** mostly from BEETLE's masks, whose own boundaries were
  machine-drawn and human-corrected. The held-out BCSS pixels are what keep the
  **evaluation** honest; the training labels remain largely a model's.
- **This model is research only.** CC BY-NC-SA via BEETLE, non-commercial via BRACS. The
  published name carries `_nc` so it cannot be mistaken for approach 1's shippable
  `invasive_tile_v1_imagenet.pt`, which is untouched by any of this.
- CPU training is the real cost, and it is **not** the ~10 h first estimated — see
  `../PIXEL_SEGMENTATION_PLAN.md` for the measured rate. A full-resolution U-Net
  forward-and-backward is far heavier than the tile model's cached-feature fit.

## What lives where

```
src/paths.py       the only module that knows a path. Shared data in, our data out.
src/classes.py     22 BCSS codes -> 3 classes + IGNORE. Asserted total at import.
src/hstain.py      white point, deconvolution, quantise, to_model_input. Parity-tested.
src/augment.py     one geometry draw applied to BOTH image and mask; epoch-varying.
src/splits.py      BCSS by institution, BRACS by hashed patient. Per-PIXEL class weights.
src/segexport.py   (image, mask) pairs, sharded and idempotent.
src/segnet.py      ResNet18 + U-Net decoder, publish/load with an arch check.
src/segtrain.py    CE(ignore_index) + soft Dice, resumable after every epoch.
src/segreport.py   per-pixel Dice/IoU + boundary F1, sources kept apart.
```
