# Implementing Each Approach — Step by Step

> Created: 3 Sep 2026 | Audience: whoever writes the code, sitting down to start.
> Companion to [`segmentation-approaches-all-seven.md`](segmentation-approaches-all-seven.md)
> (time and storage), [`segmentation-approaches-comparison.md`](segmentation-approaches-comparison.md)
> (why approaches 1–4 exist) and [`segmentation-approaches-ranked.md`](segmentation-approaches-ranked.md)
> (why 5–7 exist and how all seven rank).
>
> Approaches **1, 3, 4a and 4b already have full build plans** — this document gives their step
> sequence and defers to those for detail. Approaches **5, 6 and 7 have no plan anywhere**, so they
> are written out in full here, with code against the real APIs in `backend/app/`.
>
> Feeds: guide step 9 = **code step 8**, `backend/app/pipeline/step08_tissue_type_segmentation/`.

---

## Part 0 — Read this before starting any approach

Six of the seven approaches share the same four pieces of machinery. Writing them once, correctly, is
most of the work; each approach is then a short script on top.

### 0.1 — What steps 1–7 already give you

Steps 1–7 are implemented and run against real uploads. This is not a greenfield project — the input
contract is already written in code.

| Step | Module | What it hands you |
| --- | --- | --- |
| 1 | `app/ingestion/slide_reader.py` | `SlideReader(path)`, `.dimensions`, `.mpp`, `.best_level_for_mpp(target)`, `.read_region_pil(...)` |
| 3 | `app/services/tissue_service.py` | `tissue_service.footprint(upload_id)` — the tissue mask at ~2 µm/px |
| 4 | `app/services/calibration_service.py` | `calibration_service.white_point(upload_id)` → `WhiteReference` with `.field_for(x=, y=, size=, mpp=)` |
| 5 | `app/pipeline/step05_optical_density/tiles.py` | `read_tile(reader, x=, y=, target_mpp=, size=, base_mpp=)` → `Tile` with `.rgb` |
| 5 | `app/common/imaging.py` | `optical_density(intensity, white, floor=1.0)` |
| 6 | `app/common/stains.py` | `RUIFROK_HDAB`, `REFERENCE_BY_NAME`, `complete_basis(pair)` |
| 6 | `app/pipeline/step06_colour_deconvolution/deconvolution.py` | `separate(od, matrix)` → `Channels(haematoxylin, dab, residual)` |
| 7 | `app/services/tiling_service.py` | `tiling_service.tile_index(upload_id)` → `TileIndex` with `.tiles` of `Tile(x, y, span, kept, ...)` |

> **The rule that governs every approach below.** The exporter does **not** own a colour-deconvolution
> implementation. It imports `app.common.stains` and `step06...deconvolution.separate` — the same two
> things inference calls. A script that reaches for `skimage.color.rgb2hed` has silently created a
> second definition of "the H channel", and the failure will look like a model problem for a week
> before anyone finds the plumbing.

### 0.2 — ⚠ Gate 0: settle the tile geometry before writing anything

**The training plans and the running code disagree, and nobody can build until this is resolved.**

| Source | Tile size | At 0.5 µm/px that is | Tile area |
| --- | --- | --- | --- |
| Every training plan (approach 1 Part 1, 4a, 4b) | **224 px** | 112 µm | 0.0125 mm² |
| **`backend/app/core/config.py:62`, live** | **512 px** | 256 µm | 0.0655 mm² |

This is not cosmetic. `core/config.py` is explicit that the working resolution and tile size are one
decision for the whole pipeline:

> *"The tile size and the working resolution are deliberately not here. Both are step 1's … a second
> copy here would be a second answer to the same question, and a tile index at a resolution nothing
> else in the pipeline uses is a tile index for a different slide."*

So a model trained on 224 px tiles and served against a 512 px tile index is **serving a different
slide from the one it was trained on**. Consequences either way:

```
224 px:  8,000 tiles/slide,  287,000 across 36 slides,  31 h to embed all   (@ 0.39 s/tile)
512 px:  1,527 tiles/slide,   55,000 across 36 slides,   6 h to embed all
```

512 is **5.2× cheaper**, matches the live config, and gives more architectural context (256 µm shows a
whole duct); 224 gives 5× more training examples and a finer mask. Approach 1's own ablation **A4**
exists to settle this — *"if 512 wins, `settings.tile_size` goes back and the comments go with it."*

**Decide now, write it in one place, and have both training and inference read it from there.** If you
pick 224, `settings.tile_size` must change and steps 5 and 7 must be re-run. Do not pass `size=224` to
an exporter while leaving the config at 512.

### 0.3 — The shared H-channel functions: export stores density, load shapes it

**Two functions, and the split is load-bearing.** The stored tile is raw quantised *density*; the
shape terms are applied at **load**. That is why re-deciding them costs a feature re-cache (~1.5 h)
rather than a re-export of every tile. Put both in
`backend/app/pipeline/step08_tissue_type_segmentation/hchannel.py` and have the exporter, the training
loader and the runtime all import them.

```python
import numpy as np
from app.common.imaging import optical_density
from app.common.stains import RUIFROK_HDAB
from app.pipeline.step05_optical_density.tiles import read_tile
from app.pipeline.step06_colour_deconvolution.deconvolution import separate

OD_CLIP = 1.5              # density ceiling, and the divisor into [0, 1]
STANDARDISE_FLOOR = 0.10   # below this p99, pass the tile through unstretched

def h_channel_density(reader, white, *, x, y, size, target_mpp, base_mpp):
    """Raw haematoxylin DENSITY for one tile — this is what gets stored."""
    tile = read_tile(reader, x=x, y=y, target_mpp=target_mpp,
                     size=size, base_mpp=base_mpp)
    od = optical_density(
        tile.rgb.astype("float32"),
        white.field_for(x=tile.x, y=tile.y, size=tile.size, mpp=tile.mpp),
        floor=white.od_floor,
    )
    return separate(od, RUIFROK_HDAB).haematoxylin      # HxW float32, unclipped

def to_stored(h):
    """Density -> uint8 for disk, ~17.8 KB/tile as PNG."""
    return (np.clip(h / OD_CLIP, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

def to_model_input(stored, *, alpha=1.0, beta=0.0, gamma=1.0, standardise=True):
    """uint8 -> model tensor. `gamma` and `standardise` live HERE, not at export."""
    h = stored.astype(np.float32) / 255.0 * OD_CLIP     # back to density
    h = h * alpha + beta                                # in OD units, BEFORE the clip
    v = np.clip(h / OD_CLIP, 0.0, 1.0)                  # now in [0, 1]
    v = v ** gamma                                      # the SHAPE term, AFTER the clip
    if standardise:
        p99 = float(np.percentile(v, 99))
        if p99 >= STANDARDISE_FLOOR:                    # else: no stain to stretch
            v = np.clip(v / p99, 0.0, 1.0)
    return np.repeat(v[:, :, None], 3, axis=2)          # ResNet wants 3 channels
```

Three things this gets right that a hand-rolled version gets wrong: the white point is **per-slide and
possibly a vignette surface** (`field_for` returns whichever step 4 justified, and `optical_density`
broadcasts both), plus the **calibrated OD floor** — copy the call in `services/tiling_service.py:218`
exactly; the resample happens **in intensity space before the logarithm** inside `read_tile`; and the
stain vectors are **fixed**, never per-image.

> **Use `RUIFROK_HDAB` for H&E slides too, and do not "fix" this.** It looks wrong, and it was tested:
> rebuilding an **H-E** basis from the same `REFERENCE_BY_NAME` vectors moved the IHC/H&E channel-ratio
> gap from 23× to **18.8× — no better**. Taking `I₀` per tile instead of per slide was **worse** (36.7×).
> The live pipeline deconvolves everything with H-DAB. The gap is not a basis error.

### ⚠ 0.3b — Why `gamma` and `standardise` are not optional

The single most important finding in these documents, and the easiest to drop when re-implementing.

An IHC counterstain is **sparse and punchy** — three quarters of its tissue pixels near zero with a
hard dark tail — where H&E haematoxylin is **broad and mid-toned**. The H&E → IHC shift therefore
changes the *shape* of the distribution, not its strength: the ratio between two quantiles of one
image disagrees by 23×.

**No monotone correction can close that.** Scaling by any constant multiplies every quantile by that
constant, leaving their ratio unchanged — so widening `alpha`, and per-slide normalisation of **any**
kind, cannot work. Raising the normalised value to a power can.

| Jitter | Percentiles of the served IHC distribution inside the H&E training envelope |
| --- | --- |
| `alpha` only | **1 / 6** |
| `alpha` + `gamma` | 4 / 6 |
| `alpha` + `gamma` + `standardise` | **6 / 6 — G0b PASS** |

**1 of 6 means the model would have been extrapolating on nearly every tile it was ever served.**

```
alpha ~ U(0.80, 1.25)     # how strongly this slide was counterstained
beta  ~ U(-0.05, 0.05)    # OD units, before the clip
gamma ~ logU(0.5, 3.5)    # the SHAPE term, on the normalised value, after the clip
```

- **`gamma` after the clip, not before.** On a density it would also move the saturation point, making
  one knob do two jobs. On the `[0, 1]` value it is a pure shape change.
- **Log-uniform**, because `gamma` is an exponent: 2.0 and 0.5 are equal-sized changes in opposite
  directions, and a uniform draw would put most of its mass above 1.
- **`standardise` is an input property, not an augmentation.** It divides the tile by its own p99,
  closing the *scale* gap `alpha` cannot reach (H&E p99 0.50 vs IHC 1.03), and does nothing for shape.
  So it is set **identically on the training and inference paths** and recorded in the checkpoint
  descriptor, then checked at load.
- One cost of the uint8 store to know about: density is already clipped at `OD_CLIP` on disk, so
  `alpha > 1` saturates rather than revealing headroom. Accepted deliberately — the alternative is
  storing float32 tiles at 4× the size.

**Gate G0** — the H channel through this module equals step 6's own on the same tile.
**Gate G0b** — the served IHC distribution sits inside the training envelope at 6/6 percentiles.
**Neither approach below is worth writing until both are green.** Write
`tests/test_gamma_and_standardise.py` to hold the monotone argument as an executed test, not a comment.

### 0.4 — The shared feature cache

Frozen encoder, run once, results cached. Every approach except 2 and 6 uses this.

```python
import torch, torchvision, numpy as np

def encoder(init="imagenet"):
    if init == "imagenet":
        m = torchvision.models.resnet18(weights="IMAGENET1K_V1")   # BSD-3
    else:
        m = torchvision.models.resnet18(weights=None)
        m.load_state_dict(torch.load(MOCO_CKPT), strict=False)     # MIT, approach 3
    m.fc = torch.nn.Identity()
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m

@torch.no_grad()
def embed(model, tiles):          # tiles: NxHxWx3 float32 in [0, 1]
    x = torch.from_numpy(tiles).permute(0, 3, 1, 2)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return model(x).numpy()       # Nx512 float32
```

**Batch it.** The measured 0.39 s/tile is un-batched with a PNG decode per tile; batching is the
difference between a 6-hour job and a 31-hour one. Measure your own rate on 200 tiles before
committing to a schedule.

Cache as one `.npy` of shape `(N, 512)` plus a parallel manifest (tile id, slide, case, x, y). **The
manifest must carry the case id** — every split below is by case, never by tile.

### 0.5 — The shared head fit and the only honest split

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut

clf = LogisticRegression(max_iter=2000, class_weight="balanced")
cv  = LeaveOneGroupOut()          # groups = case id
```

- **`class_weight="balanced"` is not optional.** Without it the model answers "non-epithelium" every
  time, looks 80 % accurate, and is useless.
- **Split by case (or by source slide), never randomly by tile.** Tiles from one slide are heavily
  correlated; a random split reports a beautiful number that means nothing.
- **Report a 3 × 3 confusion matrix with the invasive ↔ non-invasive cell on its own**, never folded
  into an average. That one cell is the project.

---

## Part 1 — Approach 1: BCSS + AICAN *(plan exists — [`approach-1-training-plan.md`](approach-1-training-plan.md))*

**Step 1 — Check disk first.** `df -h /c`. Needs 10–20 GB peak against **27 GB free** today (the plan
says 33 GB; it is stale). Reclaim `data/slides/` first if needed.

**Step 2 — Download BCSS.** Clone `PathologyDataScience/BCSS`, run its download script (`--route
girder`, ~4 GB). Gets RGB H&E regions plus mask PNGs where each pixel is a class number.

**Step 3 — Read the raw label codes** from `meta/gtruth_codes.tsv`.
> ⚠️ **The trap that sinks this approach.** The widely-used "5-class" BCSS **merges DCIS into tumour**
> — the one merge this project cannot accept, and the default in most tutorials and in TIAToolbox's
> `fcn_resnet50_unet-bcss`. Use the **raw** codes where `dcis` (code 20) is separate. Also: `0
> outside_roi` is **not** class `0` — it carries zero weight, or you teach the model that unannotated
> glass is non-epithelium and inflate every number.

**Step 4 — (Optional) run AICAN on our 6 H&E slides.** `pip install pyfast`, then
`runPipeline --datahub breast-epithelium-segmentation --file <slide>`. Sample ~40 regions of 2048²
per case. **If pyFAST will not run on CPU, stop and skip it** — train on BCSS alone. You lose scanner
adaptation, not the pipeline. Do not spend three days on an installer.

**Step 5 — Export tiles** through §0.3, resampling BCSS from 0.25 to 0.5 µm/px, and label by majority
vote:
```
usable = fraction of pixels with a non-ignored label
if   usable < 0.70:            drop
elif invasive_frac    >= 0.50: label = 2
elif noninvasive_frac >= 0.50: label = 1
elif nonepi_frac      >= 0.80: label = 0
else:                          drop      # too mixed to learn from
```

**Step 6 — Cache features** (§0.4), **fit the head** (§0.5), with HED colour augmentation plus flips
and rotations. The HED jitter is what buys stain tolerance and lets you skip stain normalisation.

**Step 7 — Report**, then **delete the BCSS raw** once Notebook 02's checksums pass, before starting
anything else.

**The binding constraint, and say it in the report:** BCSS class `1` is 0.129 % of pixels with DCIS
from **one patient** — 237 : 1. Held-out BCSS **cannot** measure the invasive-vs-DCIS boundary because
it contains no DCIS. What it measures is invasive versus *normal ducts*.

---

## Part 2 — Approach 2: TIGER + BEETLE U-Net *(what you would do, and where you stop)*

1. Download TIGER (2.6 GB) and BEETLE (150.9 GB, ~300 GB peak) to an external disk.
2. Train an nnU-Net 5-fold on the merged label scheme. **Requires a GPU: 3–7 days on a card, months on
   CPU.**
3. Dense inference on the H&E: **~60–70 h per slide on CPU**, 40–80 min on a GPU.
4. VALIS-register the H&E to each IHC and transfer the mask (this half is approach 6 — Part 7).
5. Train a second IHC-specific model on the transferred masks.

**Stop at step 1.** TIGER is CC BY-NC and BEETLE CC BY-NC-SA — it cannot ship, and this feeds a test
that helps decide whether someone receives chemotherapy. Even with clean licences, steps 2 and 3 fail
on hardware grounds independently. **Keep it in `benchmarks/`, never imported by `backend/`, to learn
where the ceiling is.**

---

## Part 3 — Approach 3: MoCo initialisation *(plan exists — [`approach-3-training-plan.md`](approach-3-training-plan.md))*

1. Download Ciga & Martel's ResNet18 checkpoint (**46 MB, MIT**). It is **ResNet18**, not the
   commonly-cited ResNet50.
2. Change one line in §0.4: `encoder(init="moco")`.
3. **Re-run the feature pass** (~1.5 h) — the cache is init-specific, so it cannot be reused.
4. **Fit both heads and A/B them** on the same held-out split. Keep whichever wins.

That is the whole approach. It is ablation **A1** in approach 1's own list, ranked first there because
it is free. Expected gain: a few points, not a transformation.

---

## Part 4 — Approach 4a: BEETLE teacher, distilled *(plan exists — [`approach-4a-training-plan.md`](approach-4a-training-plan.md))*

**Step 1 — Notebook 00, the CPU smoke test. This is a hard gate.** Get the nnU-Net ensemble to predict
one 512² patch on CPU. If it cannot be made to run (Docker is the fallback), **stop** — Route A is
dead and you are choosing between 4b and everything else.

**Step 2 — Download `model.zip` (1.9 GB)**, pin its SHA-256. **Do not download the 147 GB images.**

**Step 3 — Sample 240 regions** of 2048² across the 6 H&E slides (~40 per case), spread over tissue.

**Step 4 — Run the teacher** on those regions: **6–18 h, overnight**. Measure on 3 regions first and
extrapolate before committing the night. Keep the per-model ensemble disagreement map — it is your
free uncertainty signal.

**Step 5 — Pathologist review, 30 regions, 2 hours of their time.** Judge the teacher on *whether it
correctly leaves DCIS out*, not on whether it finds tumour. Quantify the under-call; do not assume it.

**Step 6 — Export tiles** (§0.3) and **fit the head** (§0.5), with BCSS at **full weight** and the
pseudo-label rows at **lower loss weight**, tagged so they can be removed.

**Step 7 — Run the ablation.** Train with and without the pseudo-labels; compare on held-out BCSS. **If
they do not help, drop them.** Never assume the teacher was right.

> ⚠️ **Licence, unresolved.** CC BY-NC-SA 4.0, and ShareAlike arguably follows a distilled student into
> our own weights. Deprioritised by decision, not resolved. Must be closed before commercial release —
> and 4a and approach 1 cannot both sit on the 27 GB free at once, so sequence them.

---

## Part 5 — Approach 4b: BEETLE direct *(plan exists — [`approach-4b-training-plan.md`](approach-4b-training-plan.md))*

**Step 1 — Gate D0, and it is hard.** An external disk with **≥ 400 GB free**, mounted, writable, path
recorded in `data/PATHS.json`. There is no partial version of this route. **Never extract onto the
project drive "just for tiling"** — a half-extracted 147 GB archive on a disk with 27 GB free is the
most expensive failure available here, and it fails silently partway through a multi-hour job.

**Step 2 — Download** `images.zip` (147.2 GB, 3.5–13 h) with a **resumable** client (`curl -C -`).
Verify the archive checksum *before* extracting and per-slide checksums *after*.

**Step 3 — Build the splits from `data_overview.csv`**, and this is the best thing about this route:
- **R1** patient-level `GroupKFold` — model selection only, never quoted as a result.
- **R2** hold out an entire clinical centre.
- **R3** hold out an entire **scanner** — the honest proxy for our Morphle, which is not among the 7.
- **R4** hold out the 151 TCGA-BRCA slides (they *are* BCSS) — comparable with approach 1's numbers.

**Reproduce BEETLE's own degradation:** class `1` falls 0.83 internal → 0.65 external. **If R3 does not
degrade relative to R1, your split is wrong, not your model.**

**Step 4 — Export tiles** (6–10 h, disk-bound). Write the manifest as you go and **make the job
resumable**.

**Step 5 — Cache features** (~11.5 h, one overnight) and fit the head.

> **Why this is the worst-value route despite the best data:** 400 GB and six days against 4a's 1.9 GB
> and two — and dropping BCSS removes the **clean-licence fallback**, so every checkpoint it produces
> is non-commercial. **Written, not scheduled.**

---

## Part 6 — Approach 5: label ~900 of our own tiles *(no plan exists — full detail)*

The highest-ranked approach, and the only one that produces a test set on OncoStem tissue. Build it in
`step08_tissue_type_segmentation/` plus one `scripts/` entry for the clicker.

### Step 1 — Settle Gate 0 (§0.2) and write G0

Pick 224 or 512, put it in `core/config.py`, and assert the H channel matches step 6's on one tile.
Nothing below is meaningful until this passes.

### Step 2 — Build the sampler

For each of the 36 slides: get the tile index, keep only `kept` tiles, and take a **stratified random
sample of ~1,000**.

```python
from app.services.tiling_service import tiling_service

index = tiling_service.tile_index(upload_id)
kept  = [t for t in index.tiles if t.kept]
picks = rng.choice(len(kept), size=min(1000, len(kept)), replace=False)
```

> ⚠ **Balance across markers, not across slides.** N-Cadherin and Pan-Cadherin sit at 75–85 % positive.
> If they dominate, the classifier learns "lots of brown ⇒ tumour" and collapses on CD44. Draw an
> equal number per **marker**, and **keep 00267's near-blank CD44 slide** (17.5 mm² against 84.8 on its
> siblings) rather than dropping it as an outlier — it is the hardest slide we own and the one most
> likely to break in production.

### Step 3 — Export and embed the sample

Run each sampled tile through §0.3, then §0.4. **Subsample — do not embed everything:**

```
all 36 slides @ 224 px:  287,000 tiles × 0.39 s  =  31 h      ← do not do this
1,000/slide:              36,000 tiles × 0.39 s  =  3.9 h     ← one evening
(and at 512 px the full census is only 55,000 ≈ 6 h)
```

Store the H channel as uint8 PNG (~17.8 KB/tile, ~640 MB) plus the feature `.npy` (~74 MB).

### Step 4 — Cluster, so that clicking is fast

Clicking 900 random tiles is 900 decisions. Clicking 900 tiles **sorted by visual similarity** is
closer to 60, because whole screens are obviously all-fat or all-stroma.

```python
from sklearn.cluster import KMeans
labels = KMeans(n_clusters=60, n_init=10).fit_predict(features)
```

Draw ~15 tiles per cluster into 10 × 10 contact sheets.

### Step 5 — Build the clicker and click

A static HTML page over the contact sheets is enough — no server. Requirements:
- Click a tile to cycle `0 → 1 → 2 → unsure`; a key to apply one class to a **whole screen**.
- Colour-code by current class; show the tile's slide and case id on hover.
- Write out `{tile_id, case, slide, x, y, class}` as JSON/CSV, **saving incrementally**.
- **An `unsure` bucket is mandatory.** Forcing a guess on the boundary tiles is how you poison the one
  class that matters.

**60–90 minutes** for 900 tiles, most of it on the two or three screens holding the epithelium
boundary. Send those screens — about 40 tiles — to the pathologist. That is **30 minutes of their
time**, and it is the one thing you cannot self-serve.

### Step 6 — Fit and evaluate

Fit per §0.5 with `LeaveOneGroupOut` over the **6 cases**. Report the 3 × 3 confusion matrix per fold
and pooled, with the invasive ↔ non-invasive cell called out separately.

**Expect wide error bars — roughly ±0.08 on ~150 test tiles per fold.** Say so in the report. A number
with an honest interval beats a number without one.

### Step 7 — The step that is worth more than the model

**Re-score approaches 1, 3, 4a and 4b on these same clicked tiles.** They have never been evaluated on
OncoStem tissue; this is the first comparison between them that means anything. It costs one feature
pass per model and no new annotation.

### Step 8 — Back up the label file

Everything else regenerates in an evening. **Those 900 human decisions do not**, and they are worth
more than every model in these documents. Commit the label file; keep a copy off the machine.

### The limit to state plainly

**Gate zero governs class `1`.** If our six cases contain little or no DCIS, no amount of clicking
creates class-`1` examples — approach 5 then learns a two-class problem wearing a three-class head, and
the prior still has to come from BCSS, BEETLE or BRACS. **Approach 5 does not replace approaches 1/4;
it measures them.**

---

## Part 7 — Approach 6: register and propagate *(no plan exists — full detail)*

Not the shipped route. Build it as an **offline QA cross-check** and as approach 7's annotation
multiplier. New module `app/registration/`, never in the per-slide runtime path.

### Step 1 — Install VALIS and budget for the install being fiddly

MIT-licensed, CPU, but it pulls a JVM and Bio-Formats. **1–2 h, once.** Verify on one pair before
scripting 30.

### Step 2 — Produce a mask on the H&E

Any source: an approach 1/3/4 model, AICAN, or a hand-drawn QuPath outline (~20 min/slide). The warped
mask can never be better than this one, only worse.

### Step 3 — Register, per case

Register the H&E to each of its 5 IHC siblings — 5–20 min per case, ~1–2 h for all six. Save the
**transform parameters** (JSON/matrices, < 1 MB): cheap to store, expensive to recompute.

### Step 4 — ⚠ The confidence gate, which is the whole safety story

VALIS will return a transform for 00267's near-blank CD44 (17.5 mm² against 84.8) and **it will be
wrong, silently**. About 30 lines converts the worst failure mode into a visible one:

```python
ok = (matched_keypoints >= MIN_KEYPOINTS
      and residual_error_um <= MAX_RESIDUAL_UM
      and 0.5 <= tissue_area_ratio <= 2.0)
if not ok:
    raise RegistrationRefused(...)     # refuse and say why; never emit a mask
```

Calibrate the thresholds on the five well-behaved cases, then confirm 00267 **fails** them. If it
passes, the gate is too loose.

### Step 5 — Warp the mask and use it for one of three things

1. **The 6-case demo** — a mask on all 36 slides in half a day.
2. **Training labels** for an IHC model. ⚠ Registration errors become **invisible label noise** here —
   this is the objection the comparison doc raises against approach 2, and it carries over intact.
3. **The free QA cross-check**, which is where it genuinely earns its place: run the tissue model
   independently on the H&E and on the five IHC slides, register, and measure mask agreement. That
   needs **zero annotation** and directly tests the bet the whole H-channel design rests on — *does one
   model really behave the same on H&E and on IHC?* Six cases of that test are available today.

### The two hard limits

It needs a matched H&E in the same block, so it is **not a standalone-IHC product**. And serial
sections cut far enough apart can lose or gain a whole tumour focus — registration cannot invent what
is not on the slide. At 112–256 µm tiles the *geometric* error (tens of microns) is within existing
blockiness; the *content* error is not fixable.

---

## Part 8 — Approach 7: manual, then client data *(no plan exists — full detail)*

Last to finish, **first to start** — its clock belongs to someone else.

### Phase A — without the client (~3 days of ours + 2 pathologist hours)

**Step A1 — Outline the 6 H&E slides in QuPath.** Non-experts can confidently draw tumour bulk, fat,
stroma and glass. Leave the epithelium boundary loose and **flag it** rather than guessing.

**Step A2 — Propagate to the 30 IHC slides via approach 6.** One drawn slide labels six. This
multiplier is what makes a manual approach affordable at all.

**Step A3 — Pathologist reviews, does not draw.** Reviewing an outline costs a fraction of drawing one,
and it is the difference between "our best guess" and "clinically checked".

**Step A4 — Export via §0.3, fit via §0.5,** leave-one-case-out. Only the label source changed.

### Phase B — the client request (send this week; 4–12 weeks to return)

**Step B1 — Send the spec.** Half a day to write, and the longest lead time in the project. Full table
in [`segmentation-approaches-ranked.md`](segmentation-approaches-ranked.md) Part 3. The three lines
that matter more than the case count:

1. **Sparse fully-labelled ~2048² boxes, 10–20 per slide — not whole-slide outlines.** 20 min/slide
   instead of 3 h/slide, and worth the same for tile-classifier training. This framing is the
   difference between a request that gets fulfilled and one that does not.
2. **`in-situ / DCIS` as its own class.** Without it we have bought nothing — it is the exact failure
   that makes BCSS unusable for class `1`.
3. **10 cases double-read by two independent pathologists.** Two pathologists agree at roughly 0.85
   Dice on this boundary. **A model at 0.82 against a single reader may already be at human level, and
   today we have no way to know.** Cheapest item on the list, and it makes every other number in these
   documents interpretable.

Ask for GeoJSON in **level-0 coordinates** (or ImageScope XML), the same Morphle scanner, and confirm
**de-identification** before transfer — [`../handbook/admin-guide/de-identification.md`](../handbook/admin-guide/de-identification.md).

**Step B2 — Order the disk in the same week.** 50 cases × 830 MB ≈ **42 GB**; 100 cases × 2 slides ≈
**166 GB**; none deletable, and unlike BEETLE it **cannot be re-downloaded if lost**, so budget ~2× for
a backup. 27 GB free does not receive this. A 1 TB external disk also satisfies 4b's Gate D0.

**Step B3 — Write the annotation importer** while you wait: GeoJSON/XML → polygons → rasterised mask on
the level-0 grid → the same majority-vote labelling as §Part 1 Step 5. `app/annotations/`, sharing the
exporter with approach 5.

**Step B4 — QC the annotations before training on them.** Class balance per case, area per class,
polygon validity, and **inter-reader Dice on the 10 double-read cases**. That last number is the
ceiling every other number in this project should be read against.

**Step B5 — Retrain and re-measure everything.** Every estimate in every one of these documents becomes
a measurement on that day, and some will be wrong.

---

## Part 9 — Wiring any of them into step 8

Whichever approach wins, the runtime path is the same, and `step08_tissue_type_segmentation/pipeline.py`
currently raises `StepNotImplementedError`.

```python
def run(context: PipelineContext) -> StepResult:
    context.require("tissue-mask")           # step 3 — where to run
    context.require("colour-deconvolution")  # step 6 — what a tile's pixels ARE
    context.require("tiling")                # step 7 — the addresses
    ...
    return StepResult(stage_id="tissue-type-segmentation", output=report)
```

1. **Declare the requirements** above. Step 7's module docstring explains why the deconvolution one is
   not bookkeeping: the index is *addresses*, and those addresses resolve to the H channel.
2. **Load the checkpoint with a pinned SHA-256.** `app/qc/models.py` is this codebase's reference for
   loading a foreign checkpoint safely — follow it rather than inventing a second way.
3. **Predict per tile**, then smooth, threshold, apply morphology **in microns**, and **clamp to step
   3's tissue mask**.
4. **Report the region in mm²**, and flip `PipelineStage.implemented` to `True` — a stub module and the
   catalogue entry must agree.
5. **Import the same `channel.py` the exporter used.** This is where train/serve drift gets designed
   out, or designed in.

---

## Part 10 — The order to build in

| When | Do | Why now |
| --- | --- | --- |
| **First** | **Gate zero** — ask a pathologist how much DCIS is in our 6 cases | One afternoon; can retire two build plans |
| **First** | **Gate 0** — settle 224 vs 512 (§0.2) | Every exporter below depends on it |
| **Today** | **Send approach 7's annotation request** (Part 8, B1) | Longest lead time in the project; half a day to write |
| **Today** | **Order the external disk** | Approach 7 needs it, 4b's Gate D0 needs it |
| **This week** | **Approach 5** (Part 6) | 1 day, < 1 GB, and it is the test set everything else lacks |
| **This week** | **Approach 6 as a cross-check** (Part 7) | Half a day, and it needs no annotation |
| **In parallel** | **Approach 1 + 3** (Parts 1, 3) | The general-cancer prior 6 cases cannot supply |
| **Then** | Re-score 1/3/4 on approach 5's clicked tiles | The first number that means anything about OncoStem tissue |
| **Hold** | 4a on the licence; 4b on the disk; 2 in `benchmarks/` forever | |

### Three numbers to measure before trusting any schedule here

1. **The batched inference rate**, on one slide. 0.39 s/tile implies ~52 min/slide; the comparison doc
   quotes 5–8 min. **They disagree by 6×** and batching is almost certainly why — but nobody has
   measured it.
2. **Teacher inference on 3 regions**, before committing to 4a's "6–18 h" overnight.
3. **VALIS on one case**, before believing approach 6's half-day or its ~5 GB.
