# Swapping the frozen backbone for a pathology foundation model

How to replace the ResNet18 that produces step 8's 512-vectors with UNI, H-optimus-0 or
CONCH — what to change, what it costs, and what each possible result means.

**Status: not implemented.** This is the plan for the experiment that the evidence below
points at. Read the two "before you start" sections first; they are the difference
between a day's work and a month's.

---

## Why this experiment and not another

Three cheaper levers have now been tested properly, and none of them moved the number
that matters — the share of held-out in-situ tiles called invasive.

| lever | result | where |
| --- | --- | --- |
| **The label rule** (Fix 1) | 18.9 % vs v3's 17.6 % on **one** held-out set, paired McNemar **p = 0.056 — no separation** | `data/h_channel/112um/reports/09_head_to_head.json` |
| **The field of view** (224 / 448 / 672 µm) | no clean signal; the wider arms collapse to 73 and 11 held-out in-situ tiles, and a coarse window inflates measured area by 1.8× | `RESULTS_224_VS_448.md` |
| **Component aggregation** (Fix 3) | makes it *worse*. Median `p(in-situ)` among invasive-called windows is **0.006** — the classifier is confident, not incoherent, so there is nothing to aggregate | `backend/scripts/check_component_aggregation.py` |

Fix 1's real contribution was to **measurement**: it showed v3's published 12.6 % was
measured on a test set that excluded the solid and comedo DCIS the old rule deleted. On a
test set containing that morphology, v3 scores **17.6 %**. The problem was ~40 % worse
than the manifest claimed, and it is not caused by the label rule.

What has *not* been tested is whether the **features** encode the distinction at all. A
frozen ImageNet/SimCLR ResNet18 reading a haematoxylin-only 112 µm window is the weakest
remaining link. That is what this document is for.

---

## Before you start, 1: these models do not give you more context

**Every one of them is 224 × 224 pixels at 20× (~0.5 µm/px) = a 112 µm field** — exactly
the geometry the served v3 already uses.

So a foundation model gives **better features inside the same window**. It does not let
the model see a duct wall it currently cannot see. That matters because it splits the
hypothesis:

- if the failure is *feature quality at 112 µm* → this experiment fixes it;
- if the failure is *the window is too small* → this experiment will not, and feeding a
  20×-trained model a downsampled 448 µm field is out of distribution for it.

The field-of-view sweep was ambiguous about which, which is why feature quality is worth
testing next. But do not expect this to substitute for context.

> **One exception.** Virchow2 was trained multi-magnification. If you later want a wide
> *context* branch from a foundation model, that makes it a better base than H-optimus-0
> — a consideration that does not show up when ranking on single-scale accuracy.

## Before you start, 2: the serving cost is prohibitive on CPU

Measured on `CAN_00270_26_H&E` with the current two-ResNet18 concat, on CPU:

```
28,657 windows in 4,933 s  =  5.8 windows/s  =  82 minutes per slide
```

Scaling by forward-pass FLOPs (two ResNet18s ≈ 3.6 GFLOPs):

| backbone | ~GFLOPs | ratio | per slide, CPU |
| --- | --- | --- | --- |
| ResNet18 concat (today) | 3.6 | 1× | **82 min** |
| CONCH (ViT-B/16) | ~34 | ~9× | ~12 h |
| UNI (ViT-L/16) | ~120 | ~33× | ~45 h |
| H-optimus-0 (ViT-g/14) | ~560 | ~155× | **~9 days** |

Order-of-magnitude, but the conclusion is not marginal: **no foundation model belongs in
step 8's serving path on this hardware.**

Training and serving are therefore different problems:

- **Training** is one-off and cacheable. 32,030 tiles through H-optimus on a rented GPU
  is well under an hour, and only the `.npz` comes home. Affordable today.
- **Serving** is per slide, forever, on the demo box. A ViT there needs a GPU in the
  machine, or a distilled student.

**Plan the experiment as a training-only measurement.** A positive result creates a
hardware decision; it is not a drop-in upgrade. Knowing that before renting the GPU is
the point of this section.

---

## What is installed, and what each model needs

| | needs | present in `backend/.venv`? |
| --- | --- | --- |
| **UNI** | `timm`, `huggingface-hub` | **yes** — timm 1.0.28, hub 1.28.0 |
| **H-optimus-0** | `timm`, `huggingface-hub` | **yes** |
| **CONCH** | `transformers`, `open_clip_torch`, `einops`, the `conch` package | **no — none of them** |

> **Install CONCH in a separate virtualenv and hand the embeddings over as an `.npz`.**
> `backend/requirements-qc.txt` records torch ≥ 2.9 unpacking a licence tree past
> Windows MAX_PATH and failing mid-install, leaving a torch that cannot be cleanly
> uninstalled; `tissue_type_model_training/README.md` records jupyterlab doing the same
> with its Galata tree. `torch` here is pinned at **2.8.0+cpu**, and pulling
> `transformers` + `open_clip_torch` risks upgrading it. That breaks the whole
> environment, not just CONCH.

**Access:** UNI and CONCH are gated on HuggingFace, but the gate is only *"agree to
share your contact information"* - a click-through on the model page, not a reviewed
application, so access is effectively immediate. Accept the conditions, then create a
read token under **Settings -> Access Tokens** and hold it in Colab Secrets or
`huggingface_hub.login()`.

**Read the conditions rather than clicking past them.** UNI's are CC-BY-NC-ND 4.0 and
they decide whether anything you fit on it can ever be used - see the licence section at
the end, which is the part of this document most likely to change what you do.

---

## What changes in the code

The measurement path is already generic. Only the feature path is not.

### Already generic — no change needed

| file | why |
| --- | --- |
| `scripts/08_fit_concat_mlp.py` | takes the head width from `features.shape[1]` |
| `scripts/09_head_to_head.py` | same, and asserts the width against each head |
| the tile store | tiles are geometry, not features; every backbone reads the same PNGs |

### Needs work

**`src/models.py`** — three ResNet-specific assumptions:

```python
INITS: dict[str, str] = {...}        # name -> a weights FILENAME on disk
FEATURE_DIM = 512                    # hard-coded width
def resnet18_backbone(init, *, weights_dir, ...)   # builds a torchvision resnet18
torch.nn.Linear(2 * FEATURE_DIM, hidden)           # in ConcatResNet18MLP
```

Add a parallel table rather than bending `INITS`, which exists to name a *file*:

```python
@dataclass(frozen=True)
class Backbone:
    """One frozen feature extractor: how to build it, how wide, and what it eats."""
    name: str
    dim: int                       # embedding width - 512 / 1024 / 1536
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    builder: str                   # "timm_hub" | "torchvision_resnet18"
    hub_id: str | None = None      # e.g. "hf-hub:bioptimus/H-optimus-0"
```

Keep `resnet18_backbone` as-is and reachable — the published checkpoints depend on it,
and `load_pinned` reconstructs `ConcatResNet18MLP` from `CONCAT_ARCH`. **A new backbone
must not change how an existing checkpoint loads.**

**`scripts/03_features.py`** — line 94 hard-wires the ResNet:

```python
net = models.resnet18_backbone(args.init, weights_dir=backend_path.PRETRAINED_DIR)
body = models.feature_extractor(net)
print(f"\ninit={args.init}  conv1 weight norm ...")   # also ResNet-specific
```

Branch on the backbone table, and drop the `conv1` line for anything that has no
`conv1`. The cache filename already includes the init name, so `uni_std.npz` and
`imagenet_std.npz` coexist without collision.

**The transform — the subtle part.** `datasets.TileDataset` calls `to_model_input`
(`step08_tissue_type_segmentation/input.py`), which does: standardise on the tile's own
99th percentile → clip at `OD_CLIP` → scale to [0,1] → replicate to 3 channels →
normalise with **ImageNet mean/std**.

| model | expects | change needed |
| --- | --- | --- |
| **UNI** | 224 px, ImageNet mean/std | **none — your transform already matches exactly** |
| **H-optimus-0** | 224 px, its own mean/std (approx. `(0.707, 0.579, 0.704)` / `(0.212, 0.230, 0.178)` — **verify against the model card**) | normalisation override |
| **CONCH** | its own preprocessing | use its released preprocess, in its own venv |

So `to_model_input` needs an optional `mean`/`std`, defaulting to today's constants. That
is the whole change, and it must default to the current values so no existing checkpoint
moves.

**Do not touch** `models.CONCAT_ARCH`, `publish`, or `load_pinned` for this experiment.
Publishing a foundation-model head into step 8 would need the concat arch generalised
past `2 * 512`, and that is only worth doing if one of them wins.

---

## Step by step

1. **Request HuggingFace access** to UNI and CONCH. Slowest step, zero of your time.
2. **Add the `Backbone` table and the `03_features.py` branch** (~3–4 h including tests
   in this repo's style: assert the embedding width the cache claims, and refuse a
   backbone whose output width disagrees with its table entry).
3. **Add UNI** (~1 h). Its transform already matches, so this is one table row.
4. **Embed a stratified 4,000-tile subset on CPU** (~2–3 h wall clock) and score it
   before spending anything on GPU:

   ```
   python scripts/03_features.py --init uni --standardise \
       --tiles ../data/tissue_type_model_training/h_channel/112um --features ../data/tissue_type_model_training/h_channel/112um/features
   python scripts/08_fit_concat_mlp.py --tiles ../data/tissue_type_model_training/h_channel/112um --features ../data/tissue_type_model_training/h_channel/112um/features
   python scripts/09_head_to_head.py --tiles ../data/tissue_type_model_training/h_channel/112um --features ../data/tissue_type_model_training/h_channel/112um/features \
       --heads invasive_tile_fov112_fix1_concat <the new head>
   ```

   A subset makes the comparison noisier, not wrong: `09_head_to_head.py` pairs the two
   heads on the same tiles and reports McNemar, which is the right test for a subset.

5. **Only if the subset moves the number:** rent a GPU, embed all 32,030 tiles, refit,
   re-score. Then add H-optimus-0 (~1.5 h — one row plus its normalisation).
6. **CONCH separately, in its own venv**, for the zero-shot check below.

---

## The measurement, and what each outcome means

The number to beat is the one from the controlled head-to-head — **one** held-out set of
9,393 tiles, 2,270 of them in-situ, 174 slides:

| head | in-situ → invasive | 95 % CI |
| --- | --- | --- |
| `invasive_tile_v3_concat_bach` | 399/2270 = **17.6 %** | 16.1–19.2 % |
| `invasive_tile_fov112_fix1_concat` | 429/2270 = **18.9 %** | 17.3–20.6 % |

**Decision gates:**

- **Below ~14 %, paired p < 0.05** → the features were the bottleneck. Now decide the
  serving problem: a GPU in the box, or distil the foundation model into a
  ResNet-sized student. Distillation fits this codebase because it is already built on
  frozen features, but it is a project, not an afternoon.
- **Around 17–19 %** → the features are *not* the bottleneck either. Three levers plus
  this one are then exhausted, and what remains is: RGB input instead of the H channel;
  p63/CK5-6 ground truth; or reporting an uncertainty band rather than forcing a binary
  call. No serving cost has been incurred to learn this — which is the point of running
  it as a training-only measurement.
- **Worse than 19 %** → suspect the input, not the model. Replicated-grey H-channel is
  out of distribution for every one of these; see below.

---

## The RGB question, which may matter more than the model

Your tiles are the **haematoxylin channel replicated to three identical channels**. Every
one of these models is trained on **RGB H&E**. Larger models learn richer
colour-dependent features, so they can be *more* sensitive to that shift, not less — the
published ranking may not survive the transfer at all.

The H-channel exists so one model serves the H&E and all five IHC slides. But the
**invasive-versus-in-situ call only has to be made once per case, on the H&E**. So it can
be made in RGB and transferred to the IHC slides by registration, decoupling the hardest
decision from the hardest constraint.

If you run only one variant, run RGB. If you run two, run both and report the gap — that
gap is worth knowing regardless of which model wins.

---

## CONCH zero-shot: a different question, worth answering

CONCH is vision-language, so it can classify tiles from text prompts —
*"ductal carcinoma in situ"* against *"invasive ductal carcinoma"* — **with no training
and none of your labels involved.**

That is uniquely valuable here. Every in-situ label in this project traces back to
BEETLE's opinion, and BEETLE is documented as biased on exactly this axis (it disagreed
with BACH's two pathologists on 47 of 100 in-situ images, always by calling invasive). A
zero-shot result owes nothing to that. If CONCH cannot separate the two classes on your
tiles either, that is strong evidence the signal is not in a 112 µm haematoxylin window,
and it is evidence that does not depend on your supervision being right.

Run it as a standalone script in its own venv. It does not need the `Backbone` table, the
tile store's manifest, or any of the plumbing above.

---

## Licences

| model | licence | may a fitted head ship? |
| --- | --- | --- |
| **UNI** | **CC-BY-NC-ND 4.0** (gated, click-through) | **No — and the card says so explicitly** |
| H-optimus-0 | **read the model card before planning around it** | unknown |
| CONCH | non-commercial (gated) | no |
| Virchow2 | non-commercial | no |

**UNI's card closes a question this project had left open.** The BACH note in
`tissue_type_model_training/README.md` says whether a checkpoint fitted on
NoDerivatives data is itself a derivative is *"a legal question nobody here has
answered"*. UNI's licence answers it in its own text: commercial use is prohibited for
the model **"and its derivatives, which include models trained on outputs from the UNI
model"**. A head fitted on UNI embeddings is therefore a derivative by the licence's own
definition - no interpretation required - and the permitted use is *"non-commercial,
academic research purposes"*.

So the split is sharper than it looks:

- **For measurement** - deciding whether features are the bottleneck - UNI is the right
  tool and the licence permits it. The output is knowledge, not a product.
- **For anything that ships**, UNI is ruled out, including a model distilled from it or a
  dataset derived from its outputs.

**That is an argument for checking H-optimus-0's licence first and running that one
instead**, so the measurement and the deployable model are the same artefact. UNI's
advantage is that it is the most-benchmarked of the three, which is worth having for
comparison against published work and worth less if the result can never be used.

The tiles are **already** research-only regardless - BRACS non-commercial, BEETLE
CC BY-NC-SA, BACH CC BY-NC-ND - so no foundation model worsens the position for the
current corpus. What it forecloses is the one permissive route this project has: BCSS
alone plus torchvision weights.

Record whichever you use in the checkpoint manifest's `provenance`, as
`08_fit_concat_mlp.py` already does — a checkpoint that cannot be told apart from a
shippable one by looking at it is how a non-commercial dependency reaches a client.

---

## Related

- [`../segmentation_research/RESULTS_224_VS_448.md`](../segmentation_research/RESULTS_224_VS_448.md) — the field-of-view sweep and its intervals
- [`FIX1_FIX2_PLAN.md`](FIX1_FIX2_PLAN.md) — the label rule and the geometry, and why each was tried
- [`../guides/HOW_THE_PIPELINE_WORKS.md`](../guides/HOW_THE_PIPELINE_WORKS.md) — data → region → export → model, in plain language
- `backend/scripts/check_component_aggregation.py` — why post-processing is not the answer
- `tissue_type_model_training/scripts/09_head_to_head.py` — the controlled comparison harness
