# Step 2 — Quality control: setup and design notes

**Companion to** [demo-pipeline-guide.md](demo-pipeline-guide.md) § Step 2.

Step 2 is implemented and runs against a real uploaded slide. This document is
the part you cannot get from the code: what you have to install and download
before it will run, why the pieces are arranged the way they are, and which
claims in the UI are load-bearing.

---

## 1. What was built

Two independent halves, deliberately not merged.

| | What it is | What it answers | Where |
| --- | --- | --- | --- |
| **The decision** | Two pre-trained **GrandQC** segmentation models | *Which pixels are unusable, and what is wrong with them* | `backend/app/qc/inference.py` |
| **The explanation** | Classical **HistoQC-style** metrics computed directly | *Why that region looks wrong to a human* | `backend/app/qc/features.py` |

The guide's instruction is followed exactly: *use GrandQC for the result,
HistoQC's feature maps for the explanation.* Nothing the classical layer
measures is fed back into the decision. If it were, the demo would be claiming
a rigour that hand-tuned thresholds do not have.

### Both GrandQC models, in the order GrandQC requires

GrandQC's README is explicit that tissue segmentation runs first and artefact
segmentation second. Both are implemented:

1. **`Tissue_Detection_MPP10.pth`** — 2 classes, at 10 µm/px. A whole slide is
   a few thousand pixels at that scale, so this costs under a second. It is
   used to decide which patches are worth showing the expensive model, and it
   has the final say on what counts as background.
2. **`GrandQC_MPP15.pth`** (or `MPP1` / `MPP2`) — 7 classes, at 1.5 µm/px.
   Produces the per-pixel artefact map.

That ordering is why a whole-slide artefact map is affordable on a CPU at all:
the tissue pass typically excuses half to two-thirds of the grid before the
artefact model ever runs.

**Pixel coding** is GrandQC's, not ours:

```
1 tissue · 2 fold · 3 dark spot / foreign object · 4 pen marking
5 air bubble / coverslip edge · 6 out of focus · 7 background
```

Class `0` never appears in a finished mask — the models carry an unused
channel 0 — so it is treated as *not analysed* and counted as neither tissue
nor artefact.

### The classical metrics

Seven, computed on the same pixels the artefact model sees:

Tenengrad sharpness · Laplacian variance · RMS contrast · Michelson contrast ·
mean brightness · mean HSV saturation · Gabor texture energy.

They are measured in **sub-blocks of a patch**, not per patch, and that detail
turned out to matter. A 512 px patch at 2 µm/px covers a full millimetre of
tissue, and almost nothing is wrong with a whole millimetre at once — a pen
line or a fold occupies a slice of it. Measured per patch on a real slide, only
`edge` and `tissue` ever came out as the dominant class, so the explanation
table had nothing at all to say about folds, pen, dust or blur. Measured in
128 px sub-blocks, every class that is present gets a comparison.

These are the measurements HistoQC is built from, implemented directly rather
than by running HistoQC. **HistoQC is not a runtime dependency**, and that was
a deliberate call: the pipeline needs the numbers, not HistoQC's slide-level
pass/fail machinery, and a demo should not carry a second whole-slide framework
plus a Docker daemon to obtain seven filters that are twenty lines of scipy.
HistoQC remains the methodological reference and is cited as such.

---

## 2. What you have to do

Three things. The app degrades honestly at every stage in between —
`GET /api/v1/qc/capability` always answers, and the UI shows a checklist rather
than an error.

### Step A — install the QC dependencies

```bash
cd backend
.venv\Scripts\python -m pip install -r requirements-qc.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Kept out of `requirements.txt` on purpose: it pulls several hundred megabytes
of PyTorch, and every other endpoint works without it.

> **Windows path-length warning.** This project sits at a deep path, and the
> newest PyTorch releases ship a licence tree deep enough to exceed Windows'
> 260-character limit — the install fails with
> `[WinError 206] The filename or extension is too long` and leaves a broken
> half-installed `torch`. `requirements-qc.txt` therefore pins `torch<2.9`,
> which does not have that tree. If you would rather run the newest torch,
> enable long paths first (admin PowerShell, then reboot):
>
> ```powershell
> Set-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -Value 1
> ```
>
> If you already hit this, delete `.venv\Lib\site-packages\torch*`,
> `functorch`, and `torchgen` by hand before reinstalling — pip cannot
> uninstall a package whose `RECORD` never got written.

### Step B — download the two GrandQC checkpoints

They are **not** in the GitHub repository — that holds only the inference code.
The trained weights are published separately on Zenodo, and they are what
actually does the work. Nothing is trained here; these are the authors' own
checkpoints, downloaded as-is. Four files, about 26 MB each.

| What | Where | Files |
| --- | --- | --- |
| Tissue segmentation | <https://zenodo.org/records/14507273> | `Tissue_Detection_MPP10.pth` |
| Artefact segmentation | <https://zenodo.org/records/14041538> | `GrandQC_MPP15.pth` (7×, recommended), optionally `GrandQC_MPP1.pth` (10×) and `GrandQC_MPP2.pth` (5×) |

They can be fetched straight from Zenodo's API — no account, no browser:

```bash
cd models/grandqc
curl -L -o td/Tissue_Detection_MPP10.pth https://zenodo.org/api/records/14507273/files/Tissue_Detection_MPP10.pth/content
curl -L -o qc/GrandQC_MPP15.pth https://zenodo.org/api/records/14041538/files/GrandQC_MPP15.pth/content
```

They belong here:

```
tissue_scoring_demo/
└── models/
    └── grandqc/
        ├── td/
        │   └── Tissue_Detection_MPP10.pth
        └── qc/
            └── GrandQC_MPP15.pth
```

The backend also finds them in a **sibling clone of the GrandQC repository**
laid out the way their README describes, so if you already downloaded them
into `../grandqc/01_WSI_inference_OPENSLIDE_QC/models/`, they will be picked up
with no configuration. Override with `QC_MODELS_DIR` in `backend/.env`.

The minimum to run anything is *one artefact checkpoint*. Without the tissue
checkpoint the run still works, but falls back to a saturation/Otsu tissue
threshold and says so — that threshold calls pen marks tissue, which is exactly
the failure the real model exists to avoid.

### Step C — verify before touching the UI

```bash
cd backend
.venv\Scripts\python scripts\check_qc_models.py
```

This is worth running rather than skipping. It checks scipy, torch and
segmentation-models-pytorch, finds the checkpoints, **loads** them, and pushes
one blank patch through each. Exit code is 0 when step 2 can run.

A correct result looks like this: the tissue model reports 2 classes and calls
a blank pale patch `1` (background); each artefact model reports **8 output
channels** and calls it `7` (background). Eight, not seven — GrandQC's classes
are numbered 1–7 and channel 0 is unused, which is why the loader reads the
count off the segmentation head instead of assuming it.

### Why loading a GrandQC artefact checkpoint is awkward

Worth knowing, because it is the thing most likely to break on a future
upgrade. The artefact checkpoints are **pickled `nn.Module`s**, not state
dicts — `torch.save(model)` rather than `torch.save(model.state_dict())`. A
pickle stores *module paths and class names as strings*, so loading one runs
against whatever `timm` and `segmentation-models-pytorch` are installed today,
several years newer than the ones GrandQC saved with. Two things then go wrong:

1. **Names moved.** `timm.models.layers.activations` became `timm.layers.
   activations`; `timm.models.efficientnet_blocks` became
   `timm.models._efficientnet_blocks`; smp 0.5 renamed `DecoderBlock` to
   `UnetDecoderBlock`. Each surfaces as `ModuleNotFoundError` or
   `Can't get attribute`.
2. **Even after it unpickles, it will not run.** The reconstructed object gets
   *today's* class with *the old object's* attributes, so its `forward` reaches
   for things that were never stored —
   `'DepthwiseSeparableConv' object has no attribute 'has_skip'`.

Pinning every library back to GrandQC's 2023 versions is not an option here:
there is no `torch==2.0.1` wheel for Python 3.13, which this project targets.

So `app/qc/models.py` treats a pickled checkpoint as **a container of tensors,
never as code**. It unpickles behind two small shims that bridge the moved
names, lifts out the `state_dict`, discards the object, and loads those tensors
into a network built from the installed library's *current* code. The layers
that actually execute are always the maintained ones. The rebuild requires an
exact fit — zero missing and zero unexpected parameters — which is also how the
architecture is detected: the tissue model is a `UnetPlusPlus` and the artefact
models are plain `Unet`s, and only the right one absorbs a given checkpoint
cleanly.

Verified working on **smp 0.4.0 + timm 0.9.16** and **smp 0.5.0 + timm 1.0.28**,
so `requirements-qc.txt` does not pin either. If a future release breaks it,
the doctor names the class that moved.

Then start the app as usual (`start.bat`), upload a slide, and run step 1 and
step 2.

---

## 3. Runtime expectations on this machine

No CUDA GPU is present, so inference is CPU-only. That is fine and it is
planned for, but it is minutes not seconds. Measured on this machine, on
`CAN_00251_26_A.svs` (126,976 px square, 0.2222 µm/px, 28 mm of glass):

| Model | Patch grid | Inferred | Excused as glass | Wall clock |
| --- | --- | --- | --- | --- |
| 5× (2.0 µm/px) | 27 × 27 = 729 | 277 | 452 | **233 s** |
| 7× (1.5 µm/px) | 36 × 36 = 1,296 | ~490 | ~800 | ~7–8 min |

The "excused as glass" column is what the tissue pass buys: 62% of the grid
never reaches the artefact model. That is the whole reason the two-model order
is not optional.

Because of that:

- the run is a **background job**, polled — `POST /qc/{id}/run` then
  `GET /qc/{id}/run`. There is no synchronous variant, because there is no
  sensible timeout for one.
- results are **cached on disk** under `data/qc/<upload_id>/`, so stepping back
  onto step 2 is instant.
- **one run at a time**. Two slides competing for the same cores finish later
  than the same two run in sequence, and make the progress bar a lie.
- the model picker (5× / 7× / 10×) is offered in the UI before starting,
  because on a CPU that choice is the difference between three minutes and ten.

### Two deviations from upstream, both deliberate

1. **Patches are read from the nearest at-or-finer pyramid level**, not always
   from level 0 as GrandQC's scripts do. On a 40× scan, one 7× patch is
   3,456 × 3,456 level-0 pixels being downsampled to 512 — roughly 45× more
   pixels off disk than the model can use. Set `QC_READ_FROM_LEVEL_0=true` to
   reproduce upstream exactly.
2. **OpenSlide is not required.** Slides are read through `tiffslide`, which
   this project already uses and which installs on Windows without hunting for
   a binary. GrandQC's own scripts need the OpenSlide C library.

The model weights, the method, the class coding, the JPEG round-trip before
tissue detection and the 50-pixel tissue threshold are all GrandQC's, unchanged.

---

## 4. What the UI claims, and what it does not

The guide asks for a QC on/off toggle showing *two final scores side by side*.
**That is not what was built, and the difference matters.** Steps 3–16 are not
implemented, so there is no IHC score to move; showing one would make the
walkthrough describe software that does not exist — the same rule the rest of
this codebase already follows.

What the toggle shows instead is the quantity that genuinely changes today:

```
QC off →  analysable tissue area, artefacts included
QC on  →  analysable tissue area, artefacts excluded
```

That is the **denominator** the eventual score is divided by, it is measured
rather than invented, and it makes the same point. When step 16 exists, this
panel becomes the two-score comparison with no change to its argument.

Other claims held to deliberately:

- **Every ratio is against this slide's own clean tissue.** These metrics are
  not comparable between slides or between magnifications, so no absolute
  threshold appears anywhere in the UI.
- **Sample sizes are shown.** A class measured from three patches is greyed,
  because a mean of three patches is noise.
- **Scale is reported.** The grid measures at the artefact model's 1.5 µm/px;
  the region inspector re-measures at the pipeline's working resolution, and
  when the two differ *no ratio is quoted*, because comparing sharpness across
  resolutions is meaningless.
- **The Otsu fallback is never presented as GrandQC.** It is labelled in the
  API (`tissueSource`) and warned about on screen.

### What the explanation layer actually says

Measured on the slide above, as a ratio against its own clean tissue:

| Class | Sharpness | Laplacian | Contrast | Saturation | Texture |
| --- | --- | --- | --- | --- | --- |
| Dark spot | 0.18 | 0.11 | 0.84 | 0.22 | 0.49 |
| Fold | 0.36 | 0.19 | 0.84 | 0.56 | 0.66 |
| Out of focus | 0.52 | 0.25 | 0.71 | 0.47 | 0.82 |
| Pen | 1.04 | 1.54 | 1.16 | **2.53** | 0.69 |
| Edge / bubble | 0.98 | 0.84 | 1.02 | 0.83 | 1.09 |

Every one of these is the physically expected answer, which is what makes the
panel worth showing: out-of-focus regions really are half as sharp; folds are
darker and blurrier because the light went through two thicknesses; dark spots
are opaque; and pen ink is **two and a half times more saturated than any
stain**, which is exactly why the classical tissue mask of step 3 would call it
tissue.

The last row is the most useful one for the demo. Air bubbles and coverslip
edges measure ~1.0 on *everything* — no classical metric here separates them
from clean tissue, and they are 9.5% of the tissue on this slide. That is a
concrete demonstration of why the decision is GrandQC's and not a threshold's.

---

## 5. API surface

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/v1/qc/capability` | Can step 2 run; what is missing; where to get it |
| POST | `/api/v1/qc/{id}/run` | Start a run (or return the cached one) |
| GET | `/api/v1/qc/{id}/run` | Progress: phase, patches done, percentage |
| GET | `/api/v1/qc/{id}` | The report |
| GET | `/api/v1/qc/{id}/grid` | Per-patch metrics, for client-side heatmaps |
| GET | `/api/v1/qc/{id}/overlay.png` | Slide with artefacts tinted |
| GET | `/api/v1/qc/{id}/tissue.png` | Pass 1's tissue map |
| GET | `/api/v1/qc/{id}/classes.png` | Class map alone |
| GET | `/api/v1/qc/{id}/mask.png` | Indexed-colour mask, for download |
| GET | `/api/v1/qc/{id}/heatmap/{metric}.png` | One metric over the grid |
| GET | `/api/v1/qc/{id}/explain` | One region, re-measured finely |
| GET | `/api/v1/qc/{id}/explain.png` | That region's pixels |

---

## 6. Settings

All optional; all in `backend/.env`.

| Variable | Default | Notes |
| --- | --- | --- |
| `QC_MODELS_DIR` | *(search)* | Checkpoint directory with `td/` and `qc/` inside |
| `QC_MODEL_MPP` | `1.5` | 2.0 = 5×, 1.5 = 7×, 1.0 = 10× |
| `QC_MAX_PATCHES` | `6000` | Refuses a run that would take absurdly long |
| `QC_READ_FROM_LEVEL_0` | `false` | Set true to match upstream GrandQC's I/O exactly |
| `QC_DIR` | `data/qc` | Where cached reports and overlays live |

---

## 7. Attribution

GrandQC's licence requires that any package integrating it states how to cite
it and that it is non-commercial. Both appear in the API
(`/qc/capability`), on screen under the report, and here:

> Weng Z. et al. **GrandQC: a comprehensive solution to quality control problem
> in digital pathology.** *Nature Communications* (2024).
> <https://doi.org/10.1038/s41467-024-54769-y>

**GrandQC is distributed under a non-commercial licence. Use of these
checkpoints is subject to the terms of the original GrandQC licence.**

HistoQC is the methodological reference for the classical layer:
Janowczyk A. et al. *HistoQC: An Open-Source Quality Control Tool for Digital
Pathology Slides.* JCO Clinical Cancer Informatics (2019).
