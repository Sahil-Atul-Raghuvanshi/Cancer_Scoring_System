# Approach 1 — BCSS → haematoxylin-channel ResNet18

Trains the **one model** the IHC scoring pipeline needs: a 3-class tile classifier that reads
a 224 px haematoxylin-channel tile at 0.5 µm/px and answers

| class | meaning | fate |
| --- | --- | --- |
| `2` invasive epithelium | carcinoma that has broken out of the ducts | **the only thing scored** |
| `1` non-invasive epithelium | DCIS, LCIS, normal ducts and lobules | excluded |
| `0` non-epithelium | stroma, fat, inflammation, necrosis | excluded |

It feeds pipeline **step 8** (`tissue-type-segmentation`), which is inference only.

**Full build plan, with the gates and the schedule:**
[`../Breast_Cancer_IHC_Tissue_Scoring_Demo/docs/approach-1-training-plan.md`](../Breast_Cancer_IHC_Tissue_Scoring_Demo/docs/approach-1-training-plan.md)
**Why this approach and not three others:**
[`../Breast_Cancer_IHC_Tissue_Scoring_Demo/docs/segmentation-approaches-comparison.md`](../Breast_Cancer_IHC_Tissue_Scoring_Demo/docs/segmentation-approaches-comparison.md)

---

## The one idea

Every slide in a case — one H&E and five IHC — carries **haematoxylin**. Only haematoxylin.
The second dye differs (eosin on the H&E, DAB on the five IHC), and on N-cadherin or
pan-cadherin sections that DAB floods 75–85 % of the tile and drowns everything else.

So: deconvolve, throw the second dye away, keep the haematoxylin channel, and all six slides
become the same kind of image. One model serves the whole panel, and it never sees the brown
that would otherwise have to be normalised away.

That is why this folder imports the demo backend rather than reimplementing anything:

```python
from app.common.imaging import optical_density                    # step 4/5
from app.common.stains import RUIFROK_HDAB                        # step 6, fixed vectors
from app.pipeline.step06_colour_deconvolution.deconvolution import separate

od = optical_density(rgb, white)
h = separate(od, RUIFROK_HDAB).haematoxylin
```

**If a notebook here ever calls `skimage.color.rgb2hed` instead, the model is being trained
on a different definition of "haematoxylin channel" from the one it will be served** — and
the resulting accuracy drop looks like a modelling failure for a week before anyone finds the
plumbing. `src/hchannel.py` is the single definition; `tests/test_hchannel_transform.py`
asserts it agrees with the pipeline's own.

## Setup

Install into the **backend's** virtualenv, not a new one — one interpreter is what makes
`import app.*` the real thing rather than a path trick:

```powershell
cd ..\Breast_Cancer_IHC_Tissue_Scoring_Demo\backend
.venv\Scripts\python -m pip install -r ..\..\bcss_bracs_hchannel_resnet18\requirements.txt
```

Then, from this folder:

```powershell
..\Breast_Cancer_IHC_Tissue_Scoring_Demo\backend\.venv\Scripts\python -m pytest tests -q
..\Breast_Cancer_IHC_Tissue_Scoring_Demo\backend\.venv\Scripts\python -m jupyter lab notebooks
```

> **Do not upgrade torch to satisfy a notebook.** `backend/requirements-qc.txt` records why:
> torch ≥ 2.9 unpacks a licence tree deep enough to exceed MAX_PATH under this project's
> path, and pip fails mid-install leaving a torch that cannot be cleanly uninstalled.

## Two ways to run: scripts or notebooks

`scripts/` and `notebooks/` are **the same code** — every step's logic lives in `src/`, and both
are thin wrappers over it. The published checkpoints were produced by the scripts, because an
hours-long download or a ninety-minute feature pass is a poor fit for a notebook kernel: no
resume across a restart, no log to read afterwards, and a progress bar that exists only while a
browser tab is open. The notebooks are the readable version, with the figures.

```powershell
$py = "..\Breast_Cancer_IHC_Tissue_Scoring_Demo\backend\.venv\Scripts\python.exe"

& $py scripts/01_download.py                    # G0 + G1, ~1 h (network)
& $py scripts/02_export.py --limit 4            # smoke-test the exporter first
& $py scripts/02_export.py                      # G2, ~1 h (disk)
& $py scripts/03_features.py --init imagenet    # G3, ~1.5 h (CPU)
& $py scripts/04_train.py --init imagenet       # G4, minutes
& $py scripts/05_publish.py --init imagenet     # G6
```

`--limit 4` on the exporter is not a convenience. The export is the one step every later stage
trusts without re-checking, so it is worth four regions and a look at the numbers before
committing an hour to 151 of them.

> **jupyterlab will not install under this path.** It unpacks a Galata test-extension tree deep
> enough to exceed Windows MAX_PATH, and pip fails partway with `OSError errno 2` — the same class
> of problem `backend/requirements-qc.txt` records for torch ≥ 2.9. It is commented out of
> `requirements.txt` and is not needed: use the scripts, or `pip install notebook` (no Galata
> tree), or enable long paths.

## Run order

Each step consumes the previous one's *checked* output. The gate at the end of each is what makes
the next one worth running.

| # | Notebook / script | Gate | Time |
| --- | --- | --- | --- |
| 01 | `01_download_and_inspect_bcss` | **G0** the H channel matches the pipeline's · **G1** raw release, `dcis` is code 20 | ~1 h, network |
| 02 | `02_export_h_channel_tiles` | **G2** manifest sane, no slide crosses a split, tiles look like tissue | ~1 h, disk |
| 02b | `02_export.py --reuse-bcss --include-dcis` | **G2d** the borrowed DCIS tiles really are mostly class 1 · **G2c** the exporter reproduces both sources byte-for-byte | minutes |
| 03 | `03_cache_frozen_features` | **G3** deterministic, fingerprinted, carrying signal | ~1.5 h, CPU |
| 04 | `04_fit_head_and_evaluate` | **G4** confusion matrix on held-out institutions, invasive ↔ in-situ Dice ≥ 0.75 | minutes |
| — | *then* [`../moco_init_resnet18`](../moco_init_resnet18) | the A/B — one download, one feature pass | ~2 h |
| 05 | `05_optional_finetune_layer4` | **G5** only if G4 fell short, and only if the gain beats fold noise | ~15 h |
| 06 | `06_export_and_pin_model` | **G6** reloads in a fresh process and reproduces its own logits | 30 min |

Then **G7**: step 8 scores a real slide and agrees with the notebook, tile for tile. Until
that passes, the model is trained and not deployed.

## What lives where

```
src/backend_path.py    the ONE place that knows where the backend is. Import it first.
src/hchannel.py        the input transform. Exporter, dataset and step 8 all call it.
src/bcss.py            the 22 raw codes, the 3-class map, the barcode/institution parsing,
                       and the borrowed-DCIS reader + patient-level held-out rule
src/export.py          resample -> deconvolve -> tile -> majority vote -> manifest
src/datasets.py        the torch Dataset, the stain jitter, the splits that do not leak
src/models.py          backbone factory (both initialisations), publish, load_pinned
src/report.py          confusion matrix, the invasive/in-situ cell, strata, paired A/B
src/download_bcss.py   three download routes, and the release check

data/                  gitignored: bcss/ (~10 GB), dcis/ (borrowed, ~2 GB),
                       tiles/ (~300 MB), features/ (~40 MB)
reports/               gitignored: figures and JSON metrics
tests/                 59 tests. The mapping trap, the vote boundaries, the transform,
                       and every rule that keeps the borrowed class separable.
```

Checkpoints are published **outside** this folder, to
`../Breast_Cancer_IHC_Tissue_Scoring_Demo/models/tissue_type/`, so the trained model and the
served model are the same file.

## Class 1 is borrowed from BRACS, and that is the biggest change here

BCSS supplies **nine** `non_invasive_epithelium` tiles in the entire 151-region release, seven
of them from one patient — and **none at all** in the six published held-out institutions. So
the first trained head reported `non_invasive_recall 0.000` and per-class Dice `0.0` on an empty
truth row: not a bad score, no score. A three-class model cannot be fitted, and more to the
point cannot be *measured*, on that.

So class 1 comes from [`../bracs_roi_to_mask_using_beetle`](../bracs_roi_to_mask_using_beetle): BRACS DCIS
regions of interest, segmented by BEETLE's released nnU-Net, written out as **masks in BCSS's
own 22-code vocabulary** and cut by **this folder's exporter, unmodified**.

```
bracs_roi_to_mask_using_beetle/scripts/export_to_approach1.py   # BRACS + BEETLE -> data/dcis/
        │
        ▼
data/dcis/images/BRACS_1247_DCIS_1.png    RGB at 0.5 µm/px
data/dcis/masks/ BRACS_1247_DCIS_1.png    uint16 I;16, codes 1/2/4/20
data/dcis/dcis_manifest.json              resolution, provenance, every rejection
        │
        ▼
python scripts/02_export.py --reuse-bcss --include-dcis
```

Four rules keep the borrowing honest, and each is enforced rather than documented:

| | |
| --- | --- |
| **`source` column** | every row is stamped `bcss` or `bracs_dcis`. Once exported the tiles are indistinguishable — same 224 px, same uint8, same folder — so this column is the only thing that knows a person drew one label and a model guessed the other. |
| **Split by patient** | BRACS has no institutions, so `bcss.dcis_test_cases` holds out 25 % of patients by **hashing each patient id independently**. A seeded shuffle would reassign every patient the moment the export grew. |
| **The nine real tiles move to test** | and they move as **whole slides** (3 slides, 417 tiles, ~4 % of training). Moving only the nine would leave the other 408 tiles of those slides in training, and a model scored on one tile of a slide it was fitted on is scored on its own memory. This is a deliberate departure from BCSS's published split; `institution_split(holdout_real_dcis=False)` reproduces the original. |
| **Reported separately** | `report.by_source` never averages the nine human-drawn in-situ tiles into thousands of BEETLE-labelled ones. Class-1 recall against `bracs_dcis` measures **agreement with BEETLE**; only the `bcss` line measures agreement with a person. |

**What this cannot do.** It cannot make the labels true. They are BEETLE's model's opinion of a
second dataset, unreviewed by any pathologist, and BRACS's own DCIS label is per-region rather
than per-pixel. The ceiling on the borrowed tiles is BEETLE's own external non-invasive Dice.
Roughly one BRACS DCIS region in five comes back with more invasive than in-situ epithelium;
those are rejected before export and every rejection is recorded with its reason in
`dcis_manifest.json`.

## Two things the first run turned up

**The BCSS "convenient single link" does not work at scale.** Google Drive served 42 of 302 files
and then rate-limited every subsequent request. `scripts/01_download.py --route girder` — the
default — instead crops each ROI server-side from the authors' own HistomicsTK instance and pulls
its mask from figshare: ~4 GB, no quota, resumable per region, and image and mask arrive in the
same coordinate space so they align exactly. `--route gdrive` is kept but do not plan around it.

**Every filename says `MPP-0.2500` and the slides are not 0.25 µm/px.** The first one measures
**0.2521**, and TCGA scans vary. A 224 px tile is meant to be 112 µm — nine cells across — and
that is the whole argument for the tile size, so the resolution is *measured* rather than read off
a filename: the download records each slide's own `mm_x` into `data/bcss/slide_mpp.json`,
`export_region` takes `source_mpp` with **no default**, and the exporter refuses to run without
the sidecar. Pipeline step 1 states the same rule in code — convert using mpp, never a hard-coded
factor.

## Licences — **two tracks now, and only one of them ships**

| Asset | Type | Licence | Track |
| --- | --- | --- | --- |
| **BCSS** (Amgad et al. 2019) | 151 TCGA-BRCA regions, pixel labels | **CC0 1.0** | shippable |
| **torchvision ResNet18** `IMAGENET1K_V1` | weights | **BSD-3** | shippable |
| *(approach 3)* Ciga SimCLR ResNet18 | weights | **MIT** | shippable |
| **BEETLE** nnU-Net ensemble | weights, used to label `data/dcis/` | **CC BY-NC-SA 4.0** | research only |
| **BRACS** DCIS regions | images behind `data/dcis/` | **non-commercial** | research only |

This folder used to be entirely permissive. It is not any more, and the split matters:

- **Anything trained *without* `--include-dcis` is unchanged and still ships.** The BCSS-only
  path is intact — `institution_split` reproduces the published split automatically when no
  borrowed tiles are present, and the previously published
  `invasive_tile_v1_imagenet.pt` is untouched.
- **Anything trained *with* `--include-dcis` is research only.** BEETLE's ShareAlike arguably
  follows a distilled student, and BRACS is non-commercial regardless. A model fitted on those
  tiles inherits both restrictions, and so does every number computed from it.

Publish the two under different names. A checkpoint that cannot be told apart from the
permissive one by looking at it is how a non-commercial dependency reaches a client.

Nothing here touches TIGER, GrandQC, HoVer-Net, DeepLIIF, UNI, CONCH, Virchow2 or Phikon-v2.

**Two things deliberately absent:**

- **AICAN pseudo-labels** (plan input B2) — deferred, not cancelled. Every slide currently in
  `data/slides/` is the same IHC section, and AICAN needs RGB H&E. BCSS-only is a complete
  model; see the plan's Phase 5 for what to do the day an H&E slide arrives.
- **Stain normalisation** — replaced by the single-channel stain jitter in
  `src/datasets.py`. That is the whole reason pipeline step 6 can be skipped.

## Citations

Amgad M, Elfandy H, … Cooper LAD. *Structured crowdsourcing enables convolutional
segmentation of histology images.* Bioinformatics 35(18):3461–3467 (2019). Data CC0.
Ruifrok AC, Johnston DA. *Quantification of histochemical staining by colour deconvolution.*
Anal Quant Cytol Histol 23(4):291–299 (2001).
Tellez D et al. *Quantifying the effects of data augmentation and stain colour normalization…*
Medical Image Analysis 58 (2019).
