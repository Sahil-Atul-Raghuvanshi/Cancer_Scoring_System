# Cancer Scoring System

Breast cancer IHC tissue scoring from whole-slide images, plus the training
projects that produced the models it serves.

```
Cancer_Scoring_System/
├── setup.py / setup.bat                    one command to make a clone runnable
├── models.lock.json                        every checkpoint, its sha256, its source
├── tools/drive_fetch.py                    selective fetch from the Drive mirror
│
├── Breast_Cancer_IHC_Tissue_Scoring_Demo/  ← the web app
│   ├── backend/                            FastAPI, the 17-step pipeline
│   ├── frontend/                           React + Vite + OpenSeadragon
│   ├── models/                             checkpoints (downloaded, not committed)
│   └── docs/                               pipeline guide and training plans
│
├── bcss_bracs_hchannel_resnet18/           trains step 8's tile classifier
├── pixel_unet_resnet18/                    the pixel-wise U-Net alternative
└── bracs_roi_to_mask_using_beetle/         BEETLE teacher → BRACS in-situ labels
```

## Quick start

Needs **Python 3.11+** and **Node 18+** on PATH.

```bash
git clone <this-repo> Cancer_Scoring_System
cd Cancer_Scoring_System

python setup.py --drive-folder <models-folder-id>    # or setup.bat on Windows

cd Breast_Cancer_IHC_Tissue_Scoring_Demo
start.bat                                            # stop.bat to shut down
```

Then open **http://localhost:5173**. The API is on
**http://127.0.0.1:8000**, with docs at `/docs`.

## What setup.py does

Everything git does not carry, in order, reporting all problems rather than
stopping at the first:

| | Step | From |
| --- | --- | --- |
| 1 | Checks Python and Node versions | — |
| 2 | Builds `backend/.venv` | `requirements.txt`, `requirements-qc.txt` |
| 3 | Installs `frontend/node_modules` | `package-lock.json` (`npm ci`) |
| 4 | Downloads and verifies checkpoints | `models.lock.json` |
| 5 | Runs the project's own readiness checks | `check_qc_models.py`, `check_tissue_model.py` |

Step 5 is the part that matters: it loads every checkpoint, runs a forward
pass, and confirms each manifest's class order and input contract still agree
with the serving code. A green finish means the app will serve steps 2 and 8 —
not merely that `pip` exited 0.

```bash
python setup.py --check              # audit what is present, download nothing
python setup.py --models-only        # checkpoints only
python setup.py --skip-models        # code dependencies only
python setup.py --with-training      # also the training-only weights (+1.9 GB)
python setup.py --recreate-venv      # throw away .venv and rebuild it
python setup.py --clone-grandqc      # also clone upstream GrandQC for reference
```

## `backend/.venv` is disposable

It is **3.9 GB**, 95% of the backend folder, and none of it is hand-installed.
2.6 GB is static `.lib` files inside the Windows PyTorch wheel — MSVC
link-time archives that are never loaded at runtime, `dnnl.lib` alone being
2.2 GB.

Delete it whenever the disk is tight:

```bash
rm -rf Breast_Cancer_IHC_Tissue_Scoring_Demo/backend/.venv
python setup.py                      # rebuilds it, then re-verifies the models
```

`requirements-qc.txt` pins `torch>=2.6,<2.9`. The ceiling is a Windows
`MAX_PATH` constraint, not a compatibility one: torch 2.9+ ships a licence
tree deep enough that, unpacked under a long project path, `pip` fails
mid-install with `WinError 206`. Setup installs CPU wheels from
`download.pytorch.org/whl/cpu`, which are also ~2 GB smaller than the CUDA
build.

## Models

Weights are **not** in git. GrandQC is CC BY-NC, BEETLE is CC BY-NC-SA, and
this repository has no right to redistribute either. `models.lock.json`
records the exact bytes and sha256 of each file and where it comes from, so a
download either reproduces the checkpoint that was measured or fails loudly.

| Checkpoint | Size | Role | Source | Licence |
| --- | --- | --- | --- | --- |
| `grandqc/td/Tissue_Detection_MPP10.pth` | 25 MB | step 2, **required** | [Zenodo 14507273](https://zenodo.org/records/14507273) | CC BY-NC 4.0 |
| `grandqc/qc/GrandQC_MPP15.pth` | 24 MB | step 2 at 7×, **required** | [Zenodo 14041538](https://zenodo.org/records/14041538) | CC BY-NC-SA 4.0 |
| `grandqc/qc/GrandQC_MPP2.pth` | 24 MB | step 2 at 5×, optional | Zenodo 14041538 | CC BY-NC-SA 4.0 |
| `grandqc/qc/GrandQC_MPP1.pth` | 24 MB | step 2 at 10×, optional | Zenodo 14041538 | CC BY-NC-SA 4.0 |
| `tissue_type/invasive_tile_v2_imagenet_std.pt` | 43 MB | step 8, **required** | Drive mirror, or retrain | **research only** |
| `tissue_type/invasive_tile_v1_*.pt` ×3 | 43 MB ea. | step 8 alternates | Drive mirror, or retrain | mixed |
| `tissue_type/pretrained/*` ×2 | 45 MB ea. | training only | torchvision / GitHub release | BSD-3 / MIT |
| `beetle/model.zip` | 1.8 GB | training only | [Zenodo 16812932](https://zenodo.org/records/16812932) | CC BY-NC-SA 4.0 |

**Runtime total is ~350 MB.** BEETLE's 1.8 GB is a training-time dependency
only — the web app never loads it — so setup skips it unless you ask for it.

### The four trained checkpoints exist nowhere public

GrandQC, BEETLE and both pretrained backbones come from permanent archives.
The `invasive_tile_*.pt` files are this project's own training output, so the
only ways to get them are the Google Drive mirror or a retraining run
(`bcss_bracs_hchannel_resnet18/scripts/04_train.py`, then `05_publish.py`,
which writes straight into `models/tissue_type/` so the trained model and the
served model stay the same file). **Keep the Drive mirror alive.**

The Drive mirror is the `models/` folder uploaded with its tree intact.
`tools/drive_fetch.py` lists it once and pulls only the paths that are
missing, which is what keeps a routine setup at 350 MB rather than 2.2 GB.

### The default checkpoint is research-only

Step 8 defaults to `invasive_tile_v2_imagenet_std`, fitted on BCSS (CC0) plus
BRACS regions labelled by BEETLE's nnU-Net — non-commercial and ShareAlike
respectively. BCSS supplies nine non-invasive epithelium tiles in its entire
release and none in its held-out institutions, so the in-situ class that
Rule 5 gates the whole score on can be neither fitted nor measured from BCSS
alone. The model inherits both restrictions, and so does every number computed
from it.

`invasive_tile_v1_imagenet` is the commercially clean alternative and **cannot
separate in-situ from invasive disease at all**. The licence track is read out
of each manifest at load time, the most restrictive term wins, and it reaches
the screen as a blocking caveat rather than being trusted to a comment.

### Manifests are committed; weights are not

Each `.pt` has a `.manifest.json` beside it carrying its sha256, class order
and full input contract. `model.load_pinned` refuses a checkpoint if any of
the three disagrees with the serving code. Those manifests are a few KB of
text under no licence encumbrance, so they **are** in git — without them a
downloaded `.pt` cannot be loaded at all.

## Licensing

Not redistributable as a whole. The code is this project's; the model weights
and training data are not, and the non-commercial terms above travel with
every score the app produces. See
`Breast_Cancer_IHC_Tissue_Scoring_Demo/models/README.md` for the full
provenance, and each training project's README before shipping anything
distilled from BEETLE.

> Weng Z. et al. GrandQC: a comprehensive solution to quality control problem
> in digital pathology. *Nature Communications* (2024).
> https://doi.org/10.1038/s41467-024-54769-y
