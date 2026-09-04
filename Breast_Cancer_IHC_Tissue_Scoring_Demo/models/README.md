# Model checkpoints

Downloaded, not committed. Nothing here is in version control — the GrandQC
weights are ~26 MB each and are distributed under a **non-commercial licence**
that this repository has no right to redistribute.

## What goes where

```
models/grandqc/
├── td/
│   └── Tissue_Detection_MPP10.pth     https://zenodo.org/records/14507273
└── qc/
    └── GrandQC_MPP15.pth              https://zenodo.org/records/14041538
        GrandQC_MPP1.pth   (optional, 10x)
        GrandQC_MPP2.pth   (optional, 5x)
```

The backend also looks in a sibling clone of the GrandQC repository, laid out
as their README describes, so an existing checkout is picked up with no
configuration. Point `QC_MODELS_DIR` somewhere else if you prefer.

Verify with:

```bash
cd backend
.venv\Scripts\python scripts\check_qc_models.py
```

Full instructions: [../docs/step-2-quality-control.md](../docs/step-2-quality-control.md)

> Weng Z. et al. GrandQC: a comprehensive solution to quality control problem in
> digital pathology. *Nature Communications* (2024).
> https://doi.org/10.1038/s41467-024-54769-y

---

## `tissue_type/` — the model this project trains

Step 8, `invasive epithelium / non-invasive epithelium / non-epithelium` on a 224 px
haematoxylin tile at 0.5 µm/px. Build plans:
[approach 1](../docs/approach-1-training-plan.md) (BCSS, the baseline),
[approach 3](../docs/approach-3-training-plan.md) (SimCLR initialisation),
[approach 4a](../docs/approach-4a-training-plan.md) (BEETLE teacher).

### `tissue_type/*.pt` — the trained checkpoints step 8 serves

Published by `bcss_hchannel_resnet18/scripts/05_publish.py`, which writes **here** rather
than keeping its own copy — so the trained model and the served model are the same file.
Each `.pt` has a `.manifest.json` beside it carrying its sha256, its class order and its
full input contract; `model.load_pinned` refuses the checkpoint if any of the three
disagrees with the serving code, and `backend/scripts/check_tissue_model.py` replays the
manifest's own probe logits through a fresh process to prove they still match.

| File | Trained on | Licence | Ships |
| --- | --- | --- | --- |
| `invasive_tile_v1_imagenet_std.pt` **← step 8's default** | BCSS (CC0) **+ BRACS regions labelled by BEETLE's nnU-Net** | **research only** | ❌ |
| `invasive_tile_v1_simclr_std.pt` | the same, SimCLR backbone | **research only** | ❌ |
| `invasive_tile_v1_imagenet.pt` | BCSS alone | CC0 + BSD-3 | ✅ |

All three: ResNet18, frozen body, 3-class head, 224 px haematoxylin window at 0.5 µm/px,
class order `non_epithelium / non_invasive_epithelium / invasive_epithelium`.

> ⚠ **The default is not shippable, and that is a deliberate trade.** BCSS supplies
> **nine** non-invasive epithelium tiles in its entire 151-region release and **none** in
> its held-out institutions, so the in-situ class — the one Rule 5 gates the whole score
> on — can be neither fitted nor measured from it. The in-situ tiles therefore come from
> BRACS (non-commercial) labelled by BEETLE's nnU-Net (CC BY-NC-SA), and the model
> inherits both terms, as does every number computed from it. `invasive_tile_v1_imagenet`
> is the commercially clean alternative and **cannot tell in-situ from invasive at all**.
>
> The licence track is read out of each manifest at load, restrictive wins, and it travels
> to the screen as a blocking caveat. A checkpoint that cannot be told apart from the
> permissive one by looking at it is how a non-commercial dependency reaches a client.

Verify with:

```bash
cd backend
.venv\Scripts\python scripts\check_tissue_model.py
```

---

### `tissue_type/pretrained/` — backbone initialisations

Two ResNet18 bodies, downloaded and pinned. Both are **frozen** during training; only
the 1,539-parameter 3-class head is fitted.

| File | Init | Bytes | SHA-256 | Licence | Source |
| --- | --- | --- | --- | --- | --- |
| `resnet18-imagenet.pth` | `imagenet` | 46,829,803 | `dc1c4d3f57869f5a1f3c9ba7b35511534cc0d35620fdc625b24ef81014d0d240` | **BSD-3** | torchvision `ResNet18_Weights.IMAGENET1K_V1` |
| `pytorchnative_tenpercent_resnet18.ckpt` | `simclr` | 46,104,139 | `ae652d1a586c1be11b8ed7f4033094a98ac9c620e43a28998c6c86d248e750ab` | **MIT** | [ozanciga/self-supervised-histopathology](https://github.com/ozanciga/self-supervised-histopathology/releases/tag/nativetenpercent) |

**Both licences are commercially usable.** Nothing here carries the GrandQC or BEETLE
non-commercial terms — keep it that way, and see approach 4a Part 0 before shipping
anything distilled from BEETLE.

> ⚠ **`simclr` is SimCLR, not MoCo.** The design documents call it MoCo; the released
> checkpoint is from Ciga, Xu & Martel, *Self supervised contrastive learning for digital
> histopathology* (Machine Learning with Applications 7, 2022), pretrained with **SimCLR**
> across 57 histopathology datasets. MoCo is a different method by different authors. The
> filename and the manifest say `simclr` so a reader can find the paper.

> ⚠ **Take the native checkpoint, not the Lightning one.** `tenpercent_resnet18.ckpt`
> (92,148,628 bytes) is a PyTorch Lightning checkpoint: loading it needs
> `weights_only=False`, which **executes pickled code**. The native asset above is a plain
> `state_dict` and loads under `weights_only=True`. Verified: nothing in it is executed.

Verified on download (gate G3a, 2 Sep 2026): byte count matches the release asset, loads
with `weights_only=True`, and the two bodies are genuinely different networks —
`conv1` norm 12.58 vs 29.24, and `layer4.1.conv2` differs, which is what rules out a
silent partial load leaving deep layers at random init.
