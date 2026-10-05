# Folder layout changed on 2026-09-09

Two things moved. Documents in this folder written before that date use the old
names and the old data paths; they are records of what was run at the time and
have deliberately **not** been rewritten. Read them with this table beside you.

## 1. The three stage folders were renamed for what they do

| Before | Now |
| --- | --- |
| `bracs_roi_to_mask_using_beetle/` | `tissue_label_generation/` |
| `bcss_bracs_hchannel_resnet18/` | `tissue_type_model_training/` |
| `Breast_Cancer_IHC_Tissue_Scoring_Demo/` | `tissue_scoring_demo/` |

## 2. All storage moved into one `data/` tree at the workspace root

Each stage folder used to keep its own `data/`, and the source downloads sat in
a fourth directory, `original_data/`. They are now one tree, one subdirectory
per producer:

| Before | Now |
| --- | --- |
| `original_data/` | `data/original/` |
| `bracs_roi_to_mask_using_beetle/data/runs/` | `data/tissue_label_generation/runs/` |
| `bracs_roi_to_mask_using_beetle/data/regions/` | `data/tissue_label_generation/regions/` |
| `bcss_bracs_hchannel_resnet18/data/tiles/` | `data/tissue_type_model_training/tiles/` |
| `bcss_bracs_hchannel_resnet18/data/features/` | `data/tissue_type_model_training/features/` |
| `bcss_bracs_hchannel_resnet18/data/h_channel/<fov>um/` | `data/tissue_type_model_training/h_channel/<fov>um/` |
| `bcss_bracs_hchannel_resnet18/data/he/<fov>um/` | `data/tissue_type_model_training/he/<fov>um/` |
| `Breast_Cancer_IHC_Tissue_Scoring_Demo/data/` | `data/demo/` |

So a doc below that says `data/h_channel/224um` means
`data/tissue_type_model_training/h_channel/224um` today. The tile stores' own
internal layout - `<store>/{bcss,bach,beetle}/`, `tiles_manifest.csv`,
`export_summary.json`, `features/`, `reports/` - is unchanged, so a manifest's
`tile_path` still resolves against its own store.

**Why.** The irreplaceable downloads - 43 GB of BRACS, BACH, BCSS and OncoStem,
hours over services that rate-limit - used to sit in the same directories as the
exports cut from them, so "clear the exports" and "keep the sources" were the
same `rm -rf` with different intentions. Now everything derived is under
`data/<stage>/` and everything irreplaceable is under `data/original/`, and the
three stage folders hold source code only.

No code reads the old locations except as an explicit second candidate: see
`bracs_app.config.ORIGINAL_DATA_CANDIDATES` and
`backend_path.{REGIONS_DIR,BCSS_DIR}_CANDIDATES`, which let a clone with the old
layout keep working.

## Campaign artefacts also moved

`HE_CAMPAIGN_TRACKER.md`, `HE_VS_HCHANNEL.md`, `he_campaign_state.json` and
`he_vs_hchannel.json` are written by `scripts/10*.py`. They used to land at the
workspace root, among the hand-written documents; they now live beside the
per-stage logs they describe, in
`tissue_type_model_training/reports/he_campaign/`.
