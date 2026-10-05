# H&E campaign — pre-flight checklist

Run before starting the unattended campaign:

```powershell
$PY = ".\tissue_scoring_demo\backend\.venv\Scripts\python.exe"
& $PY tissue_type_model_training\scripts\10a_preflight.py
```

It prints `PASS` / `WARN` / `FAIL` / `MANUAL` per item and exits non-zero on any `FAIL`.
Stage `A0` of the driver runs it again and refuses to continue on a `FAIL`, so this is a
gate and not a suggestion. Every item here replaces a failure that would otherwise
surface hours later with nobody awake to read it.

## Mechanical — the script checks these

| # | item | why it is here |
|---|---|---|
| 1 | **Disk** — at least 12 GB free | The tightest constraint in the whole plan. This volume runs at ~96% and RGB tiles measure **3.0×** the bytes of haematoxylin ones (99.6 KB vs 33.2 KB at 224 px, measured on real region crops), so the four H&E stores come to ~4.9 GB, plus ~0.14 GB of feature caches and ~0.36 GB of checkpoints. Re-checked before **every** stage with an 8 GB floor. Peak equals final, because `07 --remove-source` moves rather than copies. |
| 2 | **torch is 2.8.x** | 2.9 and above break on Windows `MAX_PATH` in this tree. An upgrade would kill the export mid-run. |
| 3 | **`REGIONS_DIR` still resolves to `tissue_label_generation/data/regions`** | `backend_path` falls back to `DATA_DIR` if it finds a tree-shaped directory there. Neither `data/h_channel` nor `data/he` is named `dcis` or `bach_insitu`, so it should not — but a silent fallback would read the wrong trees and nothing downstream would notice. |
| 4 | **Both pretrained backbones load through `models.resnet18_backbone`** | The strict-`fc`-only assertion fires here rather than 44 minutes into the 112 µm feature pass. |
| 5 | **Six region trees: exact pair counts, `source_mpp == 0.5`, `teaches_classes` matching `bcss.BORROWED_TREES`** | dcis 239, ic 202, normal 213, bach_* 100 each. **This is the one that matters most after disk.** The `ic` tree records 115 deferred regions; if BEETLE has been re-run over any since the haematoxylin arms were cut, the tree has grown, the two exports saw different data, and `10c_assert_paired.py` refuses the pairing — 30 minutes into an export instead of in seconds here. |
| 6 | **BCSS: 150 regions and `slide_mpp.json` covering every slide** | `02_export.py` refuses to assume 0.25 µm/px from the filenames and exits. Better to know now. |
| 7 | **All four haematoxylin arms intact after the move** | Every `tile_path` resolves under `data/h_channel/<fov>um`, and both feature caches still match their manifest fingerprint. |
| 8 | **All four published haematoxylin heads load with `verify=True`, and `model_for` filters on branch** | The branch work adds `"channel"` to `_verify_input_contract`'s required keys. If any of the four stops loading, the served pipeline is broken and no H&E head may be published. |
| 9 | No other long-lived `python.exe`; the demo API not listening on 8000/8080 | A second torch job halves throughput and blows every budget. A live API holds checkpoints in its module cache and could serve a half-published one. |
| 10 | Sleep and hibernate disabled on AC; no pending Windows Update reboot | `powercfg /change standby-timeout-ac 0` and `powercfg /change hibernate-timeout-ac 0` if not. |
| 11 | The workspace root, `data/he` and `reports/` are writable | The tracker lives at the root; a read-only root means no progress record at all. |

## Manual — nobody can check these for you

- [ ] **Close the editor only after the tracker shows `RUNNING` with a non-zero `pid`.**
      The driver is launched detached, so closing the editor orphans it rather than
      killing it — but confirm it actually started first.
- [ ] Nothing else is writing to `tissue_type_model_training/data/` — no second
      session, no notebook kernel holding an `.npz` open.
- [ ] The laptop is on mains power, not battery.

## Starting it

```powershell
cd C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System
$PY = ".\tissue_scoring_demo\backend\.venv\Scripts\python.exe"
Start-Process -FilePath $PY `
  -ArgumentList "tissue_type_model_training\scripts\10_he_campaign.py","--go" `
  -WindowStyle Hidden
```

## Watching it

```powershell
Get-Content .\HE_CAMPAIGN_TRACKER.md                 # always current, single writer
& $PY tissue_type_model_training\scripts\10_he_campaign.py --watch
Get-Content -Tail 20 -Wait .\tissue_type_model_training\reports\he_campaign\E1_112_export.log
```

## If something goes wrong

Re-launching the driver is the resume: every stage whose output is already on disk
**and verifies** is skipped, so nothing is recomputed and nothing half-written is
trusted. `HE_CAMPAIGN_TRACKER.md`'s last section lists the outstanding stages as literal
commands, in order, for the case where running them one at a time is preferable.

The one thing not to do is delete a partial `data/tiles` before reading the log that
explains why it is partial — that directory is the evidence.
