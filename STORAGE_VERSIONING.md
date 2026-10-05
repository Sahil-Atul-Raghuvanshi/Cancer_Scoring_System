# Storing data and history across versions

Every time the pipeline changes in a way that changes its numbers (new model,
new step, different cut points), its outputs go into a new **version folder**:
`v1_data/`, `v2_data/`, `v3_data/` and so on. The inputs those versions all
read stay in one shared folder, so a 43 GB slide library is never copied.

Everything that is not code lives in one gitignored folder at the root, `storage/`.

This file explains what goes where, how the app chooses a version, how to start a
new version, and how to keep each version's history readable as the code changes.

---

## 1. The layout

```
Cancer_Scoring_System/
├── data_versions.py                the one module that knows this layout
├── STORAGE_VERSIONING.md           this file
├── ...code folders...
└── storage/                        GITIGNORED: every byte that is not code
    ├── data/                       SHARED, read-only inputs, used by every version
    │   ├── original/                 OncoStem slides, BRACS, BACH, BCSS,
    │   │                             annotations, reader scores
    │   ├── oncostem_docs/            client documents
    │   └── oncostem_requiremnet_docs/
    │
    ├── v1_data/                    EVERYTHING VERSION 1 PRODUCED
    │   ├── version.json              which storage layout it was written with (section 3)
    │   ├── VERSION.md                what this version is, and how it differs
    │   ├── data/                     pipeline outputs
    │   │   ├── demo/                   the app's working state (uploads, per-step caches)
    │   │   ├── history/                finished runs, filed per case (section 5)
    │   │   ├── registration/           slide alignment work
    │   │   ├── score_all_slides/       batch scoring logs, state, per-case results
    │   │   ├── score_archive/          scores kept before a re-run overwrote them
    │   │   ├── review_reports/         inputs to the review .docx builds
    │   │   ├── tissue_label_generation/ BEETLE teacher runs and regions
    │   │   ├── tissue_type_model_training/ tile and feature stores
    │   │   └── ...                     comparison runs, walkthrough shots, etc.
    │   ├── models/                   the checkpoints this version scored with
    │   ├── results/                  deliverables: score CSVs, DECISIONS.md, review .docx/.pdf
    │   └── reports/                  write-ups about this version
    │                                 (PIPELINE_PROBLEMS_*, review figures)
    │
    ├── v2_data/                    same shape, for version 2
    ├── decrecated_code/            retired code, kept for the record, never run
    └── ACTIVE_DATA_VERSION         which version the app shows (written by the web page)
```

### Shared or versioned? The rule

> **If changing the pipeline would change the file, it belongs to a version.
> If it is an input the pipeline only reads, it is shared.**

| Shared, in `storage/data/` | Versioned, in `storage/vN_data/` |
| --- | --- |
| Scanned slides (`original/oncostem_slides/`) | Anything a pipeline step wrote |
| Public datasets (BRACS, BACH, BCSS) | History of finished runs |
| Pathologist annotations and reader scores | Registration transforms |
| Client documents | Training tile/feature stores cut from the datasets |
| | Model checkpoints |
| | Score CSVs, review documents, problem write-ups |

Never write into `storage/data/`. If a new kind of input arrives (a new batch of
slides, a new reader sheet), add it under `storage/data/original/` once, and
every version can read it.

---

## 2. Choosing which version the demo shows

**In the web page.** `start.bat` takes no arguments. The page decides:

- **On opening:** if more than one version can be opened, the page asks which one
  to show and gives each version's one-line summary from its `VERSION.md`. It asks
  once per browser tab.
- **At any time:** use the **Version** menu at the top right. Picking a version
  restarts the server onto it and reloads the page. Anything still running is
  stopped first; the slides are not affected.
- **Start a new version:** the same menu has **+ Start a new version** (section 4).

The choice is saved in `storage/ACTIVE_DATA_VERSION`, so the next `start.bat`
opens the same version. If nothing has been chosen yet, the app opens **the
latest version**.

Scripts follow the same rule: `python data_versions.py --active` prints the
version they will use. To pin a batch job to one version without touching the
app's choice, set the environment variable `CSS_DATA_VERSION=v2` for that job.
When it is set for the server, the web page's menu is disabled.

---

## 3. When the code can no longer open an old version

The code will change, and at some point it will stop being able to read what an
older version wrote: a step folder gets renamed, or a record format changes. When
that happens, **nothing is deleted or converted.** The old version's folder is
kept exactly as it is, and the app runs the latest version it *can* open.

How it works:

- Each version's `version.json` records the **storage layout** it was written
  with, for example `{"version": "v1", "layout": 1, ...}`.
- `data_versions.py` declares `LAYOUT` (the layout this code writes) and
  `SUPPORTED_LAYOUTS` (the layouts it can still read).
- If the version that was chosen has a layout this code doesn't support, the app:
  - opens the **latest openable version** instead;
  - shows a notice in the page: *"v1 could not be opened. v1 was made by an older
    version of the app ... Its data is kept, untouched, in storage/v1_data.
    Showing v3, the latest version."*;
  - lists the old version in the Version menu as *cannot be opened*, so nobody
    can switch to it by mistake.
- To read old results anyway, check out the git commit recorded in that
  version's `VERSION.md`. That code reads its own layout.

**For developers, when you change how something is stored:**

1. Bump `LAYOUT` in `data_versions.py`.
2. Decide whether this code can still read the old layout:
   - **Yes:** add the new number to `SUPPORTED_LAYOUTS`, keeping the old one.
   - **No:** set `SUPPORTED_LAYOUTS` to the new number only. Versions written with
     the old layout will then be kept on disk but no longer opened.
3. Start a new version (section 4) so new runs are written with the new layout.
4. Note the change in the new version's `VERSION.md`.

---

## 4. Starting a new version

Start one when you change something that changes the numbers: a retrained or
replaced model, a new or reworked step, new cut points, a different region
source, or a storage-layout change (section 3). Do **not** start one for a UI
change or a bug fix that leaves the outputs the same.

1. **Create it.** In the app, choose **Version → + Start a new version**. This
   creates `storage/vN_data/` with `version.json` and empty `data/`, `results/`
   and `reports/`, then switches to it. A folder made by hand without
   `version.json` is treated as being in the current layout.
2. **Write `storage/vN_data/VERSION.md`** (copy the template in section 7).
   Record the date, the git commit, and what changed since the previous version.
   The page shows its first paragraph when it asks which version to open.
3. **Models:**
   - *Models unchanged:* do nothing. A version with no `models/` folder of its
     own uses the newest earlier version's.
   - *Models changed:* copy the previous `models/` folder whole into the new
     version first, then retrain or replace inside the copy:
     ```
     robocopy storage\v1_data\models storage\v2_data\models /E
     ```
     Training scripts refuse to publish into a version that has no `models/` of
     its own, because they would otherwise overwrite the earlier version's
     checkpoints. Copy the whole folder rather than one file: once `models/`
     exists, the version stops inheriting from the earlier one.
4. **Run the pipeline** on the cases you need. Outputs land in the new version,
   and the earlier one is left alone.
5. **Freeze the old version.** From now on, do not run anything against the
   earlier version. It is the record of what that version produced.

---

## 5. How history is stored

A finished run is filed into the **active version's** `data/history/` when it is
confirmed in the app, or when `score_all_slides/run_case.py` finishes a case.
Filing is a move, not a copy, out of `data/demo/`.

```
storage/vN_data/data/history/
└── CAN_00251/                       one folder per case, named by case id
    ├── manifest.json                  what was filed, when, from which slides
    ├── he_thumb.png
    ├── shared/                        H&E work every marker of the case reuses
    │   ├── tissue/  tissue_type/  tiling/  roi/  roi_selection/  roi_refinement/
    │   ├── calibration/
    │   └── slides/                    upload records only, NOT copies of slides
    └── markers/
        └── A/  F/  R/  U/  W/         one folder per IHC marker letter
            └── ihc_alignment/  nuclei/  per_cell/  binning/  scores/ ...
```

Rules for keeping history readable in future versions:

- **One case folder per version.** If v1 and v2 both scored CAN_00251, there are
  two folders, `storage/v1_data/data/history/CAN_00251/` and
  `storage/v2_data/data/history/CAN_00251/`. Never copy or merge history between
  versions; the version folder is what tells you which pipeline produced a number.
- **Re-running a case inside the same version replaces that case's filed work.**
  If you need the old numbers, copy the case folder into
  `storage/vN_data/data/score_archive/YYYY-MM-DD_<reason>/` first
  (`slide_registration/archive_scores.py` does this for batch runs).
- **Slides are referenced, never copied.** A history record points to the slide by
  its path under `storage/data/original/oncostem_slides/<CASE>/`. Keep that
  library's folder and file names stable. Renaming a case folder there breaks
  every version's records at once.
- **Paths inside records.** Records may hold absolute paths. Paths into
  `storage/data/original/` stay valid as long as the library isn't moved. Paths
  into `storage/vN_data/` point at that same version. Moving or renaming
  `storage/` or a version folder means rewriting those paths, as was done once on
  2026-10-05 (see `v1_data/VERSION.md`). A new version starts empty, so this only
  matters when an existing one is moved.
- **Deliverables go in `results/`, and write-ups go in `reports/`, of the version
  they describe.** A report that compares v1 with v2 belongs in the newer
  version's `reports/` (here, v2).
- **Name dated archives as `YYYY-MM-DD_<what>`** so they sort in time order.

---

## 6. Comparing and cleaning up versions

- **Compare:** the per-version score CSVs are
  `storage/vN_data/results/oncostem_ai_scores.csv`. Open two side by side, or run
  a review script once per version with `CSS_DATA_VERSION` set.
- **Disk:** a version without slides is small. Nearly all the bytes are in the
  shared `storage/data/original/`. `vN_data/data/demo/` holds the app's working
  state and can be cleared from the app's storage panel.
- **Delete a version** by deleting its folder. Before you do:
  - check that no later version inherits its `models/`
    (`python data_versions.py` lists each version and where its models resolve).
  - it doesn't matter if it's the one in `ACTIVE_DATA_VERSION`; the app falls
    back to the latest version.
- **Git:** all of `storage/` is gitignored. The commit that produced a version is
  recorded in its `VERSION.md`.

---

## 7. `VERSION.md` template

```markdown
# vN

- Started: YYYY-MM-DD
- Code: git commit <hash> (branch <name>)
- Based on: v<N-1>
- Models: own copy | inherited from v<N-1>
- Storage layout: <LAYOUT from data_versions.py>

## What this version is
One paragraph. The app shows it when it asks which version to open.

## What changed since v<N-1>
- ...

## Cases scored in this version
- CAN_00251 (A F R U W), ...

## Known problems
- link to reports/...
```

---

## 8. For developers

- Never write `WORKSPACE_ROOT / "data" / ...` or `ROOT / "results"` in code.
  Use [`data_versions.py`](data_versions.py):
  - `data_versions.ORIGINAL_ROOT`: the shared inputs.
  - `data_versions.data_root()`: this version's outputs.
  - `data_versions.models_root()`: models for reading (follows inheritance).
  - `data_versions.publish_models_root()`: models for writing (refuses to write
    into an inherited folder).
  - `data_versions.results_root()` and `data_versions.reports_root()`.
  - `data_versions.resolve()`: which version runs, what was asked for, and why
    they differ.
- Batch files and node scripts ask `python data_versions.py --active` rather than
  reading `ACTIVE_DATA_VERSION` themselves, so they follow the same fallback rule.
- The demo backend resolves the version once, at start-up (`app/core/config.py`),
  so switching restarts it. The endpoint rewrites `config.py`, which uvicorn's
  `--reload` (always passed by start.bat) restarts on, and then exits its own
  worker in case the reloader's Ctrl-C never reaches it. If the server was started
  without `--reload`, the page says to run stop.bat and then start.bat.
