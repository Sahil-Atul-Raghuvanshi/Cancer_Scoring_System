# Step 7 branch screen + four H&E models

## Context

Step 7 today offers one thing: a field-of-view picker hard-limited to 224/448 µm, whose choice
silently selects step 8's checkpoint. It needs to become an explicit three-way branch — **H channel
only**, **H&E ResNet segmentation**, **Run by BEETLE** (a placeholder) — with all four scales
(112/224/448/672 µm) on the two real branches, Back at every level, and step 8 running on the
committed choice with any earlier step 8 result erased.

Three findings shape the work:

1. **The H-channel branch is nearly free.** `tiling_service.FIELDS_OF_VIEW` already offers all four
   FOVs and all four H-channel checkpoints are already published in `models/tissue_type/`. Only the
   frontend is narrow (`FIELD_OF_VIEW_CHOICES = [224, 448]`).
2. **The H&E branch is genuinely new.** No H&E-vs-IHC flag exists anywhere; eosin appears only as a
   reference direction that step 6 uses to *detect faults*. And no RGB checkpoint exists — four must
   be trained.
3. **There is a live bug in the handoff.** `tissue_type_service` calls `tiling_service.report()` and
   `.tile_index()` with no parameters, so step 8 always rebuilds the grid at the **default** 224 µm
   even when the caller named the 448 µm head. The grids diverge, `inference._mark_inside` drops out
   of its fast path, and the untargeted call evicts step 7's single-entry memo. Persisting the step 7
   choice is what fixes this, so it is part of the same change.

Outcome: eight comparable checkpoints (4 H-channel × 4 H&E at one shared geometry), a step 7 screen
that explains itself to a non-expert, and an honest paired comparison of all eight at the repo root.

**Decisions taken (do not revisit):** H&E input is raw RGB with ImageNet mean/std — no optical
density, no deconvolution, no white point. The H&E option is gated automatically from step 5's point
cloud. One arm failing overnight is logged and skipped, not fatal. Changing step 7 deletes
`data/tissue_type/<id>/` outright.

**Two naming reconciliations** (the two design passes disagreed): the constant is
`CHANNEL_RGB_HE = "rgb_he"` and the function is `rgb_to_model_input`. A single
`settings.tissue_type_channel` is **not** added — `ModelBranch` with `DEFAULT_BRANCH = H_CHANNEL`
does that job.

---

## Load-bearing fact behind the whole design

`export.export_region` computes the tile grid and the class vote from the **mask** alone — the white
point, the deconvolution and `quantise` never enter the vote. So for one spec and one region list,
**the kept tile set and its `tile_id`s are identical no matter what pixels get stored.** That makes
per-tile pairing between the H and H&E stores free, and it is why the two stores are cut in two
independent passes: run that way, "the tile_ids match" is a real assertion; dual-written from one
loop it would be a tautology that verifies nothing.

---

## Phase A — the branch primitive (backend, invisible to the UI)

Must land **before** any RGB checkpoint is published: `model_for` matches on `tile_px * mpp` alone
today, so with two heads per FOV alphabetical order would decide whether step 8 runs the H or the
H&E head — and an H&E head fed optical density scores badly and *looks like a modelling result*.

- `step08_tissue_type_segmentation/input.py` — add `CHANNEL_HAEMATOXYLIN`, `CHANNEL_RGB_HE`, and
  `rgb_to_model_input(rgb)`: `float32 → transpose → /255 → ImageNet mean/std`. That is the entire
  transform. `descriptor(...)` gains `channel=CHANNEL_HAEMATOXYLIN`; the RGB dict carries
  `colour_space`/`channel_order`/`scale` and pins `invert_polarity=False`, `gamma=1.0`,
  `standardise_tile_p99=False`, raising if a caller asks otherwise (those are dye-concentration
  concepts with no meaning on a photograph).
  **The haematoxylin branch must not gain a single key** — `_verify_input_contract` iterates
  `expected.items()` and a key absent from an already-published manifest compares `None != value`,
  which would refuse all nine existing checkpoints. Highest-consequence trap in the plan.
- New `step08_tissue_type_segmentation/branches.py` (~40 lines, no torch): `ModelBranch` enum
  (`h_channel`/`he`/`beetle`), `DEFAULT_BRANCH`, `SERVED_BRANCHES`, `BRANCH_CHANNEL`,
  `branch_of_channel(None) -> H_CHANNEL`, `parse_branch`. Lives in step 8's package because the
  existing dependency direction is step 7 → step 8.
- `model.py` — `_verify_input_contract` reads `spec.get("channel", CHANNEL_HAEMATOXYLIN)` and passes
  it to `descriptor`; `"channel"` joins `required`. `Candidate` and `Pinned` each gain
  `channel: str = "haematoxylin"` plus a derived `branch` property. `_build_and_load` unchanged —
  both branches are `concat_resnet18_mlp`, 3-channel, 3-class.
- `tiling_service.py` — `model_for(fov_um, branch=DEFAULT_BRANCH)` with `candidate.branch is branch`
  in `matches()`. `FIELD_OF_VIEW_PREFERENCE` re-keyed to `dict[tuple[ModelBranch, float], str]`;
  **move, don't drop, the `(H_CHANNEL, 112.0)` entry** or the default 112 µm head silently changes to
  the legacy `invasive_tile_v3_concat_bach`. `window()` / `_offered_fields_of_view()` take the branch
  through. `FIELDS_OF_VIEW` stays one shared table — identical geometry is what makes the branches
  comparable.
- `tissue_type_service.capability()` — `served = {(branch, um): model_for(um, branch)}` over
  `SERVED_BRANCHES`; keep the existing reason substrings, `TissueTypePanel` renders them verbatim.

Nothing observable changes: every existing manifest resolves to `h_channel` and every existing call
takes the default.

**Gate:** `pytest` green in both repos, and `load_pinned(verify=True)` succeeds for all four
`invasive_tile_fov*_concat` heads.

---

## Phase B — training-repo edits (RGB export path)

- New `tissue_type_model_training/src/tilestore.py` — a frozen `Variant` dataclass and two
  instances, `H_CHANNEL` and `RGB_HE`, carrying `channel`, `stored_as`, `prepare`, `store`,
  `pil_mode`, `descriptor`, `i0_rule`, `stats`. **No arithmetic** — every callable is imported from
  `hchannel`, which re-exports the demo's `input.py`.
- `src/export.py` — `export_region` / `write_region` / `iter_regions` / `write_manifest` each gain
  keyword-only `variant: Variant = H_CHANNEL`. Inside `export_region`, the two lines computing
  `white`/`h_od` become `plane = variant.prepare(rgb, spec)`; `stored=quantise(...)` becomes
  `variant.store(pixels)`; the stat columns and `i0_rule` come from the variant. Two columns appended
  to every row on both variants: `channel`, `variant`. The grid loop, `vote()` and `resample()` are
  untouched, so **existing H-channel output stays byte-identical**.
- `scripts/02_export.py` — `--channel {haematoxylin,he}` (default `haematoxylin`), threaded through
  `run_export`, `export_borrowed`, `write_region` and **`gate_g2`** (its determinism re-cut must use
  the same variant). `reuse_bcss_rows` must `SystemExit` on a channel mismatch against
  `summary["input"]["channel"]`.
- `src/hchannel.py` — re-export the two constants and `rgb_to_model_input`. Re-export only; a second
  copy of the transform is the exact failure this module exists to prevent. No `src/rgbtile.py`.
- `src/datasets.py` — `TileDataset` gains `channel`; on RGB it opens `.convert("RGB")` and calls
  `rgb_to_model_input`. `geometric` already works on HxWx3. **Stain jitter is dropped on the RGB
  path** (alpha/beta/gamma are OD units); raise loudly rather than skip silently. New
  `channel_of(tiles_dir)` reading `export_summary.json["input"]["channel"]`, defaulting to
  haematoxylin and cross-checking the manifest column.
- `src/models.py` — `publish()` gains `channel=`.
- `scripts/03_features.py` — read `channel_of`, refuse `--standardise`/`--invert` on RGB, and write
  `.npz.tmp` + `os.replace` so a killed run cannot leave a cache `np.load` half-opens.
- `scripts/08_fit_concat_mlp.py` — `load_pair` falls back `{init}_std.npz` → `{init}.npz` and returns
  the standardise flag; **stop hard-coding `standardise=True` in `publish()`** (left unfixed, an RGB
  arm publishes a manifest asserting a transform that never ran).
- `src/report.py` — lift `score`, `wilson`, `load_head` out of `scripts/09_head_to_head.py` so
  Phase F reuses one definition.

---

## Phase C — data reorganisation

Target: `data/h_channel/{112,224,448,672}um/` and `data/he/{112,224,448,672}um/`, each with all three
sources in the existing `{bcss,beetle,bach}/<class>/` shape.

```powershell
New-Item -ItemType Directory -Force data\h_channel, data\he
Move-Item data\112um data\h_channel\112um     # same volume - a rename, not a 1.7 GB copy
Move-Item data\224_um data\h_channel\224um    # the underscore is fixed in the move
Move-Item data\448um data\h_channel\448um
Move-Item data\672um data\h_channel\672um
```

**Leave `data/tiles` and `data/features` where they are** — they are `backend_path.TILES_DIR` /
`FEATURES_DIR`, `02_export.py` has no `--out`, and `07_split_export_by_source.py` re-files out of
them.

Nothing in code depends on the old names (`tile_path` in every manifest is relative and
group-prefixed; `probe_tiles[]` in the published manifests carry no path). Verify after: every
`tile_path` resolves; both npz fingerprints still match
`manifest_fingerprint(by_source_authority(read_manifest(...)))`; `backend_path.REGIONS_DIR.name ==
"regions"` (the fallback checks for `DATA_DIR/dcis` — neither new directory is named that, but assert
it, because silently reading the wrong trees is nasty); and a `08` re-fit with no `--publish`
reproduces each recorded `08_concat_mlp.json` metric exactly (full-batch, `seed=0`, so exact is the
right bar). Stale `tiles_dir` provenance strings stay — `load_pinned` never reads them, and
hand-editing verified manifests at 3 a.m. is the class of thing that goes wrong.

---

## Phase D — the overnight campaign

**Checkpoint names:** `invasive_tile_fov{112,224,448,672}_he_concat`. The name is for humans; the
branch filter from Phase A is what makes them safe to publish.

Per arm, in order **672 → 448 → 224 → 112** (cheapest first, so a blown budget lands inside the
expensive arm with the others banked):

```powershell
& $PY scripts\02_export.py --channel he --tile-px 224 --mpp <MPP> `
      --include dcis ic normal bach_insitu bach_invasive bach_normal
& $PY scripts\07_split_export_by_source.py --dest data\he\<FOV>um --remove-source
& $PY scripts\10c_assert_paired.py --a data\h_channel\<FOV>um --b data\he\<FOV>um
& $PY scripts\03_features.py --init imagenet --tiles data\he\<FOV>um --features data\he\<FOV>um\features
& $PY scripts\03_features.py --init simclr   --tiles data\he\<FOV>um --features data\he\<FOV>um\features
& $PY scripts\08_fit_concat_mlp.py --tiles data\he\<FOV>um --features data\he\<FOV>um\features `
      --publish invasive_tile_fov<FOV>_he_concat
```

Recipe unchanged from the H arms: two frozen ResNet18 bodies (ImageNet + SimCLR) → concat 1024 →
`Linear(1024,256) → ReLU → Dropout(0.2) → Linear(256,3)`, 400 epochs, lr 1e-3, wd 1e-4, seed 0,
AdamW, full-batch, inverse-frequency class weights, τ = 0.40. Source-authority
(`datasets.SOURCE_CLASSES`) and the institution/patient splits are preserved untouched.

**Budget (scaled from measured H timings: export ×1.4 for 3× the PNG bytes, features ×1.15):**

| arm | export+split | imagenet | simclr | head | total |
|---|---|---|---|---|---|
| 672 µm | ~10 m | 1 | 1 | 2 | **~14 m** |
| 448 µm | ~12 m | 2 | 2 | 2 | **~18 m** |
| 224 µm | ~20 m | 6 | 7 | 3 | **~36 m** |
| 112 µm | ~32 m | 44 | 52 | 4 | **~2 h 12 m** |

Plus pre-flight, the moves, verification and the comparison → **≈ 4 h 10 m against 8–9 hours.**
Hard total budget **360 min**; roughly 2× headroom, which is the right amount unattended.

**672 µm will be published but is not interpretable, and the tracker must say so:** 268 tiles, 76
held out, of which the in-situ truth row is **11**. BACH contributes zero (a 672 µm window needs 1344
source px; BACH regions are 1720 × 1290, so the height fails), making it the only cell of the 4 × 2
grid with two sources instead of three. Any rate there is k/11 with a ~30-point Wilson interval.
Do not rank it against another FOV.

### Driver, tracker, pre-flight, watcher

**One resumable Python driver**, `tissue_type_model_training/scripts/10_he_campaign.py` — Python
rather than PowerShell because PS 5.1 has no `&&`, no ternary, ANSI-defaulting `Set-Content` and
here-strings that die on an indented `'@`; and because every resume predicate has to read an `.npz`.
Launched detached from the workspace root with `Start-Process -WindowStyle Hidden`, so closing the
editor orphans but does not kill it.

Stage table, each with `id`, `argv`, `done_when`, `budget_minutes`, `on_fail`:
`A0 preflight` → `A1 move` → `A2 verify` → `S0 pytest` → per FOV `E1 export`, `E2 split`,
`E3 pair_check`, `E4/E5 features`, `E6 head+publish` → `F1 compare8` → `F2 docs`.
`A0`/`A1`/`S0` abort; an `E*` failure skips the rest of that FOV and moves to the next; `E3` is a
report, not a dependency, so it continues.

**Resume** is by `done_when` predicates that actually verify, not by presence: `E4/E5` must
`np.load` the cache and match its `fingerprint` inside a `try`; `E6` must match
`sha256_of(<name>.pt)` against the manifest's recorded hash. Re-launching skips every `DONE` stage.

**Tracker — two files at the repo root, one writer, whole-file atomic rewrite** (`.tmp` +
`os.replace`). `he_campaign_state.json` is machine-readable (per-stage status/timestamps/elapsed/tile
counts/exit code/reason, `published`, `failures` with log tails, `heartbeat`, `git_dirty_at_start`).
`HE_CAMPAIGN_TRACKER.md` is rewritten in full from it after every transition and every 60 s, never
appended — the duplicated stage lines in `PROGRESS.md:84-110` are what two appenders sharing a file
produces. It leads with a one-line verdict, then the stage table, the published table, the failures
with log paths, and a **"first thing in the morning"** section holding the literal next commands.
Display timestamps are local-with-offset; **all arithmetic uses `time.monotonic()`** so an NTP step
cannot make a stage look instant.

**Pre-flight** — `HE_CAMPAIGN_PREFLIGHT.md` (root, human) + `scripts/10a_preflight.py` (mechanical,
non-zero exit on any FAIL). Sixteen items; the ones that matter:

1. **Disk is the tightest constraint in the whole plan.** C: is at **97%, 19 GB free**, measured. RGB
   PNGs are **×3.0** the size of H PNGs (99.6 KB vs 33.2 KB at 224 px, measured on real crops), so
   the he stores need ≈ 4.9 GB + 0.14 GB features + 0.36 GB checkpoints ≈ **5.4 GB**. `FAIL < 12 GB`
   at A0, re-checked `FAIL < 8 GB` before every stage. Peak equals final because
   `07 --remove-source` moves.
2. `torch.__version__` is `2.8.*` — **≥ 2.9 breaks on Windows MAX_PATH**.
3. Both pretrained backbones load through `models.resnet18_backbone` **now**, so the strict-`fc`-only
   assertion fires here and not 44 minutes into `E4_112`.
4. All six region trees: `source_mpp == 0.5`, `teaches_classes` matches `bcss.BORROWED_TREES`, and
   exact pair counts (dcis 239, ic 202, normal 213, bach_* 100 each). This is what catches the `ic`
   tree having grown — 115 regions are recorded as deferred, and if the teacher was re-run since the
   H arms were cut, the pairing assertion fails 32 minutes into an export instead of in seconds here.
5. BCSS 151 pairs + `slide_mpp.json` covering every `slide_key` (the check `02_export.py` does and
   then exits on).
6. No other `python.exe` alive > 60 s; the demo backend not listening (a live API holds `.pt` files
   in `model._CACHE`); no pending Windows Update reboot.
7. Sleep/hibernate disabled on AC (`powercfg /query SCHEME_CURRENT SUB_SLEEP` → `0x00000000`), and
   print the fix on failure.
8. All nine published manifests carry `input.channel`, and `load_pinned(verify=True)` succeeds for
   all four `fov*_concat` heads **after** the Phase A edit — the gate that stops the descriptor
   change bricking the served models.
9. `model_for(112.0).branch is H_CHANNEL` — asserts the branch filter is in place before any RGB head
   can be published. Re-asserted after every `E6`.

**Watcher — the driver is its own; do not run a second process.** It polls each stage's `Popen` every
30 s, records `log_bytes`, rewrites the heartbeat and the tracker. If the stage has not changed and
the log has not grown for **20 minutes** it sets `overall: STALLED` and banners it — `02_export.py`
prints per region and `03_features.py` per 20 batches (~40 s), so 20 minutes of silence is genuinely
wrong. **A stall is a report, not an intervention**: only the budget kills, because killing a
slow-but-working feature pass throws away an hour. On exceeding the total budget the driver
**finishes the stage it is in** (killing mid-`torch.save` is how a half-written 90 MB checkpoint gets
published), writes `BUDGET_EXCEEDED` plus the remaining commands in order, and exits 2. A `--watch`
flag on the driver itself prints the tracker top every 30 s, read-only.

Observation in the morning: `Get-Content .\HE_CAMPAIGN_TRACKER.md`.

---

## Phase E — persistence, erase, detector, screen

### E1. Persist the step 7 choice (and fix the diverging-grid bug)

An on-disk selection record, **not** threaded parameters — `_run` has four bare service calls,
`PipelineContext` is `{upload_id, artifacts}` with nowhere to put four parameters, and any caller
that forgets one reverts to the default FOV, which is *the shape of the bug being fixed*.

- New `settings.tiling_dir` → `data/tiling/<upload_id>/selection.json` holding
  `{branch, field_of_view_um, overlap, tissue_threshold, model, tile_px, mpp, chosen_at}`. Kilobytes;
  the index itself stays in memory per the step's own contract.
- `TilingService.selection` / `commit_selection` / `forget_selection`; `report`/`panel`/`tile_index`/
  `_run` gain `branch=None` and resolve **explicit argument → record → settings default**. That one
  change fixes the bug with no caller edits.
- `POST /api/v1/tiling/{id}/selection` commits — never as a side effect of a GET, or "glanced at
  672 µm" would be indistinguishable from "chose 672 µm" and every panel GET would rewrite it.
- Memo goes from one entry to a 4-entry LRU keyed `(upload_id, footprint.key, share, size, mpp,
  branch)` — the sample panel differs per branch, and one entry means a report and its two panels
  evict each other.
- `resolve_params` reads the record; a caller naming a disagreeing `branch`/`fov` gets **409**, not a
  silent run at a different geometry. **Do not forward the caller's `overlap` into
  `tiling_service.report`** — it would collapse `overlap` and `tile_overlap` and break
  `test_step_8_runs_at_step_7s_overlap_rather_than_its_own_setting` and the distinction it protects.
- The worker's four bare calls get the same parameters from `job.params`; `tissue_service.footprint`
  and `calibration_service.white_point` also currently drop the threshold — same bug one step out,
  fixed here.
- Branch-aware pixels: `haematoxylin_reader` → `window_reader(reader, white, *, base_mpp, branch)`,
  returning raw RGB with no white point and no OD on `he`; `inference.classify`'s
  `read_haematoxylin` becomes `read_window` with a `channel` argument selecting `to_model_input` vs
  `rgb_to_model_input`. Keep a deprecated alias — gate G7b's `check_tissue_geometry.py` calls it.
- `TissueTypeParams` gains `branch`, `field_of_view_um`, `input_channel`; all three join
  `_cache_key`.

### E2. Erase on change

Service level in `TissueTypeService`, on mismatch — not a new endpoint (a client that forgets to call
it is the failure mode), not `maintenance_service` (its `dry_run` semantics are administrative).

- `_discard(upload_id)` — `shutil.rmtree` the **whole** directory, and refuse while a pass is
  queued/running (a run writes into a directory being deleted underneath it). Then
  `roi_service.discard(upload_id)`.
- `_erase_stale(upload_id, params)` — same `_cache_key` → keep the cache; different → discard. The
  negative control matters: a matching choice must still return the cached pass, because that is half
  an hour of someone's CPU.
- **`restart()`'s `report.unlink()` becomes `_discard()`** — the existing bug where `classmap.npz`
  and all four PNGs survive, leaving `asset()` serving the previous choice's `map.png`. A strict
  improvement independent of this feature.
- `roi_service.discard` is necessary because `roi_service.build` caches on **its own** params only, so
  a fresh class map otherwise leaves a stale ROI that looks fresh. `maintenance_service._derived_dirs`
  gains `roi-mask` and `tiling` with `REGENERATION_COST` entries — `roi_dir` being absent today is an
  existing hole this closes.

### E3. The H&E detector

New pure module `step05_optical_density/staining.py` — takes step 5's own `Cloud`, imports no
service. `SlideStaining` enum (`he` / `haematoxylin_dab` / `single_stain` / `unknown`) and a frozen
`StainingVerdict` carrying `is_he`, a one-sentence `reason`, `source`, and every number that decided.

```
cloud is None or len(arms) < 2                              -> UNKNOWN          is_he=False
not cloud.two_armed                                          -> SINGLE_STAIN     is_he=False
any(a.nearest == "eosin" and a.degrees_from_nearest <= tol)  -> HE               is_he=True
any(a.nearest == "eosin")                                    -> UNKNOWN          is_he=False  (near-miss)
otherwise                                                    -> HAEMATOXYLIN_DAB is_he=False
```

`two_armed` is already the "is there a second dye" test — its own docstring says False is "what a
tile carrying only a counterstain looks like" — so using it is not a reinterpretation. `arms[].nearest`
is a min over all three reference vectors including eosin, so "nearest eosin" already means *closer
to eosin than to DAB*.

**Tolerance is a new setting, `he_eosin_tolerance_deg`, not `density_angular_tolerance_deg`** — that
one is 1.5° and is a per-pixel *direction-stability* bound; 1.5° against a published reference vector
would reject every real slide. **Calibrate before shipping the default**: one read-only measurement
over `CAN_00270_26_H&E.svs` (expect an eosin-nearest arm) and `CAN_00251_26_A.svs` (expect DAB) gives
the two numbers it must separate; record the measured pair in the setting's docstring. `separation` vs
`reference_separation` is evidence in the `reason`, not a gate.

The verdict is cached per upload in `density_service._staining`, written at the end of each successful
`_run`, and read by `tiling_service` as a **dict lookup that never triggers a step 5 run** — forcing
one from step 7 would read a tile, step 3's mask and step 4's white point behind a screen advertising
"one pass over a mask", and would thrash the memo step 5 shares with step 6. When it is absent the
verdict is `UNKNOWN(source="step5-not-run")` and says so.

Reconcile the prose rather than contradicting it: step 5's fault wording for an eosin-nearest arm
becomes conditional on the verdict, and step 6 gains one sentence explaining that it is deconvolving
an H&E section on an H-DAB basis and that *this is exactly why* step 7's H&E branch does not use its
channel at all.

New `GET /api/v1/tiling/{id}/branches` → `TilingBranches` (staining verdict + one `TilingBranchOut`
per branch with `enabled`, `reason`, `implemented`, and its own `fields_of_view`). Manifests plus the
cached verdict only — no pixels, no grid, no torch. `branch` joins the two existing GETs as a
`Query`; an unknown value is 422 at the boundary and `beetle` is 409.

### E4. The screen

Step 7 becomes ask-first — `isGated = isQC || isTiling || isTissueType` — not because it is expensive
(it is seconds) but because **the branch decides the grid's geometry**, and auto-starting would lay a
grid nobody chose and then have to erase step 8.

- `useTiling.ts` — `BRANCH_CHOICES`, `FIELD_OF_VIEW_CHOICES = [112, 224, 448, 672]`, and two-level
  state: `branch`, `pendingFov`, derived `level: 'branch' | 'fov' | 'report'`, plus `chooseBranch`,
  `chooseFieldOfView`, `backToFov`, `backToBranch`. `start()`, `setFieldOfView` and `setOverlap` all
  go through the **same** `commitTilingSelection`, so there is one write path and therefore one
  invalidation path.
- New `BranchPicker.tsx`, sibling of `FieldOfViewPicker.tsx`, reusing `tl-picker`. All three options
  always visible; `he` disabled renders the detector's `reason` in the `tl-picker__model` slot;
  `beetle` disabled unconditionally with **"yet to be developed"**. `BRANCH_LABELS` keyed on the
  literal union so an unlabelled branch is a compile error. Plain-language copy up front ("Blue stain
  only" / "Full colour (H&E)"), mechanism in a `<details>` — the standing preference.
- `FieldOfViewPicker.tsx` — `LABELS` entries for 112 and 672 (mandatory, compile error otherwise), a
  `branch` prop so `offered` comes from that branch's list, and a branch-aware unavailable string
  ("H&E model not trained yet at this scale"). `fovCounts` keyed `` `${branch}:${um}` `` so
  "never estimated, only measured" stays literally true.
- Back at every level: `← Choose a different approach` → `backToBranch()`, `← Change the field of
  view` → `backToFov()`, alongside the existing page `← Back` to step 6 and the untouched rail.
- **One missing primitive:** `usePipelineRun` cannot un-complete a step, so a changed step 7 leaves
  step 8 reading "Completed" with nothing behind it. Add `invalidateAfter(index)` dropping
  `completed`, `skipped` **and `attempts`** past it — clearing attempts is what makes step 8's button
  read "Classify every patch" again rather than "Try again".
- **The sample panel must follow the branch.** `_sample_png` renders step 6's H channel
  unconditionally; add `overlay.rgb_sample_png` and branch on `run.branch`, with the caption
  switching on `params.inputChannel`. Otherwise the H&E branch shows a picture of an input it does
  not use.
- `useTissueType` takes `branch` and `fieldOfView`, and its six potential clear-effects collapse to
  one derived `inputsKey = [uploadId, tissueThreshold, tileOverlap, modelName, branch, fieldOfView]`.
  This is what fixes the reported symptom — `modelName` is missing from the list today, so changing
  the FOV leaves a stale step 8 report on screen.

### E5. Before the H&E heads exist

No special-casing needed; the existing `available` mechanism covers it. `model_for(fov, HE)` → `None`
→ `window()` falls back to the planned geometry → step 7 still lays and prices a **real** grid with
`model: null`, and the picker renders four disabled scales. `TilingBranchOut` for `he` stays
`implemented=True, enabled=<staining gate>` so an H&E slide can select the branch and *see* that
training is in progress. Publishing the four RGB checkpoints later needs **no code change**.

### E6. Schemas, types, catalogue

Python: `schemas/tiling.py` (`TilingBranchName`, `TilingStaining`, `TilingBranchOut`,
`TilingBranches`, `TilingSelection`, `TilingSelectionIn`; `TilingParams.branch`/`.input_channel`;
`TilingReport.staining`/`.branches`), `schemas/tissue_type.py`, `schemas/density.py`
(`DensityReport.staining` — the verdict belongs on the screen that owns the evidence, so step 7
quotes step 5 rather than asserting something of its own). Mirrored camelCase in
`frontend/src/types/{tiling,tissueType,density}.ts`; `branch` into `api/tiling.ts`'s `query()` so
every panel URL is per-branch.

`backend/app/data/pipeline_steps.py`: step 7's `input_label` "Haematoxylin-channel tissue region" is
**a lie after this change** → "Tissue region, as the chosen model sees it"; `action_label` → "Choose
an approach and tile"; step 8's `how` hard-codes "haematoxylin channel … 224 px at 0.5 um/px" and
must name both branches and four scales. Then re-run `backend/scripts/sync_frontend_catalogue.py` —
never hand-edit `stages.ts`.

---

## Phase F — compare all eight models

New `scripts/10b_compare_eight.py`. `09_head_to_head.py` is the right scoring *kernel* and the wrong
*driver*: its premise is one export, one feature space, heads differing only in fitting data — and
that fails across channels (an H&E head consumes RGB-derived vectors) and across FOVs.

Per cell of the 4 × 2 grid: held-out rows from `institution_split` carrying `tile_ids`, predictions at
**both argmax and τ = 0.40**, then accuracy, per-class recall, macro/min recall, Dice
invasive-vs-in-situ, `insitu_called_invasive` **with its k/n and Wilson interval**, the full 3 × 3
confusion matrix, and a `by_source` breakdown of the held-out in-situ row — because "9 human in-situ
tiles in all of BCSS" is the fact that makes every class-1 number in this project fragile.

Within a FOV, across channels — the honest paired comparison the two-pass export bought:
**McNemar on the in-situ rows** (`only_h_wrong` / `only_he_wrong`, continuity-corrected) as the
headline test; **Wilcoxon signed-rank on per-slide Dice** via the repo's own `paired_comparison`
(which raises below 3 shared slides — at 672 µm report *"not computable, n_slides = k"* rather than
omitting it); and a paired accuracy delta bootstrapped **over slides, not tiles** — resampling tiles
would treat 200 near-duplicate tiles of one slide as 200 independent observations. Report
`len(A)`, `len(B)`, `len(A∩B)` even though `E3` asserted they are equal, so a silent divergence
appears in the document and not only in a log.

Across FOVs: point estimates only, each labelled **uncontrolled**.

Output **`HE_VS_HCHANNEL.md`** + `he_vs_hchannel.json` at the repo root, with four warnings **above**
the tables, not in a footnote: window-size selection (448 µm needs 896 source px, 672 µm needs 1344,
so smaller regions contribute nothing — systematic, and nothing here measures whether small regions
are harder); BACH absent at 672 µm; n = 76 held out at 672 µm of which 11 are in-situ; and held-out
in-situ row sizes differing ~200× across FOVs, so every percentage carries its k/n. The verdict
section must be allowed to say "no separation", as `RESULTS_224_VS_448.md` already does at p = 0.055.

Then: **`models.lock.json`** gains all **eight** tissue-type heads (it is stale — it never got the
four `fov*_concat` ones) with real bytes, the `sha256` each manifest already records, and
`sources: []` because they are not downloadable. Pointer sections in `RESULTS.md` and
`RESULTS_224_VS_448.md`; **one** section in `PROGRESS.md` pointing at the tracker rather than
duplicating it; path updates in `FIX1_FIX2_PLAN.md` and `HOW_THE_PIPELINE_WORKS.md`.

---

## Execution order

| # | block | unattended? | budget |
|---|---|---|---|
| 1 | Phase A + Phase B edits, new driver/preflight/compare/pair scripts. `pytest` green both repos; all four H heads still `load_pinned(verify=True)` | no | — |
| 2 | Copy this plan to the repo root as `STEP7_BRANCH_PLAN.md`; write `HE_CAMPAIGN_PREFLIGHT.md` | no | — |
| 3 | Run `10a_preflight.py`, fix every FAIL, launch the campaign detached | no | 5 m |
| 4 | Campaign: Phase C moves → verify → 672 → 448 → 224 → 112 → Phase F | **yes, overnight** | 360 m hard |
| 5 | Phase E, in order E1 → E2 → E3 → E4 → E6, while the campaign runs | no | — |
| 6 | Morning: read the tracker, flip `he` to selectable, run step 8 on both branches | no | — |

**Block 5 runs concurrently with block 4 safely** because the campaign imports only
`step08.../{input,model,classes}.py` and `app/common/*` from the demo, all of which are finished in
block 1. **Do not touch those four files while the campaign is running** — everything else (services,
schemas, endpoints, the whole frontend) is free.

---

## Verification

**Backend, per phase:**
`cd tissue_scoring_demo/backend; .venv\Scripts\python -m pytest tests -q` after
each of A, E1, E2, E3. The new cases that carry the plan:

- `test_two_heads_at_one_field_of_view_are_told_apart_by_their_branch` — both branches published at
  224 µm, each resolving to its own, neither shadowing the other. The case the geometry rule used to
  resolve by accident.
- `test_a_manifest_with_no_channel_is_the_haematoxylin_branch` — the backward-compatibility pin.
- `test_step_8_rebuilds_step_7s_index_at_the_committed_field_of_view` — commit 448 µm, assert the
  index has 448's `size`/`mpp`/`span`/`stride`, that `_mark_inside` takes its identity fast path, and
  `grid.gated_out == 0`. **This is the regression test for the whole motivation; it fails at 224
  today.**
- `test_the_he_branch_needs_two_arms_and_one_of_them_eosin` — table-driven over synthetic `Cloud`s, no
  slide needed. Plus `test_the_reason_names_the_number_that_decided` and
  `test_the_gate_does_not_run_step_5_as_a_side_effect` (monkeypatch `density_service._run` to raise).
- `test_changing_the_field_of_view_erases_the_cached_class_map`, `..._the_branch_...`,
  `test_a_matching_choice_still_returns_the_cached_pass` (the negative control),
  `test_restart_removes_the_whole_directory_and_not_only_the_report`,
  `test_erasing_step_8_also_erases_the_region_built_from_it`,
  `test_erasing_refuses_while_a_pass_is_running`.
- `test_the_he_branch_is_offered_with_every_scale_unavailable_when_nothing_is_published` and
  `test_publishing_an_he_head_lights_up_its_scale_with_no_code_change`.

**Existing tests that will break and must be fixed in the same commit:** `test_tiling.py`'s
`_stub_candidate` needs `channel`/`branch` (all four stubs go `None` without it) and the tuple-keyed
preference lookups; `test_tissue_type.py:1163` and `:1420`'s `lambda upload_id:` stubs `TypeError`
the moment `resolve_params` passes keywords → `lambda upload_id, **_:`;
`test_a_cache_written_before_a_setting_existed_invalidates_rather_than_crashing` needs the three new
`_cache_key` entries.

**Training repo:** `python -m pytest tests -q`, plus the Phase C reproduction check (each moved arm's
`08` re-fit reproducing its recorded metrics exactly) and `10c_assert_paired.py` per arm — an exact
sorted `tile_id` list comparison plus per-tile agreement on `label`, `source`, `slide_id`,
`institution`, `x`, `y` and the three vote fractions. Every paired statistic in Phase F rests on this
and nothing else would notice it failing.

**Frontend:** no test runner. `npx tsc --noEmit` plus the repo lint. Two changes are
compile-error-by-design and act as the net: `LABELS` keyed on `FIELD_OF_VIEW_CHOICES` and
`BRANCH_LABELS` keyed on `BRANCH_CHOICES`.

**End to end, in the running app** (`/run` or the documented start commands): load
`CAN_00270_26_H&E.svs`, walk to step 7, confirm the H&E option is **enabled** with a reason naming
the eosin angle; load `CAN_00251_26_A.svs` and confirm it is **disabled** with the DAB reason; confirm
"Run by BEETLE" is disabled reading "yet to be developed". Then: choose H channel / 448 µm, run step
8, note the tumour content; Back to step 7, switch to 224 µm, confirm `data/tissue_type/<id>/` is gone
and step 8's panel is empty rather than showing the old map; run again and confirm the numbers differ.
Repeat on the H&E branch once the heads are published, confirming the sample panel shows the colour
photograph and `params.inputChannel == "rgb_he"`.
