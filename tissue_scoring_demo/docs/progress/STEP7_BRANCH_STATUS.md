# Step 7 branch + H&E models — what was built

Companion to [`../design/STEP7_BRANCH_PLAN.md`](../design/STEP7_BRANCH_PLAN.md) (the plan) and
[`HE_CAMPAIGN_TRACKER.md`](HE_CAMPAIGN_TRACKER.md) (the overnight training run, which is
the live document — read that one first for arm-by-arm status).

## Where things stand — done

**Training: 4 of 4 published, zero failures, 156 minutes** against a 360-minute budget.
Every arm exported, split, **passed the pairing assertion**, extracted both feature
caches and published. Each H&E store held exactly the same tile count as its
haematoxylin twin — 36,539 / 6,907 / 1,157 / 268 — with identical tile ids and identical
votes, which is what makes the paired comparison below sound rather than merely
plausible.

**Integration.** 453 backend tests under random ordering, 84 in the training repo,
`tsc -b` clean, `eslint --max-warnings 0` clean, `setup.py --check` green over all 20
locked checkpoints, and **all eight heads pass G6** (worst logit delta 2.5e-06 to
5.5e-06).

## The result

`insitu -> invasive` is an in-situ tile called invasive — the confusion the whole score
is gated on. Lower is better.

| fov | held out | H insitu→inv | **H&E insitu→inv** | H acc | **H&E acc** |
|---|---|---|---|---|---|
| 112 µm | 9,393 / 174 slides | 18.9% [17-21] | **10.5% [9-12]** | 0.846 | **0.880** |
| 224 µm | 1,848 / 164 slides | 15.3% [12-19] | **13.6% [11-17]** | 0.853 | **0.874** |
| 448 µm | 320 / 116 slides | 6.8% [3-15] | 6.8% [3-15] | 0.881 | 0.872 |
| 672 µm | 76 / 40 slides | 9.1% [2-38] | 18.2% [5-48] | 0.882 | 0.921 |

**At 112 µm — the only well-powered cell — keeping the colour is a large, unambiguous
win on the tile level.** The in-situ error nearly halves and the intervals do not
overlap; McNemar on the in-situ row is 327 tiles only haematoxylin gets wrong against
100 only H&E gets wrong, p < 1e-16.

**And the per-slide test disagrees, which is the part worth reading twice.** Over 172
slides the mean per-slide Dice difference is +0.016 with a bootstrap interval of
[-0.007, +0.039] — inside the noise. That is not a contradiction: McNemar counts tiles
and there are 9,393 of them, but tiles from one slide are not independent; the bootstrap
counts slides, which is the conservative and clinically relevant unit. Read the tile
test as "the labels changed, a lot" and the slide test as "how much a case's answer
moved". Both are in `HE_VS_HCHANNEL.md` because neither on its own is the answer.

At 224 µm the same direction, weaker (p = 0.09 on the in-situ row). At 448 µm the two
are indistinguishable. At 672 µm n = 11 in-situ tiles and nothing there is readable.

**Nothing has been switched over.** `config.tissue_type_model` remains a haematoxylin
head, and the H&E branch is offered on step 7 only where step 5 found eosin. Whether a
10.5%-versus-18.9% tile-level gain that does not clear a slide-level interval is worth
changing the default is a decision, not a measurement.

## The thing worth checking first

The two branches now share all four geometries, and each resolves to its own head:

```
   112 um  h=invasive_tile_fov112_fix1_concat   he=<as published>
   224 um  h=invasive_tile_fov224_concat        he=<as published>
   448 um  h=invasive_tile_fov448_concat        he=invasive_tile_fov448_he_concat
   672 um  h=invasive_tile_fov672_concat        he=invasive_tile_fov672_he_concat
```

Verified after the first publish: `model_for(672, HE)` returns the new head,
`model_for(672, H_CHANNEL)` still returns the old one, and neither shadows the other.
This was the highest-consequence risk in the whole plan — geometry alone stopped being a
unique key, and letting `discover()`'s alphabetical ordering decide would have fed a
colour model optical density. That does not raise; it scores badly and reads as a
modelling result.

## What changed, by area

### The branch is the input contract

`step08.../input.py` gained `CHANNEL_HAEMATOXYLIN` / `CHANNEL_RGB_HE` and
`rgb_to_model_input`, which is four lines: `/255`, move the channel axis, ImageNet
normalise. No optical density, no white point, no deconvolution, no clip, no gamma, no
polarity, no per-tile standardisation — every one of those is a statement about dye
concentration and this branch makes none of them. `descriptor(channel="rgb_he")`
*refuses* a manifest claiming any of them.

The haematoxylin descriptor gained **nothing**, deliberately.
`_verify_input_contract` compares key by key, and a key absent from an already-published
manifest compares `None != value` — one added field would have refused all nine existing
checkpoints. All four `fov*_concat` heads were re-verified after the change.

New `step08.../branches.py`: `ModelBranch`, `DEFAULT_BRANCH`, `SERVED_BRANCHES`,
`branch_of_channel(None) → h_channel` (the backward-compatibility rule that keeps every
legacy checkpoint resolvable).

### The choice is persisted, which fixes a live bug

`data/tiling/<id>/selection.json` holds branch, field of view, overlap and threshold.

Step 8's worker used to ask `tiling_service` for the index with **no arguments**, so it
rebuilt the grid at the configured 224 µm no matter which head the caller had named. The
grids diverged, `inference._mark_inside` silently left its identity fast path, and the
untargeted call evicted step 7's single-entry memo on the way past. Every read now
resolves *explicit argument → committed record → settings default*, so the no-argument
call is correct by construction rather than by everyone remembering.

Also fixed alongside it: `tissue_service.footprint` and
`calibration_service.white_point` were called bare in the same worker, dropping the
threshold. The memo went from one entry to a four-entry LRU keyed on the branch too,
because the sample panel differs between branches even where the geometry does not.

### Changing step 7 erases step 8 — properly

`discard()` removes the whole directory, not just `report.json`. The old `restart()` left
`classmap.npz` and all four PNGs behind, so `asset()` kept serving the previous choice's
`map.png` and `class_map()` handed step 9 the previous choice's labels. `data/roi/<id>/`
goes too, because step 9 caches on its *own* parameters and cannot notice that the labels
underneath it changed.

`_cache_key` gained `branch`, `field_of_view_um` and `input_channel`. `discard` refuses
while a pass is running. `maintenance_service` now inventories `tiling_dir` and
`roi_dir` — `roi_dir` was missing entirely, so a region outlived the slide it was drawn
from.

### The H&E gate is measured, not guessed

`step05_optical_density/staining.py` reads step 5's existing point cloud: two separate
dye directions, one of them within `he_eosin_tolerance_deg` of Ruifrok's eosin vector.

Measured over 18 tissue tiles, three on each of six of this project's sections:

| sections | eosin-nearest arm | separation |
| --- | --- | --- |
| 2 × H&E | **every tile**, 5.0–9.4° | 18.8–38.1° |
| 4 × IHC | **none of the twelve** | 48.6–63.5° |

18 of 18 correct at the shipped 15° bound. The real discriminator turned out to be
*which* published vector an arm lands nearest; the tolerance is the safety bound on top.
Both are pinned as a regression table in `tests/test_staining.py`.

**One guess the measurement refuted, and it mattered.** An H&E wedge here is
*narrower* than an immunostained one, not wider. Had `separation` been made a criterion
on the intuition that two real dyes open a wider wedge, it would have rejected every H&E
section and accepted every IHC one. It is reported as evidence, never as a gate.

Step 6 still treats an eosin-nearest arm as a fault, and that is not a contradiction:
step 6 asks whether this is the H-DAB section it was told to unmix, step 7 asks whether
the section has two dyes at all.

### The screen

Step 7 is now ask-first (`isGated` gained `isTiling`) with two levels and Back at each:
`BranchPicker` → `FieldOfViewPicker` → the report, which keeps both pickers as the direct
route. All four scales are offered (was `[224, 448]`); "Run by BEETLE" is visible and
disabled reading "yet to be developed"; an unavailable H&E option renders the detector's
own sentence.

`useTiling`'s three write paths collapse to one `commit`, so there is one invalidation
path. `useTissueType`'s five potential clear-effects collapse to one derived `inputsKey`
— `modelName` was missing from that list, which is the reported symptom: changing the
field of view left a stale class map on screen.

`usePipelineRun.invalidateAfter` is new. Without it a changed step 7 left step 8 in
`completed`, so the rail read "Completed" with nothing behind it.

The sample panel follows the branch (`overlay.rgb_sample_png`). On the H&E branch it is
the photograph, because that is what the model gets — showing the deconvolved plane
there would be a perfectly plausible picture of an input that model never sees.

### Training repo

`src/tilestore.py` is a two-entry variant table with **no arithmetic** — every callable
comes from `hchannel`, which re-exports the backend's `input.py`. `export.py` gained one
keyword-only `variant` argument; the grid, the vote and `resample` are untouched, so the
haematoxylin output is byte-identical. Verified by re-fitting the 672 µm arm after the
data move and reproducing its recorded metrics exactly.

`08_fit_concat_mlp.py` no longer hard-codes `standardise=True`. On an RGB arm that would
have published a manifest asserting a transform that never ran — and the new descriptor
raises on it, which is the desired loud failure, but only once the value is read from the
cache. `03_features.py` writes `.npz.tmp` then `os.replace`, so a killed run cannot leave
a cache that `np.load` half-opens.

Data reorganised to `data/h_channel/{112,224,448,672}um` and `data/he/…` (the `224_um`
underscore fixed in the move). All four haematoxylin arms verified intact afterwards:
every tile path resolves, both feature fingerprints match.

## Serving reproduces training — verified, on both branches

`backend/scripts/check_tissue_model.py`'s G6 gate replays a checkpoint's own recorded
probe logits through the **serving** transform in a fresh process. It is the only check
that can catch a training/serving input mismatch: a wrong transform costs accuracy
silently, and the loss is invisible in every number the training run computed, because
that run applied the transform consistently to itself.

| head | probes | worst logit delta |
| --- | --- | --- |
| `invasive_tile_fov112_fix1_concat` | 21 | 3.6e-06 |
| `invasive_tile_fov224_concat` | 21 | 5.5e-06 |
| `invasive_tile_fov448_concat` | 21 | 3.4e-06 |
| `invasive_tile_fov672_concat` | 21 | 3.5e-06 |
| `invasive_tile_fov448_he_concat` | 21 | **2.5e-06** |
| `invasive_tile_fov672_he_concat` | 21 | **5.1e-06** |

The two bold rows are the ones that matter here: they prove
`input.rgb_to_model_input` — the four lines step 8 will actually run — reproduces
exactly what the RGB arms were fitted under.

**The gate was dead, and reviving it found something.** It looked for a `tile_path` key
the manifests have never written, so it silently reported "none of the probe tiles are in
the store" on every checkpoint; and it pointed at `data/tiles`, the staging tree the
exporter empties after each run. It now resolves probes by `tile_id` through the store's
own manifest, and picks the store from the checkpoint's own channel and geometry — so a
head published at a new field of view needs no change there.

What it found: **`invasive_tile_v3_concat_bach` cannot reproduce its own logits** — all
twelve of its probes resolve and the worst difference is **1.0e+01**, four orders of
magnitude past tolerance, while its Fix 1 successor at the identical geometry against the
identical store reproduces to 3.6e-06. So the store is not at fault; v3's recorded logits
were written either against a differently resampled export or by a publish path that no
longer exists. Its weights may well be fine — the point is that they cannot be checked.

That mattered because `settings.tissue_type_model` still named it: a research-only
checkpoint, at 112 µm, while `tiling_field_of_view_um` is 224. The default is now
`invasive_tile_fov224_concat`, which passes its gate and matches the default field of
view, so the two settings describe one grid instead of two. `model_for` had already
stopped selecting v3 — the preference table pins the Fix 1 head at 112 µm — so this only
affects `discover()`'s ordering and a caller who names nothing.

`scripts/check_tissue_model.py` now reports **step 8 is ready**.

## The whole-slide path runs, on both branches

G6 above replays *stored training tiles*. This is the other half: `window_reader` →
`rgb_reader` → `inference.classify(channel="rgb_he")` → `plan_block`'s addressing, over
real pixels off `CAN_00270_26_H&E.svs` (126,976² at 0.2222 µm/px), the central 4×4 block
of the real full-slide grid at each scale.

| head | branch | mean confidence | invasive share |
| --- | --- | --- | --- |
| `fov672_he_concat` | rgb_he | 0.940 | 12.5% |
| `fov672_concat` | haematoxylin | 0.954 | 56.2% |
| `fov448_he_concat` | rgb_he | 0.999 | 18.8% |
| `fov448_concat` | haematoxylin | 0.889 | 50.0% |
| `fov224_he_concat` | rgb_he | 0.948 | 12.5% |
| `fov224_concat` | haematoxylin | 0.984 | 12.5% |
| `fov112_fix1_concat` | haematoxylin | 0.991 | 0.0% |

**What this establishes: the path runs.** Every head loads, reads, classifies and
reports through the code step 8 will actually execute, and the RGB heads do it without
touching a white point or a deconvolution.

**What it does not establish: anything about accuracy.** Sixteen windows at one
arbitrary location on one slide, with no ground truth and a stand-in white point. The
two branches disagreeing by 44 points at 672 µm and agreeing exactly at 224 µm is not a
finding — it is sixteen windows. The place that question gets answered is
`HE_VS_HCHANNEL.md`, on held-out tiles, paired per tile, with McNemar and a k/n beside
every rate.

## Two bugs the campaign itself found

**G2c re-cut a region that yields no tiles.** The determinism gate blindly re-exported
`per_region[0]`; at 672 µm only 189 of 1,104 BCSS regions produce a single tile and the
first produces none, so the gate failed asserting that zero tiles equal zero tiles. It
now picks the first region that actually yielded something, on both the BCSS and the
borrowed halves. Latent since the larger fields of view were added.

**The staging tree was not cleared between exports.** Tile ids are
`<roi>__r<row>c<col>`, and row/column index *this* geometry's grid — so the same id names
a different square at a different field of view, and a previous export's PNGs sit at
exactly the filenames the next one writes. `02_export.py` now clears `data/tiles` first.

## Left to do

- [ ] **A fresh clone can no longer complete `setup.py`.** `invasive_tile_fov224_concat`
      is now `role: runtime` in the lock file — it is what the app loads — and like
      every `fov*` head it is fitted on BACH, so `sources: []`: it cannot be
      redistributed. Previously the runtime head was v2, which is fetchable from Drive.
      This is the BACH ND constraint becoming visible rather than something new, and
      `retrain_with` on each entry gives the remedy, but it does change what a clone
      does. Worth a deliberate decision.
- [ ] **Licence tracks on the `fov*` heads read `LICENCE UNKNOWN`**, where v1/v2/v3 read
      `RESEARCH ONLY`. All eight `fov*` heads are fitted on the BACH trees, so the
      unresolved CC BY-NC-ND question applies to them exactly as it does to v3 — their
      manifests simply do not carry a `provenance.licences` block for `_licence_track`
      to read. Not changed tonight, because relabelling a licence is the project
      owner's call, not a cleanup. It should be a deliberate decision before any of
      these leave the machine.
- [x] ~~`models.lock.json`~~ — done. Twenty entries, nine added (the four `fov*`, the
      four `fov*_he`, and v3, which had never been listed), plus a stale
      "the default checkpoint" note corrected on v2. `setup.py --check` verifies every
      hash. Written by `scripts/10e_update_models_lock.py`, which reads each sha256 out
      of the manifest that produced it rather than recomputing a second answer.
- [ ] Walk the **UI** once: load an H&E slide, confirm the H&E option is enabled with the
      eosin angle in its reason; load an IHC slide and confirm it is disabled with the
      DAB reason; switch the field of view and confirm `data/tissue_type/<id>/` is gone
      rather than showing the old map. The service path and the model path are both
      verified above; what has not been exercised is the screen and the four panels.

      **One thing this walk would have found has since been fixed without it** - see
      "The two screens, after the walk was still outstanding" below. The rest of the
      walk still stands.
- [x] ~~`HE_VS_HCHANNEL.md`~~ — written, all eight cells scored. Read the four warnings
      above its tables before the tables.

      One bug of mine caught here: `compare_done()` used `any(head in text)`, so the
      document I generated part-way through the campaign — three arms published, the
      fourth not — satisfied the predicate and the campaign **skipped** its final
      comparison. The 112 µm cell, the one with five times the held-out tiles of any
      other, was missing. Now `all(...)` over the published arms, and re-run.
- [x] ~~Root docs~~ — done. `PROGRESS.md` points at the tracker rather than duplicating
      it; `RESULTS.md` and `RESULTS_224_VS_448.md` carry supersession notes;
      `HOW_THE_PIPELINE_WORKS.md` shows the eight-store layout and the
      `foundation-model-features` doc's runnable commands take the new paths. The
      historical logs in `PROGRESS.md` and `FIX1_FIX2_PLAN.md` were **not** rewritten to
      match the move - they are records, and editing them to agree with a later change
      would make them useless as records.

## The two screens, after the walk was still outstanding

Two statements above stopped being true, both on the screen rather than in the service.

**Step 5 now names the staining, and it is the reason the H&E option can be trusted.**
`DensityStaining` was on the wire from the beginning and step 5's own screen rendered
none of it - it drew the arms, the angles and the wedge and then said nothing about what
they meant, so **step 7 was the first screen in the walkthrough to use the word H&E**,
and it used it beside a greyed-out button. A reader met the consequence before the
measurement. `features/density/StainingVerdict.tsx` states the verdict in plain words
between the arm card and the tile chooser, quotes the server's own `reason` verbatim -
the same string step 7 puts under a disabled option - and keeps the angles in a
disclosure. It restates; it computes nothing.

**And the H&E option was disabled on H&E slides, for a reason that had nothing to do
with the gate.** `tiling_service.branches` reads step 5's verdict out of a per-slide
cache and deliberately never computes one, which is right. But `useTiling` fetched that
payload on `[uploadId]` alone, as soon as a slide existed - which is *before* step 5 runs
in every walkthrough - and never fetched it again. So the payload was read while the
answer was still `step5-not-run`, and an H&E slide reached step 7 with the full-colour
option off and the reason telling the reader to go and run step 5, which they had just
done. The verdict is now an argument to the hook and therefore a dependency of that
fetch, so the option enables itself the moment the measurement exists. Only `branches` is
written on that refetch: a viewer who has already chosen is not moved off their choice.

**BEETLE is now gated on step 5's verdict, like the full-colour option.** It was the one
option on that screen that ignored what step 5 measured: enabled purely on
`beetle.available()`, so an immunostained slide with the archive on disk was offered it.
It reads the colour photograph with no deconvolution and no stain normalisation -
`dataset.json` names all three channels `rgb_to_0_1` and `to_model_input` is a division
by 255 - so the dyes on the section *are* its input distribution, and a DAB section is
brown where that distribution is pink. It would not fail there; it would return a
confident wrong segmentation.

**The claim is made from the archive and not from the paper**, deliberately: the channel
names are checkable in the download, and what the training slides were stained with is
recorded nowhere in it. And the gate matters more here than for `he` because there is no
recourse - these are released weights with no version fitted on anything else, where our
own heads could at least be retrained.

`_offered_branches` therefore asks the two questions **in order: the slide, then the
machine.** Only the second is fixable, and naming a 1.9 GB download first would send a
reader after it for a section that could never use the result. On an IHC slide the reason
is step 5's own sentence, quoted rather than restated, preceded by one sentence saying why
BEETLE needs the colour. Six tests in `test_staining.py` pin it, including the ordering,
the fail-closed behaviour on an unmeasured slide, and the fact that a disabled branch
still lists its four scales - branch `enabled` is "can this slide take it", field-of-view
`available` is "does a model exist at this scale", and collapsing the two would lose
information.

**BEETLE's copy was also still the placeholder's.** `branches.py` has had BEETLE in
`SERVED_BRANCHES` since it was built, and `_offered_branches` has resolved it through
`beetle.available()`, so the server has been offering it correctly - but the field-of-view
picker still gave `not developed yet` as its unavailable reason, and that is now the one
wrong answer: it sends a reader looking for training that will never happen instead of
for a 1.9 GB download. It reads `BEETLE's weights are not installed`. The branch picker,
`stages.ts`, the type comments and the hook's own notes carried the same stale claim and
now describe what BEETLE is instead: five classes per pixel at its own fixed 0.5 µm/px,
one release covering all four extents.

Two tests asserted the placeholder rule - that `beetle` was a 409 on the API and a
`TilingError` at commit - and are **inverted rather than deleted**, because the geometry
they now pin is the part worth pinning: `commit_selection("beetle", fov=224)` must record
448 px at 0.5 µm/px and not 224 px, since our geometry under BEETLE's name would feed the
network tissue at four times its trained scale and return a confident wrong segmentation.
The 409 contract keeps a test of its own, with `SERVED_BRANCHES` narrowed so it is
reachable at all - it is what a fourth name added to `ModelBranch` before its serving path
exists would hit.

Writing that test found one more thing: `conftest`'s directory redirect covered every
per-slide cache **except `tiling_dir` and `roi_dir`**, so one POST to `/selection` for the
upload id `nope` wrote a real `data/demo/tiling/nope/selection.json` for a slide that has
never existed. Both are redirected now. They are also the two directories
`maintenance_service` inventories, so that leak would have surfaced a second time as an
orphan the app offered to clean.

518 backend tests pass under random ordering. `eslint --max-warnings 0` is clean and
`vite build` bundles; `tsc` reports errors only in `features/tissueType/`, which is being
edited concurrently for BEETLE's per-pixel `paintedMasks` and is untouched here.
