# `bracs_roi_to_mask_using_beetle/` — borrowing the class BCSS cannot teach

A standalone app that takes BRACS DCIS regions of interest, segments them with BEETLE's
released nnU-Net, writes the result as a **BCSS-coded mask**, and cuts approach 1's own
tiles from it — to answer one question:

> BCSS gives approach 1 **nine** `non_invasive_epithelium` tiles, seven of them from a
> single patient. Its held-out set contains **none**, so class 1 recall is 0.000 and per
> class Dice is 0.0. Can that class be taught from a different dataset instead?

**Short answer: yes on volume, the tiles are drop-in, and there are two caveats that
matter more than the headline.** Thirty regions from thirty patients produced **148**
class-1 tiles in 2.6 minutes — 16.4x BCSS's entire corpus — from 4.5 % of the 665
available regions. But 20 % of those regions disagree with their own DCIS label, and the
class-1 tiles differ from BCSS's in distribution *shape*, which no stain normalisation
fixes. Both are measured below. And none of it establishes that the labels are
*correct*: see [What this does not prove](#what-this-does-not-prove).

```
start.bat          # http://127.0.0.1:8100
stop.bat
```

---

## The two corrections this folder makes to approach 4a

Both were found by reading the released archive rather than the paper, and both are the
kind that produce a working pipeline and a wrong answer.

### 1. BEETLE's class codes 2 and 3 are the other way round

`beetle_teacher_resnet18/src/beetle.py` records, from the paper:

```python
BEETLE_CODES = {"unannotated": 0, "other": 1,
                "invasive_epithelium": 2, "non_invasive_epithelium": 3, "necrosis": 4}
```

`dataset.json` **inside `model.zip`**, and the `init_args` inside every fold's
checkpoint, both say:

```json
{"background": 0, "other": 1,
 "non-invasive epithelium": 2, "invasive epithelium": 3, "necrosis": 4}
```

That module's own `assert_codes_verified` was written to fail closed until exactly this
check was done, and its docstring calls the codes "unconfirmed". They are now confirmed,
and they were wrong. With the paper's ordering every in-situ duct would have been
exported as `invasive_epithelium` and every invasive focus as class 1 — a training set
**inverted on the one boundary the region model exists to draw**, which would have
trained to a plausible accuracy and been wrong about the only thing that matters.

`backend/bracs_app/labels.py` reads the codes from the archive at run time and refuses
to start if they are not the ordering it was written against.

### 2. The teacher has five output channels, not four

`beetle.py`'s `TEACHER_CLASSES` is a 4-tuple, on the reasoning that the released
ensemble "predicts four classes and never `unannotated`". Every fold's
`decoder.seg_layers.6.weight` is `(5, 32, 1, 1)`. Channel 0 is real and learned — it is
not an abstention but "this looks like the unannotated parts of my training slides",
mostly glass. `remap_teacher` would raise on any output containing it.

---

## What it does

```
BRACS DCIS ROI (RGB PNG, ~0.25 µm/px, up to 40 MPx)
        │  resample ONCE, area-averaged, to 0.5 µm/px
        │  (BEETLE's training spacing AND approach 1's tile spacing — same array,
        │   so mask and image cannot drift out of register)
        ▼
BEETLE PlainConvUNet, 46.3 M params, 512² patches, half-step sliding window,
Gaussian blending, softmax averaged over 1–5 folds            ~0.6 s / patch / fold
        ▼
5-class argmax  ──►  BCSS's own 22-code vocabulary  ──►  uint16 I;16 PNG
        │            unannotated → outside_roi (0)
        │            other       → stroma (2)
        │            non-invasive→ dcis (20)          ← the point of the exercise
        │            invasive    → tumor (1)
        │            necrosis    → necrosis_or_debris (4)
        ▼
bcss_bracs_hchannel_resnet18/src/export.py::write_region   ← approach 1's, UNMODIFIED
        ▼
224 px haematoxylin-density tiles, by class, + manifest
```

The mask is a byte-for-byte equivalent kind of file to
`TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500.png` — single channel, uint16, code
valued — which is what makes the last step possible at all.

### Why write BCSS codes rather than our three classes

Because `bcss.remap` is where approach 1's judgement calls live: `dcis` and
`normal_acinus_or_duct` to class 1, `outside_roi` to `IGNORE`, and the assertion that
`tumor` and `dcis` never collapse together. Emitting three classes directly would mean a
second definition of class 1 — the precise failure `beetle_teacher_resnet18`'s
`backend_path.py` is written to prevent one layer down.

The one judgement call this folder does make is `other → stroma`, which flattens
BEETLE's catch-all onto one BCSS code. Lossless *for this pipeline* — approach 1 sends
sixteen codes to class 0 anyway — and documented in `labels.py` as not reusable by
anything that separates stroma from inflammation.

---

## Results so far

`scripts/batch.py --per-case 1 --limit 30 --folds 0` — 30 regions, one from each of 30
different patients, 2.6 minutes total on CPU. `reports/pilot.json`.

| `non_invasive_epithelium` tiles | from | |
| --- | --- | ---: |
| BCSS, entire 151-region corpus | 3 patients, 7 of 9 from one | **9** |
| BRACS, 30 regions | 30 patients, one each | **148** |

**16.4× BCSS's whole corpus, from 4.5 % of the available regions, in under three
minutes.** Volume is not the constraint. One larger region, `BRACS_1247_DCIS_1`, gives 38
class-1 tiles on its own in 24 s.

The tiles are format-identical to approach 1's — 224 px, uint8, mode `L` — because they
were cut by approach 1's exporter.

### 20 % of regions disagree with their own label

Six of the 30 came back with **more invasive than in-situ epithelium** in a region BRACS
annotates as DCIS. This is not a size artefact: the flagged regions' median area is
2.20 MP against 2.21 MP for the clean ones. The split is clean rather than gradual —
regions land at either 45–91 % in-situ or 0–27 %, with nothing in between — so it is a
per-region disagreement, not a noisy boundary.

Without a pathologist there is no way to tell which side is wrong: BEETLE may be reading
solid DCIS as invasive nests (a genuinely hard call that needs myoepithelial staining),
or BRACS's region-level label may cover fields that also contain invasion. Either way the
practical answer is the same and it is automatic — `pipeline.verdict` flags them, and
they should be excluded before training rather than argued about.

### Do the new tiles look like the old ones?

Screen 3 measures this, because volume is worthless if the model has to extrapolate on
every new tile. BRACS density divided by BCSS density, across the quantile ladder:

| class | p50 | p75 | p90 | p95 | p99 | spread | reading |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `invasive_epithelium` | 1.05 | 1.06 | 0.97 | 0.89 | 0.78 | 1.35× | strength, **inside** `alpha` |
| `non_invasive_epithelium` | 1.25 | 1.13 | 1.00 | 0.89 | 0.73 | 1.70× | **shape** |
| `non_epithelium` | 0.61 | 0.60 | 0.58 | 0.57 | 0.64 | 1.13× | strength, **outside** `alpha` |

Three different answers, and they need different things:

- **Invasive tiles need nothing.** A near-constant 0.97× sits inside approach 1's
  `alpha` jitter of (0.80, 1.25), so the existing augmentation already covers them.
- **Non-epithelium tiles are uniformly 0.60× — a pure strength difference, but outside
  `alpha`.** Monotone, so `standardise=True` (divide each tile by its own p99) removes it
  by construction.
- **Class 1 — the one being harvested — is a 1.7× *shape* difference**, and approach 1's
  gate G0b already established that no monotone correction removes one: not stain
  normalisation, not a per-slide white point, not `alpha`, and not `standardise` either.
  The only knob that moves shape is `gamma`, whose training range of (0.5, 3.5) has to
  cover it.

Both mitigations already exist in `bcss_bracs_hchannel_resnet18/src/`, so the recommendation is
to train the BRACS-augmented head as the existing `04_head_imagenet_std` variant
(`standardise=True`, `gamma` jitter on) rather than to add preprocessing here.

The class-1 row rests on BCSS's nine tiles, so read it as a direction rather than a
measurement — the app says so too, and will keep saying so until BCSS has more class 1
tiles, which is the thing that will never happen and the reason this folder exists.

---

## The web page: one image, brought by the user

The page runs the pipeline above on any H&E image somebody uploads, shows the original
and the overlay side by side, and says what each colour means. It exists because "show
me what the model does with this" is the first question every reader of this report
asks, and answering it by editing a path in `batch.py` is not answering it.

**It used to be four screens.** A picker over the 665 BRACS regions, a mask viewer, a
stain comparison, and this. The first three drove the experiment through a browser, and
they were removed: the experiment is finished, its numbers are in this file, and
re-running any of it is a `scripts/batch.py` invocation rather than an afternoon of
clicking. The endpoints those screens called are all still there, still tested, and
still what `batch.py` and `export_to_approach1.py` use — `/api/rois`, `/api/runs`,
`/api/results`, `/api/compare`. What was deleted is roughly 400 lines of frontend, not
any capability — and note that this folder is not under version control, so those 400
lines are gone rather than recoverable. The endpoints they called are what would make
rebuilding them a frontend job rather than a rewrite, which is why they stayed.

What is left is a side door onto the same model rather than a step of the experiment,
and three things are arranged to keep it that way:

| | BRACS region | uploaded image |
|---|---|---|
| output directory | `data/runs/<roi_id>/` | `data/uploads/<upload_id>/` |
| tiles cut | yes, by approach 1's exporter | **no** |
| counted in `/api/compare` | yes | **no** |
| `verdict` | `usable: true` / `false` + notes | `usable: null`, no notes |

The directory split is the load-bearing one. Everything that reads `data/runs/` treats
what it finds there as a BRACS DCIS region: `compare.compare_all` pools its tiles into
the BRACS side of the stain comparison, and `scripts/export_to_approach1.py` copies them
into approach 1's training set. An image somebody dropped on a web page is neither, so it
never lands there.

The `verdict` row is the subtler one, and `pipeline.verdict` now branches on
`manifest["source"]` to get it right. Both of its notes are *disagreements with BRACS's
own annotation* — "the teacher found in-situ epithelium on only 3 % of a region BRACS
annotates as DCIS" is a red flag precisely because two things that should agree do not.
Nobody has said what an uploaded image contains, so the same 3 % is evidence of nothing,
and `usable` is `null` rather than `true` so that no caller can read a passed check out
of an unperformed one. The page says so in place of a verdict, and points at the check
a reader can actually make: fade the colours and see whether the amber follows the
outlines of the ducts.

Two honest limitations are stated on the page rather than left to be discovered:

* **The resolution field decides what the model sees.** Tell it 0.25 µm/px for an image
  that is really 0.5 and every nucleus arrives at twice its trained size. Nothing errors.
* **There is no "I don't know" output.** An argmax over five channels always names a
  winner, so BEETLE returns a full three-colour mask for an IHC slide, another organ, or
  a photograph of a screen. The confidence figure measures how peaked the softmax was,
  which is not the same as being right.

The colours are not restated in the frontend. `labels.colour_legend` pushes one pixel of
each BEETLE class through `to_bcss_codes` and approach 1's `bcss.remap` and reads off
where it lands, so the legend is derived from the functions that painted the picture —
a hand-written grouping would be a third definition of class 1 and the one most likely
to be believed, because it is the one on the screen. Only the plain-language sentences
live in `app.js`, including the one everybody omits: BEETLE's non-invasive class is its
own documented mix of "healthy glands and DCIS ... LCIS, atypical ductal hyperplasia,
apocrine metaplasia", so **amber means "lining, still inside", not "cancer"**. On a BRACS
DCIS region the dataset's annotation carries that distinction; on an arbitrary upload
nothing does, so it has to be said.

```
POST   /api/uploads                        multipart: file, source_mpp, folds -> job id
GET    /api/uploads                        every upload, newest first, with `ready`
GET    /api/uploads/{id}                   upload.json + manifest.json (or null)
GET    /api/uploads/{id}/image/{which}     original | overlay | mask | input
DELETE /api/uploads/{id}                   409 while its job is still running
```

`original` is the resampled image capped at 1400 px for the browser; `input` is that
same resampled image at full working resolution, which is the only one the downloadable
uint16 mask lines up with pixel for pixel. Uploads are capped at 120 MPx and 400 MB,
checked while the body is still streaming, and a file that Pillow cannot open is refused
with a sentence rather than a stack trace from the worker thread.

One asymmetry worth knowing about: `/api/health` still reports every input the project
needs, `bracs_regions` included, and `health.ok` is an AND over all of them. The page
reads none of those regions, so it checks only `beetle_weights`, `approach1_src` and
`demo_backend` (`NEEDED_HERE` in `app.js`) — a chip reading "missing: bracs_regions"
above a working upload form would be false. The endpoint stayed honest; the chip got
narrower.

---

## What this does not prove

**These labels are not ground truth and nothing here should be quoted as though they
were.** They are three steps removed from a pathologist:

1. BEETLE's own training annotations are model-generated and human-*corrected* — a
   two-phase epithelium U-Net drew the boundaries, a human selected and fixed them.
2. What this app runs is BEETLE's *model's* opinion of a *different dataset* it has
   never seen, with no human in the loop at all.
3. BRACS's own DCIS label is per-region, not per-pixel. "This region contains DCIS" does
   not certify which pixels are duct.

So the ceiling on their accuracy is BEETLE's external non-invasive Dice, and a held-out
set built from these tiles would measure **agreement with BEETLE, not with the truth**.

The only honest test is: train on BCSS + BRACS, and evaluate on **BCSS's real class-1
tiles held entirely out of training**. Nine tiles is a thin test set, but it is a real
one, and it is the number that answers the original question. `institution="BRACS"` is
deliberately not a two-letter code so these regions can never land in approach 1's
held-out institutions by accident.

Gate zero from approach 4a still stands and is unaffected by any of this: *a pathologist
has not confirmed there is DCIS in the six clinical cases at all.*

---

## The six-slide comparison — do we still need BEETLE?

`start.bat`, tab **"Our six slides"**. One row per clinical case: the tissue, our trained
tile classifier's answer, and BEETLE's, on the same 2048 px window at 0.5 µm/px.

Both models must run at 0.5 µm/px, and a whole 28 mm slide at that spacing is ~400 MPx —
about an hour per fold. So one window per slide, picked as the most **nucleus-dense** one
by `beetle_teacher_resnet18/src/regions.py` (a stain measurement, not a model output, so
it favours neither model). About 75 s per slide.

**Agreement is scored per tile**, against BEETLE's majority label in the same 224 px tile.
BEETLE labels pixels; our model labels tiles. Scoring per pixel would charge our model for
its output format rather than its judgement — that number is reported too, as the
pessimistic one.

### Result: 60.9 % tile agreement over 486 tiles

| case | agree | BEETLE in-situ → ours invasive | ours (other/in-situ/inv) | BEETLE |
| --- | ---: | ---: | --- | --- |
| CAN_00267_26 | 86.4 % | 0 | 86/7/6 | 100/0/0 |
| CAN_00251_26 | 79.0 % | 0 | 78/2/20 | 99/0/1 |
| CAN_00865_26 | 70.4 % | 0 | 40/1/59 | 63/0/37 |
| CAN_00303_26 | 64.2 % | 18 | 10/36/54 | 9/51/41 |
| CAN_00270_26 | 61.7 % | 21 | 15/2/83 | 25/28/47 |
| **CAN_00259_26** | **3.7 %** | **26** | 68/0/32 | **4/96/0** |

**The disagreement is one-directional and it is the pair that matters.** 65 tiles where
BEETLE says in-situ and ours says invasive, against 7 the other way. Our model has a
strong invasive prior — unsurprising, since 5,282 of its training tiles are BCSS invasive
against 1,852 in-situ.

### And on the worst slide, BEETLE looks like the one that is wrong

`CAN_00259_26` is not a subtle case: BEETLE calls **96 %** of the window in-situ
epithelium. The tissue is a **confluent sheet of tumour cells threaded with thin fibrous
bands** — no duct outlines, no lumens, no basement membranes anywhere in the field. That
is what invasive carcinoma looks like, and an in-situ call over almost all of it is very
hard to justify. Our model said 32 % invasive and 68 % other tissue: also wrong in its own
way — it under-calls tumour — but it did not make the in-situ error.

So the naive reading of this screen ("disagreement = the student failed to learn") is
**backwards on at least one of the six slides**. That is precisely why the screen shows
the tissue in the first column rather than only the two masks and a number.
**A pathologist should adjudicate these six windows.** Until one has, neither column is
ground truth and this table is two opinions, not a score.

### What it does and does not license

- It does **not** license shipping our model as a BEETLE replacement. 60.9 % agreement on
  the only three-class boundary that matters is not equivalence.
- It does show the student is not blindly reproducing the teacher, and that BEETLE itself
  is unreliable on this laboratory's tissue — a domain neither model was trained on.
- It makes the pathologist review a **small, concrete** task: six windows, ~1 mm each,
  with the two models' disagreements already marked.

## Layout

```
start.bat / stop.bat        one service on :8100; stop.bat reuses the demo's
                            stop-service.ps1 rather than copying its logic
backend/
  bracs_app/
    unet.py                 nnU-Net PlainConvUNet in plain torch, built FROM plans.json
    teacher.py              archive reading, fold loading, sliding-window inference
    labels.py               BEETLE ↔ BCSS translation — the crux; read this first
    pipeline.py             one region end to end, and the `verdict`
    compare.py              are BRACS tiles interchangeable with BCSS tiles?
    student.py              our published tile classifier, run over a region
    sixslides.py            both models on our own H&E slides, and the agreement
    jobs.py                 one-at-a-time background worker
    config.py               paths, and the assumptions marked as assumptions
    main.py                 FastAPI + static frontend
  tests/                    30 tests, none of which load the 1.9 GB archive
frontend/                   three static files, one screen, no build step, no Node
                            index.html / app.js / styles.css, ~950 lines together
scripts/
  verify_teacher.py         proves the rebuilt network behaves like the released one
  check_resolution.py       measures BRACS's µm/px instead of assuming it
  batch.py                  headless runs, sampled per patient
data/
  dcis/{train,val,test}/    the BRACS regions (665 in train, 60 patients)
  runs/<roi_id>/            mask, overlay, tiles/, manifest.json
  uploads/<upload_id>/      the same, minus tiles/, for images a user brought
```

### Why the package is `bracs_app` and not `app`

Approach 1's `hchannel` imports `app.common.imaging` from the demo backend. Two packages
named `app` on one `sys.path` resolve to whichever was imported first, and the symptom is
`ModuleNotFoundError: No module named 'app.common'` raised from a file that has nothing
to do with the collision.

### Why nnU-Net is not a dependency

The released archive is five checkpoints plus `plans.json`; nnU-Net itself is needed for
exactly one thing, turning those plans into a `torch.nn.Module`. Installing it pulls in
`dynamic_network_architectures`, `batchgenerators`, `acvl_utils` and SimpleITK, several
of which have no Python 3.13 wheel. `unet.py` rebuilds the architecture in 200 lines and
`load_beetle_weights` refuses to load unless the checkpoint's key set and the model's are
*identical* — so a wrong architecture raises instead of loading 80 % of its weights
cleanly and segmenting noise. `scripts/verify_teacher.py` covers the wiring errors that
shape-checking cannot.

---

## Checks worth running

```bash
PY=../Breast_Cancer_IHC_Tissue_Scoring_Demo/backend/.venv/Scripts/python.exe

$PY -m pytest backend/tests/ -q          # 30 tests, ~3 s, no weights needed
$PY scripts/verify_teacher.py            # the network really is BEETLE's
$PY scripts/check_resolution.py          # is BRACS actually 0.25 µm/px?
$PY scripts/batch.py --per-case 1        # 60 patients, one region each

# the six-slide comparison, headless
$PY -c "import sys;sys.path.insert(0,'backend');from bracs_app import config,sixslides;config.install_beetle_path();config.ensure_dirs();[print(c, sixslides.compare_slide(c)['agreement']['tile_agreement']) for c in sixslides.cases()]"
```

`check_resolution.py` measures nuclear size against BCSS's known 0.25 µm/px. It currently
reports an implied **0.226 µm/px**, consistent with the 0.25 assumption — which matters,
because that number sets whether a 224 px tile really covers the 112 µm the tile size is
argued from.

---

## Licence

Two non-commercial dependencies meet in this folder and nothing produced here is clean:

- **BEETLE** weights and data — CC BY-NC-SA 4.0. The Apache-2.0 on its GitHub repo
  covers only inference code. ShareAlike arguably follows a distilled student.
- **BRACS** — non-commercial, research use only.

Used here for testing at the user's explicit direction. Everything under `data/runs/` is
derived from both and inherits both restrictions. Approach 1's
`invasive_tile_v1_imagenet.pt` is the clean-licence model and must stay intact.
