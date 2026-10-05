# Unattended session checkpoint — 2026-09-15

Working while the user sleeps (~3-4h). This file is updated after every task so the
work is recoverable if the session is interrupted. Newest status at the top.

## Status: IN PROGRESS

## The four asks

| # | Task | State |
|---|------|-------|
| A | Switching the H&E/IHC tab must actually RUN that slide's step — internals and display both change | **DONE** |
| B | Everything except QC auto-runs on switch; QC asks, and when run, runs for BOTH slides | **DONE** |
| C | Cannot advance to the next step until the previous steps have run — for both slides where the step is per-slide | **DONE** |
| D | Step 8 defaults to H&E @ 224 um; the other options leave the pipeline and the option menu is removed (code kept, not deleted) | **DONE** |

## Verification gate (must be green before each checkpoint)

    cd backend && ./.venv/Scripts/python.exe -m pytest -q
    cd backend && ./.venv/Scripts/python.exe -m ruff check <touched files>
    cd frontend && npm run build && npm run lint

Known pre-existing failure, NOT caused by this work and not in scope:
`tests/test_roi_borders.py::test_region_bbox_matches_the_outer_ring_extent`
(off-by-one, (97,65) vs (96,64)).

## Log

- Session start. Baseline green: 698 passed / 1 pre-existing failure; frontend builds and lints clean.

### A / B / C — done (frontend)

Design change from the earlier pass: the IHC hook instances are **no longer gated on
which tab is selected**. A step that declares two slides has two answers the pipeline
needs, and if the second only exists while its tab is open then step 4 can never
truthfully ask whether step 3 has run.

- `features/slideRole/useBothSlides.ts` — folds a step's two slides into ONE task.
  The pipeline marks a step complete when its task resolves, so this is what makes
  "step 3 is done" mean "on both slides". H&E first, always: `qc_service` holds a
  single global run lock and refuses a second run while any run is going, and
  `runQualityControl` resolves only on completion, so awaiting in order is the only
  thing that works for QC.
- `DemoPage.tsx` — `task` for steps 2-6 is now the both-slides callable. Auto-start
  therefore runs both; QC stays gated and, when the button is pressed, runs both.
- Gating: `hasTissueMask` / `hasWhitePoint` / `hasDensity` now require BOTH slides
  (`bothOrLone`, which tolerates a lone uploaded file with no IHC half).
- The picker now only changes what is DISPLAYED — it starts nothing, so it stays live
  during a run. Both cards carry a real headline on every per-slide step now (QC's
  removed share, step 3's tissue share, step 4's I0, step 5's staining verdict, step
  6's DAB peak — which is the per-slide argument in one number: full of signal on the
  stained slide, near-empty on the H&E).
- Removed `slideChoiceOffersBoth` (its premise — one step fetching both while others
  fetched only the displayed one — no longer exists).

Green: frontend builds and lints clean.

### Open decision, flagged for the user

"except the QC all should run automatically" — taken literally this includes step 8,
which is ~30 minutes of CPU. Auto-launching that on navigation is the exact thing its
gate exists to prevent. Current plan: auto-run 3,4,5,6 and (once its option menu is
gone, task D) step 7; keep QC gated as asked, and keep **step 8 gated**. Flag in the
final report so it can be overridden.

### D — done (both halves)

Verified first that the default is real: `invasive_tile_fov224_he_concat` is on disk and
`model_for(224.0, committed_branch())` resolves to it.

**Backend.** New `settings.tiling_branch = "he"`, and `tiling_service.committed_branch()`
reads it. Deliberately NOT a change to `DEFAULT_BRANCH`: that constant answers "what does
a manifest with no `input.channel` key mean" and must stay `h_channel` for ever, because
that is what every pre-H&E checkpoint silently is. The two only looked like one question
while there was one branch. `tiling_field_of_view_um` was already 224.

Also added: step 7's report now carries a loud note if the colour model is ever run on a
section step 5 says is not H&E. `commit_selection` validates that a branch is *served*,
not that it suits the slide, and with the branch committed rather than picked there is no
longer a picker to refuse it - on a DAB slide the colour model reads an eosin direction
pointing at nothing, and it scores badly rather than failing.

**Frontend.** `COMMITTED_BRANCH`/`COMMITTED_FOV_UM` in `useTiling`, seeded into state so
`start()` is valid on arrival. `BranchPicker` and `FieldOfViewPicker` are no longer
rendered but are still on disk with a header saying why; the hook still exposes
`chooseBranch`/`chooseFieldOfView`/`backToBranch`/`backToFov`. `TilingView` states what
ran where the picker was. The rail's "Approach" decision fork is gone (a fork whose value
can never change teaches the reader something false).

**Step 7 is no longer gated** — its gate existed to stop a grid being laid that nobody had
chosen, and there is no unchosen grid now. It is seconds, so it auto-starts like the other
cheap steps. **Step 8 stays gated** (~30 min of CPU).

Two tests updated: `test_selection_and_erase.py` asserted `ModelBranch.H_CHANNEL` literally
under the name "the default selection is the configured one", so it was testing one
configuration rather than the sentence in its own name. It now reads `committed_branch()`.

### End-to-end smoke test on a real case — PASSED

Ran the real services against `CAN_00270` (H&E + CD44). This is the check the unit
tests cannot make: whether an actual slide survives the wiring.

| | H channel p99 | **DAB p99** | staining verdict | I0 |
|---|---|---|---|---|
| H&E | 0.947 | **0.199** | `he` | #c4c4c4 |
| CD44 IHC | 1.934 | **3.186** | `haematoxylin_dab` | #c1c1c1 |

That table is the whole argument for the per-slide prefix, measured: a 16x difference
in the DAB channel - the channel step 14's score is made of - between the two slides of
one case, and two genuinely different white points. Step 6 shown only on the H&E was
showing the 0.199 column and calling it the picture the score is measured from.

Step 7 resolved the committed configuration with nobody choosing it:
`branch=he  fov=224.0 um  model=invasive_tile_fov224_he_concat  tile_px=224  mpp=1.0`,
15,876 squares down to 7,202.

### Heavy verification — RUNNING

Background job `heavy.py`: QC on both slides sequentially (tests the global run lock
and the "QC runs for both" requirement), then step 8 on the H&E with the committed
he @ 224 um head. Expect 20-45 min.

## Status: ALL FOUR TASKS DONE — heavy verification running

### Polish added after the four tasks

- **Per-slide progress on the picker.** A per-slide step runs on both slides as one
  unit of work, so mid-run one is finished and the other is not - and the only way to
  find out used to be clicking the other tab. Each card now carries `done` / `running`
  / `waiting`. (`SlidePicker` `state` prop, `prefixState` in `DemoPage`.)
- **Documentation caught up with the committed configuration**: step 7's README, step
  7's and step 8's catalogue prose (tagline, `what`, `how`, `action_label`, `rule`),
  and the bundled frontend catalogue re-synced.

### Pre-existing problems found and fixed along the way (not asked for, but blocking)

- The **frontend did not build** before this session: five type errors in the steps
  14-17 panels (`'warning'` is not a `BadgeTone` - it is `warn`; and
  `bars[bars.length - 1]` possibly undefined). `npx tsc --noEmit` passes these;
  only `npm run build` (`tsc -b`) catches them. Fixed.
- `maintenance_service` inventoried 11 of 15 artefact directories - `per_cell`,
  `binning` and `scores`, i.e. the pipeline's actual output, were never swept. Added,
  and the `store` fixture's own guard caught it exactly as designed.

### Still open / for the user to decide

1. **Step 8 remains gated.** "Except the QC all should run automatically" taken
   literally includes it, but it is ~30 min of CPU and auto-launching that on a
   navigation is what its gate exists to prevent. Steps 3-7 auto-run; QC and step 8 ask.
   One-line change in `DemoPage`'s `isGated` if that is the wrong call.
2. **`tests/test_roi_borders.py::test_region_bbox_matches_the_outer_ring_extent`**
   still fails - off-by-one, `(97,65)` vs `(96,64)`, in step 9's region bbox. It was
   failing before this session and is in files none of this work touched.
3. Some files in the working tree were modified before this session started
   (`nuclei_service.py`, `score_service.py`, `sampling.py` at 11:04-11:23). Not mine.

### A bug found while reviewing, and fixed: the dead toggle

`runs_on=[he, ihc]` is true of SEVEN steps, but it means two different things:

- steps 2-6 have a *separate answer per slide*, so a switch between them is real;
- step 1 opens both pyramids so the readouts land side by side, and step 10 registers
  one slide onto the other. Neither has a "which slide" to choose.

The first pass rendered the picker whenever a step named two slides, which would have
put an `H&E | CD44 IHC` toggle on steps 1 and 10 that changed nothing while implying
it could. Fixed with `reads_slides_together` on `PipelineStage` (true for exactly
`read-slide` and `ihc-alignment`), honoured by `slideChoiceFor`. Two new tests pin it,
including one that a step claiming to read two slides together must actually name two.

The badge on those two steps now reads `H&E + CD44 IHC` rather than naming only the
first, which was the same half-truth the badge exists to remove.

### Verification status

- Backend: 700 pass, 1 pre-existing failure (`test_roi_borders`, off-by-one, untouched
  by this work). Ruff clean on every file this session touched.
- Frontend: `npm run build` and `npm run lint` both clean.
- Real-case smoke test: PASSED (table above).
- Heavy job (QC on both slides, then step 8 on the committed head): running. QC on the
  H&E is done and cached - artefacts removed 0.41% of the tissue, 1.09 mm2.
- Playwright is available (`playwright` is already a devDependency and chromium
  launches), so a real browser walkthrough is possible; `scratchpad/ui_check.mjs` is
  written and waiting for the heavy job to finish so the two do not contend for CPU
  or race on the same caches.

### Note: your backend was already running

Port 8000 had a live uvicorn (PID 16568) with `--reload` throughout this session, so
every backend edit restarted it. Nothing was in flight, so no harm - but it is worth
knowing that the running server has picked up all of this work already, and
`GET /api/v1/pipeline/stages` on it returns the new `runsOn` / `readsSlidesTogether`
fields correctly.

Consequence to respect: do not edit backend files while a long run is in progress, or
the reload will kill it mid-request. All backend editing for this session is finished.

### Real browser walkthrough — 8/9, and the 9th was a wrong assertion

Playwright is a devDependency already and chromium launches, so the UI was verified
for real rather than argued about.

    PASS  home: counts come from the catalogue, not the word "sixteen"
    PASS  home: the pill names the panel, not one marker  [breast - immunohistochemistry - five markers]
    PASS  step 1: a slide badge names the slide  [H&E + CD44 IHC]
    PASS  a per-slide step offers two slides
    PASS  the two options are the H&E and the marker slide
    PASS  each option reports its own progress  [H&E done 0.4% removed | CD44 IHC waiting]
    PASS  switching the tab changes the slide badge
          [H&E CAN_00270_26_H&E.svs  ->  CD44 IHC CAN_00270_26_A.svs]
    FAIL  switching the tab changes the pictures
    PASS  the IHC badge names the marker

The FAIL was my test, not the app: the walk landed on **quality control**, where the
H&E had a report (the heavy job had just written it) and the IHC did not, because QC
on the IHC slide was still running at that moment. No picture for a slide that has not
been quality-controlled is the correct answer. The 409s in the console are the
documented "this slide has simply not been run yet" response - `useQualityControl`
says so in its own comment.

Two things that could only be confirmed in a browser:
- `H&E + CD44 IHC` on step 1 is `reads_slides_together` working end to end.
- The badge carries the FILENAMES, so the switch is demonstrably showing two different
  physical slides rather than relabelling one.

Re-running the image comparison on step 3, where both slides really do have data.

## The measurement that justifies the whole thing

Quality control run on BOTH slides of `CAN_00270`, sequentially, for the first time:

| slide | tissue removed as artefact | area |
|---|---|---|
| H&E | 0.41% | 1.09 mm2 |
| **CD44 IHC** | **12.63%** | **26.40 mm2** |

A 31x difference, and the wrong way round from where the attention was. The H&E - the
slide that is *not* scored - was the only one ever quality-controlled. The
immunostained slide, the one every number in steps 11-16 is measured on, had **26.4
mm2 of folded, blurred or pen-marked tissue** that nothing had ever looked at. Every
nucleus in it went into the denominator.

`qc_gated` on the H&E now reads `True / grandqc`, where before both slides reported
`False` - and step 4's own on-screen note ("Step 2 has not run, so no artefacts were
excluded from the glass... a pen mark anywhere outside the tissue is pulling I0 down
right now") was therefore true of the IHC slide on every run, unseen.

This is not a hypothetical the change guards against. It was happening on the demo case.

### Requirement B, verified end to end

After QC ran on both slides, step 4 picks up the gate on both:

    [H&E]   I0=(196.0, 196.0, 196.0)  qc_gated=True  source=grandqc
    [A IHC] I0=(193.0, 193.0, 193.0)  qc_gated=True  source=grandqc

And the effect propagates, which is the part worth noticing: step 7's kept-square count
moved from **7202 to 7189** once the H&E's artefact map existed. Those 13 squares are
forward passes through the region model that were previously spent on damaged tissue.
The chain QC -> tissue mask -> tiling -> step 8 is real, not declared.

### Processes this session left running

- Your **backend on :8000** was already running when the session started (PID 16568,
  `--reload`); it has picked up every change.
- A **frontend dev server on :5173** was started here for the browser checks and has
  since been **stopped**, so `start.bat` will not collide with it. Port 5173 is free;
  port 8000 is still yours.

Screenshots from the browser runs are in the session scratchpad under `shots/`
(`qc-he.png`, `qc-ihc.png`, `04-step3.png`, `05-step3-ihc.png`).

### Requirement A, verified in a browser — 5/5

Asked on step 2 once BOTH slides had a quality-control report:

    PASS  on quality control
    PASS  BOTH slides now report done
          H&E done 0.4% removed | CD44 IHC done 12.6% removed
    PASS  the badge follows the switch
          H&E CAN_00270_26_H&E.svs  ->  CD44 IHC CAN_00270_26_A.svs
    PASS  the PICTURES follow the switch
          before: qc/uC8eGezagEjBHNSoxjUF1Q/overlay.png, .../heatmap/tenengrad.png
          after:  qc/alEtP1bYCEZkbuqS3oekRA/overlay.png, .../heatmap/tenengrad.png
    PASS  and they point at two different upload ids

Every image source changes upload id on the switch, so the internals and the display
both follow it - not a relabelled picture.

`shots/qc-ihc.png` is worth looking at: the H&E's artefact overlay is nearly clean, and
the CD44 IHC slide beside it is covered in magenta and green. That 12.6% was invisible
before, on the slide every score is measured on.

## A real bug the browser check found (and it would have bitten on every second run)

On step 2 with a quality-control report already on disk, the ONLY actions were
`Back` and `Run quality control`. No continue, no skip. The viewer was stuck on a
step that costs seven minutes, with the answer already on screen.

The mechanism is worth writing down because it is invisible from any one file:

- quality control is **gated**, so arriving never launches it, so the pipeline never
  marks the step complete - hence no "Continue";
- `QCPanel` shows a cached report the moment one exists, and the "run it or skip it"
  offer that carries the **Skip** button is part of the *setup* view it just
  replaced - hence no "Skip".

Neither half is wrong on its own. Together they are a dead end, and it is reachable
on the second visit to any slide that has ever been quality-controlled - which, now
that QC runs on both slides, is the normal state.

**Fix:** an effect that marks a gated step complete when its result exists for every
slide it declares. Mirrors the one step 7 already had. After it, step 2 offers
`Back`, `Continue to step 03`, `Run it again`.

This is the one thing in this session that could only have been found by driving the
real UI. It was not in scope, was not asked for, and would have made the walkthrough
unusable on a second run.

## Requirement D, verified end to end — step 8 ran on the committed head

    ok in 611s
    model=invasive_tile_fov224_he_concat  branch=he
      Not tumour tissue             82.72%   mean conf 0.981
      Tumour still inside the duct   4.69%   mean conf 0.888
      Tumour that has broken out    11.49%   mean conf 0.880
      Cannot be determined           1.10%   mean conf 0.642
    ALL ASSERTIONS PASSED

11.49% invasive on the H&E, against the 12.7% the h_channel head gave on the same
slide - so the colour head is behaving sensibly, not merely running.

## Requirement A/B/C, verified in a browser — 9/9 on the full walk

    per-step "done" counts: {Quality control: 2, Tissue mask: 2, White calibration: 2,
                             Optical density: 2, Colour deconvolution: 2}
    PASS  every per-slide step ran on BOTH slides
    PASS  reached tiling
    PASS  step 7 states the committed configuration
          "Running the colour model at 224 um per square - invasive_tile_fov224_he_concat"
    PASS  no branch/scale menu remains
    PASS  step 7 auto-ran

## Stale prose the screenshots caught

Reading the rendered pages found three sentences that the committed branch made false,
all now branch-aware:

- step 7: "it is the blue-stain picture from step 6, **not the colour photo**" - exactly
  backwards once the colour head is what runs;
- step 7: "because the approach **you chose** looks at colour" - nobody chooses now;
- step 8's technical readout: "Input: **the haematoxylin channel**..." stated
  unconditionally, which on the committed head is a factual error about what the model
  reads. `branch` is now mirrored on the frontend `TissueTypeParams` so the screen reads
  the manifest's own `input.channel` rather than guessing from the model's name.

### The dead end was in BOTH gated steps, not one

The first fix was written for quality control, where it was found. Step 8 is gated the
same way and `TissueTypePanel` returns its cached report before it ever renders the
run/skip offer - so it had the identical trap, with half an hour of CPU as the price of
escaping it instead of seven minutes. The effect now covers both, differing only in how
many slides have to be done: quality control needs both, step 8 needs the H&E.

Worth noting for anyone adding a third gated step: the rule is "a gated step whose
result already exists is complete". Forgetting it does not fail loudly - it produces a
screen with a result on it and no way forward.

### Final walk to step 8 — the important check passed; three assertions were mine

    PASS  never stuck on a gated step        <- the fix, confirmed
    PASS  reached tissue-type segmentation
    FAIL  step 8 shows its cached class map
    FAIL  step 8 describes the COLOUR input
    FAIL  step 8 offers a way forward

All three FAILs were bad assertions, not defects - worth writing down so nobody
re-investigates them:

- `invasive_tile_v1_imagenet` appears on that screen inside a **licensing note** ("the
  BCSS-only checkpoint ... is the commercially clean alternative"), not as the model
  that runs. My regex matched the wrong sentence.
- The colour-input description lives in the *result* view; step 8 was showing its
  *setup* view, because it had not run for this cache key in the browser session.
- It does offer a way forward - "Classify tissue types" and "Skip for now". The check
  only looked for the word "continue".

The one thing that needed confirming, confirmed directly from disk:

    data/demo/tiling/<he>/selection.json
    {"branch": "he", "field_of_view_um": 224.0,
     "model": "invasive_tile_fov224_he_concat", "tile_px": 224, "mpp": 1.0}

That is what step 7 committed in the browser and what step 8 runs.

### Final confirmation against the shipped build

The last two frontend changes (the generalised gated-step fix, and mirroring `branch`
on `TissueTypeParams`) landed after the previous walk, so they were only covered by
build and lint. Re-walked:

    VISITED: Read the slide > Quality control > Tissue mask > White calibration >
             Optical density > Colour deconvolution > Tiling > Tissue-type segmentation
    RUNTIME ERRORS: none
    REACHED STEP 8: true

Eight steps in order, no runtime errors, no dead end.

## Status: COMPLETE — all four tasks implemented and verified
