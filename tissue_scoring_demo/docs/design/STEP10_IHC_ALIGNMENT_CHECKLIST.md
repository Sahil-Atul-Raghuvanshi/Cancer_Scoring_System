# Step 10 — Align ROI to IHC: unattended build checklist

Live progress log. Updated as each item completes. Started 2026-09-10.

Legend: `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked//needs a human

---

## Phase 0 — Environment

- [x] Python 3.11.9 installed (winget, user scope)
- [x] Isolated venv at `tissue_scoring_demo/valis_service/.venv`
- [x] `valis-wsi` 1.2.0 + deps installed into it (numpy 1.26.4, SimpleITK, pyvips, scyjava, jpype1)
- [x] A JVM VALIS can reach — Temurin 21 JDK via `cjdk`, no admin needed
- [x] `pyvips-binary` for the libvips DLLs (the `pyvips` wheel alone does not ship them)
- [x] Two Windows-specific traps fixed in `register.py`:
      scyjava defaults to a **JRE** (no `jar` tool → jgo hangs), and jgo looks for
      `JAVA_HOME/bin/java` with no `.exe`, so the JDK's `bin` must be on PATH
- [x] `from valis import registration` imports and starts a JVM

## Phase 1 — Prove VALIS works, and pin down its real API

- [x] `Slide.warp_xy_from_to(xy, to_slide_obj, src_slide_level, src_pt_level, dst_slide_level, non_rigid)`
- [x] Errors (`rigid_D`/`non_rigid_D`) are already in physical units — no manual mpp maths
- [x] Smoke test on synthetic images: known 7° + (25,−15) recovered to **0.05 px** median error,
      377 matched keypoints, round-trip 0.0 µm

## Phase 2 — Register the real slide pair

- [x] `CAN_00270_26_H&E.svs` ↔ `CAN_00270_26_A.svs` (CD44) registers in ~5 min at 2048 px
- [x] Diagnostics captured. **First result was poor**: 12 matched keypoints, and the
      non-rigid pass reported a *worse* error than the rigid one (31 µm → 94 µm) —
      both signs that VALIS's defaults suit this cross-stain pair badly
- [x] Registrar pickle reload fixed — VALIS nests it at `<dst>/<name>/data/…`, not
      `<dst>/data/…`, so "reuse" had silently been re-registering every time
- [~] Tuning sweep to pick the configuration by measurement (`_tune.py`)
      - first scoring attempt (tissue overlap) **discarded**: a saturation threshold
        called 24% of the H&E tissue vs 4% of the IHC, so the score was bounded by the
        mask mismatch, not the alignment — every candidate would have tied
      - replaced with mutual information on absorbance, which needs no mask and does
        not assume the two stains look alike

## Phase 3 — Backend

- [x] `valis_service/register.py` — the isolated worker (registers, warps rings, writes JSON)
- [x] `app/registration/exceptions.py` — `RegistrationRefused`, `RegistrationUnavailable`
- [x] `app/registration/gate.py` — four independent measures, all reported on refusal
- [x] `app/registration/transform.py` — densify before the warp, simplify after
- [x] `app/registration/valis_align.py` — subprocess bridge into the 3.11 venv
- [x] `app/pipeline/step12_ihc_alignment/` — pipeline entry, region selection, overlay, README
- [x] `app/services/ihc_alignment_service.py` — job + cache + artefacts + the confirm gate
- [x] `app/schemas/ihc_alignment.py`, `app/api/v1/endpoints/ihc_alignment.py`, router wiring
- [x] `scripts/run_case_alignment.py` (headless end-to-end) and
      `scripts/check_alignment_geometry.py` (did the polygons survive the warp)

## Phase 4 — Renumbering 10–16 → 11–17

- [x] `schemas/pipeline.py` index ceiling 16 → 17
- [x] `data/pipeline_steps.py` insert `ihc-alignment` at 10, bump the seven stubs
- [x] Rename stub packages `step10_…`–`step16_…` → `step11_…`–`step17_…`, docstrings included
- [x] `pipeline/runner.py` STEP_FUNCTIONS
- [x] Regenerate `frontend/src/features/pipeline/data/stages.ts` (17 stages)

## Phase 5 — Frontend

- [x] `api/ihcAlignment.ts`, `types/ihcAlignment.ts`
- [x] `features/ihcAlignment/useIhcAlignment.ts`
- [x] `IhcAlignmentPanel.tsx` — side-by-side border overlay, blocking confirm, 3 IHC crops
- [x] Wired into `DemoPage.tsx`, gated on step 9's regions + a loaded IHC slide,
      invalidated by step 9's `generatedAt`

## Phase 6 — Verify

- [x] Backend suite: 541 passing before this work, plus 20 new tests for step 10.
      Two catalogue tests updated (16→17 stages, 9→10 implemented). One pre-existing
      failure left alone: `test_region_bbox_matches_the_outer_ring_extent` expects the
      bbox on the exact cell boundary while `class_regions` insets rings by
      `BORDER_GAP_CELLS = 0.06` — unrelated to this work, and step 9's published
      geometry should not be changed to satisfy a test
- [x] Frontend typecheck + lint + **production build** clean
- [x] **`npm run typecheck` was checking nothing** — the root tsconfig has `files: []`
      and only references, so `tsc --noEmit` silently passed on an empty project.
      Changed to `tsc -b`, which exposed 3 pre-existing `noUncheckedIndexedAccess`
      errors in `TissueTypePainting.tsx` that were blocking any production build.
      Fixed (behaviour-preserving)
- [x] **Full end-to-end run on real slides, and it works mechanically.**
      `scripts/run_case_alignment.py CAN_00270 A`:
      case resolved → steps 2–9 on the H&E in 24 min → **104 invasive regions** →
      top 3 selected → 2,520 densified vertices → registered in 304 s → gated.
- [x] **The gate refused it**, on the one measure that failed:
      *only 12 matched features, need at least 50*. Everything else was healthy —
      tissue ratio 0.672 (in range), 99.8% of probe points landed on the IHC slide,
      0 vertices clamped off-slide, round trip 0.0 µm.
      No regions were emitted, which is the designed behaviour.
- [x] **The accuracy metric was itself validated, before trusting its verdict.**
      Run against the synthetic pair, where the alignment is known good (0.05 px),
      it reports NMI **1.605 registered vs 1.157 unregistered — a gain of +0.45**.
      On the real H&E/CD44 pair it reported **1.008 vs 1.005, a gain of +0.003**.
      So the metric swings hard when alignment is genuinely right, and barely moved
      here. The refusal is correct **on the merits**, not merely on a proxy count.
      (A cross-stain pair will never reach 1.6 — the synthetic pair is one image
      transformed, so its textures are identical. The *gain over unregistered* is
      the fair comparison, and that is 0.003 against 0.45.)
- [x] **Root cause found, and it is the preprocessing, not the detector.**
      Swapping detectors made it *worse*, not better — SuperPoint+SuperGlue found
      **0** features where DISK+LightGlue found 12. Two independent learned matchers
      nearly failing on the same pair is not a detector problem.
      Looking at what VALIS actually feeds them settles it: the processed H&E is
      mid-grey with filled structures, the processed CD44 slide is near-black with
      thin bright edges. **Same architecture, drawn two different ways** — the same
      nodule cluster, ducts and voids are visible in both, so the tissue does
      correspond; the renderings do not. VALIS's default is optical density with
      `adaptive_eq=False`.
- [x] **Fixed, by one flag.** `OD` with `adaptive_eq=True` instead of VALIS's
      `adaptive_eq=False`:

      | | default | with equalisation |
      | --- | --- | --- |
      | matched features | 12 | **67** |
      | error after registering | 31 µm rigid → **94 µm** non-rigid | 41 µm rigid → **43 µm** non-rigid |

      67 clears the gate's threshold of 50 honestly, and the non-rigid pass now
      *holds* instead of degrading. Shipped as
      `settings.registration_processor{,_adaptive_eq}`, with the measurement in
      the comment so nobody silently reverts it.
      Ruled out along the way, each with a number: SuperPoint+SuperGlue (**0**
      features), DISK at 4096 px (**0** features, 20 min), MicroRigidRegistrar
      (killed at 47 min / 2.3 CPU-hours).
- [x] **A limit of my own metric, stated rather than hidden.** `alignment_nmi`
      scored 1.0121 for *both* configurations. It samples at ~62 µm/px, so it
      cannot distinguish a 41 µm residual from a 94 µm one — both are sub-pixel
      to it. It is a gross-correspondence check (it swings 1.157 → 1.605 on the
      synthetic pair) and must be read alongside the keypoint count and residual,
      not instead of them.
- [x] **A silently-disabled gate, found and fixed.** The real run returned a null
      residual error, and the reason was Windows' 260-character path limit: the
      registrar pickle path came to **259 chars** because the pair key was used as
      the VALIS run name and VALIS nests it twice. So VALIS wrote no `data/`
      directory, no summary — which cost both the registration cache *and* the
      residual measurement, leaving **three of the four gates doing the work of
      four**. Fixed by using a short run name, and the gate now **refuses** on an
      unmeasured residual rather than skipping the check.
      DISK+LightGlue usually returns hundreds across a slide pair, and the mutual
      information barely moved (1.008 registered vs 1.005 unregistered), so the
      transform is not adding much. Sweep of VggFD and SuperPoint+SuperGlue running
      to find out. **The threshold was deliberately not lowered to make the refusal
      go away** — that would be exactly the failure the gate exists to prevent.
- [x] **Re-run end to end with the fix, and it PASSES.**
      `state: ready` — 67 keypoints, residual **42.6 µm**, tissue ratio 0.672,
      round trip 0.015 µm median, **0 of 2,520 vertices off-slide**, 3 regions
      carried (22.9, 2.1, 2.1 mm²), `confirmed: false` awaiting a person.
      The residual is populated again, which also confirms the path-length fix
      restored the gate that had been silently inert.
- [x] **Checked by eye, which is the point of the step.** On the H&E the three
      borders sit on dark cellular tumour; on the IHC they land on the **brown
      CD44-positive nodule** and its two smaller foci, visibly rotated and warped
      relative to the H&E outlines rather than copied. `invasive1.png` is a clean
      crop of brown-stained tumour — exactly the tissue the measurement steps need.
- [x] **Geometry checked numerically**: area preserved at 0.97–0.99×, centroids
      moved 0.68–1.08 mm (two sections placed differently on their glass), no
      vertices off-slide.
      One false alarm found and fixed in my own checker: it flagged all three
      warped rings as self-intersecting, but **the source rings already are** —
      step 9 traces rectilinear tile boundaries and diagonally-adjacent cells
      pinch. The check now compares against the source and only reports a ring
      the warp actually broke.
- [x] **Gate refuses a deliberately bad pair.** CAN_00270's H&E against
      **CAN_00303's** — two different patients, so no correct alignment exists.
      Refused on two independent grounds:

      | | true pair (H&E ↔ CD44) | different patients |
      | --- | --- | --- |
      | matched features | 67 | **6** |
      | residual | 42.6 µm | **4,836 µm** |
      | NMI vs unregistered | +0.007 | **−0.004** (registering made it worse) |

      A 113× separation in residual. The gate is not merely refusing everything —
      it accepted the true pair the same afternoon.
- [x] **A bug in that test, found because the result looked wrong.** It first
      reported *"PASSED THE GATE"*. It had not: the harness shelled out to plain
      `python`, which resolved to the VALIS interpreter, which has no pydantic, so
      importing the gate died — and the non-zero exit was read as "no objections".
      **A crash and a verdict were the same signal.** Fixed to try interpreters
      until one can import the backend, and to say "harness failure" rather than
      ever inventing a pass. Worth noting that the failure mode was a *false
      accept* on two different patients' slides, which is the worst direction for
      it to fail in.

---

## Done. Where it landed

Step 10 works end to end on your real data. `CAN_00270` H&E → CD44:
**67 matched features, 42.6 µm residual, 3 invasive regions carried onto the IHC
slide with their areas preserved to 0.97–0.99×**, panels rendered, crops cut, and
the step correctly parked at `confirmed: false` waiting for a person.

The one change that made registration work was **optical density with adaptive
equalisation** (`settings.registration_processor_adaptive_eq`). VALIS's default
renders the H&E mid-grey and the IHC near-black — the same tissue drawn two ways
no matcher can pair. That took matched features 12 → 67.

To see it: run the demo, pick **A / CD44**, point at
`data/original/oncostem_slides/CAN_00270`, and walk to step 10. Steps 2–9 are
already cached for that slide, so only step 10 will take time (~8 min). Or
headless:

```
cd tissue_scoring_demo/backend
python scripts/run_case_alignment.py ../../data/original/oncostem_slides/CAN_00270 A
python scripts/check_alignment_geometry.py <he-id> <ihc-id>
```

### What I would look at next

1. **Only one case could be tested.** `CAN_00270` is the only case with its IHC
   slides on this disk; the other three are H&E-only. The thresholds are
   calibrated against one good pair and one deliberately-bad pair, which is thin.
2. **Four preprocessors were queued, one was measured.** `Luminosity`,
   `StainFlattener` and `BgColorDistance` never ran — `_tune.py` still has them.
   One may beat 67 features.
3. **`round_trip_max_um` was 174 µm** on the passing run against a 0.015 µm
   median, so a small part of the deformation field is much less invertible than
   the rest. Worth a look if a border ever seems locally wrong.

## One thing to confirm when you are back

You said *"224 H&E tissue type segmentation is default now"*. Steps 2–9 do now run
on the **H&E slide** at the default **224 µm** field of view — that part is done and
is what step 10 consumes.

But the default *branch* is `h_channel`, not the H&E-RGB one:
`tiling_service.DEFAULT_BRANCH = h_channel`, with `tissue_type_model =
invasive_tile_fov224_concat`. So step 8 sees the haematoxylin channel of the H&E,
not its raw RGB. That is the existing configured default and the design that lets
one model serve H&E and IHC alike, so it was left alone — switching it changes
which checkpoint runs and therefore the numbers, which is not a call to make
unattended. Step 10 is unaffected either way: it consumes step 9's regions
whatever produced them.

## Log

- Phase 0 started.
