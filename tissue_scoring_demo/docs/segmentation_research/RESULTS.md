# Tissue-type model — every approach tried, what works, and what is still wrong

Final update 2026-09-06. Held-out numbers are on tiles from **held-out BCSS
institutions and held-out BRACS/BACH patients** — one fixed test set per run that never
varies between configurations. Only what goes into *training* changes.

---

> **v3 is no longer what step 8 loads.** Eight heads are published now — four fields of
> view times two input channels — and which one runs is step 7's committed choice rather
> than a setting. `config.tissue_type_model` moved to `invasive_tile_fov224_concat`,
> which matches the default field of view and reproduces its own probe logits; v3 does
> not (worst logit difference 1.0e+01 against its Fix 1 successor's 3.6e-06 on the same
> store). See [`../progress/STEP7_BRANCH_STATUS.md`](../progress/STEP7_BRANCH_STATUS.md), and
> [`HE_VS_HCHANNEL.md`](HE_VS_HCHANNEL.md) for the eight-way comparison.
>
> Everything below stands as the record of how the tile-based approach was arrived at,
> and the approach ledger further down is still the reason each ingredient is in it.

## DEPLOYED — the tile-based approach now serves v3

**`invasive_tile_v3_concat_bach`** is published to
`models/tissue_type/` and is what step 8 loads.

ImageNet **and** SimCLR ResNet18 bodies, both frozen, their 512-vectors concatenated to
1024, one hidden layer (256, ReLU, dropout 0.2), three classes. Trained on BCSS + all
six borrowed trees with `bracs_ic` capped to 25% of its patients, balanced class
weights. **Invasive is decided by `p(invasive) ≥ 0.40`, not by a three-way argmax.**

| | value |
| --- | --- |
| **accuracy** | **0.864** |
| **weakest class recall** | **0.827** |
| stroma / in-situ / invasive | 0.916 / 0.827 / 0.839 |
| Dice invasive vs non-invasive | 0.892 |
| DCIS called invasive | 22.4% |
| cross-validated weakest class | 0.802 ± 0.031 |

Highest accuracy of every configuration measured (~200 per sweep, four sweeps).

### What deploying it changed

Three things, because a two-body model does not fit the path v1/v2 were served on:

| file | change |
| --- | --- |
| `tissue_type_model_training/src/models.py` | `ConcatResNet18MLP`; `publish` records `arch`; `load_pinned` branches on it |
| `…/step08_tissue_type_segmentation/model.py` | builds the concat architecture when the manifest says so; `Pinned.tau` |
| `…/step08_tissue_type_segmentation/inference.py` | `_decide()` — τ threshold instead of argmax |
| `…/services/tissue_type_service.py` | passes `pinned.tau` through |
| `…/backend/app/core/config.py` | `tissue_type_model` default flipped v2 → v3 |

`settings.tissue_type_model` is the switch: set it back to
`invasive_tile_v2_imagenet_std` to serve the NC/SA-only model, with no other change.

`tau=None` for every earlier checkpoint, so v1 and v2 are still served exactly as they
were validated. Verified: **G6 reproduces the manifest's probe logits to 5e-7** through
the fully assembled two-body network, 360 demo tests and 64 approach-1 tests pass.

### ⚠ Two things that ship with it

**Inference cost roughly doubles.** Two ResNet18 forward passes per tile. Step 8's wall
clock on a whole slide goes up accordingly, for ever.

**The licence question is open and was overridden, not resolved.** BACH is CC BY-NC-ND
4.0 — NoDerivatives — and whether a model fitted on it is a derivative work has not been
decided by anyone. This was published at the project owner's explicit direction with
that question open; the position is recorded in the checkpoint's own manifest under
`training_data.licence_position`. **Do not distribute it externally until the ND term is
settled.** `invasive_tile_v2_imagenet_std` and the pre-BACH heads in `reports/` carry
only NC/SA terms.

### ⚠ Do not read the model picker as a ranking

Step 8's checkpoint list shows each manifest's own `held_out.accuracy`, and those were
measured on **different test sets at different times**. Side by side they look like a
league table and are not one:

| shown in the picker | actually |
| --- | --- |
| `invasive_tile_v1_imagenet` **0.915** | the *worst* model here — scored on a population where class 1 is effectively absent, and the one that scored **0.19 Dice** on invasive |
| `invasive_tile_v2_imagenet_std` **0.849** | re-scored on v3's tiles it gets **0.721** |
| `invasive_tile_v3_concat_bach` **0.864** | on the same tiles as that 0.721 |

The only comparisons in this document that mean anything are the ones where the test
population is stated and identical. v1's 0.915 is the highest number on that screen and
picking it would be a serious mistake.

### If you would rather not take those two

`reports/best_head_imagenet_std.pt` is a drop-in `fc` state dict — single backbone, no
BACH, accuracy **0.823** against v2's 0.721 — needing only the τ change, which is now
already in place. It gives up 0.041 accuracy to avoid both problems.

### Against the shipped model

v2 re-scored on the same features and the same held-out tiles (pre-BACH run, where the
test population matched v2's exactly):

| | v2 (shipped) | this |
| --- | --- | --- |
| stroma recall | 0.856 | 0.904 |
| in-situ recall | 0.829 | 0.832 |
| invasive recall | 0.592 | **0.821** |
| invasive, second laboratory | 0.410 | **0.753** |
| accuracy | 0.721 | **0.853** |
| DCIS called invasive | **10.3%** | 19.7% |

These are the **pre-BACH** figures, because that run's test population is identical to
v2's — 6,208 tiles of held-out BRACS and BCSS, no BACH. The post-BACH model scores 0.864
on its own larger test set, which is not the same population and must not be compared
with v2's 0.721.

v2 holds DCIS→invasive at 10.3% by being bad at invasive generally — it finds 59% of
invasive carcinoma and 41% of a second laboratory's. That is not Rule 5 respected, it is
Rule 5 untested. **If 22.4% is unacceptable, τ = 0.80 returns it to 10.0%** at a cost of
invasive recall 0.839 → 0.707. The whole curve is in `reports/sweep_heads_bach.json`.

---

## On the slide the problem was found on

`CAN_00251_26_H&E`, 170 tissue windows, 13,770 tiles, same body and transform, only the
head swapped:

| | v2 | candidate |
| --- | --- | --- |
| **false in-situ field** (tissue-gated) | 23.8% | **14.1%** |
| all tiles | 32.6% | 20.5% |
| in-situ mean confidence | 0.782 | 0.715 |
| invasive found | 1.7% | 3.3% |
| in-situ runner-up = stroma | 84.6% | 81.7% |

v2 reproduces the original failure. The candidate cuts the false in-situ field by ~40%
and drops its confidence. **But the residual in-situ is still firing on stroma 81.7% of
the time** — reduced, not fixed. *(Run with the earlier `imagenet+linear` candidate;
`concat+mlp` needs both backbones and the script loads one.)*

---

## Every approach, and whether it worked

| # | approach | verdict |
| --- | --- | --- |
| 1 | τ filter on p(invasive) | ✅ **works** — τ=0.40; +0.02–0.05 weakest-class recall over argmax |
| 2 | Bigger field of view (224 µm) | ❌ **no** — scored higher on a smaller, easier test set |
| 3 | Second laboratory's in-situ (BACH) | ➖ **+0.002 accuracy** on comparable data; broke the confound; **measured a 0.294 in-situ recall on unseen tissue** |
| 4 | Rebalance the `ic` tree | ✅ **works** — 25% of patients beats 100% |
| 5 | τ calibrated for Rule 5 | ✅ measured — τ=0.80 restores 10.0% |
| 6 | Unweighted ablation | ❌ **no** — wrecks a class |
| 7 | MLP head | ✅ **works** — +0.010 CV |
| 8 | ImageNet+SimCLR concatenation | ✅ **biggest single gain** — +0.020 CV |
| 9 | Ensembling | ➖ ties, higher variance, 3× cost |
| 10 | SimCLR initialisation | ➖ loses alone (CV 0.690), **helps concatenated** |

### Cross-validated ranking (τ=0.40, 5-fold GroupKFold inside training)

| family | CV weakest class | sd | CV accuracy |
| --- | --- | --- | --- |
| **concat+mlp** | **0.802** | 0.031 | 0.862 |
| ensemble | 0.788 | 0.038 | 0.860 |
| imagenet+mlp | 0.778 | 0.036 | 0.837 |
| concat+linear | 0.777 | 0.026 | 0.842 |
| simclr+mlp | 0.764 | 0.038 | 0.832 |
| imagenet+linear | 0.735 | 0.023 | 0.815 |
| simclr+linear | 0.690 | 0.044 | 0.789 |

Selection is by cross-validation, never by the held-out sweep. That matters: the first
winner this project picked scored 0.788 on held-out and cross-validated at
0.751 ± 0.070 — the fold spread was the only thing that flagged it in advance.

### Which trees are mandatory — all of them

| training set | weakest class | in-situ | invasive | DCIS→inv |
| --- | --- | --- | --- | --- |
| bcss+dcis | 0.592 | 0.829 | 0.592 | 10.3% |
| bcss+dcis+ic | 0.636 | 0.636 | 0.806 | 26.3% |
| bcss+dcis+normal | 0.587 | 0.864 | 0.587 | 12.1% |
| **bcss+dcis+ic+normal** | **0.754** | 0.754 | 0.805 | 31.8% |
| bcss+ic+normal (no dcis) | 0.581 | **0.581** | 0.880 | 62.4% |

Class weights are mandatory too: `bcss+dcis+ic` unweighted gets in-situ recall **0.356**
and calls **52.6%** of DCIS invasive. That resolves the suspicious ablation — unweighted's
higher Dice was bought entirely by over-calling invasive.

### Field of view: not shown to help

| | 112 µm | 224 µm |
| --- | --- | --- |
| tiles kept | 22,630 | 4,538 |
| test tiles | 6,208 | 1,241 |
| **dropped as `mixed`** | 17.8% | **22.1%** |

It scored higher (0.810 vs 0.788) and that is not evidence. The wider window discards
*more* candidate windows for having no clean majority label — and a window with no clean
majority is a duct edge, precisely the case the boundary lives in. Its test set is five
times smaller **and** easier by construction.

---

## What BACH actually bought

### It did NOT improve accuracy on the data we already had

Training with and without the three `bach_*` trees, identical recipe, scored on the
**same** held-out BRACS+BCSS tiles (6,208):

| | without BACH | with BACH |
| --- | --- | --- |
| accuracy | 0.853 | 0.855 |
| weakest class | 0.821 | 0.821 |
| in-situ recall | 0.832 | 0.827 |
| invasive recall | 0.821 | 0.821 |

Noise. **The 0.853 → 0.864 improvement quoted elsewhere in earlier drafts of this file
was the test set changing, not the model getting better** — adding BACH put BACH tiles
into both halves. `reports/did_bach_help.json` is the corrected comparison.

### What it did instead: measure the cross-laboratory collapse

Scored on **held-out BACH** — a laboratory the BACH-free model has never seen:

| | without BACH | with BACH | |
| --- | --- | --- | --- |
| **in-situ recall** | **0.294** | 0.828 | +0.534 |
| invasive recall | 0.772 | 0.946 | +0.174 |
| accuracy | 0.674 | 0.909 | +0.236 |
| Dice invasive | 0.728 | 0.934 | +0.206 |

**A model trained without BACH recovers 29.4% of in-situ carcinoma on an unseen
laboratory, and 77.2% of invasive.** That is the cross-laboratory failure, measured
directly, and it mirrors the teacher exactly (BEETLE: 99% agreement on BACH invasive,
53% on BACH in-situ). Invasive travels between laboratories; in-situ does not.

> **Read the right column with care.** Once BACH is in training it is no longer an
> unseen laboratory, so 0.828 is an in-domain number and is *not* evidence of
> generalisation to a fourth laboratory. The number that predicts behaviour on your
> clinical slides is the **left** column: ~0.29 in-situ recall on unseen tissue.

### It broke the confound

| class | before | after |
| --- | --- | --- |
| non-epithelium | 84.1% BCSS / 15.9% BRACS | 63.0% BCSS / 11.9% BRACS / **25.1% BACH** |
| **in-situ** | 0.3% BCSS / **99.7% BRACS** | 0.2% BCSS / **67.4% BRACS / 32.4% BACH** |
| invasive | 51.9% BCSS / 48.1% BRACS | 42.7% BCSS / 39.6% BRACS / **17.7% BACH** |

The shortcut "looks like a BRACS crop ⇒ in-situ" is genuinely dead: BRACS now supplies
invasive and stroma too, and a third of in-situ comes from another continent.

### It measured the real problem, which is worse than the confound

BEETLE — the teacher behind *every* borrowed label here — was asked to segment BACH's
images, and its agreement with BACH's pathologists is wildly asymmetric:

| BACH tree | BEETLE agrees |
| --- | --- |
| **invasive** | **99%** (99/100) |
| normal | 74% |
| **in-situ** | **53%** (47 rejected, every one for calling invasive where humans said in-situ) |

Compare BRACS: `ic` 10% rejected, `dcis` 22%, `normal` 30%. **The teacher recognises
invasive carcinoma on an unfamiliar scanner almost perfectly and fails on in-situ half
the time.** Every in-situ label in this project is BEETLE's output, so every in-situ
number — including this model's 0.827 — inherits that.

### ⚠ It did NOT make the features laboratory-invariant

Within **every** class, a probe separates BRACS from BACH at:

| class | BRACS vs BACH |
| --- | --- |
| non-epithelium | 0.994 |
| in-situ | 0.994 |
| invasive | 0.997 |

0.5 is chance. Laboratory identity is almost perfectly readable *inside* each class, so
a head fitted on these features is free to learn per-laboratory rules that need not
transfer to a fourth laboratory. **Killing the `laboratory ⇒ class` correlation is not
the same as removing the laboratory signal, and only the first was achieved.**

### The leakage check is still INCONCLUSIVE

`06_leakage_check.py` projects out one source direction and reports honestly that it
did not work (source separability 0.898 → 0.877; chance is 0.5). I tried a stronger
version and **it failed for the same reason**: a 3-class linear probe yields a rank-3
subspace, so "40 directions removed" actually removed 3 of 512. The flat curve in
`reports/leakage_two_labs.json` is an artefact of my method, not a property of the data.

Neither test settles the question. The shipped check's own closing line is right: what
settles it is in-domain tiles clicked by a pathologist.

---

## ⚠ LICENCE — this checkpoint cannot currently be published

| source | licence |
| --- | --- |
| BCSS | CC0 1.0 |
| torchvision ResNet18 | BSD-3 |
| Ciga SimCLR ResNet18 | MIT |
| BEETLE nnU-Net | CC BY-NC-SA 4.0 — non-commercial |
| BRACS | non-commercial |
| **BACH / ICIAR-2018** | **CC BY-NC-ND 4.0 — NoDerivatives** |

**ND is the binding term and it is unresolved.** Whether a model fitted on BACH images
is a "derivative work" is a legal question nobody here has answered. A model trained
*without* the three `bach_*` trees carries only the NC/SA terms — `imagenet+mlp` or
`concat+mlp` on the pre-BACH tile set (CV 0.792 ± 0.014, accuracy 0.853) is the
publishable-subject-to-NC fallback, and it is only ~0.01 behind.

---

## What is still wrong

- **In-situ labels are BEETLE's, and BEETLE gets in-situ wrong ~half the time
  off-distribution.** This is the root cause and nothing here fixes it. Pathologist-drawn
  in-situ on in-domain tissue is the only thing that would.
- **Laboratory identity is readable at 0.994+ inside every class.**
- **Held-out patients come from the same three laboratories as training.** Only the
  slide test speaks to a fourth, and it covers one slide.
- **DCIS→invasive is measured against BEETLE's labels too** — BRACS DCIS regions can
  contain genuine invasive foci, so part of the 22.4% is correct behaviour counted as error.
- **`concat+mlp` has not been run over the slide** — needs both backbones.
- **Two backbones at inference** is a serving cost the pipeline does not currently pay.

## Files

| | |
| --- | --- |
| chosen head | `reports/best_head_concat_mlp_bach.pt` (+ `.json`) |
| pre-BACH equivalent (NC-only licence) | `reports/best_head_concat_mlp.pt` |
| single-backbone fallback | `reports/best_head_imagenet_std.pt` |
| head sweeps / CV | `reports/sweep_heads_bach.json`, `confirm_heads_bach.json` |
| tree & τ sweeps | `reports/sweep_112um.json`, `sweep_simclr.json`, `sweep_224um.json`, `sweep_bach_*.json` |
| **slide test** | `reports/slide_test_CAN_00251.json`, `slide_test.log` |
| leakage | `reports/06_leakage_with_bach.log`, `leakage_two_labs.json` |
| confound before/after | `reports/bach_chain_state.json` |
| v2's preserved inputs | `data/v2_baseline/` |
| BACH licence note | `tissue_label_generation/data/bach/LICENCE_NOTE.md` |
