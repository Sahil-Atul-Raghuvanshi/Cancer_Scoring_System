# Fix 1 + Fix 2 — live progress

Appended by the runner as each stage completes. Newest entries at the bottom.
Timestamps are local. Written by
`scratchpad/run_fix1_fix2.sh`; per-stage logs are in `scratchpad/logs/`.

**Plan:** [../design/FIX1_FIX2_PLAN.md](../design/FIX1_FIX2_PLAN.md) · **Results (written at the end):**
`RESULTS_224_VS_448.md`


### S1 - region trees (Fix 1: the consensus decides the lesion)

14:35:33 | S1 bach_insitu: starting
14:35:37 | S1 bach_insitu: kept 100 regions | overruled the teacher on 100 of 100 regions, median 47% of their epithelium
14:35:37 | S1 bach_invasive: starting
14:35:41 | S1 bach_invasive: kept 99 regions | overruled the teacher on 19 of 99 regions, median 4% of their epithelium
14:35:41 | S1 bach_normal: starting
14:35:47 | S1 bach_normal: kept 100 regions | overruled the teacher on 90 of 100 regions, median 13% of their epithelium
14:35:47 | S1 normal: starting
14:46:20 | S1 normal: kept 213 regions | overruled the teacher on 158 of 213 regions, median 0% of their epithelium
14:46:20 | S1 ic: starting

### Re-planned at 15:02 for a 17:00 deadline

The sequential chain would have finished S1 at ~17:00 and S2 at ~18:15. Two changes:

- **`ic`'s 131 fresh segmentations are deferred**, via a new `--cached-only`. It supplies
  class 2, which BCSS already has in abundance, so an hour of teacher spent there buys
  far less than the same hour spent on DCIS. 189 of its 320 regions are cached or
  re-codable and go in now; the rest are a follow-up run, and the manifest records them
  as `skipped` rather than rejected.
- **S2's expensive half runs now, in parallel with S1.** Cutting the 151 BCSS regions
  does not depend on any region tree - those tiles come straight from `original_data` -
  so it overlaps the DCIS segmentation and `--reuse-bcss` picks the rows up afterwards.
  The two load different things (nnU-Net threads vs numpy and disk), so they contend
  less than two teachers would.

15:04:33 | follow-on: waiting for the DCIS segmentation and the BCSS cut
15:06:42 | follow-on: waiting for the DCIS segmentation and the BCSS cut
### Unattended from 15:10 — what is running and when it lands

Three processes, all detached. Closing the editor orphans them but does not kill them —
that was tested at 14:58, when the app was closed accidentally and the run continued.

| | process | state at 15:10 |
| --- | --- | --- |
| P1 | DCIS segmentation, 239 regions | 36 done, 15 freshly segmented · **~15:54** |
| P2 | BCSS tiles at 224 µm, 150 regions | 92 done · **~15:15** |
| W | follow-on watcher, polling both logs every 30 s | waiting |

**Machine checks before leaving it:** sleep and hibernate are both disabled on AC
(`STANDBYIDLE = 0x0`), so it will not drop out mid-run. 24.8 GB free against roughly
5 GB of remaining writes — the new DCIS segmentations in `runs/`, their copies in
`regions/dcis/`, and the tiles.

**The order is a priority order, not a pipeline order.** 224 µm is carried all the way
to a fitted head before 448 µm is started, and within 224 µm the imagenet head is fitted
and its results written before the simclr feature pass begins. Each feature pass is
~25 min, so doing both first would have put the first real number after 17:30. If the
machine stops at any point, what survives is the most useful thing finishable by then.

### Expected timeline

| time | milestone |
| --- | --- |
| ~15:15 | P2 done — BCSS tiles at 224 µm on disk |
| ~15:54 | P1 done — **S1 complete**, every DCIS region segmented under Fix 1 |
| ~15:58 | `ic` assembled from cache (131 regions deferred) |
| ~16:20 | borrowed trees cut, reusing BCSS — **S2 complete**, `data/224_um/` exists |
| ~16:45 | features, imagenet |
| **~16:50** | **224 µm head fitted · RESULTS_224_VS_448.md carries a real number** |
| ~17:40 | simclr features + the concat re-fit (comparable with served v3) |
| ~18:30 | 448 µm export, features, head — **everything done** |

15:12:28 | follow-on: waiting for the DCIS segmentation and the BCSS cut
15:27:34 | follow-on: still waiting, 15 min - P1 at 83/239 dcis regions
15:42:39 | follow-on: still waiting, 30 min - P1 at 137/239 dcis regions
15:57:48 | follow-on: still waiting, 45 min - P1 at 191/239 dcis regions
16:12:55 | follow-on: still waiting, 60 min - P1 at 228/239 dcis regions
16:14:26 | follow-on: both halves are done

### S1b - the invasive tree, assembled from cache

16:14:29 | follow-on: waiting for the DCIS segmentation and the BCSS cut
16:14:30 | follow-on: both halves are done

### S1b - the invasive tree, assembled from cache

16:14:30 | follow-on: waiting for the DCIS segmentation and the BCSS cut
16:14:31 | follow-on: both halves are done

### S1b - the invasive tree, assembled from cache

16:15:22 | S1b ic: kept 202 regions | SKIPPED 115 regions deferred

### S2 - 224 um: the borrowed trees, reusing the BCSS cut

16:15:22 | S1b ic: kept 202 regions | SKIPPED 115 regions deferred

### S2 - 224 um: the borrowed trees, reusing the BCSS cut

16:15:22 | S1b ic: kept 202 regions | SKIPPED 115 regions deferred

### S2 - 224 um: the borrowed trees, reusing the BCSS cut

16:22:04 | S2: G2 gate reported a problem - tiles written, continuing. See s2_borrowed224.log
16:22:18 | S2: G2 gate reported a problem - tiles written, continuing. See s2_borrowed224.log
16:22:18 | S2: G2 gate reported a problem - tiles written, continuing. See s2_borrowed224.log
16:22:38 | S2: split FAILED - see s2_split.log
16:22:38 | S2: split FAILED - see s2_split.log
16:22:51 | S2: split into data/224_um/{bcss,beetle,bach}
      bach       1322 tiles  non_epithelium=444  non_invasive_epithelium=491  invasive_epithelium=387
      bcss       2253 tiles  non_epithelium=1039  non_invasive_epithelium=1  invasive_epithelium=1213
      beetle     3332 tiles  non_epithelium=526  non_invasive_epithelium=1280  invasive_epithelium=1526

### S4/S5 - 224 um: features and the head


### S4/S5 - 224 um: features and the head (resumed 16:27)

16:25:00 | resumed: data/224_um holds 6,907 tiles - 1,772 of them class 1, of which 1,771 are borrowed
16:31:24 | S4 224/imagenet: 6,377 tiles, manifest fingerprint ea8212250393cbdf
16:31:52 | S5 224/imagenet: **HEAD FITTED - there is now a 224 um model**
16:31:52 | results written with the 224 um imagenet head in place
16:34:36 | S4 224/imagenet: 6,377 tiles, manifest fingerprint ea8212250393cbdf
16:35:07 | S5 224/imagenet: **HEAD FITTED - there is now a 224 um model**
      [imagenet] fold 1: dice invasive vs in-situ 0.734 (906 tiles)
      [imagenet] fold 2: dice invasive vs in-situ 0.845 (906 tiles)
      [imagenet] fold 3: dice invasive vs in-situ 0.730 (906 tiles)
      [imagenet] fold 4: dice invasive vs in-situ 0.804 (905 tiles)
        dice non_epithelium             0.866
        dice non_invasive_epithelium    0.743
        dice invasive_epithelium        0.849
      tumour content      tiles   accuracy   dice inv
      ablation, no class weights: accuracy 0.830 (vs 0.826), dice invasive 0.878 (vs 0.860)
    == G4: dice invasive vs non-invasive 0.860 (target >= 0.75) ==
16:35:08 | RESULTS_224_VS_448.md written with the 224 um imagenet head
16:39:57 | S4 224/simclr: done
16:40:33 | S5 224: **both inits fitted** (comparable with the served concat v3)
16:40:34 | **224 um IS COMPLETE.** Everything below is the 448 um arm.

### S3 - 448 um export

16:42:01 | S4 224/simclr: done
16:42:31 | S5 224: **both inits fitted** (comparable with the served concat v3)
16:42:31 | **224 um IS COMPLETE.** Everything below is the 448 um arm.

### S3 - 448 um export

16:50:15 | concat-448 watcher: waiting for both 448 um feature caches
16:50:22 | S3: G2 gate reported a problem - tiles written, continuing
16:50:24 | S3: split into data/448um/{bcss,beetle,bach}
      bach        177 tiles  non_epithelium=49  non_invasive_epithelium=68  invasive_epithelium=60
      bcss        424 tiles  non_epithelium=178  non_invasive_epithelium=0  invasive_epithelium=246
      beetle      556 tiles  non_epithelium=48  non_invasive_epithelium=209  invasive_epithelium=299

### S4/S5 - 448 um: features and the head

16:51:21 | S4 448/imagenet: done
16:51:51 | S3: 
16:51:53 | S3: split into data/448um/{bcss,beetle,bach}
      bach        177 tiles  non_epithelium=49  non_invasive_epithelium=68  invasive_epithelium=60
      bcss        424 tiles  non_epithelium=178  non_invasive_epithelium=0  invasive_epithelium=246
      beetle      556 tiles  non_epithelium=48  non_invasive_epithelium=209  invasive_epithelium=299

### S4/S5 - 448 um: features and the head

16:52:19 | S4 448/simclr: done
16:52:26 | concat-448: both caches present, fitting the concat+MLP
16:52:41 | S5 448: **head fitted**
16:52:42 | **ALL DONE** - RESULTS_224_VS_448.md has both arms
16:52:46 | **concat+MLP fitted at 448 um**
    == held out, argmax ==
      dcis_called_invasive             0.0685
      recall_insitu                    0.8904
      recall_invasive                  0.8770
      recall_stroma                    0.8833
      min_recall                       0.8770
      macro_recall                     0.8836
      dice_invasive_vs_non_invasive    0.9619
    
    == held out, tau=0.4 ==
      dcis_called_invasive             0.0685
      recall_insitu                    0.8904
      recall_invasive                  0.8877
      recall_stroma                    0.8833
      min_recall                       0.8833
      macro_recall                     0.8871
      dice_invasive_vs_non_invasive    0.9623
    
    wrote data\448um\reports\08_concat_mlp.json
16:52:46 | **RESULTS_224_VS_448.md now carries both concat heads** - the comparison is complete
16:52:54 | S4 448/imagenet: done
16:53:40 | S4 448/simclr: done
16:53:49 | S5 448: **head fitted**
16:53:50 | **ALL DONE** - RESULTS_224_VS_448.md has both arms

### 672 um arm - 224 px at 3.0 um/px

17:03:09 | 672 um: BACH yields 0 tiles at this geometry and 196 of 239 DCIS regions yield 0. Expect ~400 tiles.
17:08:45 | 672 um export: 
17:08:46 | 672 um: split into data/672um/
      bcss        122 tiles  non_epithelium=45  non_invasive_epithelium=0  invasive_epithelium=77
      beetle      146 tiles  non_epithelium=7  non_invasive_epithelium=38  invasive_epithelium=101
17:09:06 | 672 um features/imagenet: done
17:09:24 | 672 um features/simclr: done
17:09:30 | **672 um concat+MLP fitted**
    == held out, argmax ==
      dcis_called_invasive             0.0909
      recall_insitu                    0.8182
      recall_invasive                  0.9259
      recall_stroma                    0.7273
      min_recall                       0.7273
      macro_recall                     0.8238
      dice_invasive_vs_non_invasive    0.9804
    
    == held out, tau=0.4 ==
      dcis_called_invasive             0.0909
      recall_insitu                    0.8182
17:09:30 | **RESULTS_224_VS_448.md now carries all three fields of view**

### 112 um under Fix 1 - the controlled test of the label rule

17:38:54 | 112 um: same geometry as the served v3, new label rule. Expect ~30k tiles - the largest arm.

### 112 um under Fix 1 - the controlled test of the label rule

20:52:48 | 112 um: same geometry as the served v3, new label rule. Expect ~30k tiles - the largest arm.
21:14:41 | 112 um export: 
21:15:48 | 112 um: split into data/112um/
      bach       8852 tiles  non_epithelium=3874  non_invasive_epithelium=2754  invasive_epithelium=2224
      bcss      10726 tiles  non_epithelium=5435  non_invasive_epithelium=9  invasive_epithelium=5282
      beetle    16961 tiles  non_epithelium=4192  non_invasive_epithelium=5905  invasive_epithelium=6864
21:53:52 | 112 um features/imagenet: done
22:38:34 | 112 um features/simclr: done
22:40:15 | **112 um concat+MLP fitted and published as invasive_tile_fov112_fix1_concat**
    == held out, argmax ==
      dcis_called_invasive             0.1890
      recall_insitu                    0.7784
      recall_invasive                  0.8439
      recall_stroma                    0.9079
      min_recall                       0.7784
      macro_recall                     0.8434
      dice_invasive_vs_non_invasive    0.8872
    
    == held out, tau=0.4 ==
      dcis_called_invasive             0.2211
      recall_insitu                    0.7467
    served-vs-fitted agreement on 21 probe tiles: worst |difference| 0.00e+00
    published invasive_tile_fov112_fix1_concat.pt and invasive_tile_fov112_fix1_concat.manifest.json
22:40:16 | **RESULTS_224_VS_448.md now carries 112/224/448/672 - the label rule is isolated at 112**
22:41:14 | **112 um published** - checking step 7 hands it to step 8

| field of view | checkpoint step 8 will run |
| --- | --- |
| 112 um | `invasive_tile_fov112_fix1_concat` |
| 224 um | `invasive_tile_fov224_concat` |
| 448 um | `invasive_tile_fov448_concat` |
| 672 um | `invasive_tile_fov672_concat` |

**112 um now serves the Fix 1 head.**
22:41:15 | verify112: done

---

## 2026-09-09 — the H&E campaign

Four RGB heads beside the four haematoxylin ones, and step 7 turned into a branch
screen. **Live status is in [`HE_CAMPAIGN_TRACKER.md`](HE_CAMPAIGN_TRACKER.md)** — one
writer, rewritten in full after every stage, which is why it is not appended to here.
The integration record is [`STEP7_BRANCH_STATUS.md`](STEP7_BRANCH_STATUS.md) and the
plan behind both is [`../design/STEP7_BRANCH_PLAN.md`](../design/STEP7_BRANCH_PLAN.md).

One thing to know when reading the log above: the four field-of-view directories it
names (`data/112um`, `data/224_um`, `data/448um`, `data/672um`) now live under
`data/h_channel/`, with the `224_um` underscore normalised. The paths in this file are
left as they were written — it is a log, and rewriting history to match a later move
would make it useless as one.
