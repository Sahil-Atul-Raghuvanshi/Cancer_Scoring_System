# Step 10 - Align the ROI to the IHC slide

## Input

Step 9's invasive-carcinoma regions, in the H&E slide's level-0 pixels, plus
the case's IHC slide for the chosen antibody.

## Output

The same regions in the **IHC slide's** level-0 pixels, two border-only panels
to judge them on, and three crops taken from the IHC slide itself.

## Why this step exists, and why it is the awkward one

Tissue typing runs on the H&E: that is where structure is legible and where
the model was trained. The brown that actually gets scored is on the IHC
section. Those are two different physical slices of one block, cut seconds
apart - **structurally similar, not the same cells** - so a region found on one
cannot simply be read off at the same coordinates on the other.

This reverses a decision recorded elsewhere in the docs, and the reversal
should be understood rather than discovered.
[`docs/guides/images-to-scores-mapping.md`](../../../../docs/guides/images-to-scores-mapping.md)
Part 3 and [`docs/design/five-marker-implementation.md`](../../../../docs/design/five-marker-implementation.md)
§1.8 both chose *not* to register, on the grounds that registration error at a
region boundary turns straight into scoring error, and that pathologists
re-find the tumour on each slide by eye instead.
[`docs/segmentation_research/segmentation-approaches-implementation.md`](../../../../docs/segmentation_research/segmentation-approaches-implementation.md)
Part 7 kept registration only as an offline QA cross-check. This step promotes
that cross-check to the main path, so the objection it was filed under has to
be answered rather than ignored - which is what the gate and the confirmation
below are for.

## What it refuses to do

VALIS returns a transform for a pair it could not really register, and that
transform is wrong *silently*. Four independent measurements decide whether the
result is emitted at all (`app/registration/gate.py`):

| Measure | Refuses when | Why it is not redundant |
| --- | --- | --- |
| matched keypoints | too few to fit anything | the fit had nothing to work from |
| residual error (µm) | features still far apart after warping | the warp did not bring them together |
| tissue-area ratio | one section holds far less tissue | catches a near-blank IHC section, and needs no successful registration to be meaningful |
| round-trip error (µm) | warping across and back does not return | measured on the ROI's own vertices, not on the keypoints the fit flattered itself with |

A refusal is stored as a result, with its numbers, and **no regions**.

## What it will not decide on its own

Passing the gate does not mean the alignment is right. The step finishes
`confirmed=false` and stays there until a person looks at the two panels side
by side and says the regions landed on the same tissue. Nothing downstream
should measure inside these regions before that.

## What has actually been measured

On `CAN_00270`'s H&E against its CD44 (`_A`) slide, both 126,976 px square at
0.2222 µm/px, registering at 2048 px:

| | VALIS's default preprocessing | **`OD` + `adaptive_eq`, what this ships** |
| --- | --- | --- |
| matched features | 12 | **67** |
| error between matched features | 1039 µm → 31 µm rigid → **94 µm** non-rigid | 949 µm → 41 µm rigid → **43 µm** non-rigid |
| round trip over the warped regions | 0.004 µm median | 0.005 µm median, 0.7 µm worst |
| time | ~5.5 min, ~20 s to reload a cached one | ~14 min under load |

**The preprocessing is the whole story here, not the detector.** Without
equalisation, VALIS renders the H&E mid-grey with filled structures and the
CD44 slide near-black with thin bright edges — the same tissue architecture
drawn two ways no matcher can pair up. You can see it in VALIS's own
`processed/` thumbnails. Swapping detectors does not help and makes it worse:
SuperPoint+SuperGlue found **0** features on this pair, and DISK at 4096 px
also found 0, in twenty minutes.

Note the second column's non-rigid error *holds* at 43 µm rather than
degrading to 94 µm. On twelve keypoints that degradation was mostly noise —
both figures are medians over the very points the transform was fitted to,
which is exactly why the gate does not rely on them alone.

**A limit of `alignment_nmi` worth knowing.** It scored 1.0121 for *both*
configurations above, and that is not a failure of the alignment but of the
probe's resolution: it samples the slides at ~62 µm per pixel, so it cannot
see the difference between a 41 µm and a 94 µm residual — both are sub-pixel
to it. It is a check on *gross* correspondence (on the synthetic pair, where
misalignment is large, it swings from 1.157 to 1.605) and the keypoint count
and residual are what discriminate fine alignment. Read them together.

**A tissue-overlap score was tried first and thrown away.** A saturation
threshold called 24% of the H&E tissue against 4% of the IHC, so the fraction
was bounded by the mask disagreement rather than by the alignment, and every
configuration would have tied at the ceiling. Absorbance-based mutual
information replaced it: no mask, and no assumption that the two stains look
alike.

**`MicroRigidRegistrar` is not used.** It refines the rigid fit against the
slides at full resolution and was killed after 47 minutes and 2.3 CPU-hours on
this pair, against 5.5 minutes without it. This step is a screen somebody waits
at, so that is disqualifying whatever it would have scored.
`valis_service/_tune.py` records this and compares the remaining
configurations.

## Accuracy notes

*   **Rings are densified before warping** (`app/registration/transform.py`).
    A non-rigid warp does not map a straight line to a straight line, so
    warping only the corners of a tile-grid-traced ring keeps the corners
    honest and drops every deformation between them.
*   **Registration resolution is a setting, not a default.** VALIS registers at
    512 px out of the box, which on a 127,000 px slide is one pixel per ~55 µm.
    `settings.registration_max_dim_px` raises it.
*   **`tissue_hit_fraction` is the honest accuracy number.** It measures how
    much of the H&E's tissue lands on IHC tissue over the whole section, and is
    reported next to the same figure with no registration at all - a warp that
    does not beat doing nothing has not earned the name.

## Status

Built. Carries regions through the stored Mattes transform that
`slide_registration/fit_best.py` fitted for the pair, applied with SimpleITK in the
isolated environment in [`valis_service/`](../../../../valis_service/README.md). VALIS
itself is retired to `decrecated_code/valis_registration/`. Without that environment the step
reports that registration is unavailable rather than pretending to align.
