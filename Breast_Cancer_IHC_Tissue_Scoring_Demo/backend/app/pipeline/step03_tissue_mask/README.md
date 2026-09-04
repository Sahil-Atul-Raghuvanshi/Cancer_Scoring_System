# Step 03 - Tissue mask

`app/data/pipeline_steps.py` (`id="tissue-mask"`) · implemented.

## Input

Usable tissue, as step 2 left it

## Output

Binary tissue mask, plus the histogram and the threshold that produced it

## Contents

- `mask.py` - the algorithm. Read the module docstring first: it states which
  two rules from the pipeline guide are enforced in code rather than left to a
  comment, and where.
- `overlay.py` - the four panels (thumbnail, saturation, mask, overlay).
- `pipeline.py` - `run(context)`. Adapts `app.services.tissue_service`, and
  enforces that step 2 ran first.

Otsu, Zack's triangle rule and the saturation channel live in
`app/common/imaging.py`, not here, because step 2's tissue-detection fallback is
built on the same primitives. That fallback is a stand-in *for* this step's
method; two private copies would let them drift apart, at which point the
fallback stops being the thing it claims to be.

## The three decisions worth arguing about

**Saturation, not brightness.** Glass is bright and colourless; pale tissue is
dim and coloured. They share brightness and do not share saturation, so a
brightness threshold has to choose between losing pale tissue and keeping bright
glass. Saturation has to do neither.

**QC first, and before the histogram.** Pen ink is darker and far more saturated
than any stain. Subtracting step 2's artefact map *after* thresholding would
still leave the ink in the histogram that chose the threshold, and on a faintly
stained slide that can push the cut clean past the tissue. So the artefacts come
out first and the histogram is counted over what remains.

**Not always Otsu.** This is the one place the step departs from
`docs/demo-pipeline-guide.md`, and it was measured rather than assumed.

Otsu's method is derived for a histogram of two classes, each with a spread, and
finds the split maximising the variance between them. On the demo's CD44 slide
the haematoxylin counterstain is so weak that **82% of the frame sits at
saturation exactly 0** and the rest is a monotone tail with no valley in it. A
spike has no variance, so Otsu's criterion is maximised far out in the tail:

| Rule | Cut | Tissue found | IoU vs step 2's tissue map | Recall |
| --- | --- | --- | --- | --- |
| Otsu | 27 | 46.8 mm² | 0.32 | 33% |
| Zack triangle | 1 | 157.8 mm² | **0.73** | **96%** |
| best possible on this channel | 3 | 148.4 mm² | 0.74 | 94% |

So saturation was never the problem — the threshold *rule* was. Both rules are
always computed and always reported; `choose_threshold` picks between them on the
modal level's share of the histogram (`tissue_spike_share`, default 0.35), which
is the assumption violation stated directly. Neither rule dominates: on a
genuinely bimodal histogram the triangle rule lands well below the valley and
Otsu is clearly better, which is why the choice is made from the data and not
fixed in advance.

Things that were tried and measured *worse*, recorded so nobody repeats them:
excluding step 2's background class from the histogram (Otsu moves to 31, IoU
0.24) and median-blurring the saturation channel first, as CLAM does (Otsu moves
to 20 at a 7 px radius, IoU 0.21).

## What this step does not do

It does not remove fat. Fat is tissue; at this stage the only thing known is
tissue versus glass, and dropping pale regions here would also drop pale tumour
and washed-out areas with no record of what went. Fat leaves at step 8, as a
named class you can toggle on screen.

What the step does do about fat is fill enclosed pale regions under
`tissue_fill_hole_max_mm2`, and report the area it reclaimed. Fat at the cut edge
of the section is still lost, and no morphology recovers it - which is the
honest limit of a threshold, and the reason step 8 is a model.

## Settings

All in `app/core/config.py`, all physical:

| Setting | Default | What it is |
| --- | --- | --- |
| `tissue_mask_mpp` | 2.0 | Resolution to work at |
| `tissue_mask_max_px` | 4096 | Cap on the mask's longest edge; binds on whole excisions |
| `tissue_close_um` | 30 | Closing radius - bridges gaps within one specimen |
| `tissue_open_um` | 12 | Opening radius - drops dust and precipitate |
| `tissue_min_component_mm2` | 0.02 | Smallest connected region kept |
| `tissue_fill_hole_max_mm2` | 1.0 | Largest enclosed void filled back in |
| `tissue_spike_share` | 0.35 | Modal-level share at which Otsu gives way to the triangle rule |

None of these is a pixel count, so none of them changes meaning on a different
scanner. The report carries each one's pixel equivalent at the resolution
actually used, so the conversion is checkable on screen.
