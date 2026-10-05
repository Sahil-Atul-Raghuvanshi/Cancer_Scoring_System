# Step 12 - Cell typing

## Input

Step 11's nuclei: area, perimeter, circularity, eccentricity and mean
haematoxylin, per nucleus, for the ones step 11 actually counted.

## Output

One of three classes per nucleus, the share of each, and — as much a part of the
output as the classes — how much of that share the thresholds are deciding.

| class | what it is | what it looks like |
| --- | --- | --- |
| `tumour` | the denominator | large, irregular |
| `lymphocyte` | immune infiltrate | small, round, very dark |
| `spindle` | fibroblasts, endothelium | elongated |

## Why this is a rule and not a model

The guide's recipe is to fit a small classifier on a few hundred hand-checked
nuclei, and it is right that this is the easy classification problem — unlike the
region problem, a non-expert can label it.

**But nobody has labelled any.** So there was a choice between a rule with its
judgements written out, and a model fitted on labels somebody invented. The
second is worse, and not marginally: it would carry exactly the same judgements
with an accuracy figure attached that implied they had been measured. A reader
cannot argue with 0.91; they can argue with "a lymphocyte nucleus is under
35 µm²".

So every threshold is a constant with its reasoning beside it in `classify.py`,
in microns, and **the sensitivity of the answer to each one is published with the
answer**. That is the guide's own standard for a parameter like this, applied
here rather than only to compartment widths: a tumour share that swings ten
points between two defensible thresholds is a share with a hidden parameter in
it, and the only honest thing is to show the swing.

When a few hundred nuclei do get labelled, `fit()` replaces `classify()` and
these thresholds become the baseline it has to beat.

## The order the rules are applied in, and why

1. **Lymphocyte first.** It is the most specific rule — small *and* round *and*
   dark, three conditions at once — and the costs are asymmetric: a lymphocyte
   left in the denominator dilutes the percentage, which is the error this whole
   step exists to remove.
2. **Spindle second**, and only on what is not already a lymphocyte.
3. **Tumour is the default.** Inside a region step 9 already called invasive
   carcinoma, the prior is that a cell is tumour unless it looks like something
   else. Making tumour a positive test instead would quietly shrink the
   denominator every time a nucleus was unusual.

Darkness is **relative to the field's median**, not absolute. Stain uptake varies
between slides, so an absolute cutoff would mean a different thing on each one —
and this rule has to work on an H&E and on five IHC sections.

## The check that can refuse the whole answer

Typing is size and shape arithmetic, so it inherits every fault in the
segmentation underneath it. If step 11 under-called the nuclei, these rules are
sorting fragments and a confident stacked bar would be the worst possible output.

`trustworthy` is false when the median nuclear area falls below **25 µm²**. A
breast epithelial nucleus is 7–10 µm across, so 40–80 µm²; measured on this
project's own H&E the median is 43. Well below that, the objects being sorted are
not cells.

This is not hypothetical on this panel. The CD44 section's median comes out at
15 µm², and step 11 independently reports 72% fewer nuclei per mm² than the H&E
of the same block. Two measurements, one conclusion, and the step says so rather
than drawing the bar.

## Status

Built, and request-shaped rather than job-shaped: it opens no slide and runs no
model, so it answers in well under a second. That is what makes the thresholds
live on screen.
