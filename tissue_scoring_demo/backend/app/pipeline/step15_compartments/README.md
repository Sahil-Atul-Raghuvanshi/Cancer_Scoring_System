# Step 13 - Build compartments

## Input

Step 11's nucleus label maps, step 12's classes, and the antibody letter.

## Output

Three disjoint regions per tumour cell — nucleus, cell, and the compartment the
marker is measured in — plus how much that compartment changes if the width does.

## The fork, and where it actually lives

**This is the first step in the pipeline that reads the antibody letter.**

| markers | compartment | how it is built |
| --- | --- | --- |
| A CD44, F ABCC4, R ABCC11 | membrane ring | grow the nucleus by ~4 µm, keep the outer band, subtract the nucleus and every neighbour |
| U N-cadherin, W pan-cadherin | cytoplasm band | a wider expansion, ~6 µm, nucleus subtracted, **not** thinned to a ring |

The fork is **not defined here**. It is read from
[`app/panel.py`](../../panel.py), which already carries `compartment`,
`compartment_width_um` and `second_measure` for all five antibodies. A second
copy in this folder would be a second place to change it, and the two could
differ while both looked right.

It is also **not a request parameter**. `GET /compartments/{he}` takes a width,
because that is a genuine unknown, but it cannot be asked for a ring on a
cadherin. The guide names that specific regression, and the cleanest way to
prevent it is to make it unreachable rather than to validate against it.

**Why it is not a small error.** A thin 3 µm ring on a cytoplasmic marker samples
the outer edge of the cell, which is where it meets the next cell. What comes out
correlates with how tightly the tissue is packed rather than with how much
N-cadherin the cell has made — and it will look entirely plausible while doing
it.

## Voronoi-constrained expansion

Plain dilation of each nucleus by 4 µm walks straight into the neighbouring cell
in crowded epithelium. Two cells then both claim the same pixels, so a strongly
stained cell bleeds its signal into a negative neighbour and the score drifts
with cell packing.

The fix is one distance transform instead of one dilation per cell. Growing every
nucleus at once from the full label map, with each pixel going to whichever
nucleus is nearest, makes double-claiming impossible by construction — the same
thing QuPath's cell expansion does. `contested_px` reports how many pixels the
constraint took away from a cell that plain dilation would have given it, so the
size of the error is measured rather than asserted.

It matters more for U and W than for A/F/R, exactly as the guide says: a 6 µm
band collides with every neighbour a 4 µm ring does, and more.

## Two details that are easy to get backwards

**Non-tumour cells are removed *before* the expansion, not after.** A lymphocyte
removed afterwards has already taken its share of the pixels between it and the
tumour cell beside it, so the tumour cell's compartment comes out dented by a
cell nobody is measuring. Removing it first lets the tumour cell grow into that
space — which is what would have happened had the lymphocyte never been detected.

**Widths are in microns and converted through the field's own mpp.** At this
project's 0.2222 µm/px a 4 µm ring is 18 px; on another scanner it is not.

## The honesty term

Neither default width is established fact. Both come from QuPath's convention
plus the physical size of a breast epithelial cell. So `width_sensitivity` sweeps
the width and reports what the compartment area would have been at each — the
guide's own instruction for this parameter, because a score that swings fifteen
points between a 3 µm and a 5 µm ring is a score with a hidden parameter in it.

## Status

Built, and request-shaped like step 12: it reads the label maps step 11 stored
and runs two distance transforms per field, which is fast enough to answer while
somebody drags the width slider. It refuses to run until step 12 has sorted the
cells, because compartments are built for tumour cells only and measuring a
lymphocyte's membrane would put a cell in the numerator that is not in the
denominator.
