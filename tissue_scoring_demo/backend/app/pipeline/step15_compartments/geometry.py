"""Growing a cell outward from its nucleus, and stopping where it should.

The marker does not live in the nucleus. CD44, ABCC4 and ABCC11 sit in the
**membrane**; N-cadherin and pan-cadherin sit in the **cytoplasm**. So before
anything can be measured, each nucleus has to be turned into the region of that
cell where the brown is supposed to be - and which region that is depends on the
antibody. That fork is `app.panel`'s, not this module's; this module builds
whichever shape it is handed.

**Voronoi-constrained expansion, not plain dilation, and the difference is the
point.** Dilating a nucleus by 4 microns in crowded epithelium walks straight
into the neighbouring cell, so a strongly stained cell bleeds its signal into a
negative neighbour and the score drifts upward with cell packing rather than
with the marker. Constraining the growth to the midline between neighbours is
what QuPath's cell expansion does, and it costs one extra distance transform:
grow every nucleus at once from the full label map, and let each pixel go to
whichever nucleus is nearest. Two cells can then never claim the same pixel.

It matters more for U and W than for A/F/R, exactly as the guide says, because
the wider the expansion the more neighbours you collide with - a 6 micron band
collides with everything a 4 micron ring does and more.

**Every distance is in microns.** At this project's 0.2222 um/px a 4 um ring is
18 px and on another scanner it is not, so the conversion happens once, here,
from the mpp of the field the labels were produced at.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class Compartments:
    """The three regions of every cell in one field, as label maps.

    Each array holds the nucleus id that owns the pixel, or 0. They are disjoint
    by construction, which is what makes "the brown in the membrane" a question
    with one answer per pixel.
    """

    #: The segmentation as it came from step 11.
    nucleus: np.ndarray
    #: Everything the cell owns, nucleus included, out to the expansion distance.
    cell: np.ndarray
    #: What the marker is measured in: `cell` minus `nucleus`, and for a membrane
    #: marker thinned to a ring at the cell's outer edge.
    measured: np.ndarray
    #: Microns the expansion actually reached, for the report.
    expansion_um: float


def _expand(labels: np.ndarray, distance_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Grow every label outward at once, each pixel going to the nearest nucleus.

    `distance_transform_edt` with `return_indices` gives, for every background
    pixel, both how far the nearest labelled pixel is and which one it is - so
    one pass yields the distance cutoff and the ownership at the same time. That
    is the Voronoi constraint: ownership is decided by proximity, so no pixel can
    be claimed twice however far the expansion is asked to go.
    """
    background = labels == 0
    distance, (rows, cols) = ndimage.distance_transform_edt(
        background, return_indices=True
    )

    grown = labels.copy()
    within = background & (distance <= distance_px)
    grown[within] = labels[rows[within], cols[within]]
    return grown, distance


def build(
    labels: np.ndarray,
    *,
    mpp: float,
    expansion_um: float,
    ring: bool,
    ring_um: float | None = None,
) -> Compartments:
    """Nucleus, cell and measured compartment for every object in `labels`.

    `ring` keeps only the outer `ring_um` of the expansion; False keeps the whole
    cell body outside the nucleus.

    **The two paths differ in shape, not only in width.** A membrane marker is
    measured in the outer `ring_um` of its expansion - by default 1.5 um of a
    4 um cell body, so the shell sits from 2.5 to 4.0 um out with cytoplasm
    between it and the nucleus. A cytoplasmic marker is measured across the whole
    body.

    This used to be width alone: `ring_um` equalled `expansion_um`, so the "ring"
    was the entire annulus and the flag changed nothing. That is the guide's
    literal recipe - "dilate the nucleus by ~3-5 um, subtract the nucleus,
    subtract every neighbouring nucleus" - and it is not what ASCO/CAP mean by
    "complete, intense membrane staining", which is about the rim. It also made
    the screen impossible to draw honestly: there was no cytoplasm to colour,
    because the membrane compartment already was the cell body.

    The thickness is `app.panel`'s, like the width, and it is provisional.

    The regression the guide names is still about the **width**: run a cytoplasmic
    marker at the membrane's 4 um and the compartment samples the outer edge of
    the cell, which is where it meets the next cell. What comes out correlates
    with how tightly the tissue is packed rather than with how much N-cadherin
    the cell has made - and it will look entirely plausible. That is why both the
    width and the shell are read from `app.panel` by the antibody letter and
    never from the request.
    """
    if mpp <= 0:
        raise ValueError("compartments are built in microns, so the field needs a scale")

    distance_px = expansion_um / mpp
    cell, distance = _expand(labels, distance_px)

    nucleus = labels
    body = np.where(nucleus > 0, 0, cell)

    if ring:
        thickness = (ring_um if ring_um is not None else expansion_um) / mpp
        # The band at the outer edge: far enough from the nucleus that it is not
        # nuclear membrane, and within `thickness` of where the growth stopped.
        outer = (distance > max(0.0, distance_px - thickness)) & (distance <= distance_px)
        measured = np.where(outer, body, 0)
    else:
        measured = body

    return Compartments(
        nucleus=nucleus,
        cell=cell,
        measured=measured,
        expansion_um=expansion_um,
    )


@dataclass(frozen=True)
class Anatomy:
    """One cell's three regions, drawn, plus the other fork's for comparison.

    `build` above answers "what does THIS antibody measure". This answers "what
    are the parts of this cell", which is what a screen needs - and the two must
    agree, so the region named `measured` here is exactly the mask `build`
    returns.

    **The three are disjoint.** Nucleus, then cytoplasm, then the membrane shell
    at the outer edge. They used to be nested annuli that differed only in width,
    which meant there was no cytoplasm to draw at all: the "membrane" was the
    whole cell body, so a screen asked to show nucleus-cytoplasm-membrane could
    only ever show two colours. Splitting the body at `body_um - shell_um` is
    what makes the third one exist.

    `alternate` is the region the *other* kind of antibody would have measured on
    this same cell, drawn dashed. It is the fork made visible rather than
    asserted, and it is the one thing here that may overlap the others.
    """

    #: The segmentation, unchanged.
    nucleus: np.ndarray
    #: The cell body inside the shell. For a cytoplasmic marker, the whole body.
    cytoplasm: np.ndarray
    #: The shell at the outer edge. Empty for a cytoplasmic marker, which has none.
    membrane: np.ndarray
    #: What the other fork would measure. May overlap the two above.
    alternate: np.ndarray

    #: "membrane" or "cytoplasm" - which of the two this antibody actually uses.
    measured: str
    body_um: float
    shell_um: float
    alternate_um: float


def anatomy(
    labels: np.ndarray,
    *,
    mpp: float,
    body_um: float,
    shell_um: float,
    alternate_body_um: float,
    alternate_shell_um: float,
) -> Anatomy:
    """Split one field's cells into nucleus, cytoplasm and membrane.

    One distance transform for all of it. Voronoi ownership is decided by which
    nucleus is nearest and not by how far the growth was allowed to go, so every
    band below is an exact slice of the same expansion - running it once per band
    would give the same answer and cost four times as much.
    """
    if mpp <= 0:
        raise ValueError("compartments are built in microns, so the field needs a scale")

    reach = max(body_um, alternate_body_um) / mpp
    cell, distance = _expand(labels, reach)
    body = np.where(labels > 0, 0, cell)

    def band(low_um: float, high_um: float) -> np.ndarray:
        low = low_um / mpp
        high = high_um / mpp
        return np.where((distance > low) & (distance <= high), body, 0)

    inner_um = max(0.0, body_um - shell_um)

    return Anatomy(
        nucleus=labels,
        cytoplasm=band(0.0, inner_um),
        membrane=(
            band(inner_um, body_um) if shell_um > 0 else np.zeros_like(labels)
        ),
        alternate=(
            band(alternate_body_um - alternate_shell_um, alternate_body_um)
            if alternate_shell_um > 0
            else band(0.0, alternate_body_um)
        ),
        measured="membrane" if shell_um > 0 else "cytoplasm",
        body_um=body_um,
        shell_um=shell_um,
        alternate_um=alternate_body_um,
    )


def unconstrained(labels: np.ndarray, *, mpp: float, expansion_um: float) -> np.ndarray:
    """How many nuclei reach each pixel under plain dilation, ignoring neighbours.

    Anything above one is a pixel two cells would *both* have counted - one
    cell's stain credited to another. Never used for measurement: this exists to
    measure the error the Voronoi constraint prevents, and to draw it.

    Costs one dilation per nucleus, so it is run on a sample of fields rather
    than all of them. `contested_px` below is the number it produces.
    """
    distance_px = expansion_um / mpp
    structure = ndimage.generate_binary_structure(2, 2)
    overlaps = np.zeros(labels.shape, dtype=np.int32)

    for index in np.unique(labels):
        if index == 0:
            continue
        grown = ndimage.binary_dilation(
            labels == index, structure=structure, iterations=max(1, int(round(distance_px)))
        )
        overlaps += grown.astype(np.int32)

    return overlaps


def contested_px(labels: np.ndarray, *, mpp: float, expansion_um: float) -> int:
    """Pixels that two or more cells would both have claimed under plain dilation.

    The size of the error the Voronoi constraint prevents, measured rather than
    asserted. Exact: it counts the pixels where the unconstrained expansions
    actually overlap, not a proxy for them.

    An earlier version of this tried to derive the figure from the constrained
    result alone, by subtracting the assigned pixels from the reachable ones -
    and those two sets are identical by construction, so it reported zero on
    every field including ones where two cells were plainly colliding. The
    number has to come from doing the wrong thing and looking at it.
    """
    if not labels.any():
        return 0
    overlaps = unconstrained(labels, mpp=mpp, expansion_um=expansion_um)
    return int(np.count_nonzero(overlaps > 1))


def areas_um2(compartment: np.ndarray, *, mpp: float) -> dict[int, float]:
    """Area of each cell's share of a compartment, in square microns."""
    flat = compartment.ravel()
    counts = np.bincount(flat[flat > 0])
    scale = mpp * mpp
    return {int(index): float(count) * scale for index, count in enumerate(counts) if count}


__all__ = [
    "Anatomy",
    "Compartments",
    "anatomy",
    "areas_um2",
    "build",
    "contested_px",
    "unconstrained",
]
