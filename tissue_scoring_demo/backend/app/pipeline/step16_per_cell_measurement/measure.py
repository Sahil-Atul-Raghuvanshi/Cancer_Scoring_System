"""The arithmetic of step 14: how much brown is in this cell, and is it a ring.

This is the first moment anything is actually measured. Everything before it was
deciding *what* to measure - which cells count, which part of each one, at what
scale. So the arithmetic here is deliberately plain: mask the DAB channel by the
compartment and take a mean. The work went into earning the right to do that.

**Mean, not max, and it does not change between markers.** Max is what QuPath
uses by default; it is more sensitive and noisier, and on a 4 um ring of maybe
two hundred pixels one hot pixel moves it. Mean is stable and it is what every
number this pipeline emits is built from. `INTENSITY_STATISTIC` says so in one
place so a report can print it, because a statistic that silently differed
between two markers would make their intensities incomparable while looking
identical on screen.

**The fork is real, not a parameter.** Membrane markers get ring completeness;
cytoplasmic markers get the stained fraction of their band. `completeness()` is
not called for a cytoplasmic marker at all - not called and defaulted, not
called and discarded. Cytoplasmic staining has no circumference, so a
completeness number for it is not a harder measurement, it is a meaningless one,
and the specific way this goes wrong is a shared code path computing it anyway
and quietly feeding a near-zero value into the positivity rule. A cell would
then be negative because of a geometry it never had. `measure_field` takes the
compartment and dispatches; there is no argument that makes it do both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Mean across the compartment's pixels. Fixed for every marker - see the module
#: docstring - and reported beside every number so the choice travels with it.
INTENSITY_STATISTIC = "mean"

#: 36 angular bins of 10 degrees, walking around the ring. The guide's number,
#: and the resolution at which "34 of 36 stained" and "6 of 36 stained" are
#: visibly different cells rather than two points on a smooth curve.
RING_BINS = 36

#: A bin holding fewer pixels than this has no mean worth trusting, so it is
#: counted as *unoccupied* rather than as unstained. At a 4 um ring and 0.5 um/px
#: a full bin holds roughly a dozen pixels; three is thin but real.
MIN_BIN_PIXELS = 3


@dataclass(frozen=True)
class CellMeasurement:
    """One cell, one row. Everything steps 15 and 16 need, and nothing else.

    Deliberately flat and pixel-free: once this exists, no later step reopens a
    slide. Step 15 bins `intensity_od`, step 16 counts and averages. Keeping the
    rows is also what makes the scatter on screen clickable - `x`/`y` are where
    to crop the image from.
    """

    #: The nucleus id step 11 gave this cell, unique within its field.
    cell_id: int
    region_rank: int
    field_index: int

    #: Level-0 slide coordinates of the nucleus centroid, for the crop.
    x: float
    y: float

    #: Number 1, always: mean DAB optical density across the compartment's pixels.
    intensity_od: float
    #: The same compartment's maximum. Reported, never used - it is here so the
    #: choice of mean can be checked rather than taken on trust.
    max_od: float

    #: Number 2, whichever this marker gets: ring completeness for a membrane
    #: marker, stained fraction of the band for a cytoplasmic one. One field
    #: rather than two, because a cell has exactly one second number and two
    #: fields would leave one of them null for every cell in every slide.
    second: float
    #: Which of the two `second` is. Carried per cell so a row is self-describing.
    second_measure: str

    #: Pixels in the compartment, and its area. A cell whose ring is a handful of
    #: pixels has a mean that means very little, and step 16 says so.
    pixels: int
    area_um2: float

    #: Membrane only: how many of the 36 bins held enough pixels to be judged.
    #: A cell crowded on three sides by neighbours cannot show a complete ring
    #: however strongly it is stained, and this is what makes that visible
    #: instead of it looking like weak staining. None for a cytoplasmic marker.
    occupied_bins: int | None = None

    #: Whether step 14 called this cell tumour (P-04). None when the typing was not
    #: available. Carried per cell so the score can report a tumour-only sensitivity
    #: beside the all-cells figure from the same rows.
    tumour: bool | None = None


def _centroids(
    labels: np.ndarray, ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Row and column centroid of each id in `labels`, in the given order.

    Taken from the **nucleus** map rather than the measured compartment, because
    the angle a ring pixel sits at is only meaningful about the cell's own
    centre. A crescent-shaped remnant of a ring has a centroid out on the
    crescent, and angles measured from there would spread a one-sided arc across
    every bin and report a complete ring.
    """
    flat = labels.ravel()
    keep = flat > 0
    if not keep.any():
        zeros = np.zeros(ids.shape, dtype=np.float64)
        return zeros, zeros.copy()

    rows, cols = np.nonzero(labels)
    size = int(flat.max()) + 1
    counts = np.bincount(flat[keep], minlength=size).astype(np.float64)
    row_sum = np.bincount(flat[keep], weights=rows.astype(np.float64), minlength=size)
    col_sum = np.bincount(flat[keep], weights=cols.astype(np.float64), minlength=size)

    safe = np.maximum(counts, 1.0)
    return (row_sum / safe)[ids], (col_sum / safe)[ids]


def _per_cell_sums(
    cell_index: np.ndarray, values: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sum and pixel count of `values` per cell, for cells numbered 0..count-1."""
    totals = np.bincount(cell_index, weights=values.astype(np.float64), minlength=count)
    pixels = np.bincount(cell_index, minlength=count).astype(np.float64)
    return totals[:count], pixels[:count]


def completeness(
    cell_index: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    od: np.ndarray,
    centre_rows: np.ndarray,
    centre_cols: np.ndarray,
    *,
    count: int,
    positivity_od: float,
    bins: int = RING_BINS,
    min_bin_pixels: int = MIN_BIN_PIXELS,
) -> tuple[np.ndarray, np.ndarray]:
    """Walk around each ring in `bins` angular bins and count the stained ones.

    A bin is stained when the mean DAB optical density of its pixels clears the
    positivity cut. Returns `(stained / bins, occupied)` per cell.

    **The denominator is `bins`, not the occupied bins**, and that is the guide's
    definition: a cell at 34 of 36 has genuine complete membrane staining, one at
    6 of 36 has non-specific speckle that only looks positive once you average it
    away. Dividing by the occupied bins instead would rescue the second cell by
    declaring most of its circumference unmeasurable, which is exactly the
    judgement the number is supposed to make.

    `occupied` comes back alongside precisely because that denominator is
    unforgiving in one legitimate case: a cell pressed on three sides by
    neighbours has had those bins taken from it by the Voronoi constraint, and it
    cannot show a complete ring however strongly it is stained. That is a
    geometry problem, not a staining one, and step 16 reports how many cells are
    in it rather than letting them read as weak.

    ASCO/CAP wrote this distinction into the HER2 guideline - "complete, intense
    membrane staining" is the definition of 3+ - and OncoStem's own deck
    distinguishes complete from partial membrane staining. It is not a refinement
    we are adding; it is the measurement they already make.
    """
    if cell_index.size == 0:
        return np.zeros(count), np.zeros(count, dtype=np.int64)

    angle = np.arctan2(
        rows.astype(np.float64) - centre_rows[cell_index],
        cols.astype(np.float64) - centre_cols[cell_index],
    )
    # atan2 gives (-pi, pi]; shift to [0, 2pi) then scale to bin indices. The
    # modulo catches the single pixel that can land exactly on 2pi.
    bin_index = np.floor((angle + np.pi) / (2.0 * np.pi) * bins).astype(np.int64) % bins

    key = cell_index * bins + bin_index
    size = count * bins
    totals = np.bincount(key, weights=od.astype(np.float64), minlength=size)[:size]
    pixels = np.bincount(key, minlength=size)[:size]

    grid_pixels = pixels.reshape(count, bins)
    occupied = grid_pixels >= min_bin_pixels
    means = np.divide(
        totals.reshape(count, bins),
        np.maximum(grid_pixels, 1),
        out=np.zeros((count, bins)),
        where=grid_pixels > 0,
    )

    stained = occupied & (means > positivity_od)
    return stained.sum(axis=1) / float(bins), occupied.sum(axis=1)


def stained_fraction(
    cell_index: np.ndarray,
    od: np.ndarray,
    *,
    count: int,
    positivity_od: float,
) -> np.ndarray:
    """Share of a cytoplasm band's pixels that clear the positivity cut.

    The honest analogue of ring completeness for a marker whose stain has no
    shape. It answers "is the brown spread through this cell's body, or is it one
    speckle?" without pretending the brown has a circumference it does not have -
    and one speckle in a 6 um band is exactly the artefact a mean optical density
    on its own cannot tell from faint even staining.
    """
    if cell_index.size == 0:
        return np.zeros(count)
    above = (od > positivity_od).astype(np.float64)
    totals, pixels = _per_cell_sums(cell_index, above, count)
    return np.divide(totals, pixels, out=np.zeros(count), where=pixels > 0)


def measure_field(
    nucleus: np.ndarray,
    measured: np.ndarray,
    dab: np.ndarray,
    *,
    membrane: bool,
    positivity_od: float,
    mpp: float,
    region_rank: int,
    field_index: int,
    x0: float = 0.0,
    y0: float = 0.0,
    level0_scale: float = 1.0,
) -> list[CellMeasurement]:
    """Every cell in one field, measured once.

    `nucleus` and `measured` are step 13's label maps - the nucleus, and the ring
    or band the marker is supposed to be in - and `dab` is the DAB optical
    density of the same pixels at the same scale. All three must be the same
    shape; they are, because they are produced from one field read at one mpp,
    and a mismatch here would silently measure one cell's stain in another cell's
    ring.

    `x0`, `y0` and `level0_scale` put the centroid back into slide coordinates so
    a dot on the scatter can be clicked through to its crop.
    """
    if nucleus.shape != measured.shape or nucleus.shape != dab.shape:
        raise ValueError(
            "the nucleus map, the compartment map and the DAB channel must describe the "
            f"same pixels; got {nucleus.shape}, {measured.shape} and {dab.shape}"
        )
    if mpp <= 0:
        raise ValueError("a cell's area is in square microns, so the field needs a scale")

    ids = np.unique(measured)
    ids = ids[ids > 0]
    if ids.size == 0:
        return []

    count = int(ids.size)
    # Dense 0..count-1 indices, so every bincount below is over a compact range
    # rather than over the largest label id in the field.
    lookup = np.zeros(int(measured.max()) + 1, dtype=np.int64)
    lookup[ids] = np.arange(count)

    rows, cols = np.nonzero(measured)
    cell_index = lookup[measured[rows, cols]]
    od = dab[rows, cols].astype(np.float64)

    totals, pixels = _per_cell_sums(cell_index, od, count)
    mean_od = np.divide(totals, pixels, out=np.zeros(count), where=pixels > 0)

    maxima = np.zeros(count)
    np.maximum.at(maxima, cell_index, od)

    centre_rows, centre_cols = _centroids(nucleus, ids)

    if membrane:
        second, occupied = completeness(
            cell_index,
            rows,
            cols,
            od,
            centre_rows,
            centre_cols,
            count=count,
            positivity_od=positivity_od,
        )
        second_name = "ring_completeness"
        occupied_out: list[int | None] = [int(value) for value in occupied]
    else:
        # Not computed, not defaulted, not discarded. See the module docstring.
        second = stained_fraction(
            cell_index, od, count=count, positivity_od=positivity_od
        )
        second_name = "stained_fraction"
        occupied_out = [None] * count

    area_scale = mpp * mpp
    return [
        CellMeasurement(
            cell_id=int(ids[index]),
            region_rank=region_rank,
            field_index=field_index,
            x=float(x0 + centre_cols[index] * level0_scale),
            y=float(y0 + centre_rows[index] * level0_scale),
            intensity_od=float(mean_od[index]),
            max_od=float(maxima[index]),
            second=float(second[index]),
            second_measure=second_name,
            pixels=int(pixels[index]),
            area_um2=float(pixels[index] * area_scale),
            occupied_bins=occupied_out[index],
        )
        for index in range(count)
    ]


__all__ = [
    "INTENSITY_STATISTIC",
    "MIN_BIN_PIXELS",
    "RING_BINS",
    "CellMeasurement",
    "completeness",
    "measure_field",
    "stained_fraction",
]
