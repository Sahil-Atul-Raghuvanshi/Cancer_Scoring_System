"""Step 7's algorithm: the slide cut into patches, and the ones worth keeping.

Plumbing, and the guide says so - there is no science in laying a grid over a
rectangle. What there *is* here is a funnel, and the funnel is the whole point of
the step running where it does. A whole-slide scan holds hundreds of thousands of
tiles at working magnification; a section holds a small fraction of that, and the
part of the section quality control did not object to holds slightly less again.
Every one of those tiles is a forward pass through the region model at step 8,
which is the most expensive thing in the pipeline, so the count that leaves this
step is the compute bill for everything after it.

  every tile     the grid over the whole canvas, including the two thirds of an
                 Aperio frame that is empty glass.
  on tissue      tiles whose tissue share clears `min_tissue_share`, from step 3's
                 mask. This is the big cut and it is why the step runs after step 3.
  clean          of those, tiles whose share of step-2-approved pixels clears
                 `min_clean_share`. Smaller, and the reason the step runs after
                 step 2: an out-of-focus or folded tile produces a confident,
                 wrong class rather than no class.

`min_tissue_share` is now the pipeline's only tissue gate on a square - step 8's
window gate sees the same square and so removes nothing - which is why it carries
step 8's 0.5 rather than the 0.1 this step used when the two grids differed.

The three counts travel out together, because a filtered list and an audited one
are different things and only the second lets a reader check the rule.

**Three parameters, and only one of them is this step's to choose.**

  size       the region model's, read from its checkpoint's manifest by
             `tiling_service.window` - 224 px for the published model. Not a
             preference and not this step's: the model's field of view is a
             property of the checkpoint, and feeding it a differently-scaled field
             is the same class of failure as feeding it the wrong channel.
  resolution the same manifest's, for the same reason. Note this is the one place
             the pipeline's working magnification does not rule: steps 4 to 6 work
             at `target_mpp` and this grid works at the checkpoint's, and when the
             two differ it is the checkpoint that is right about what the model
             sees.
  overlap    25-50% for segmentation, and the one real choice here. A model has no
             context beyond a tile's edge, so its predictions there are its worst;
             overlapping lets those edges be averaged away instead of being
             stitched into visible seams across the class map.

**The grid this step lays is the grid step 8 runs.** It was not always: this step
emitted 512 px tiles at the pipeline's working resolution and step 8 laid its own
224 px windows over them, so the funnel below priced 4,952 squares and the model
then made 98,262 passes over the same slide. Two grids meant two tile counts, two
tissue gates and two answers to "how much work is this", and only step 8's were
real. One grid means the count that leaves this step is the compute bill, in the
literal sense of being the number of forward passes.

**The tiles are addressed, not stored.** What leaves this step is coordinates -
level-0 origin, level, size - and never pixels. Two reasons, and the second is
the one that matters. A tile index for a whole slide is kilobytes where the
pixels are gigabytes; and more importantly, the *pixels* a tile resolves to are
step 6's haematoxylin channel rather than RGB, computed on demand by the one
deconvolution function this codebase has. Materialising RGB tiles here would put
a second, silent answer to "what does the model see" on disk, which is exactly
the drift the guide warns about.

Slideflow, arXiv 2304.04142, for the tile-index shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class TilingError(ValueError):
    """The slide cannot be tiled - and why, in words."""


#: Tiles returned in the sampled listing, at most. The index itself may hold tens
#: of thousands; a list of tens of thousands is not something a person reads, and
#: shipping it would make the response larger than the picture of it.
MAX_LISTED = 240

#: Grid cells drawn on the map panel per axis, at most. Beyond this the drawn grid
#: is finer than the lines used to draw it, so the picture stops being a grid and
#: becomes a texture - see `overlay.map_png`, which aggregates instead.
MAX_DRAWN = 220


@dataclass(frozen=True)
class Tile:
    """One patch of the grid, addressed rather than stored.

    `x` and `y` are level-0 pixel coordinates - the frame of reference that does
    not move when a resolution changes - and `span` is the level-0 extent the tile
    covers, which is what makes the address resolvable at any pyramid level.
    """

    col: int
    row: int
    x: int
    y: int
    span: int

    #: Share of the tile that step 3 called tissue, and share that step 2 left in
    #: play. Both measured on step 3's grid, which is coarser than the tile - see
    #: `build_index` for why that is honest rather than approximate.
    tissue_share: float
    clean_share: float

    kept: bool
    #: Which gate turned it away: "tissue", "clean", or None when it was kept.
    rejected_by: str | None

    #: Bounds as fractions of the slide, so a browser can draw the tile over a
    #: thumbnail of any size without knowing this step's resolution. Same contract
    #: as steps 4 and 5's patch bounds.
    fx: float
    fy: float
    fw: float
    fh: float


@dataclass(frozen=True)
class Funnel:
    """How many tiles survived each gate - the step's own headline.

    Counts and not shares, deliberately. "93% dropped" is a statistic; "412,164
    tiles became 24,908" is the compute bill for step 8, and the second is what
    this step exists to reduce.
    """

    every: int
    on_tissue: int
    clean: int

    #: How many times fewer tiles reach the model than the grid holds.
    reduction: float


@dataclass(frozen=True)
class TileIndex:
    """Step 7's whole output: the grid, the survivors, and the arithmetic."""

    tiles: tuple[Tile, ...]
    funnel: Funnel

    #: The grid's shape and geometry, in the units each one is a statement about.
    cols: int
    rows: int
    size: int
    span: int
    stride: int
    overlap: float
    mpp: float
    tile_um: float

    #: Unique slide area the kept tiles cover, in mm^2, counted once however many
    #: tiles overlap it - see `_covered_mm2`. Reported beside the tile count
    #: because with overlap the two are not proportional, and a reader comparing
    #: two overlap settings needs the one that does not move.
    covered_mm2: float
    tissue_mm2: float

    #: Resolution of step 3's mask, which is what the shares were measured on.
    mask_mpp: float
    qc_gated: bool

    #: The kept mask on the grid, for the renderer. Not serialised.
    grid_kept: np.ndarray
    grid_tissue: np.ndarray


def build_index(
    *,
    tissue: np.ndarray,
    considered: np.ndarray | None,
    mask_mpp: float,
    base_mpp: float,
    slide_size: tuple[int, int],
    target_mpp: float,
    size: int,
    overlap: float,
    min_tissue_share: float,
    min_clean_share: float,
    max_tiles: int,
) -> TileIndex:
    """Lay the grid over the slide and score every cell against the two gates.

    **The shares are measured on step 3's grid, not on the tile's own pixels, and
    that is a decision rather than a shortcut.** Step 3's mask is at ~2 um/px and a
    tile is at ~0.5, so one mask pixel covers about sixteen tile pixels: the shares
    below therefore move in steps of one mask pixel rather than continuously, and the
    report states the size of that step for the grid actually built - see
    `share_quantisation`, which depends on the square and so moves with the
    checkpoint. The alternative is
    reading every tile off the pyramid to measure its own tissue fraction, which is
    reading the whole slide at working magnification - the precise thing this step
    exists to avoid, and for a number whose only use is a comparison against a
    threshold of 0.1. So the coarse measurement is the correct one, and the
    quantisation is reported rather than hidden.

    `max_tiles` is a refusal and not a cap: a grid larger than this is silently
    either a corrupt slide dimension or a resolution mistake, and truncating it
    would hand step 8 an arbitrary two thirds of a section.
    """
    if size < 16:
        raise TilingError("a tile smaller than 16 px is not a patch, it is a pixel")
    if not 0.0 <= overlap < 1.0:
        raise TilingError(
            f"overlap must be at least 0 and less than 1, not {overlap}; at 1 the "
            "stride would be zero and the grid would never advance"
        )

    width, height = slide_size
    if width <= 0 or height <= 0:
        raise TilingError("this slide reports no dimensions, so there is nothing to tile")

    # Everything in level-0 pixels, because that is the only frame of reference
    # that does not move when a pyramid level is chosen. `span` is what one tile
    # covers there; `stride` is how far the grid steps between tiles.
    span = max(1, int(round(size * target_mpp / base_mpp)))
    stride = max(1, int(round(span * (1.0 - overlap))))

    # Ceiling division, so the last partial tile at each edge is still a tile. A
    # floor would drop a strip up to one tile wide down two sides of every slide,
    # and on a small biopsy that strip is a real part of the specimen.
    cols = max(1, int(np.ceil(max(1, width - span) / stride)) + 1)
    rows = max(1, int(np.ceil(max(1, height - span) / stride)) + 1)

    every = cols * rows
    if every > max_tiles:
        raise TilingError(
            f"this grid would hold {every:,} tiles, past the {max_tiles:,} this step will "
            f"build. At {target_mpp:g} um/px with {overlap:.0%} overlap that is "
            f"{cols:,} x {rows:,} over a {width:,} x {height:,} slide. Either the slide "
            "dimensions are wrong or the working resolution is finer than this pipeline "
            "was configured for - refusing rather than truncating, because an arbitrary "
            "two thirds of a section is worse than no tile index at all"
        )

    in_play = considered if considered is not None else np.ones_like(tissue, dtype=bool)
    mask_height, mask_width = tissue.shape

    # One conversion, level-0 pixels to mask pixels. Through microns rather than
    # through a pixel ratio, because step 3 caps its own longest edge and on a
    # large scan its mask is coarser than it asked to be - a ratio computed from
    # the requested resolution would then address the wrong rows.
    to_mask = base_mpp / mask_mpp

    # **Both shares for the whole grid at once, through summed-area tables.** The
    # loop this replaces cost three numpy calls per grid cell: nothing at 20,000
    # cells, and sixteen seconds at 250,000 - which is the size this grid reaches as
    # soon as the tile is the model's window rather than the screen's field of view.
    # A cell's box is the product of a column range and a row range, which is what
    # makes a table applicable; the arithmetic is the same rational number as the
    # slice-and-mean it replaces rather than an approximation of it.
    tissue_sat = _integral(tissue)
    clean_sat = _integral(tissue & in_play)

    # Origins per axis, computed once. Same clamp as the address in `Tile`: the last
    # tile on each axis is pulled back onto the slide rather than hanging off it.
    cols_x = np.minimum(np.arange(cols, dtype=np.int64) * stride, max(0, width - span))
    rows_y = np.minimum(np.arange(rows, dtype=np.int64) * stride, max(0, height - span))

    # Mask-pixel bounds per axis. `maximum(m0 + 1, ...)` keeps a box at least one
    # mask pixel wide before the clamp, so a tile narrower than one mask pixel still
    # reads the pixel it sits in; the clamp can then still empty it at the far edge,
    # which is the "off the mask entirely" case and is recorded as no tissue.
    mx0 = np.clip(np.floor(cols_x * to_mask).astype(np.int64), 0, mask_width)
    my0 = np.clip(np.floor(rows_y * to_mask).astype(np.int64), 0, mask_height)
    mx1 = np.clip(
        np.maximum(mx0 + 1, np.ceil((cols_x + span) * to_mask).astype(np.int64)),
        0,
        mask_width,
    )
    my1 = np.clip(
        np.maximum(my0 + 1, np.ceil((rows_y + span) * to_mask).astype(np.int64)),
        0,
        mask_height,
    )

    empty = (mx1 <= mx0)[None, :] | (my1 <= my0)[:, None]
    area = ((my1 - my0)[:, None] * (mx1 - mx0)[None, :]).astype(np.float64)
    tissue_sum = _box_sum(tissue_sat, my0, my1, mx0, mx1).astype(np.float64)
    clean_sum = _box_sum(clean_sat, my0, my1, mx0, mx1).astype(np.float64)

    shares = np.divide(tissue_sum, area, out=np.zeros_like(area), where=area > 0)
    # Over the tile's *tissue*, not over the whole tile. A tile that is half glass
    # would otherwise be credited for the glass being clean, and glass is always
    # clean - it has nothing on it to be wrong.
    cleans = np.divide(
        clean_sum, tissue_sum, out=np.zeros_like(area), where=tissue_sum > 0
    )
    shares[empty] = 0.0
    cleans[empty] = 0.0

    on_tissue = shares >= min_tissue_share
    grid_kept = on_tissue & (cleans >= min_clean_share)
    grid_tissue = shares.astype(np.float32)

    # The union of the kept tiles, accumulated on step 3's own grid - see
    # `_covered_mm2` for why that grid and not a stride-sized one.
    covered = _union(grid_kept & ~empty, my0=my0, my1=my1, mx0=mx0, mx1=mx1, shape=tissue.shape)

    # The addressed listing. Still every cell and not only the survivors: a filtered
    # list and an audited one are different things, and only the second lets a reader
    # check the rule - see the module docstring. Built row by row off the arrays
    # above, with one `tolist` per row rather than a numpy scalar per field, because
    # at this grid size the boxing is the cost.
    tiles: list[Tile] = []
    x_of = cols_x.tolist()
    for row in range(rows):
        y = int(rows_y[row])
        fy = y / height
        fh = min(span, height - y) / height
        tissue_row = shares[row].tolist()
        clean_row = cleans[row].tolist()
        kept_row = grid_kept[row].tolist()
        on_row = on_tissue[row].tolist()

        for col in range(cols):
            x = x_of[col]
            kept = kept_row[col]
            tiles.append(
                Tile(
                    col=col,
                    row=row,
                    x=x,
                    y=y,
                    span=span,
                    tissue_share=tissue_row[col],
                    clean_share=clean_row[col],
                    kept=kept,
                    # Named in gate order, so a tile off the section reads as "not
                    # on tissue" rather than as an artefact - it is not one.
                    rejected_by=None if kept else ("tissue" if not on_row[col] else "clean"),
                    fx=x / width,
                    fy=fy,
                    fw=min(span, width - x) / width,
                    fh=fh,
                )
            )

    on_tissue_count = int(on_tissue.sum())
    clean_count = int(grid_kept.sum())

    if clean_count == 0:
        raise TilingError(
            f"no tile of this slide is at least {min_tissue_share:.0%} tissue and "
            f"{min_clean_share:.0%} clear of step 2's artefacts, so there is nothing for the "
            f"region model to run on. {on_tissue_count:,} of {every:,} tiles cleared the "
            "tissue gate. Either step 3's threshold has claimed almost nothing as tissue, or "
            "step 2 flagged most of what there was"
        )

    return TileIndex(
        tiles=tuple(tiles),
        funnel=Funnel(
            every=every,
            on_tissue=on_tissue_count,
            clean=clean_count,
            reduction=every / max(1, clean_count),
        ),
        cols=cols,
        rows=rows,
        size=size,
        span=span,
        stride=stride,
        overlap=overlap,
        mpp=target_mpp,
        tile_um=size * target_mpp,
        covered_mm2=_covered_mm2(covered, mask_mpp=mask_mpp),
        tissue_mm2=float(tissue.sum()) * (mask_mpp / 1000.0) ** 2,
        mask_mpp=mask_mpp,
        qc_gated=considered is not None,
        grid_kept=grid_kept,
        grid_tissue=grid_tissue,
    )


def _integral(mask: np.ndarray) -> np.ndarray:
    """Summed-area table of a boolean mask, padded with a leading zero row and column.

    `table[r, c]` is the number of true pixels above and left of `(r, c)`, so the sum
    over any box is four lookups and no slice.

    `int32` and a contiguous intermediate, both for speed rather than for space: step
    3 caps its mask at 4,096 px on the long edge, so the largest total is 16.8 million
    and cannot approach the type's limit, and cumulative sums written straight into a
    padded view run slower than ones written into their own buffer and copied.
    """
    running = np.cumsum(mask, axis=0, dtype=np.int32)
    np.cumsum(running, axis=1, out=running)
    table = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int32)
    table[1:, 1:] = running
    return table


def _box_sum(
    table: np.ndarray,
    my0: np.ndarray,
    my1: np.ndarray,
    mx0: np.ndarray,
    mx1: np.ndarray,
) -> np.ndarray:
    """Inclusion-exclusion over every (row range, column range) pair at once.

    The four terms broadcast a row vector against a column vector, so the result is
    the whole grid rather than one cell - which is the point of building the table.
    """
    return (
        table[my1[:, None], mx1[None, :]]
        - table[my0[:, None], mx1[None, :]]
        - table[my1[:, None], mx0[None, :]]
        + table[my0[:, None], mx0[None, :]]
    )


def _union(
    kept: np.ndarray,
    *,
    my0: np.ndarray,
    my1: np.ndarray,
    mx0: np.ndarray,
    mx1: np.ndarray,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Which mask pixels at least one kept tile covers.

    Through a 2D difference array rather than a slice assignment per tile: overlapping
    tiles mean a hundred thousand slice writes over the same mask, where four corner
    increments each and one double cumulative sum is a fixed cost in the mask's size.
    Only whether the count is positive is read, so the counts themselves never matter.
    `int16` is deliberate and bounded: a running value here is at most the number of
    kept tiles in one column of the grid, thousands at the very most against a limit
    of 32,767, and halving the width of the two cumulative sums over a mask of
    millions of pixels is most of this function's cost.
    """
    diff = np.zeros((shape[0] + 1, shape[1] + 1), dtype=np.int16)
    rows, cols = np.nonzero(kept)
    if rows.size:
        top, bottom = my0[rows], my1[rows]
        left, right = mx0[cols], mx1[cols]
        np.add.at(diff, (top, left), 1)
        np.add.at(diff, (top, right), -1)
        np.add.at(diff, (bottom, left), -1)
        np.add.at(diff, (bottom, right), 1)

    np.cumsum(diff, axis=0, out=diff)
    np.cumsum(diff, axis=1, out=diff)
    return diff[: shape[0], : shape[1]] > 0


def _covered_mm2(covered: np.ndarray, *, mask_mpp: float) -> float:
    """Unique slide area the kept tiles cover, counting overlap once.

    Not `count * span^2`, and the difference is the whole reason this exists. At
    50% overlap that product is four times the area actually covered, so a reader
    comparing two overlap settings would see the "area" quadruple while the section
    stayed the same size - and would reasonably conclude that overlapping tiles
    look at more of the slide, which is exactly backwards. Overlap buys reliable
    tile edges at the cost of more forward passes over the *same* tissue.

    **Counted on step 3's grid, and the choice of grid is load-bearing.** The
    obvious implementation paints the tiles onto a grid at *stride* resolution,
    and it is wrong in a way that defeats the purpose: the cell size then changes
    with the overlap, so a coarser stride over-estimates the union by more, and the
    figure moves when the overlap moves. Measured on the demo's slide that put 25%
    overlap at 218 mm2 and 50% at 208 - a *decrease* from tiling more finely, which
    is an artefact of the measurement and not a fact about the slide.

    Step 3's grid is fixed for a given mask, so the same tissue covered by any
    overlap returns the same area. It is also the grid `tissue_mm2` is measured on,
    which makes the coverage ratio a comparison of two counts on one grid rather
    than of two quantisations.
    """
    return float(covered.sum()) * (mask_mpp / 1000.0) ** 2


def sample(index: TileIndex, *, limit: int = MAX_LISTED) -> tuple[Tile, ...]:
    """A representative slice of the index, for a response a browser can hold.

    Evenly spaced through the kept tiles rather than the first `limit` of them.
    The grid is generated in row order, so the head of the list is the top edge of
    the section and nothing else - a reader shown that would be looking at one
    strip of the slide and told it was a sample.
    """
    kept = [tile for tile in index.tiles if tile.kept]
    if len(kept) <= limit:
        return tuple(kept)

    picks = np.linspace(0, len(kept) - 1, limit).round().astype(int)
    return tuple(kept[position] for position in dict.fromkeys(picks.tolist()))
