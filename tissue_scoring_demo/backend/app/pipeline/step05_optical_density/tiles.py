"""Which tile does step 5 run on, and why that one?

Steps 3 and 4 work on the whole slide at about 2 um/px, because their questions -
where is the tissue, what does the glass measure - are questions about
millimetres. Step 5's question is not. A density is per pixel, and the picture it
has to produce is the pipeline guide's triptych: an RGB tile, the same tile as a
density heatmap, and the density point cloud with two visible arms. All three are
about one field of view at working magnification.

So step 5 needs a tile, and the choice of tile is not neutral. Two thirds of a
stained section is counterstain and stroma, and a tile of that has *one* arm.
Show it and the central claim of the whole measurement branch - that the two arms
are the two stains - has no picture to stand on, through no fault of the maths.
Picking at random and hoping is how a demo ends up asserting something its own
screen contradicts.

This module therefore scores every candidate tile and says why it chose the one it
did, from the thumbnail step 3 and step 4 already have in memory - which is the
only reason this is cheap. Two factors, multiplied:

  stain    the mean optical density over the block. A tile has to have stain in
           it before it can have two of them.
  mixing   how much of the block's density cloud lies off its own dominant
           direction, as the square root of the ratio of the second eigenvalue to
           the first. This is the same geometry `density.build_cloud` performs at
           full resolution, run at thumbnail scale as a screening test: one stain
           is one ray and scores near zero, two stains open a wedge and score
           high.

Multiplied and not added, because either one alone is a bad tile: a strongly
stained block of pure counterstain has nothing to un-mix, and a block whose
handful of stained pixels point in all directions is measuring noise.

That second failure is not hypothetical, and it is why there is a gate as well as
a score. `mixing` is a ratio of two eigenvalues of a cloud, and a ratio is
meaningless until the cloud is bigger than the noise it sits in - so a faint
region of false colour (a scanning artefact, chroma ringing along a high-contrast
edge) scores *higher* on mixing than real haematoxylin and DAB do, because two
wrong colours are still two colours. On the demo's slide the top-scoring block
before this gate existed was exactly that: a patch of lurid purple and yellow that
step 2 had not flagged, at 0.11 mean OD against 0.21 for a real stained duct.

Hence `min_stain`, expressed as a multiple of the optical-density noise floor step
4 measured on this slide's own glass rather than as a constant. A block whose stain
does not clear the floor of what this acquisition can measure at all cannot have a
meaningful direction, let alone two.

Every candidate's score, and both factors, travel out with the choice. The viewer
can pick a different tile and watch the arms close up - which is a better argument
for the scoring rule than the rule's own description.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from app.common.imaging import optical_density


class TileError(ValueError):
    """No tile on this slide can be used, and why."""


#: Blocks scored per axis at most, so a very large thumbnail does not turn the
#: screening pass into the expensive part. At 256 um a tile and 2 um/px, 64 blocks
#: an axis covers 16 mm - the whole of most sections.
MAX_BLOCKS = 64

#: Candidates returned to the caller, best first. Enough for the viewer to try a
#: few alternatives; far short of the hundreds scored, because a list of hundreds
#: is not a choice a person can make.
MAX_CANDIDATES = 12


@dataclass(frozen=True)
class Candidate:
    """One possible tile, scored from the thumbnail."""

    col: int
    row: int
    #: Origin in level-0 pixel coordinates - what the reader is asked for.
    x: int
    y: int

    tissue_share: float
    #: Share of the block step 2 left in play. 1.0 when quality control has not run.
    considered_share: float

    #: Mean optical density over the block, from the thumbnail.
    stain: float
    #: sqrt(second eigenvalue / first) of the block's density cloud - how far the
    #: cloud spreads off a single ray, and so whether there is more than one stain.
    mixing: float
    score: float
    usable: bool

    #: Bounds as fractions of the slide, so the browser can draw the block over a
    #: thumbnail of any size without knowing this step's resolution. Same contract
    #: as step 4's patch bounds.
    fx: float
    fy: float
    fw: float
    fh: float


def rank_tiles(
    *,
    rgb: np.ndarray,
    tissue: np.ndarray,
    considered: np.ndarray | None,
    white: np.ndarray | tuple[float, float, float],
    mpp: float,
    base_mpp: float,
    slide_size: tuple[int, int],
    tile_um: float,
    min_tissue_share: float,
    min_stain: float,
    od_floor: float,
    limit: int = MAX_CANDIDATES,
) -> list[Candidate]:
    """Score every block of the thumbnail and return the best few, best first.

    `mpp` is the thumbnail's resolution and `base_mpp` the slide's own, and both
    are needed because the score is computed on the thumbnail while the answer -
    a tile origin - has to be in level-0 coordinates. Converting through microns
    rather than through a pixel ratio is what keeps the origin correct when step
    3's size cap has made the thumbnail coarser than it asked to be.

    The screening runs in optical density and not in RGB, and that is the point of
    doing it here rather than in step 3: "how much stain" and "how many stains"
    are both statements about densities, and asking them of intensities would
    measure the illumination as much as the section.
    """
    height, width = tissue.shape
    block = max(4, int(round(tile_um / mpp)))

    cols = min(MAX_BLOCKS, max(1, width // block))
    rows = min(MAX_BLOCKS, max(1, height // block))
    if cols * rows == 0:
        raise TileError("this slide is smaller than one tile at the working magnification")

    # One density pass over the whole thumbnail rather than one per block. The
    # blocks are a partition, so this is the same arithmetic done once.
    density = optical_density(rgb.astype(np.float32), white, floor=od_floor)
    mean_od = density.mean(axis=-1)

    in_play = considered if considered is not None else np.ones_like(tissue, dtype=bool)

    candidates: list[Candidate] = []
    for row in range(rows):
        for col in range(cols):
            y0, x0 = row * block, col * block
            y1, x1 = min(height, y0 + block), min(width, x0 + block)

            tissue_block = tissue[y0:y1, x0:x1]
            if tissue_block.size == 0:
                continue

            tissue_share = float(tissue_block.mean())
            considered_share = float(in_play[y0:y1, x0:x1].mean())

            # Scored over the tissue in the block only. A block that is half glass
            # would otherwise be rewarded for the glass's own low density, and
            # punished for it in the mixing term - both meaningless.
            selection = tissue_block & in_play[y0:y1, x0:x1]

            # Both shares have to clear the bar, and the second is not redundant.
            # The score is computed over clean pixels, but the *tile* that gets read
            # covers the whole block - so a block that is mostly tissue and a third
            # artefact would be scored on its good part and then transformed
            # including its bad part. A fold is dark and a pen mark is darker, and
            # dark reads as stain here: either would widen the point cloud with
            # directions that are not stains, which is the one thing this step's
            # picture must not do. Step 2 exists to keep them out, and this is where
            # step 5 honours it.
            eligible = (
                tissue_share >= min_tissue_share
                and considered_share >= min_tissue_share
                and int(selection.sum()) >= 32
            )

            stain, mixing = 0.0, 0.0
            if eligible:
                stain = float(mean_od[y0:y1, x0:x1][selection].mean())
                mixing = _mixing(density[y0:y1, x0:x1][selection])

            # The stain gate is applied after measuring, not before, so a block
            # rejected for being too faint still reports the figure it was rejected
            # on - which is the difference between a filtered list and an audited
            # one. See the module docstring for why the gate exists at all.
            usable = eligible and stain >= min_stain

            candidates.append(
                Candidate(
                    col=col,
                    row=row,
                    # Through microns, not through a pixel ratio - see the docstring.
                    x=int(round(col * block * mpp / base_mpp)),
                    y=int(round(row * block * mpp / base_mpp)),
                    tissue_share=tissue_share,
                    considered_share=considered_share,
                    stain=stain,
                    mixing=mixing,
                    score=stain * mixing if usable else 0.0,
                    usable=usable,
                    fx=x0 / width,
                    fy=y0 / height,
                    fw=(x1 - x0) / width,
                    fh=(y1 - y0) / height,
                )
            )

    scored = [entry for entry in candidates if entry.usable and entry.score > 0.0]
    if not scored:
        raise TileError(
            f"no block of this slide is at least {min_tissue_share:.0%} tissue, "
            f"{min_tissue_share:.0%} clear of step 2's artefacts, and stained to at least "
            f"{min_stain:.3f} mean optical density - so there is nowhere to measure a "
            "density that would mean anything. Either step 3's threshold has claimed almost "
            "nothing as tissue, step 2 has flagged most of what there was, or this section "
            "is too faintly stained for the scanner's own noise floor"
        )

    scored.sort(key=lambda entry: entry.score, reverse=True)
    return scored[:limit]


def _mixing(vectors: np.ndarray) -> float:
    """How far a block's density cloud spreads off its own dominant direction.

    `sqrt(lambda2 / lambda1)` of the cloud's second-moment matrix, uncentred for
    the same reason `density.build_cloud` is uncentred: the cloud emanates from
    the origin, and the origin - no stain - is a point the geometry has to contain.

    Uncentred also makes the number bounded and comparable across blocks. One
    stain at any range of concentrations is one ray and returns near zero; two
    stains open a wedge and the ratio rises with its width. So this is a screening
    test for "is there more than one absorber here", computed at thumbnail scale,
    and it is the same statistic the full-resolution cloud reports as its own arm
    separation - which is what makes the choice checkable rather than magic.
    """
    if vectors.shape[0] < 8:
        return 0.0

    samples = vectors.astype(np.float64)
    gram = samples.T @ samples / float(samples.shape[0])
    eigenvalues = np.linalg.eigvalsh(gram)

    first = float(eigenvalues[-1])
    second = float(eigenvalues[-2])
    if first <= 1e-12:
        return 0.0
    return float(np.sqrt(max(0.0, second) / first))


def nearest(candidates: list[Candidate], *, x: int, y: int) -> Candidate:
    """The candidate whose origin is closest to a requested one.

    A viewer clicking the slide map does not land on a block boundary, and the
    honest response is to snap to the block that was actually scored rather than
    to read an unscored tile and report a score belonging to its neighbour.
    """
    if not candidates:
        raise TileError("there are no candidate tiles to choose from")
    return min(candidates, key=lambda entry: (entry.x - x) ** 2 + (entry.y - y) ** 2)


# --- reading the tile itself --------------------------------------------------


@dataclass(frozen=True)
class Tile:
    """The pixels step 5 actually transformed, and where they came from."""

    rgb: np.ndarray
    #: Level-0 origin and the extent read there, so the tile can be located again.
    x: int
    y: int
    span: int
    #: Pyramid level read and its downsample factor.
    level: int
    downsample: float
    #: Resolution achieved, in microns per pixel. Equal to the target unless the
    #: pyramid could not supply it.
    mpp: float
    target_mpp: float
    size: int
    #: True when the read had to be area-averaged down to the target resolution.
    resampled: bool


def read_tile(
    reader: object,
    *,
    x: int,
    y: int,
    target_mpp: float,
    size: int,
    base_mpp: float,
) -> Tile:
    """One tile at the working magnification, area-averaged if the pyramid overshoots.

    Two decisions here, and both are about not corrupting a density.

    **The level is chosen by microns, never by index.** `best_level_for_mpp` is
    step 1's rule and it holds here: the same level number is a different
    resolution on two scanners, and a tile read at the wrong scale would produce
    perfectly precise densities of the wrong thing.

    **Any resampling happens in intensity space, before the logarithm, with area
    averaging.** That ordering is not a detail. A coarser sensor averages the light
    arriving over a larger area, so averaging *transmissions* is what a coarser
    scan physically is; the density of the average then follows. Averaging
    densities instead would take the mean of logarithms, which is the logarithm of
    a *geometric* mean of transmission - a different number, biased low, and one no
    instrument would ever record. Hence `Image.Resampling.BOX` on the RGB, and the
    density computed afterwards.
    """
    span = max(1, int(round(size * target_mpp / base_mpp)))

    width, height = reader.dimensions  # type: ignore[attr-defined]
    x = max(0, min(int(x), max(0, width - span)))
    y = max(0, min(int(y), max(0, height - span)))

    level = reader.best_level_for_mpp(target_mpp)  # type: ignore[attr-defined]
    downsample = float(reader.level_downsamples[level])  # type: ignore[attr-defined]
    read = max(1, int(round(span / downsample)))

    image = reader.read_region_pil((x, y), level, (read, read))  # type: ignore[attr-defined]

    resampled = read != size
    if resampled:
        image = image.resize((size, size), Image.Resampling.BOX)

    return Tile(
        rgb=np.asarray(image.convert("RGB")),
        x=x,
        y=y,
        span=span,
        level=int(level),
        downsample=downsample,
        # After the resize the tile covers `span` level-0 pixels in `size` of its
        # own, so its resolution is that ratio and not the level's - which is the
        # number every physical extent downstream has to be divided by.
        mpp=base_mpp * span / max(1, image.size[0]),
        target_mpp=target_mpp,
        size=int(image.size[0]),
        resampled=resampled,
    )
