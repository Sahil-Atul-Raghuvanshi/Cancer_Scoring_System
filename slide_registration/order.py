"""Phase 2 - recover the cut order of a block's sections, and their relative rotations.

    python order.py CAN_00303

**The question this answers is not on record anywhere.** The OncoStem documents confirm
that the six slides of a case are serial sections of one block, sharing a block ID on the
label, and SOP clause 1.5 pins the H&E to one *end* of the ribbon: *"A first slide and an
H&E review after IHC sectioning (7th/9th slide) confirms invasive tumour is still
adequate."* The order of A, F, R, U and W within the ribbon is nowhere - not in the SOP,
not in the deck, and not on the glass, whose labels read `Ab-A` and `Ab-U`, the marker
letter only with no section number.

So it is measured. Two sections cut next to each other in a block look more alike than two
cut five sections apart, which makes the cut order recoverable as the ordering that
maximises total similarity between consecutive slides - a shortest-Hamiltonian-path
problem on six nodes with one endpoint fixed, which is 120 permutations and needs no
cleverness at all.

**Why bother, when VALIS can sort images itself.** Because we need the *rotations* too,
and because a measured order with a stated margin is evidence, while VALIS's internal
sort is a side effect we would never see. If the margin over the runner-up is thin, this
says so rather than asserting an order - a wrong order is worse than no order, since it
would chain the registration through the wrong neighbours.

**Similarity is measured on the tissue outline, not on the stain.** An H&E and a DAB
section of the same tissue look nothing alike pixel for pixel, which is the entire reason
the feature matcher fails on these cases; any similarity that reads colour would rank the
five IHC slides as mutually close and the H&E as far from all of them, recovering the
staining protocol rather than the cutting order.
"""

from __future__ import annotations

import argparse
import itertools
import pathlib
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402

#: Side of the square grid every mask is rasterised into for comparison. 160 px over a
#: normalised tissue extent is about 1% of the tissue's width per cell - finer than the
#: section-to-section shape change being measured, and coarse enough that 145 rotations
#: times 15 pairs is seconds rather than minutes.
GRID = 160

#: Half-width of that grid in normalised units (tissue area is scaled to 1.0), chosen so
#: an ordinary section fills most of the frame without its corners falling off it.
HALF = 1.5

#: How many rotations are tried. 145 steps over the full turn is 2.5 degrees apart, which
#: is finer than the residual a rigid registration will then remove anyway.
ROTATIONS = 145

#: How much overlap a rotation may give up to be preferred for being smaller. Set
#: against the observed gap between a real flip and a symmetry artefact: CAN_00267's
#: genuinely inverted sections beat their unturned score by far more than this, while
#: the spurious half-turns that prompted the tie-break were within a couple of points.
TIE_TOLERANCE = 0.03

#: Most tissue pixels the rotation search will look at per slide. The comparison grid
#: is GRID x GRID = 25,600 cells, so this saturates it four times over and the result
#: is indistinguishable from using every pixel - it is purely a cost ceiling.
POINT_BUDGET = 100_000


def load_mask(path: pathlib.Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def points(mask: np.ndarray) -> np.ndarray:
    """Tissue pixels, centred on their centroid and scaled to unit area.

    Normalising away position and size is what makes the comparison about *shape*. Two
    sections of one block differ a little in both - the microtome wanders, and a section
    picked up on glass sits where it lands - and neither difference says anything about
    how far apart they were cut.
    """
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return np.zeros((0, 2))

    # Centroid and scale are computed on every pixel - they are cheap and exact.
    centre_x, centre_y = cols.mean(), rows.mean()
    scale = np.sqrt(len(rows))

    # The rotation search is not. It rasterises this cloud 145 times per pair, 15 pairs per
    # case, and the cloud grows with the square of the render resolution: CAN_00267's
    # 3742 px canvas holds about four million tissue pixels and took two and a half minutes
    # per pair, and the chained 4 um/px retry would have had sixteen times that. Subsampling
    # costs nothing in accuracy because the destination is a 160x160 grid - 25,600 cells,
    # which POINT_BUDGET already saturates several times over - so the extra millions of
    # points were being thrown away by the rasteriser anyway, just slowly.
    if rows.size > POINT_BUDGET:
        # A *random* sample, not every Nth pixel. `np.nonzero` returns raster order, so
        # striding it walks the tissue row by row and lays down a regular lattice that
        # aliases against the comparison grid - measured on CAN_00267's A-F pair, striding
        # moved the answer from 184 degrees to 199 and dropped the overlap from 0.886 to
        # 0.855. Seeded, so the same slides always give the same number.
        picked = np.random.default_rng(0).choice(rows.size, POINT_BUDGET, replace=False)
        rows, cols = rows[picked], cols[picked]

    return np.column_stack([(cols - centre_x) / scale, (rows - centre_y) / scale])


def raster(pts: np.ndarray) -> np.ndarray:
    grid = np.zeros((GRID, GRID), bool)
    if pts.size == 0:
        return grid
    index = ((pts + HALF) / (2 * HALF) * GRID).astype(int)
    index = index[(index >= 0).all(1) & (index < GRID).all(1)]
    grid[index[:, 1], index[:, 0]] = True
    return ndimage.binary_fill_holes(ndimage.binary_closing(grid, np.ones((3, 3))))


def best_rotation(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """The rotation of `a` onto `b` that maximises mask overlap, and that overlap.

    Brute force over the whole turn rather than a gradient method, because the thing this
    has to catch is a section mounted the other way up - a 180 degree difference, which
    is a global maximum a local search started near zero will never find, and which is
    exactly what two of CAN_00267's slides turned out to have.
    """
    target = raster(b)
    scored: list[tuple[float, float]] = []
    for angle in np.linspace(0, 2 * np.pi, ROTATIONS, endpoint=False):
        cos, sin = np.cos(angle), np.sin(angle)
        turned = raster(a @ np.array([[cos, sin], [-sin, cos]]))
        union = (turned | target).sum()
        scored.append((float(angle), (turned & target).sum() / union if union else 0.0))

    best_score = max(score for _, score in scored)

    # Among rotations that are as good as the best within `TIE_TOLERANCE`, take the one
    # closest to no rotation at all.
    #
    # A section is often near enough symmetric that turning it half a turn overlaps almost
    # as well as leaving it alone, and the argmax then picks between them on noise. That
    # is not a cosmetic problem: this angle is *applied* to the image before registration,
    # so a coin-toss 180 degrees would hand the matcher an upside-down section and destroy
    # a case that was fine. Observed on CAN_00303's A-U pair, which scored 345 degrees on
    # one mask and 181 on another with nearly the same overlap.
    #
    # This cannot hide a genuine flip. When a section really is mounted the other way up -
    # as two of CAN_00267's are - leaving it unturned scores far worse than
    # `TIE_TOLERANCE`, so the half-turn wins on merit rather than on a tie-break.
    contenders = [(angle, score) for angle, score in scored if best_score - score <= TIE_TOLERANCE]
    angle, score = min(contenders, key=lambda row: abs(np.pi - abs(row[0] - np.pi)))
    return angle, score


#: Above this many non-reference slides the orderings are ranked greedily rather than
#: exhaustively. Six slides means five to permute - 120 chains, instant. Ten would mean
#: 362,880 and twelve would mean forty million, and this has to keep working on panels
#: nobody has shown us yet, so the exhaustive search gets a ceiling rather than a
#: comment saying it should be fine.
EXHAUSTIVE_LIMIT = 8


def _rank_orders(others: list[str], iou: dict) -> list[tuple[float, float, tuple]]:
    """Candidate cut orders, best first, as (total IoU, weakest link, chain).

    The H&E is always the first element: SOP clause 1.5 puts it at one end of the ribbon
    ("a first slide and an H&E review after IHC sectioning"), so it is an endpoint rather
    than a free node, which halves the search and rules out chains that put it in the
    middle.
    """
    chains: list[tuple[float, float, tuple]] = []

    if len(others) <= EXHAUSTIVE_LIMIT:
        candidates = [("HE",) + one for one in itertools.permutations(others)]
    else:
        # Greedy nearest-neighbour from the H&E, then the same from every other possible
        # second slide, so the result is not hostage to one early choice. Not optimal,
        # but the ranking only has to be good enough to route the registration through
        # similar neighbours, and the direct-to-reference rung is the safety net.
        candidates = []
        for first in others:
            chain, remaining = ["HE", first], [c for c in others if c != first]
            while remaining:
                nearest = max(remaining, key=lambda c: iou[chain[-1]][c])
                chain.append(nearest)
                remaining.remove(nearest)
            candidates.append(tuple(chain))

    for chain in candidates:
        links = [iou[chain[i]][chain[i + 1]] for i in range(len(chain) - 1)]
        chains.append((sum(links), min(links), chain))
    chains.sort(key=lambda row: (-row[0], -row[1]))
    return chains


def measure_case(case: str, log=None) -> dict:
    """All-pairs shape similarity, the recovered cut order, and each slide's rotation."""
    manifest = common.read_json(common.case_dir(case) / "render.json")
    if manifest is None:
        raise ValueError(f"{case}: no render.json - run phase 1 first")

    render_dir = common.case_dir(case) / "render"
    codes = sorted(manifest["slides"])
    # The permissive mask, not the strict one. The question here is "how alike are
    # these two sections in shape", which is about the tissue rather than about how
    # well it took the stain. Using the strict mask makes a near-negative section look
    # like a different specimen: CAN_00865's CD44 slide scored 0.36 IoU against every
    # sibling and was assigned a 253 degree rotation fitted to a few specks, which
    # `stack.py` would then have applied to the real image before registering it.
    masks = {
        c: load_mask(
            render_dir
            / manifest["slides"][c].get("visibleFile", manifest["slides"][c]["maskFile"])
        )
        for c in codes
    }
    clouds = {c: points(masks[c]) for c in codes}

    iou: dict[str, dict[str, float]] = {c: {} for c in codes}
    angle: dict[str, dict[str, float]] = {c: {} for c in codes}
    for first, second in itertools.combinations(codes, 2):
        turn, score = best_rotation(clouds[first], clouds[second])
        iou[first][second] = iou[second][first] = round(score, 4)
        angle[first][second] = round(float(np.degrees(turn)), 2)
        angle[second][first] = round(float((-np.degrees(turn)) % 360), 2)
        common.say(f"{case}:   {first}-{second} IoU {score:.3f} at {np.degrees(turn):6.1f} deg", log)

    others = [c for c in codes if c != "HE"]
    scored = _rank_orders(others, iou)

    best_total, best_weakest, best_chain = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else best_total
    margin = best_total - runner_up

    # Rotations are expressed against the H&E, because that is the slide every ROI is
    # drawn on and therefore the frame everything else has to be brought into.
    rotations = {"HE": 0.0}
    for code in others:
        rotations[code] = angle[code]["HE"]

    result = {
        "case": case,
        "codes": codes,
        "iou": iou,
        "rotationToHE": rotations,
        "pairwiseAngle": angle,
        "order": list(best_chain),
        "orderTotalIou": round(best_total, 4),
        "orderWeakestLink": round(best_weakest, 4),
        # How much better the winning order is than the next one. A thin margin means the
        # shape simply does not distinguish these orderings, and the caller should treat
        # the order as unmeasured rather than trusting it.
        "orderMargin": round(margin, 4),
        "orderConfident": bool(margin >= 0.01 and best_weakest >= 0.5),
        "runnersUp": [
            {"order": list(chain), "totalIou": round(total, 4)} for total, _, chain in scored[1:4]
        ],
    }
    common.write_json(common.case_dir(case) / "order.json", result)
    common.say(
        f"{case}: cut order {' -> '.join(best_chain)} "
        f"(total {best_total:.3f}, weakest link {best_weakest:.3f}, margin {margin:.3f}, "
        f"{'confident' if result['orderConfident'] else 'NOT confident'})",
        log,
    )
    turned = {c: r for c, r in rotations.items() if min(r, 360 - r) > 20}
    if turned:
        common.say(
            f"{case}: sections mounted well off the H&E's orientation: "
            + ", ".join(f"{c} {r:.0f} deg" for c, r in turned.items()),
            log,
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case")
    args = parser.parse_args()
    common.ensure_dirs()
    measure_case(args.case, log=common.RUN_LOG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
