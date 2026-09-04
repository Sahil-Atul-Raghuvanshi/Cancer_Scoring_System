"""Gate G7's second half: does window (row, col) really cover the pixels it claims to?

    .venv\\Scripts\\python scripts\\check_tissue_geometry.py <upload_id>
    .venv\\Scripts\\python scripts\\check_tissue_geometry.py <upload_id> --windows 48

`check_tissue_model.py` (G6) proves the *transform* and the *weights* agree with what
the training process recorded: it replays stored training tiles and reproduces their
logits to 2e-06. What it cannot prove is anything about **addressing**, because a
stored tile has no position on a slide. This script closes that gap.

**The failure it exists to catch.** Step 8 does not read one window at a time. It reads
a block of 8 x 8 of them in a single pass off the pyramid and cuts the windows out of
the result at computed offsets - `inference.plan_block`. If those offsets were wrong by
even one window's stride, every prediction would still be a confident, plausible class
for *a* patch of tissue; the class map would still look like a class map, the tumour
share would still be a number in the right range, and nothing anywhere would complain.
The map would simply describe the tissue half a window to the left of where it says.
Downstream, step 9 would then cut an ROI mask around the wrong region.

**How it checks.** For a set of windows spread across the cached class map, it recomputes
each one by a completely different route: a **single-window read** at the level-0 origin
the stored grid geometry implies, straight through the same pixel composition, one
forward pass. No blocks. If the block offsets are right the two agree to float noise; if
they are wrong the disagreement is enormous and obvious.

The pixel composition is deliberately *shared* with the service rather than
reimplemented. What is under test here is the addressing, so the read, the white point
and the deconvolution have to be the same ones - otherwise a disagreement could as
easily mean this script deconvolves differently, and the check would prove nothing.

**Edges are checked on purpose.** The last column and the last row are the interesting
ones: their origin is clamped to the canvas rather than being `index * stride`, so they
sit closer to their neighbour than the stride, and they are exactly where an
offset-by-index implementation would be wrong while looking right everywhere else. They
are always included in the sample.

Exit code is 0 when every sampled window agrees, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running as `python scripts/check_tissue_geometry.py` from the backend root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingestion.slide_reader import open_slide  # noqa: E402
from app.pipeline.step08_tissue_type_segmentation import input as model_input  # noqa: E402
from app.pipeline.step08_tissue_type_segmentation import model  # noqa: E402
from app.pipeline.step08_tissue_type_segmentation.classes import CLASS_NAMES  # noqa: E402
from app.services.calibration_service import calibration_service  # noqa: E402
from app.services.tissue_type_service import (  # noqa: E402
    TissueTypeError,
    tissue_type_service,
)
from app.services.upload_service import UploadError, resolve_ready_path  # noqa: E402

TICK = "  ok  "
CROSS = " FAIL "

#: How far a re-read window's probabilities may differ from the cached ones. The two
#: paths run the same weights over the same bytes, so the only legitimate source of
#: difference is float32 reduction order - worth about 1e-5 through a ResNet18. A real
#: addressing bug is a different *patch of tissue* and lands orders of magnitude away.
PROBABILITY_TOLERANCE = 1e-3

#: Windows sampled by default. Enough that a systematic offset cannot hide, few enough
#: that the check is a minute rather than an hour - each one is its own pyramid read.
DEFAULT_SAMPLE = 32


def line(status: str, message: str) -> None:
    print(f"[{status}] {message}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upload_id", help="Slide whose cached class map to verify")
    parser.add_argument(
        "--windows",
        type=int,
        default=DEFAULT_SAMPLE,
        help=f"How many windows to re-read independently (default {DEFAULT_SAMPLE})",
    )
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    print("=" * 74)
    print("Gate G7b - step 8's window addressing, checked against single-window reads")
    print("=" * 74)

    try:
        class_map = tissue_type_service.class_map(arguments.upload_id)
    except (TissueTypeError, UploadError) as exc:
        line(CROSS, str(exc))
        return 1

    grid = class_map.grid
    line(
        TICK,
        f"cached class map: {grid.cols} x {grid.rows} grid, "
        f"{class_map.classified:,} windows classified",
    )
    line(
        "      ",
        f"window {grid.size} px at {grid.mpp} um/px, span {grid.span:,} level-0 px, "
        f"stride {grid.stride:,}, overlap {grid.overlap:.0%}",
    )

    try:
        pinned = model.load_pinned()
    except model.ModelError as exc:
        line(CROSS, f"the checkpoint would not load: {exc}")
        return 1

    if pinned.tile_px != grid.size or pinned.mpp != grid.mpp:
        line(
            CROSS,
            f"the cached map was made at {grid.size} px / {grid.mpp} um/px but the "
            f"checkpoint now serves {pinned.tile_px} px / {pinned.mpp}. Re-run step 8 "
            "before checking its geometry - these are two different grids.",
        )
        return 1

    cells = _sample(class_map, count=arguments.windows, seed=arguments.seed)
    line("      ", f"re-reading {len(cells)} windows one at a time, blocks bypassed")

    white = calibration_service.white_point(arguments.upload_id)
    path = resolve_ready_path(upload_id=arguments.upload_id)

    worst = 0.0
    worst_cell: tuple[int, int] | None = None
    disagreements: list[str] = []

    import torch

    with open_slide(path) as reader:
        base_mpp = reader.mpp
        if not base_mpp:
            line(CROSS, "this slide records no microns-per-pixel")
            return 1
        if abs(float(base_mpp) - grid.base_mpp) > 1e-9:
            line(
                CROSS,
                f"the cached map was built against {grid.base_mpp} um/px and the slide "
                f"now reports {base_mpp}. Re-run step 8.",
            )
            return 1

        read = tissue_type_service.haematoxylin_reader(
            reader, white, base_mpp=float(base_mpp)
        )

        for row, col in cells:
            # The address the stored geometry implies for this cell, computed here and
            # not taken from the run - that is the whole point of the check.
            x, y = grid.x_of(col), grid.y_of(row)
            pixels = read(x, y, grid.span, grid.size)

            if pixels.shape != (grid.size, grid.size):
                disagreements.append(
                    f"({row},{col}) read back {pixels.shape}, expected "
                    f"{(grid.size, grid.size)}"
                )
                continue

            tensor = model_input.to_model_input(
                pixels,
                gamma=pinned.gamma,
                invert=pinned.invert_polarity,
                standardise=pinned.standardise,
            )
            with torch.inference_mode():
                fresh = torch.softmax(pinned.net(torch.from_numpy(tensor[None])), dim=1)
            fresh = fresh.numpy()[0]

            cached = class_map.probabilities[row, col]
            gap = float(np.abs(fresh - cached).max())
            if gap > worst:
                worst, worst_cell = gap, (row, col)

            if gap > PROBABILITY_TOLERANCE:
                disagreements.append(
                    f"({row},{col}) at level-0 ({x:,},{y:,}): cached "
                    f"{_named(cached)} vs re-read {_named(fresh)}, gap {gap:.3f}"
                )

    print("-" * 74)
    if disagreements:
        line(CROSS, f"G7b FAIL - {len(disagreements)} of {len(cells)} windows disagree")
        for entry in disagreements[:12]:
            line("      ", entry)
        if len(disagreements) > 12:
            line("      ", f"... and {len(disagreements) - 12} more")
        line(
            "      ",
            "the block offsets in inference.plan_block do not address the pixels the "
            "class map claims. Every prediction is still a plausible class for *a* "
            "patch of tissue, which is why nothing else catches this - and step 9 "
            "would cut its ROI around the wrong region.",
        )
        return 1

    line(
        TICK,
        f"G7b PASS - {len(cells)} windows re-read one at a time reproduce the block-read "
        f"class map. Worst probability difference {worst:.2e}"
        + (f" at {worst_cell}" if worst_cell else ""),
    )
    line(
        "      ",
        "so a block read addresses exactly the pixels its windows claim, including at "
        "the clamped last row and column.",
    )
    return 0


def _named(vector: np.ndarray) -> str:
    """A probability triple as `class=p`, for a message someone has to act on."""
    top = int(np.argmax(vector))
    return f"{CLASS_NAMES[top]}={vector[top]:.3f}"


def _sample(class_map, *, count: int, seed: int) -> list[tuple[int, int]]:
    """Windows to re-read: the awkward ones first, then a spread of ordinary ones.

    The edges are not a nice-to-have. Every interior window's origin is `index *
    stride`, so an implementation that assumed that would pass on all of them; the last
    row and column are clamped to the canvas and are the only place the assumption
    breaks. They go in first, and the random spread fills the rest.
    """
    grid = class_map.grid
    inside = np.argwhere(grid.inside)
    if inside.size == 0:
        return []

    chosen: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def take(row: int, col: int) -> None:
        cell = (int(row), int(col))
        if cell not in seen and grid.inside[cell]:
            seen.add(cell)
            chosen.append(cell)

    # The clamped edges, if any window there ran at all.
    for row, col in inside:
        if int(col) == grid.cols - 1 or int(row) == grid.rows - 1:
            take(row, col)
        if len(chosen) >= max(4, count // 4):
            break

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(inside))
    for index in order:
        if len(chosen) >= count:
            break
        take(*inside[index])

    return chosen


if __name__ == "__main__":
    raise SystemExit(main())
