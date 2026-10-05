"""Step 7 - Tiling.

Implemented. See README.md in this folder, `docs/guides/demo-pipeline-guide.md` (step 8
there), and `app.data.pipeline_steps` (id="tiling") for what this step does and
why it runs at this point in the pipeline.

`build_index` produces addresses, never pixels. A tile's pixels are step 6's
haematoxylin channel, computed on demand through the one deconvolution function
this codebase has - see `app.pipeline.step06_colour_deconvolution`.
"""

from .index import Funnel, Tile, TileIndex, TilingError, build_index, sample

__all__ = [
    "Funnel",
    "Tile",
    "TileIndex",
    "TilingError",
    "build_index",
    "sample",
]
