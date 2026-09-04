"""Step 6 - Colour deconvolution.

Implemented, and the fork. See README.md in this folder,
`docs/demo-pipeline-guide.md`, and `app.data.pipeline_steps`
(id="colour-deconvolution") for what this step does and why it runs here.

`deconvolve` is the one function the whole pipeline separates stains with. Every
consumer - this step's service, the tile exporter, and the region model's runtime
when it is built - imports it from here. Nothing is permitted to reach for
`skimage.color.rgb2hed` on its own: two implementations drift, and when they do the
IHC slides score nothing like the H&E.
"""

from .deconvolution import (
    BASES,
    CHANNELS,
    ESTIMATED,
    FIXED,
    Basis,
    Channels,
    DeconvolutionError,
    Separation,
    deconvolve,
    separate,
)

__all__ = [
    "BASES",
    "CHANNELS",
    "ESTIMATED",
    "FIXED",
    "Basis",
    "Channels",
    "DeconvolutionError",
    "Separation",
    "deconvolve",
    "separate",
]
