"""The model's input: RGB in, three-channel haematoxylin tensor out.

**The arithmetic is no longer here.** It lives in the backend, at
`app.pipeline.step08_tissue_type_segmentation.input`, and this module re-exports it.
That is the point rather than an accident of refactoring.

This module used to be the definition, and step 8 was going to grow its own copy of the
same twenty lines. Two copies of the input transform is the failure this file has always
existed to prevent: a model fitted on one definition of "haematoxylin channel" and
served another scores nothing like its validation number, and the drop looks like a
modelling failure for a week before anyone finds the plumbing. A test asserting the two
copies agree would help, but only while somebody keeps running it - and the copy that
goes stale is always the one nobody is looking at.

So there is one definition, and it sits in the backend because **the backend is what
serves it**:

  - the exporter, the torch Dataset and pipeline step 8 all call the same function
    object, not three functions that ought to be equal;
  - the direction of the dependency stays research -> backend, which is the direction it
    already ran for `optical_density`, `RUIFROK_HDAB` and `separate`. Nothing in the
    backend imports this folder, so a non-commercial training input cannot creep into
    the shipped app through an import;
  - `descriptor` is written into every checkpoint's manifest here and **checked** there
    at load, by `model.load_pinned`, against the same function's output. A checkpoint
    whose recorded contract disagrees with the serving code is refused rather than
    served, which is a guarantee no test schedule is needed to keep.

What the transform is, and why, is documented at the definition. Read
`step08_tissue_type_segmentation/input.py` for the ordering of the clip, the shape term
and the standardisation, and for the measurement that made `gamma` necessary.

Only `white_point` is still defined here, because it is a fact about *training* data
rather than about the model: a BCSS region of interest is an extracted rectangle of
tissue with no glass in it, so step 4's white calibration - which samples the empty
glass around a section - has nothing to sample and this stands in for it. A served slide
always has glass, and step 8 uses step 4's own estimate.

Ruifrok AC, Johnston DA. Quantification of histochemical staining by colour
deconvolution. Anal Quant Cytol Histol 23(4):291-299 (2001).
Tellez D et al. Quantifying the effects of data augmentation and stain colour
normalization in convolutional neural networks for computational pathology. Medical
Image Analysis 58 (2019) - the augmentation Part 6 of the plan derives from.
"""

from __future__ import annotations

import numpy as np

import backend_path  # noqa: F401 - imported for the sys.path side effect

from app.pipeline.step08_tissue_type_segmentation.input import (
    CHANNEL_HAEMATOXYLIN,
    CHANNEL_RGB_HE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    OD_CLIP,
    OD_FLOOR,
    STANDARDISE_FLOOR,
    STANDARDISE_TARGET,
    dequantise,
    descriptor,
    from_stored,
    haematoxylin_od,
    quantise,
    rgb_to_model_input,
    to_model_input,
)


def white_point(rgb: np.ndarray, percentile: float = 99.0) -> tuple[float, float, float]:
    """`I0` for an image with no glass in it, per channel.

    Step 4 measures the white point from the empty glass around the section, which is
    the right way to do it and needs a whole slide. A BCSS region of interest is an
    extracted rectangle of tissue with no glass in it at all, so there is nothing for
    that machinery to sample and this stands in: the high percentile of each channel
    over the region.

    Per channel and not over the luminance, for step 4's own reason - a scanner's lamp
    and coverslip are not neutral, and collapsing three numbers into one would bake
    that cast into every density downstream.

    The 99th rather than the maximum: a single hot pixel or a compression overshoot
    would otherwise set a denominator for the whole region.

    **Which rule produced `I0` travels in the tile manifest.** A training tile from
    BCSS and an inference tile from an OncoStem slide get their white point from
    different rules, and that is a real domain difference - worth being able to point
    at later rather than discovering in a confusion matrix.
    """
    flat = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)
    values = np.percentile(flat, percentile, axis=0)
    return (float(values[0]), float(values[1]), float(values[2]))


__all__ = [
    "CHANNEL_HAEMATOXYLIN",
    "CHANNEL_RGB_HE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "OD_CLIP",
    "OD_FLOOR",
    "STANDARDISE_FLOOR",
    "STANDARDISE_TARGET",
    "dequantise",
    "descriptor",
    "from_stored",
    "haematoxylin_od",
    "quantise",
    "rgb_to_model_input",
    "to_model_input",
    "white_point",
]
