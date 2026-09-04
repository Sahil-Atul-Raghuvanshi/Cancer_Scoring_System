"""The region model's input, defined once: RGB in, three-channel haematoxylin tensor out.

**This module is the contract between training and serving, and it lives here rather
than beside the training code on purpose.** The tile exporter, the torch Dataset that
fitted the head, and this step all have to compute the input the same way; a model
fitted on one definition of "haematoxylin channel" and served another scores nothing
like its validation number, and the failure looks like a modelling problem for a week
before anyone finds the plumbing. So there is one definition, it sits in the backend
because the backend is what serves it, and `bcss_hchannel_resnet18/src/hchannel.py`
imports it rather than restating it. The dependency runs research -> backend and never
the other way, which is also what keeps a non-commercial training input out of the
backend's own imports.

The transform, in the order the physics requires:

  step A   RGB to optical density, `OD = -log10(I / I0)`. Beer-Lambert. Done by
           `app.common.imaging.optical_density`, which steps 4 and 5 already share
           for the same reason this module exists.
  step B   colour deconvolution onto Ruifrok's fixed haematoxylin-DAB basis, keeping
           column 0. Done by `step06_colour_deconvolution.deconvolution.separate`.
           **Fixed vectors, never per-image estimates** - an estimate rescales itself
           to whatever slide it is given, so "0.4 haematoxylin" would mean a different
           amount of dye on every slide and the model's input distribution would move
           under it.
  step C   optional per-tile standardisation, then clip to [0, OD_CLIP], scale to
           [0, 1], the shape term, the polarity, replicate to three channels, and
           normalise with the statistics the pretrained weights were fitted under.

Why the haematoxylin channel at all: it is the one dye present on all six slides of a
case - the H&E and all five IHC - so dropping the second dye makes every slide the same
kind of image and one model can serve the whole panel. The brown that floods an
N-cadherin section at 75-85% positive is entirely in the channel this throws away.
That is Rule 2 of the pipeline guide, and step 6 is where the fork happens.

Why three identical channels: ResNet expects three. It is handed the same grayscale
image three times, which works fine and is cheaper than reasoning about a modified
first convolution.

**Every serving-time value here is recorded in the checkpoint's manifest and checked
at load** - see `model.load_pinned` and `descriptor` below. A checkpoint whose
descriptor disagrees with this module is a checkpoint being served something it never
saw, so the disagreement is a refusal rather than a warning.

Ruifrok AC, Johnston DA. Quantification of histochemical staining by colour
deconvolution. Anal Quant Cytol Histol 23(4):291-299 (2001).
Tellez D et al. Quantifying the effects of data augmentation and stain colour
normalization in convolutional neural networks for computational pathology. Medical
Image Analysis 58 (2019).
"""

from __future__ import annotations

import numpy as np

from app.common.imaging import optical_density
from app.common.stains import RUIFROK_HDAB
from app.pipeline.step06_colour_deconvolution.deconvolution import separate

#: Optical density at which the haematoxylin channel is considered saturated. Above
#: this a nucleus is simply "as dark as nuclei get", and letting the tail run free
#: would hand the whole input range to a handful of pixels. Comfortably above the
#: 99.5th percentile of a normally counterstained tile.
OD_CLIP: float = 1.5

#: Intensity floor for the Beer-Lambert transform, so a black pixel gives a large
#: finite density rather than an infinity. Step 4's `calibration_od_floor`, restated
#: as a constant so this module needs no Settings object to read one float - and
#: asserted equal to it by the tests, so the two cannot drift.
OD_FLOOR: float = 1.0

#: Lower bound on a tile's own p99 density before `standardise` will rescale it. A
#: tile of bare glass or loose stroma has no stain to normalise against, and dividing
#: by its p99 would stretch sensor noise across the whole input range - inventing
#: texture where the slide has none. Below this the tile is left alone.
STANDARDISE_FLOOR: float = 0.10

#: Where a standardised tile's own p99 is placed inside [0, 1]. Not 1.0: the tail
#: above p99 needs somewhere to go, and pinning p99 to the top would clip every
#: nucleus in every tile.
STANDARDISE_TARGET: float = 0.60

#: The scale torchvision's ImageNet weights were fitted under. Used for **both**
#: backbone initialisations, including the SimCLR one: Ciga's own code normalises this
#: way too, and more importantly an A/B that changed the normalisation alongside the
#: weights would not be measuring the weights.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

_MEAN = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
_STD = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)


def haematoxylin_od(
    rgb: np.ndarray,
    white: np.ndarray | tuple[float, float, float],
    *,
    od_floor: float = OD_FLOOR,
) -> np.ndarray:
    """The haematoxylin channel of an HxWx3 image, in optical density units.

    Steps A and B, through the pipeline's own two functions and no others. Returns
    HxW float32, unclipped and unscaled - the physical quantity, before any decision
    about what a network wants to see. `to_model_input` makes that decision, and
    keeping the two apart is what lets a stored training tile be re-interpreted later
    without re-reading a slide.

    `white` may be one triple or a full HxWx3 field, because step 4 may have justified
    an illumination surface rather than a flat white point. `optical_density`
    broadcasts both, so this never branches on which.
    """
    od = optical_density(rgb, white, floor=od_floor)
    return separate(od, RUIFROK_HDAB).haematoxylin.astype(np.float32)


def quantise(h_od: np.ndarray) -> np.ndarray:
    """Haematoxylin density to the uint8 the training tile store holds.

    Serving never calls this - a served tile goes straight from density to tensor -
    but the gate that proves serving matches training does: it replays a stored
    training tile through `from_stored` and compares logits against the manifest. So
    the quantisation has to be defined where the check lives.

    The stored artefact is the **model's input range**, not the raw density: `[0,
    OD_CLIP]` mapped onto `[0, 255]`. One level is 0.0059 OD, finer than the 8-bit
    intensity data the density was computed from can resolve, so the quantisation is
    lossless in every way that matters.
    """
    scaled = np.clip(np.asarray(h_od, dtype=np.float32), 0.0, OD_CLIP) / OD_CLIP
    return np.rint(scaled * 255.0).astype(np.uint8)


def dequantise(stored: np.ndarray) -> np.ndarray:
    """The stored uint8 back to optical density. The inverse of `quantise`."""
    return np.asarray(stored, dtype=np.float32) / 255.0 * OD_CLIP


def to_model_input(
    h_od: np.ndarray,
    *,
    alpha: float = 1.0,
    beta: float = 0.0,
    gamma: float = 1.0,
    invert: bool = False,
    standardise: bool = False,
) -> np.ndarray:
    """Haematoxylin density to the 3xHxW float32 tensor the network sees.

    **At serving time `alpha`, `beta` and `gamma` sit at their identity values and only
    `standardise` is live.** The first three are the training-time stain jitter, and
    they are parameters of *this* function rather than a separate augmentation step for
    a reason worth stating: "HED augmentation" on a single channel is a stain-strength
    jitter. Tellez-style augmentation perturbs the haematoxylin, eosin and DAB
    concentrations of an RGB image and recomposes it; our input is the H channel alone,
    so there is nothing to recompose, and applying `rgb2hed`-style jitter to three
    identical channels would perturb three copies of one number in three different
    directions and produce a colour cast no slide can have. The honest single-channel
    equivalent is what varying stain strength actually does to the haematoxylin channel
    - scale it, shift it a little - applied **before the clip**, because a strongly
    counterstained slide really does saturate more of its nuclei.

    `gamma` is the **shape** term and it exists because of a measurement. Comparing one
    of this project's H&E slides against the IHC section of the same case, the IHC/H&E
    haematoxylin density ratio runs from 0.087 at p75 to 2.03 at p99 - a 23x spread
    where a pure strength difference would be one constant. Two candidate explanations
    were tested and both rejected: deconvolving the H&E with an H-E basis instead of
    H-DAB (18.8x, no better) and taking I0 once per slide from glass instead of per
    tile (36.7x, worse). The two distributions differ in **shape**: an IHC counterstain
    is sparse and punchy, three quarters of its tissue pixels near zero with a hard
    dark tail, where H&E haematoxylin is broad and mid-toned.

    **No `alpha` fixes that, and neither would any per-slide normalisation**, because
    every such correction is monotone and a monotone map cannot change the ratio
    between two quantiles of the same image. `gamma` can: applied to the value *after*
    it is scaled into [0, 1], `x ** gamma` with gamma > 1 pushes mid-tones down while
    leaving the top of the range fixed, which is exactly the H&E -> IHC direction. The
    training jitter spans roughly 0.5 to 3.5 rather than trying to hit one value; the
    point is not to map one stain onto the other but to make the training distribution
    *contain* the serving one, so the model never has to extrapolate.

    `standardise` divides the tile by its own p99 before the clip, which closes the
    *scale* gap the jitter cannot reach (H&E p99 0.50 against IHC 1.03 - twice, well
    outside the alpha range) but, being monotone, does nothing for the shape. It is the
    one non-identity serving parameter, it is recorded per checkpoint, and it is read
    from the manifest rather than chosen here.

    `invert` flips the polarity. This H channel puts nuclei *bright* - more dye, higher
    value - where a grayscale photograph of an H&E slide puts them dark, which is the
    polarity ImageNet's filters were fitted against. Which convention a checkpoint was
    fitted under is recorded in its manifest, so serving cannot disagree.

    Order is fixed and not interchangeable: standardise, then scale and offset in
    density units, then the clip, then the shape term on the normalised value, then
    polarity, then ImageNet normalisation. `gamma` after the clip rather than before is
    what keeps it a pure shape change - applied to a density it would also move the
    saturation point, and then one knob would be doing two jobs.
    """
    x = np.asarray(h_od, dtype=np.float32)

    if standardise:
        ref = float(np.percentile(x, 99.0))
        if ref >= STANDARDISE_FLOOR:
            x = x * np.float32(STANDARDISE_TARGET * OD_CLIP / ref)

    x = np.clip(x * np.float32(alpha) + np.float32(beta), 0.0, OD_CLIP) / OD_CLIP

    if gamma != 1.0:
        # x is already in [0, 1], so the power is a shape change and nothing else.
        x = np.power(x, np.float32(gamma), dtype=np.float32)

    if invert:
        x = 1.0 - x

    stack = np.repeat(x[np.newaxis, :, :], 3, axis=0)
    return (stack - _MEAN) / _STD


def from_stored(
    stored: np.ndarray,
    *,
    alpha: float = 1.0,
    beta: float = 0.0,
    gamma: float = 1.0,
    invert: bool = False,
    standardise: bool = False,
) -> np.ndarray:
    """A stored training tile's uint8 straight to a model input.

    Only the gate uses this - `scripts/check_tissue_model.py` replays the tiles the
    manifest recorded logits for and checks this process reproduces them. The stored
    tile is raw quantised density, so `gamma` and `standardise` are load-time decisions
    rather than export-time ones, which is why the same tile store can serve a
    checkpoint fitted either way.
    """
    return to_model_input(
        dequantise(stored),
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        invert=invert,
        standardise=standardise,
    )


def descriptor(
    *,
    tile_px: int,
    mpp: float,
    invert: bool,
    gamma: float = 1.0,
    standardise: bool = False,
) -> dict[str, object]:
    """The input contract, as the JSON a checkpoint's manifest carries.

    Written by the training run and checked here at load. Recorded as data rather than
    left as a convention because a convention cannot be compared: `model.load_pinned`
    asserts the manifest's block equals this function's output for the manifest's own
    tile size, resolution and polarity, and refuses the checkpoint otherwise.

    The jitter *ranges* are deliberately absent. They are a training detail; what
    serving must reproduce exactly is `gamma`, `standardise` and the geometry.
    """
    return {
        "channel": "haematoxylin",
        "stain_basis": "ruifrok_hdab_fixed",
        "od_clip": OD_CLIP,
        "od_floor": OD_FLOOR,
        "tile_px": int(tile_px),
        "mpp": float(mpp),
        "replicate_to_3ch": True,
        "invert_polarity": bool(invert),
        # Serving-time values. What inference must reproduce exactly.
        "gamma": float(gamma),
        "standardise_tile_p99": bool(standardise),
        "standardise_floor": STANDARDISE_FLOOR,
        "standardise_target": STANDARDISE_TARGET,
        "normalisation": "imagenet",
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
    }


__all__ = [
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
    "to_model_input",
]
