"""What a stored training tile *is* - one table, two entries, no arithmetic.

The exporter cuts a grid, votes each square against the mask, and writes what it kept.
Only the last of those three depends on which input contract the tiles are for, and
this module is that dependency isolated: a `Variant` says how to turn a resampled RGB
region into the plane the tiles are cut from, how to store one window, what the
manifest should call it, and which per-tile statistics are worth recording.

**Every callable here is imported from `hchannel`, which re-exports the backend's
`input.py`.** There is deliberately no arithmetic in this file. A second copy of the
input transform is the failure `hchannel`'s docstring is about, and it would be no less
a failure for being spelled as a lambda in a table.

Why this lives in the research repo when the transforms live in the backend: a *variant
table* is a training-only concept. Serving is handed one checkpoint whose manifest names
one contract; it never chooses between two ways of storing a tile, because it stores
nothing. So the fork the exporter needs has no serving counterpart, and putting it here
keeps the backend's own imports free of it.

The two variants:

  `H_CHANNEL`  Ruifrok's haematoxylin channel, quantised to uint8 over [0, OD_CLIP] and
               written as a single-channel PNG. Every checkpoint published before the
               H&E branch. The white point is measured per region by `hchannel.white_point`
               because a BCSS crop has no glass in it.

  `RGB_HE`     the resampled sRGB itself, written as a three-channel PNG. No white
               point, no optical density, no deconvolution: the H&E model is shown the
               photograph, and the honest store for that is the photograph.

**The kept tile set is identical under both.** `export.export_region` computes the grid
and the vote from the *mask* alone - the white point, the deconvolution and the
quantisation never enter the vote - so for one spec and one region list the two variants
produce the same `tile_id`s with the same labels. That is what makes a per-tile paired
comparison between the two stores possible, and it is why the two stores are cut in two
independent passes rather than dual-written from one loop: run separately, "the tile ids
match" is an assertion worth making; produced by one loop it would be a tautology.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

import hchannel


@dataclass(frozen=True)
class Variant:
    """One way of storing a tile, and everything the exporter needs to know about it."""

    #: The input contract's name, as `input.descriptor` records it. This is the value
    #: every downstream branch reads - `datasets.channel_of`, the published manifest,
    #: and step 7's branch filter - so it is the one field that must never drift.
    channel: str

    #: What the bytes on disk are, as the manifest's `variant` column. Distinct from
    #: `channel` because a channel is a claim about *meaning* and this is a claim about
    #: *encoding*; one day there may be two encodings of one channel.
    stored_as: str

    #: `(rgb_resampled, spec) -> plane`. HxW float32 for the H channel, HxWx3 uint8 for
    #: RGB. Called once per region, never per tile: the transform is per pixel, so
    #: doing it once is the same arithmetic done fewer times and it guarantees
    #: neighbouring tiles share a white point.
    prepare: Callable[[np.ndarray, Any], np.ndarray]

    #: `window -> uint8 array to save`.
    store: Callable[[np.ndarray], np.ndarray]

    #: The PIL mode to save under, or `None` to let PIL infer it from the shape.
    pil_mode: str | None

    #: `(*, tile_px, mpp, invert) -> dict`. The `input` block of the export summary,
    #: and the same function the publish step writes into a checkpoint manifest.
    descriptor: Callable[..., dict[str, object]]

    #: How `I0` was obtained, as a manifest string. A real rule for the H channel; the
    #: absence of one, stated, for RGB - recording "none" is more useful than a gap,
    #: because a gap is indistinguishable from a field nobody filled in.
    i0_rule: str

    #: `window -> dict` of extra per-tile manifest columns. Per variant because the
    #: interesting summary of a density plane is not the interesting summary of a
    #: photograph, and recording a mean optical density for pixels that are not optical
    #: densities would be a number nobody could interpret.
    stats: Callable[[np.ndarray], dict[str, object]]


H_CHANNEL = Variant(
    channel=hchannel.CHANNEL_HAEMATOXYLIN,
    stored_as="h_od_uint8",
    prepare=lambda rgb, spec: hchannel.haematoxylin_od(
        rgb, hchannel.white_point(rgb, percentile=spec.white_percentile)
    ),
    store=hchannel.quantise,
    pil_mode="L",
    descriptor=hchannel.descriptor,
    i0_rule="p{white_percentile:g}_per_channel_over_region",
    stats=lambda window: {
        "mean_h_od": round(float(window.mean()), 4),
        "p99_h_od": round(float(np.percentile(window, 99.0)), 4),
    },
)


def _rgb_descriptor(**kwargs: Any) -> dict[str, object]:
    """`hchannel.descriptor` pinned to the RGB channel.

    A named function rather than a `partial` so the traceback names it when the
    descriptor refuses a transform this branch does not have.
    """
    return hchannel.descriptor(channel=hchannel.CHANNEL_RGB_HE, **kwargs)


RGB_HE = Variant(
    channel=hchannel.CHANNEL_RGB_HE,
    stored_as="rgb_uint8",
    # Identity: `export.resample` has already put the region at the target mpp, and
    # there is nothing else to do to a photograph.
    prepare=lambda rgb, spec: np.asarray(rgb, dtype=np.uint8),
    store=lambda window: window,
    pil_mode=None,
    descriptor=_rgb_descriptor,
    i0_rule="none - raw sRGB stored; no white point and no deconvolution at export",
    stats=lambda window: {
        "mean_rgb": [round(float(value), 2) for value in window.reshape(-1, 3).mean(0)],
        "p99_gray": round(float(np.percentile(window.mean(axis=2), 99.0)), 2),
    },
)


#: What `--channel` accepts. `rgb_he` is spelled both ways because the manifest records
#: `rgb_he` and a human types `he`; accepting only one of the two is a papercut with no
#: upside.
VARIANTS: dict[str, Variant] = {
    "haematoxylin": H_CHANNEL,
    "he": RGB_HE,
    "rgb_he": RGB_HE,
}


def resolve(name: str) -> Variant:
    """A `--channel` value to its variant, refusing anything else by name."""
    try:
        return VARIANTS[str(name)]
    except KeyError:
        raise SystemExit(
            f"--channel {name!r} is not one of: {', '.join(sorted(VARIANTS))}"
        ) from None


__all__ = ["H_CHANNEL", "RGB_HE", "VARIANTS", "Variant", "resolve"]
