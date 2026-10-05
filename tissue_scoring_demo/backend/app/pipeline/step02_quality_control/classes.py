"""The classes a GrandQC artefact model emits, in one place.

Pixel coding is GrandQC's, not ours - it is fixed by the checkpoints and is
documented in their README:

    1 tissue - 2 fold - 3 dark spot / foreign object - 4 pen marking
    5 edge / air bubble - 6 out of focus - 7 background

Class 0 never appears in a finished mask. The models carry an unused
channel-0, and the padding written around the patch grid is zeroed, so 0 means
"not analysed" and is counted as neither tissue nor artefact.

The colours are GrandQC's published palette, kept deliberately: a mask rendered
here is then directly comparable with the figures in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Written into the mask where the patch grid did not reach (right/bottom edge).
UNANALYSED = 0

TISSUE = 1
FOLD = 2
DARKSPOT = 3
PEN = 4
EDGE = 5
FOCUS = 6
BACKGROUND = 7


@dataclass(frozen=True)
class QCClass:
    """One pixel class, with the wording and colour the UI shows for it."""

    id: int
    key: str
    label: str
    blurb: str
    colour: tuple[int, int, int]
    is_artefact: bool
    #: The classical metric that most directly corroborates this artefact.
    #: Used by the explanation panel to pick which feature to lead with; it is
    #: a presentation hint, never an input to the decision.
    explained_by: str | None = None

    @property
    def hex_colour(self) -> str:
        red, green, blue = self.colour
        return f"#{red:02x}{green:02x}{blue:02x}"


QC_CLASSES: tuple[QCClass, ...] = (
    QCClass(
        id=TISSUE,
        key="tissue",
        label="Clean tissue",
        blurb="Tissue with no artefact detected. This is what survives into step 3.",
        colour=(128, 128, 128),
        is_artefact=False,
    ),
    QCClass(
        id=FOLD,
        key="fold",
        label="Tissue fold",
        blurb=(
            "The section folded over itself during floating or mounting, so the light "
            "passed through two thicknesses. Everything reads darker and denser than it is."
        ),
        colour=(255, 99, 71),
        is_artefact=True,
        explained_by="rms_contrast",
    ),
    QCClass(
        id=DARKSPOT,
        key="darkspot",
        label="Dark spot / foreign object",
        blurb=(
            "Dust, a hair, or debris trapped under the coverslip. Opaque, so it is "
            "indistinguishable from very dense chromatin to anything counting nuclei."
        ),
        colour=(0, 255, 0),
        is_artefact=True,
        explained_by="brightness",
    ),
    QCClass(
        id=PEN,
        key="pen",
        label="Pen marking",
        blurb=(
            "Ink a pathologist drew on the glass to ring a region of interest. Dark and "
            "strongly coloured, so a tissue mask built on saturation will happily call it tissue."
        ),
        colour=(255, 0, 0),
        is_artefact=True,
        explained_by="saturation",
    ),
    QCClass(
        id=EDGE,
        key="edge",
        label="Air bubble / coverslip edge",
        blurb=(
            "An air bubble under the coverslip, or the coverslip's own edge. Produces "
            "high-contrast rims that look like membranes and pale voids that look like fat."
        ),
        colour=(255, 0, 255),
        is_artefact=True,
        explained_by="texture_energy",
    ),
    QCClass(
        id=FOCUS,
        key="focus",
        label="Out of focus",
        blurb=(
            "The scanner's focus map failed here, usually on a fold or a thick region. "
            "Nuclear boundaries dissolve, so segmentation under-counts and intensity is smeared."
        ),
        colour=(75, 0, 130),
        is_artefact=True,
        explained_by="tenengrad",
    ),
    QCClass(
        id=BACKGROUND,
        key="background",
        label="Background",
        blurb="Empty glass. Never part of any denominator.",
        colour=(255, 255, 255),
        is_artefact=False,
    ),
)

#: Just the artefacts, in mask order - what the bar chart iterates.
ARTEFACT_CLASSES: tuple[QCClass, ...] = tuple(item for item in QC_CLASSES if item.is_artefact)

_BY_ID = {item.id: item for item in QC_CLASSES}
_BY_KEY = {item.key: item for item in QC_CLASSES}

#: Classes that sit on tissue: clean tissue plus every artefact. This is the
#: denominator for "% of tissue rejected", and the reason it excludes
#: background is that glass was never analysable in the first place.
TISSUE_CLASS_IDS: tuple[int, ...] = tuple(
    item.id for item in QC_CLASSES if item.id != BACKGROUND and item.id != UNANALYSED
)


def class_by_id(class_id: int) -> QCClass | None:
    return _BY_ID.get(class_id)


def class_by_key(key: str) -> QCClass | None:
    return _BY_KEY.get(key)


def palette_rows() -> list[tuple[int, int, int]]:
    """A 256-entry RGB palette for writing the mask as an indexed PNG."""
    rows = [(0, 0, 0)] * 256
    for item in QC_CLASSES:
        rows[item.id] = item.colour
    return rows
