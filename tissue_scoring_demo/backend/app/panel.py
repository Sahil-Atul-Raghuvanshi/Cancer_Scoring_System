"""The five antibodies this system scores, and what each one implies.

The single trailing letter in a slide's filename is the antibody - confirmed by
OncoStem's CAB deck, the 28 July call, and the physical slide labels (see
`docs/design/five-marker-implementation.md` and `docs/guides/images-to-scores-mapping.md`).
Every per-marker decision resolves through this module, so adding a sixth
antibody is a new `PANEL` entry rather than a change to five call sites.

Compartment is the field that forks downstream measurement: three membrane
markers take a ring, two cytoplasmic markers take a band. N-cadherin and
pan-cadherin are cytoplasmic, not membranous - an earlier draft had this wrong.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Marker(str, Enum):
    """The antibody a slide was stained with, keyed by its filename letter."""

    CD44 = "A"
    ABCC4 = "F"
    ABCC11 = "R"
    N_CADHERIN = "U"
    PAN_CADHERIN = "W"
    HE = "HE"  # not scored; used for orientation only


class Compartment(str, Enum):
    """Where in the cell the brown is expected to be."""

    MEMBRANE = "membrane"
    CYTOPLASM = "cytoplasm"
    NONE = "none"


@dataclass(frozen=True)
class MarkerSpec:
    """One antibody, and every downstream decision that follows from it."""

    letter: str
    name: str
    full_name: str
    compartment: Compartment
    scored: bool

    # In MICRONS, never pixels: at 0.2222 um/px a 4 um ring is 18 px, and on
    # another scanner it is not.
    #
    # How far the cell is grown from its nucleus - the outer edge of the cell
    # body, for either kind of marker.
    compartment_width_um: float

    # Membrane markers get ring completeness over 36 angular bins; cytoplasmic
    # markers get the stained fraction of the band.
    second_measure: str

    # The reporting range from OncoStem's own SOP, clause 9.1. Explicitly
    # descriptive: "the expected population ranges below are descriptive, not
    # thresholds, and a case outside them is not by itself an error". So it is a
    # flag for human review and never a cutoff - clause 9.2 is equally explicit
    # that no positivity cut-off is applied at the scoring stage and that the raw
    # percentage and intensity pass straight to the prediction algorithm.
    expected_percent: tuple[int, int]

    # For a MEMBRANE marker, how thick the measured shell at that outer edge is.
    # Zero for a cytoplasmic marker, which measures the whole body.
    #
    # **This is a sampling choice and not a measurement of a membrane.** A cell
    # membrane is about 10 nm thick, forty times finer than one pixel at this
    # project's 0.2222 um/px, so nothing here can resolve it. What this number
    # decides is how much of the cell's outer edge is averaged to stand for it.
    #
    # It used to equal `compartment_width_um`, which made the "ring" the entire
    # 4 um annulus - the whole cell body - so a membrane marker and a cytoplasmic
    # one differed only in how wide their band was, never in shape. That is
    # defensible as the guide's literal recipe and it is not what ASCO/CAP mean
    # by "complete, intense membrane staining", which is about the rim. At 1.5 um
    # the shell sits from 2.5 to 4.0 um out, clear of the nucleus, with cytoplasm
    # genuinely between the two.
    #
    # Provisional, like the widths: nobody has established it, and the screen
    # sweeps and reports it rather than hiding it.
    membrane_shell_um: float = 0.0


#: How thick a membrane marker's measured shell is, in microns. One number for
#: the three membrane markers because nothing distinguishes them here - the
#: antibodies differ in what they bind, not in where a cell's rim is.
MEMBRANE_SHELL_UM = 1.5

PANEL: dict[str, MarkerSpec] = {
    "A": MarkerSpec("A", "CD44", "CD44", Compartment.MEMBRANE, True, 4.0, "ring_completeness", (0, 85), MEMBRANE_SHELL_UM),
    "F": MarkerSpec("F", "ABCC4", "ABCC4 (MRP4)", Compartment.MEMBRANE, True, 4.0, "ring_completeness", (35, 70), MEMBRANE_SHELL_UM),
    "R": MarkerSpec("R", "ABCC11", "ABCC11 (MRP8)", Compartment.MEMBRANE, True, 4.0, "ring_completeness", (30, 70), MEMBRANE_SHELL_UM),
    "U": MarkerSpec("U", "N-cadherin", "N-cadherin (CDH2)", Compartment.CYTOPLASM, True, 6.0, "stained_fraction", (70, 85)),
    "W": MarkerSpec("W", "Pan-cadherin", "Pan-cadherin", Compartment.CYTOPLASM, True, 6.0, "stained_fraction", (70, 85)),
    "HE": MarkerSpec("HE", "H&E", "Haematoxylin & eosin", Compartment.NONE, False, 0.0, "none", (0, 0)),
}

SCORED_MARKERS = tuple(letter for letter, spec in PANEL.items() if spec.scored)


def spec(letter: str) -> MarkerSpec:
    """The spec for a letter, or a KeyError naming what was asked for."""
    try:
        return PANEL[letter.upper()]
    except KeyError as exc:
        raise KeyError(f"unknown marker {letter!r}; expected one of {sorted(PANEL)}") from exc


# `<anything>` then one or more of `_`/`-` then the letter code, then `.svs`.
# Tolerant of the mixed separator `CAN_00865_26-_H&E.svs` ("the filename trap",
# docs/guides/images-to-scores-mapping.md) because `[_-]+` accepts any run of either
# character, not just one.
_FILENAME_RE = re.compile(r"^.+?[_-]+(?P<letter>H&E|[AFRUW])\.svs$", re.IGNORECASE)


def infer_from_filename(filename: str) -> str | None:
    """Guess the antibody from a filename, or None if it cannot be read.

    Returns a *suggestion*, never a decision - the filename convention is
    OncoStem's, and a slide from anywhere else will not follow it. Silently
    mis-detecting a marker produces a plausible number measured in the wrong
    compartment, which is the worst kind of wrong: it does not look like an
    error. Callers must let a human confirm, not act on this alone.
    """
    match = _FILENAME_RE.match(filename)
    if not match:
        return None
    letter = match.group("letter").upper()
    key = "HE" if letter == "H&E" else letter
    return key if key in PANEL else None


__all__ = [
    "MEMBRANE_SHELL_UM",
    "PANEL",
    "SCORED_MARKERS",
    "Compartment",
    "Marker",
    "MarkerSpec",
    "infer_from_filename",
    "spec",
]
