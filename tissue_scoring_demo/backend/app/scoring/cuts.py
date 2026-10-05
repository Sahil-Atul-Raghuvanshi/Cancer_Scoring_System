"""The cut points: five independent sets, one per antibody, loaded from one file.

**One cut-point table cannot serve five antibodies, and the observed data says
so loudly.** Each has its own antibody concentration, incubation time, detection
chemistry and dynamic range, so the optical density that means "moderate" for
CD44 is not the one that means "moderate" for pan-cadherin. In the reader data
CD44 spans 5-85 % across six cases while N-cadherin is 80 % on every single
read. Sharing one threshold set across the panel guarantees that at least three
of the five markers are cut in the wrong place - which is why `CUTS` is a dict
keyed by letter and why `for_marker()` is the only way to reach one.

**The cuts are absolute, on the calibrated optical-density scale.** Not
per-slide percentiles. A percentile cut asks "which of this slide's pixels are
the brownest", and the answer is the same shape on a weak slide and a strong one
- so the two score identically and the diagnosis is normalised away. Step 4
already estimated what zero stain means on this particular slide, which is what
makes an absolute cut meaningful across slides in the first place. `compare()`
below computes the percentile answer as well, purely so a screen can show the
two side by side and the reader can see the failure rather than be told about
it.

**They are provisional and say so.** The honest calibration is the one the guide
describes: run steps 1-14 on the six cases in `6Slide Reports2.xlsx`, then choose
each marker's cut points to minimise disagreement with that marker's reader
consensus, holding cases out and reporting the fit and the held-out result
separately. That sheet is not on disk, so no fit has been done, and every number
these functions return carries `provisional=True` out to the report and onto the
screen. `tools/calibrate_marker_cuts.py` is the hook that does the fit when the
sheet appears; nothing else needs to change when it does, because the numbers
live in a data file rather than in code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from app import panel

#: Where the versioned file lives. `backend/config/marker_cuts.v1.json`.
#: `app/scoring/cuts.py` -> app -> backend.
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "marker_cuts.v1.json"

#: The only intensities OncoStem's sheet permits. 119 of its 120 readings land
#: exactly on one of these, which is what a reader picking one band per slide
#: looks like - so this is the shape of the answer, not a rounding of it.
PERMITTED_BANDS: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 1.75, 2.0)

#: How a cell whose membrane is only partly stained enters the percentage. Q1 in
#: `docs/guides/images-to-scores-mapping.md`, and the open question with the largest
#: effect on our numbers.
PARTIAL_RULES: tuple[str, ...] = ("count", "half", "exclude")


class CutsError(ValueError):
    """The cut-point file is missing, malformed, or has nothing for this marker."""


@dataclass(frozen=True)
class MarkerCuts:
    """One antibody's cut points. Nothing here is shared with another antibody."""

    letter: str
    name: str
    compartment: str

    #: The three optical-density cuts that separate 0 from 1+, 1+ from 2+ and
    #: 2+ from 3+. `od[0]` doubles as the **positivity** cut: it is what a
    #: membrane bin has to clear to count as stained, and what a cytoplasm pixel
    #: has to clear to count in the stained fraction.
    od: tuple[float, float, float]

    #: The second cut, and which of the two second measures it applies to.
    #: Exactly one of these is ever set, decided by the compartment - a
    #: cytoplasmic marker has no circumference, so a completeness minimum for it
    #: would be a threshold on a quantity that does not exist.
    completeness_min: float | None = None
    stained_fraction_min: float | None = None

    #: Ascending `(od_below, band)` pairs; the last has `od_below = None` and
    #: catches everything above. Per marker, because "how dark is strong" is a
    #: property of the antibody and its detection chemistry.
    od_to_band: tuple[tuple[float | None, float], ...] = ()

    provisional: bool = True
    anchored_on: tuple[str, ...] = ()
    notes: str = ""

    @property
    def positivity_od(self) -> float:
        """The cut a pixel or a bin has to clear to be called stained at all."""
        return self.od[0]

    @property
    def second_measure(self) -> str:
        """Which second number this marker gets. Resolved from the compartment."""
        return "ring_completeness" if self.compartment == "membrane" else "stained_fraction"

    @property
    def second_min(self) -> float:
        """The cut on the second number, whichever of the two this marker has."""
        value = (
            self.completeness_min
            if self.compartment == "membrane"
            else self.stained_fraction_min
        )
        if value is None:
            raise CutsError(
                f"{self.name} ({self.letter}) is a {self.compartment} marker but its "
                f"cut-point entry sets no {self.second_measure} minimum. Positivity is "
                "two conditions for every marker in this panel, so a missing second cut "
                "is a marker that would be called positive on optical density alone."
            )
        return float(value)

    def band(self, od: float) -> float:
        """Map a mean optical density onto the nearest permitted band.

        The banding step is not optional polish. Our software naturally produces
        a continuous number; 119 of the 120 real readings land exactly on a band
        value. Emitting 1.34 does not look more precise to the reader receiving
        it - it looks like a different measurement from the one they asked for,
        and it cannot be compared against their sheet.
        """
        for ceiling, value in self.od_to_band:
            if ceiling is None or od < ceiling:
                return float(value)
        return float(self.od_to_band[-1][1]) if self.od_to_band else 0.0

    def bin_of(self, od: float) -> int:
        """Per-cell 0 / 1+ / 2+ / 3+ for one cell's mean optical density.

        **Internal machinery, and deliberately a different thing from `band`.**
        This one counts positives and feeds the H-score; `band` is what gets
        reported. Conflating the per-cell bin with the slide-level band is the
        common bug this pair of methods exists to keep apart - they run on
        different scales (0-3 against 0-2) and they are read by different people.
        """
        if od < self.od[0]:
            return 0
        if od < self.od[1]:
            return 1
        if od < self.od[2]:
            return 2
        return 3


@dataclass(frozen=True)
class CutSet:
    """Every marker's cuts, plus the settings that are panel-wide rather than per-marker."""

    version: int
    status: str
    provenance: str
    markers: dict[str, MarkerCuts] = field(default_factory=dict)
    partial_membrane_rule: str = "count"
    band_labels: dict[float, str] = field(default_factory=dict)
    band_conflict: str = ""
    source_path: str = ""

    def for_marker(self, letter: str) -> MarkerCuts:
        key = letter.upper()
        try:
            return self.markers[key]
        except KeyError as exc:
            raise CutsError(
                f"no cut points for marker {letter!r}; the file has "
                f"{sorted(self.markers)}. A marker with no cuts cannot be scored, and "
                "borrowing another antibody's would put the cut in the wrong place - "
                "which is the exact failure five separate sets exist to prevent."
            ) from exc

    def label_for(self, band: float) -> str:
        return self.band_labels.get(float(band), "")


def _tuple3(values, where: str) -> tuple[float, float, float]:
    try:
        first, second, third = (float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise CutsError(f"{where}: 'od' must be three numbers, got {values!r}") from exc
    if not first < second < third:
        raise CutsError(
            f"{where}: the three optical-density cuts must ascend, got "
            f"{first}, {second}, {third}. Out of order they would put a darker cell in a "
            "lower bin than a paler one."
        )
    return first, second, third


def load(path: Path | None = None) -> CutSet:
    """Read and validate the cut-point file. Raises rather than falling back.

    No default is substituted for a missing or broken file, on purpose. A score
    computed against silently invented cut points is the failure mode this whole
    module is organised against: it would look exactly like a real result.
    """
    source = path or CONFIG_PATH
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CutsError(
            f"no cut-point file at {source}. Every intensity call this pipeline makes "
            "reads its thresholds from that file, so there is nothing to score with."
        ) from exc
    except ValueError as exc:
        raise CutsError(f"the cut-point file at {source} is not valid JSON: {exc}") from exc

    bands = payload.get("bands", {})
    table = tuple(
        (None if row[0] is None else float(row[0]), float(row[1]))
        for row in bands.get("od_to_band", [])
    )
    labels = {float(key): str(value) for key, value in bands.get("labels", {}).items()}

    permitted = tuple(float(value) for value in bands.get("permitted", PERMITTED_BANDS))
    for _, value in table:
        if value not in permitted:
            raise CutsError(
                f"the band table maps an optical density onto {value}, which is not one "
                f"of the permitted values {list(permitted)}. OncoStem's sheet has no "
                "column that can hold it."
            )

    rule = str(payload.get("partial_membrane_rule", "count"))
    if rule not in PARTIAL_RULES:
        raise CutsError(
            f"partial_membrane_rule is {rule!r}; expected one of {list(PARTIAL_RULES)}"
        )

    provisional = str(payload.get("status", "provisional")).lower() != "calibrated"
    anchored = tuple(str(entry) for entry in payload.get("anchored_on", []))

    markers: dict[str, MarkerCuts] = {}
    for letter, entry in payload.get("markers", {}).items():
        key = letter.upper()
        where = f"marker {key}"
        try:
            marker_spec = panel.spec(key)
        except KeyError as exc:
            raise CutsError(f"{where}: not an antibody this panel knows about") from exc

        compartment = str(entry.get("compartment", marker_spec.compartment.value))
        if compartment != marker_spec.compartment.value:
            # The compartment is `app.panel`'s decision and nothing else's. A cut-point
            # file that disagrees is a file describing a different pipeline, and
            # accepting it would measure the wrong part of the cell.
            raise CutsError(
                f"{where}: the cut-point file calls this a {compartment} marker, but "
                f"app.panel says {marker_spec.compartment.value}. The compartment is "
                "panel's decision; a config file cannot move it."
            )

        markers[key] = MarkerCuts(
            letter=key,
            name=str(entry.get("name", marker_spec.name)),
            compartment=compartment,
            od=_tuple3(entry.get("od", ()), where),
            completeness_min=(
                float(entry["completeness_min"]) if "completeness_min" in entry else None
            ),
            stained_fraction_min=(
                float(entry["stained_fraction_min"])
                if "stained_fraction_min" in entry
                else None
            ),
            od_to_band=table,
            provisional=provisional,
            anchored_on=anchored,
            notes=str(entry.get("notes", "")),
        )
        # Fail here rather than at measurement time: a marker with no second cut is
        # a marker that would be called positive on optical density alone, and that
        # is a silent widening of the definition of "positive". `second_min` raises
        # for the marker that is missing one, which is the check - the value itself
        # is not wanted here, only the fact that it resolves.
        _ = markers[key].second_min

    missing = [letter for letter in panel.SCORED_MARKERS if letter not in markers]
    if missing:
        raise CutsError(
            f"the cut-point file has no entry for {missing}. Every scored antibody needs "
            "its own set - see this module's docstring for why one table cannot serve five."
        )

    return CutSet(
        version=int(payload.get("version", 0)),
        status=str(payload.get("status", "provisional")),
        provenance=str(payload.get("provenance", "")),
        markers=markers,
        partial_membrane_rule=rule,
        band_labels=labels,
        band_conflict=str(bands.get("conflict", "")),
        source_path=str(source),
    )


@lru_cache(maxsize=1)
def cut_set() -> CutSet:
    """The loaded cut points, read once per process."""
    return load()


def for_marker(letter: str) -> MarkerCuts:
    """One antibody's cuts. The only way a step reaches a threshold."""
    return cut_set().for_marker(letter)


def reload() -> CutSet:
    """Re-read the file, for a test or a recalibration that has just written it."""
    cut_set.cache_clear()
    return cut_set()


__all__ = [
    "CONFIG_PATH",
    "PARTIAL_RULES",
    "PERMITTED_BANDS",
    "CutSet",
    "CutsError",
    "MarkerCuts",
    "cut_set",
    "for_marker",
    "load",
    "reload",
]
