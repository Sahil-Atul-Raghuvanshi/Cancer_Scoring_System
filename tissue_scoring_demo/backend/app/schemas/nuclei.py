"""Schemas for step 11 - finding every nucleus inside the carried regions."""

from typing import Literal

from pydantic import Field

from app.schemas.common import APIModel

NucleiState = Literal["queued", "running", "ready", "failed"]

#: Which segmenter produced a result. Both are always available; the second is
#: the comparison that shows why instance segmentation is its own problem.
Engine = Literal["instanseg", "watershed"]

#: What the segmenter was shown. `haematoxylin` is the shipped path and the only
#: one whose counts feed a denominator; `rgb` exists so the screen can show what
#: detecting on the measured stain does to the number.
Channel = Literal["haematoxylin", "rgb"]


class NucleiRun(APIModel):
    """Live state of one pass. Polled while it runs."""

    he_upload_id: str
    ihc_upload_id: str
    state: NucleiState
    message: str | None = None
    #: 0-1 across all regions, so the bar means the same thing throughout.
    progress: float = 0.0
    started_at: str | None = None
    finished_at: str | None = None
    duration: float | None = None
    error: str | None = None


class NucleusOut(APIModel):
    """One nucleus. Geometry in the IHC slide's level-0 pixels."""

    id: int
    x: float
    y: float
    area_um2: float
    perimeter_um: float
    circularity: float
    eccentricity: float
    haematoxylin: float
    #: Outer ring first, then holes.
    rings: list[list[list[float]]]
    #: False when the nucleus sits in the border band and is excluded from counts.
    counted: bool


class FieldOut(APIModel):
    """One segmented square of slide, and what was found in it."""

    index: int
    x: int
    y: int
    #: Side in level-0 pixels - what was read off the slide.
    span: int
    #: Side in model pixels - what was segmented.
    size: int
    #: Microns per pixel the model actually saw, after resampling.
    mpp: float
    tissue: float
    detected: int
    counted: int
    #: Area the counted nuclei were counted over: the field less its border band.
    counted_mm2: float
    density_per_mm2: float


class RegionNuclei(APIModel):
    """One carried region's sample, and the estimate it supports."""

    rank: int = Field(description="1-based, matching step 10's crops and panels")
    index: int
    area_mm2: float

    fields: list[FieldOut]
    #: How many fields the region could have offered. The pair `sampled of
    #: available` is what stops the count being read as a census.
    fields_available: int
    sampled_mm2: float
    #: Share of the region actually looked at. Small on purpose - see the step's
    #: README - and printed next to every number derived from it.
    sampled_share: float

    detected: int
    counted: int
    density_per_mm2: float
    #: Spread of the per-field densities, as a coefficient of variation. A high
    #: value means the region is heterogeneous and the sample is thin, which is a
    #: statement about confidence rather than a fault.
    density_cv: float

    median_area_um2: float
    median_circularity: float


class StainReport(APIModel):
    """Which basis un-mixed the tile, and what the alternative would have said."""

    basis: str
    gain: float
    #: Macenko's estimate, computed even when it is not used, because its drift
    #: is the evidence for using the fixed basis on this panel.
    macenko_explained: float | None = None
    macenko_haematoxylin_drift_deg: float | None = None
    macenko_dab_drift_deg: float | None = None
    notes: list[str] = Field(default_factory=list)


class ComparisonOut(APIModel):
    """One field segmented every way, so the screen can put the counts together."""

    field_index: int
    region_rank: int
    instanseg_haematoxylin: int
    instanseg_rgb: int
    watershed_haematoxylin: int


class NucleiReport(APIModel):
    """Step 11's result: the nuclei, and how much of the region they came from."""

    he_upload_id: str
    ihc_upload_id: str
    marker: str | None = None
    state: NucleiState
    generated_at: str

    #: The stamp of the step 10 result these nuclei were segmented inside.
    #:
    #: Carried so this report can tell whether it is still describing something
    #: that exists. Step 10 re-runs whenever the region selection changes, and a
    #: segmentation measured in the previous selection is not a stale *number* -
    #: it is a measurement of a different piece of tissue.
    #:
    #: Optional because reports written before this field existed have no stamp;
    #: those are treated as stale, which costs one re-run and cannot mislead.
    alignment_generated_at: str | None = None

    model_name: str | None = None
    model_version: str | None = None
    model_licence: str | None = None
    model_mpp: float | None = None

    regions: list[RegionNuclei]
    stain: StainReport
    comparison: list[ComparisonOut] = Field(default_factory=list)

    #: Totals across the three regions.
    detected: int = 0
    counted: int = 0
    sampled_mm2: float = 0.0
    density_per_mm2: float = 0.0

    #: The cross-slide QC the guide asks for: serial sections of one block should
    #: land in the same neighbourhood, so this carries the same figure for every
    #: other marker of this case already segmented, keyed by letter.
    density_by_marker: dict[str, float] = Field(default_factory=dict)

    #: The same measurement on this case's **H&E** slide, inside the same regions.
    #:
    #: The reference the IHC figure is read against, and it is measured on every
    #: run rather than waiting for a second marker to exist - otherwise the first
    #: marker of a case has nothing to be compared with and the check that is
    #: supposed to catch a segmentation failure only starts working after one has
    #: already gone unnoticed. Cheap: a handful of fields on a slide that is
    #: already open in the pipeline.
    he_density_per_mm2: float | None = None
    he_median_area_um2: float | None = None
    #: How far the IHC density falls short of the H&E's, as a fraction. The guide
    #: names 30% as the level at which this is a segmentation failure rather than
    #: biology.
    density_shortfall: float | None = None

    seconds: float | None = None
    notes: list[str] = Field(default_factory=list)
