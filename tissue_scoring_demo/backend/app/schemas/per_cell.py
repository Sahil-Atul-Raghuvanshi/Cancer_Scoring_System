"""Schemas for step 14 - the first step that measures anything.

Two numbers per cell: the DAB optical density in its compartment, always, and a
second number whose *meaning* depends on the antibody. `second_measure` names
which it is, and it is carried on the report and on every point rather than left
for the client to infer from the marker letter - a scatter whose y-axis label is
derived in the browser is a second place for the fork to be got wrong.
"""

from typing import Literal

from pydantic import Field

from app.schemas.common import APIModel

#: Which second number this marker's cells carry. Resolved from the compartment,
#: which is `app.panel`'s decision.
SecondMeasure = Literal["ring_completeness", "stained_fraction"]


class MeasurementParams(APIModel):
    """What the measurement was made with, so a number can be reproduced."""

    #: "mean". Fixed for every marker; reported so it cannot change silently
    #: between two markers whose intensities are then compared.
    intensity_statistic: str

    #: The compartment this was measured in, and how wide it was.
    compartment: Literal["membrane", "cytoplasm"]
    expansion_um: float
    ring_um: float | None = None

    #: The optical density a bin or a pixel has to clear to count as stained.
    #: Read from this antibody's own cut-point set, never shared across markers.
    positivity_od: float
    #: 36 bins of 10 degrees. Null for a cytoplasmic marker, which has no ring.
    ring_bins: int | None = None

    #: The cut on the second number - ring completeness or stained fraction,
    #: whichever this marker has. Carried here so the scatter's decision boundary
    #: is the server's number rather than one the browser hard-codes: a boundary
    #: drawn at a different threshold from the one being applied is a picture of
    #: a decision nobody is making.
    second_min: float

    cuts_version: int
    #: True until the cut points have been fitted against reader scores. It
    #: travels with every number produced under them.
    cuts_provisional: bool = True


class CellPoint(APIModel):
    """One cell, as the scatter draws it: a dot you can click through to a crop."""

    cell_id: int
    region_rank: int
    field_index: int
    #: Level-0 slide coordinates of the nucleus centroid.
    x: float
    y: float

    #: x-axis: mean DAB optical density in this cell's compartment.
    intensity_od: float
    #: Reported, never used in a score.
    max_od: float
    #: y-axis: whichever second number this marker gets.
    second: float
    second_measure: SecondMeasure

    pixels: int
    area_um2: float
    #: Membrane only: bins that held enough pixels to be judged, out of 36.
    occupied_bins: int | None = None


class FieldMeasurement(APIModel):
    region_rank: int
    index: int
    cells: int
    mean_od: float
    mean_second: float


class RegionMeasurement(APIModel):
    rank: int
    #: Area of the whole invasive region this field sample was drawn from. Step
    #: 16 needs it for the area-weighted average.
    area_mm2: float
    cells: int
    mean_od: float
    median_od: float
    mean_second: float
    fields: list[FieldMeasurement] = Field(default_factory=list)


class ODHistogramBin(APIModel):
    """One bar of the DAB optical-density histogram the cut lines are drawn on."""

    lower: float
    upper: float
    count: int


class PerCellReport(APIModel):
    """Step 14's result: one measurement row per tumour cell."""

    he_upload_id: str
    ihc_upload_id: str
    marker: str
    marker_name: str
    second_measure: SecondMeasure

    generated_at: str
    nuclei_generated_at: str | None = None

    params: MeasurementParams
    regions: list[RegionMeasurement] = Field(default_factory=list)

    cells: int = 0
    mean_od: float = 0.0
    median_od: float = 0.0
    mean_second: float = 0.0

    #: The histogram the cut lines sit on. Two humps - unstained and stained -
    #: are what a well-placed cut sits between, and this is what lets a viewer
    #: see whether it does.
    histogram: list[ODHistogramBin] = Field(default_factory=list)

    #: A sample of cells for the scatter. Sampled rather than complete because a
    #: hundred thousand dots is not a picture; `cells` above is the real count
    #: and `sampled` says how many of them are drawn.
    points: list[CellPoint] = Field(default_factory=list)
    sampled: int = 0

    #: Membrane markers only: cells whose ring was too crowded by neighbours for
    #: completeness to mean anything. Reported, not dropped.
    crowded_cells: int | None = None

    notes: list[str] = Field(default_factory=list)
