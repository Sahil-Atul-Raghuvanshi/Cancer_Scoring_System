"""Schemas for step 15 - the translation layer between the machine and the reader.

Two things live here and they are deliberately separate types. `BinCount` is
per-cell 0 / 1+ / 2+ / 3+: internal machinery that counts positives and feeds an
H-score. `BandTableRow` is the 0-2 scale OncoStem actually reports. They run on
different scales and are read by different people, and a single type spanning
both is how the two get conflated.
"""

from pydantic import Field

from app.schemas.common import APIModel


class BinCount(APIModel):
    """One of the four per-cell levels. Internal, not the reported scale."""

    bin: int
    label: str
    count: int
    share: float
    #: The optical-density range this bin covers. Null above on the top bin.
    od_from: float
    od_to: float | None = None


class BandTableRow(APIModel):
    """One row of the antibody's optical-density-to-band table."""

    od_below: float | None
    band: float
    label: str


class CutLine(APIModel):
    """A cut drawn on the histogram, and what it separates."""

    od: float
    label: str
    separates: str


class SchemeComparison(APIModel):
    """What a rejected threshold scheme would have reported on the same cells."""

    scheme: str
    label: str
    od_cuts: list[float]
    counts: list[int]
    positive_share: float
    #: Points of percent-positive away from the answer this pipeline reports.
    delta_points: float
    note: str


class BinningParams(APIModel):
    marker: str
    marker_name: str
    #: Always "absolute". Recorded rather than assumed, because the whole
    #: argument of this step is that it is not per-slide percentiles.
    scheme: str
    od_cuts: list[float]
    second_measure: str
    second_min: float
    partial_rule: str
    cuts_version: int
    cuts_provisional: bool
    cuts_source: str


class ODHistogramBin(APIModel):
    lower: float
    upper: float
    count: int


class BinningReport(APIModel):
    """Step 15's result: cells sorted into levels, with the cuts shown."""

    he_upload_id: str
    ihc_upload_id: str
    marker: str
    marker_name: str

    generated_at: str
    measured_at: str | None = None

    params: BinningParams
    cells: int = 0
    bins: list[BinCount] = Field(default_factory=list)

    #: Positivity is two conditions, so the count of cells over the density cut
    #: and the count over both are different numbers and both are shown.
    over_od_cut: int = 0
    positive_cells: float = 0.0
    positive_share: float = 0.0

    histogram: list[ODHistogramBin] = Field(default_factory=list)
    cut_lines: list[CutLine] = Field(default_factory=list)
    #: Where the valley between the unstained and stained humps actually is, if
    #: there are two humps. The check on whether a cut landed in the right place.
    valley_od: float | None = None

    band_table: list[BandTableRow] = Field(default_factory=list)
    comparisons: list[SchemeComparison] = Field(default_factory=list)

    notes: list[str] = Field(default_factory=list)
