"""Schemas for step 12 - which cells belong in the denominator."""

from pydantic import Field

from app.schemas.common import APIModel


class TypingRules(APIModel):
    """The thresholds the answer rests on. Judgements, not measurements.

    Sent back with every result, and accepted on the request, so the screen can
    move one and watch the mix change - which is the honest way to present a
    parameter nobody has fitted.
    """

    lymphocyte_max_area_um2: float = 35.0
    lymphocyte_min_circularity: float = 0.72
    lymphocyte_min_darkness: float = 1.05
    spindle_min_eccentricity: float = 0.85
    spindle_max_area_um2: float = 90.0


class TypeCount(APIModel):
    """One class, and how much of the mix it is."""

    type: str
    label: str
    count: int
    share: float
    #: `r,g,b` as the overlay draws it, so the legend and the picture agree.
    colour: tuple[int, int, int]


class RegionTyping(APIModel):
    """One carried region's cell mix."""

    rank: int
    area_mm2: float
    counted: int
    counts: list[TypeCount]
    #: Cells per mm2 of the tumour class alone - the density that matters for a
    #: denominator, as against step 11's all-cells figure.
    tumour_per_mm2: float
    median_tumour_area_um2: float
    median_lymphocyte_area_um2: float


class SensitivityPoint(APIModel):
    value: float
    tumour_share: float


class SensitivitySweep(APIModel):
    """What the tumour share would have been at other defensible thresholds."""

    parameter: str
    baseline: float
    points: list[SensitivityPoint]
    #: Largest minus smallest tumour share across the sweep. This is the number
    #: that says whether the threshold is doing the work or the data is.
    swing: float


class CellTypingReport(APIModel):
    """Step 12's result: the mix, and how much of it is the thresholds' doing."""

    he_upload_id: str
    ihc_upload_id: str
    marker: str | None = None
    generated_at: str
    #: Step 11's stamp. A re-segmentation is different nuclei, so a typing of the
    #: old ones is describing cells that no longer exist.
    nuclei_generated_at: str | None = None

    rules: TypingRules
    regions: list[RegionTyping]

    counted: int = 0
    counts: list[TypeCount] = Field(default_factory=list)
    tumour_share: float = 0.0

    sensitivity: list[SensitivitySweep] = Field(default_factory=list)

    #: Whether the nuclei underneath are good enough for this to mean anything.
    #:
    #: Cell typing is size and shape arithmetic, so it inherits every fault in
    #: the segmentation it reads. On a slide whose nuclei came out far smaller
    #: than an epithelial nucleus can be, the size rules are sorting debris and
    #: the mix is not a mix of cells.
    trustworthy: bool = True
    trust_reason: str | None = None
    median_area_um2: float = 0.0

    notes: list[str] = Field(default_factory=list)
