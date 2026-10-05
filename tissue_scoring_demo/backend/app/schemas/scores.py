"""Schemas for step 16 - the deliverable, and everything shown beside it.

`percent` and `intensity` are the contract. They are the first two fields of
`MarkerScoreOut` for that reason, and every other field on it is context, a
check, or the field's standard vocabulary. A client that renders only those two
has rendered the whole deliverable.
"""

from pydantic import Field

from app.schemas.common import APIModel


class RegionScoreOut(APIModel):
    """One invasive region's own answer, before the regions are combined."""

    rank: int
    area_mm2: float
    cells: int
    positive_cells: float
    percent_raw: float
    intensity_raw: float


class HeterogeneityTile(APIModel):
    """One sampled field, and the local percent positive inside it.

    What makes "50 %" legible as an average over a very non-uniform field
    rather than a property the slide has evenly.
    """

    region_rank: int
    field_index: int
    x: float
    y: float
    span: int
    cells: int
    percent: float


class CascadeStep(APIModel):
    """One line of the arithmetic, written out with the actual numbers in it.

    The screen shows the whole cascade at once - counts, then the formula with
    the numbers substituted, then the rounding, then the pair - so there is no
    hidden arithmetic between the cells and the answer.
    """

    label: str
    expression: str
    value: str


class MarkerScoreOut(APIModel):
    """One antibody's numbers. The first two are what OncoStem receives."""

    # --- the deliverable ----------------------------------------------------
    marker: str
    marker_name: str
    percent: int
    intensity: float
    intensity_label: str

    # --- the same two before rounding, so the step can be shown --------------
    percent_raw: float
    intensity_raw: float

    #: The pooled figures - every measured cell counted once, ignoring which
    #: region it came from. Reported beside the area-weighted ones because the
    #: gap between them is the size of the sampling imbalance, not a rounding.
    percent_pooled: float = 0.0
    intensity_pooled: float = 0.0

    compartment: str
    second_measure: str

    cells: int
    positive_cells: float
    bin_counts: list[int]
    bin_shares: list[float]

    # --- reference vocabulary, never the deliverable ------------------------
    h_score: float
    allred_proportion: int
    allred_intensity: int
    allred_total: int
    her2_call: str | None = None
    her2_note: str = ""

    # --- the open questions, answered both ways -----------------------------
    percent_area_weighted: float
    percent_plain_mean: float
    averaging_gap_points: float
    averaging_used: str = "area_weighted"

    #: Which cells `percent` and `intensity` are counted over - "all" in the region or
    #: "tumour" only (P-04) - and the tumour-only figures beside them when the reported
    #: ones are all-cells. A sensitivity, not an alternative answer: it rests on a cell
    #: typing that fails its own check, which is why it is not the reported figure.
    population: str = "tumour"
    percent_tumour_only: int | None = None
    intensity_tumour_only: float | None = None
    cells_tumour_only: int | None = None
    partial_rule: str = "count"
    percent_by_partial_rule: dict[str, int] = Field(default_factory=dict)

    regions: list[RegionScoreOut] = Field(default_factory=list)
    heterogeneity: list[HeterogeneityTile] = Field(default_factory=list)
    cascade: list[CascadeStep] = Field(default_factory=list)

    od_cuts: list[float] = Field(default_factory=list)
    second_min: float = 0.0
    cuts_provisional: bool = True

    #: Things a reader has to know before using these numbers: a provisional cut
    #: point, a machine-confirmed alignment, a nuclei shortfall.
    caveats: list[str] = Field(default_factory=list)

    #: What the two numbers are, in one machine-readable word (P-17):
    #:   `measured`           nothing below undermines them;
    #:   `provisional`        computed as designed, but a step a person should have
    #:                        checked was not, or a calibration is not fitted yet;
    #:   `not_a_measurement`  a condition under which the numbers carry no information -
    #:                        too few cells, an alignment that failed its gate or cannot
    #:                        be read, a cell typing that failed its own check.
    #: Caveats explain; this decides. Before it, a row produced with every check skipped
    #: read exactly like a fully checked one.
    status: str = "provisional"
    status_reasons: list[str] = Field(default_factory=list)


class ScoreReport(APIModel):
    """Step 16's result for one (H&E, IHC) pair - that is, for one marker."""

    he_upload_id: str
    ihc_upload_id: str
    generated_at: str
    measured_at: str | None = None

    score: MarkerScoreOut
    notes: list[str] = Field(default_factory=list)
    #: What this score was made with - see `app.core.provenance` (P-15). Null on a report
    #: written before stamping existed.
    provenance: dict | None = None


class CaseScoreRow(APIModel):
    """One marker's row in the ten-number grid OncoStem actually receives."""

    marker: str
    marker_name: str
    percent: int | None = None
    intensity: float | None = None
    intensity_label: str = ""
    cells: int = 0
    state: str = "missing"
    detail: str = ""
    #: The score's own status - `measured`, `provisional` or `not_a_measurement` - when
    #: it was scored (P-17). `state` says whether a number exists; this says whether it
    #: is one.
    status: str | None = None


class CaseScoreReport(APIModel):
    """The whole case: five markers, ten numbers, as the grid they are sent as."""

    case_id: str
    generated_at: str
    rows: list[CaseScoreRow] = Field(default_factory=list)
    complete: bool = False
    notes: list[str] = Field(default_factory=list)
