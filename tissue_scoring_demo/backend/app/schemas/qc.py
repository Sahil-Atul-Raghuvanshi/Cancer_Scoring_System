"""Schemas for step 2 - quality control.

Two things are reported separately on purpose, and the split runs through every
model here:

  the decision      GrandQC's artefact classes and the tissue area they remove.
  the explanation   classical metrics, always as a ratio against *this slide's*
                    own clean tissue, never against a threshold baked into the
                    code. These metrics are not comparable between slides or
                    between resolutions, so an absolute number would be a
                    number-shaped lie.

Nothing here reports an IHC score. Steps 3-16 are not built, so there is no
score to move; what the QC on/off comparison reports is the analysable tissue
area, which is the denominator that score would eventually be divided by.
"""

from pydantic import Field

from app.schemas.common import APIModel

# --- capability --------------------------------------------------------------


class QCModelInfo(APIModel):
    """One GrandQC checkpoint: what it is and whether it is on disk."""

    role: str = Field(description="'tissue' or 'artefact'")
    name: str = Field(description="Checkpoint filename GrandQC publishes")
    found: bool
    path: str | None = None
    mpp: float | None = Field(default=None, description="Resolution it was trained at")
    magnification: str | None = Field(default=None, description="The same, as a magnification")
    classes: int | None = Field(
        default=None,
        description="Output channels, read off the checkpoint once it has been loaded",
    )


class QCDownload(APIModel):
    """Where to get a checkpoint that is missing."""

    label: str
    url: str
    files: list[str]
    target_dir: str


class QCCapability(APIModel):
    """Whether step 2 can run, and precisely what is missing when it cannot."""

    ready: bool = Field(description="Both GrandQC models present and torch importable")
    mode: str = Field(
        description=(
            "'full' - both GrandQC models plus the classical metrics. 'degraded' - the "
            "artefact model runs, but the tissue checkpoint is missing so tissue comes from "
            "a saturation/Otsu threshold that mistakes pen marks for tissue. "
            "'unavailable' - nothing can run."
        )
    )
    reason: str

    torch_installed: bool
    torch_version: str | None = None
    smp_version: str | None = None
    device: str = Field(description="Device inference would run on: cuda, mps or cpu")
    device_name: str | None = None

    features_available: bool
    features_problem: str | None = None

    models_root: str | None = None
    searched_paths: list[str] = Field(default_factory=list)
    models: list[QCModelInfo] = Field(default_factory=list)
    available_model_mpps: list[float] = Field(default_factory=list)
    default_model_mpp: float
    downloads: list[QCDownload] = Field(default_factory=list)

    citation: str
    licence_note: str


# --- run state ---------------------------------------------------------------


class QCParams(APIModel):
    """The settings a run actually used."""

    mode: str
    model_mpp: float
    model_label: str = Field(description="The artefact model's magnification, e.g. '7x'")
    tissue_model_mpp: float
    patch_size: int
    read_from_level_0: bool
    collect_features: bool


class QCRun(APIModel):
    """Progress of one QC job. A whole-slide run on CPU takes minutes."""

    upload_id: str
    state: str = Field(
        description=(
            "'idle', 'queued', 'running', 'ready', 'cancelled' or 'failed'. "
            "'cancelled' is its own state rather than a failure - a decision someone "
            "made, not a problem to report."
        )
    )
    phase: str | None = Field(
        default=None, description="'tissue', 'artefacts', 'rendering' or 'summarising'"
    )
    message: str | None = None
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    done: int = 0
    total: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    error: str | None = None
    params: QCParams | None = None


# --- report ------------------------------------------------------------------


class QCClassShare(APIModel):
    """How much of the tissue one class accounts for."""

    id: int
    key: str
    label: str
    blurb: str
    colour: str
    is_artefact: bool
    pixels: int
    share_of_tissue: float = Field(
        description="Fraction of tissue pixels, where tissue excludes glass entirely"
    )
    area_mm2: float | None = None
    explained_by: str | None = Field(
        default=None,
        description="Metric key the explanation panel leads with. Presentation only.",
    )


class QCTissue(APIModel):
    """The pixel accounting behind every percentage in the report."""

    tissue_pixels: int = Field(description="Clean tissue plus artefacts; glass excluded")
    clean_pixels: int
    artefact_pixels: int
    background_pixels: int
    unanalysed_pixels: int = Field(
        description="Right/bottom margin narrower than one patch, so never visited"
    )

    tissue_area_mm2: float | None = None
    clean_area_mm2: float | None = None
    rejected_area_mm2: float | None = None

    rejected_share: float = Field(description="Artefact pixels over tissue pixels")
    tissue_share_of_slide: float
    tissue_source: str = Field(
        description="'grandqc' or 'otsu' - the fallback must never be presented as the model"
    )


class QCGate(APIModel):
    """The QC on/off comparison, over the quantity that actually exists today."""

    qc_off_pixels: int
    qc_on_pixels: int
    qc_off_area_mm2: float | None = None
    qc_on_area_mm2: float | None = None
    removed_area_mm2: float | None = None
    removed_share: float
    note: str = Field(
        description="Says in words what this compares, and what it does not compare"
    )


class QCMetricSummary(APIModel):
    """One classical metric, per artefact class, against clean tissue."""

    key: str
    label: str
    unit: str
    description: str
    low_is_bad: bool
    measured_at_mpp: float
    clean_mean: float | None = None
    by_class: dict[str, float] = Field(default_factory=dict)
    ratio_to_clean: dict[str, float] = Field(
        default_factory=dict,
        description="Class mean over clean-tissue mean. 1.0 means indistinguishable.",
    )
    sample_sizes: dict[str, int] = Field(
        default_factory=dict, description="Cells measured per class - guards over-reading"
    )


class QCGridMeta(APIModel):
    """Shape of the two grids: forward passes, and measurement cells.

    They are not the same thing. A patch is one run of the model. A cell is a
    sub-block of a patch, and it is what the metrics are measured on - a whole
    patch covers most of a millimetre, and almost nothing is wrong with a whole
    millimetre of tissue, so classing patches would leave the explanation panel
    with nothing to say about folds, pen or blur.
    """

    cols: int = Field(description="Measurement cells across")
    rows: int = Field(description="Measurement cells down")
    patch_cols: int = Field(description="Model patches across - one forward pass each")
    patch_rows: int
    blocks_per_patch: int = Field(description="Cells per patch edge")

    patch_size: int
    cell_mpp: float
    cell_extent_px: int = Field(description="One cell's extent in level-0 pixels")
    cell_extent_um: float

    measured_cells: int = Field(description="Cells that contained tissue and were measured")
    skipped_cells: int = Field(description="Cells with no tissue in them")
    patches_inferred: int = Field(description="Patches the artefact model actually ran on")
    patches_skipped: int = Field(description="Patches the tissue pass excused as glass")


class QCReport(APIModel):
    """Everything step 2 produced for one slide."""

    upload_id: str
    filename: str
    generated_at: str

    params: QCParams
    run: QCRun

    classes: list[QCClassShare]
    tissue: QCTissue
    gate: QCGate
    metrics: list[QCMetricSummary]
    grid: QCGridMeta
    models: list[QCModelInfo]

    notes: list[str] = Field(
        default_factory=list,
        description="Caveats that belong on screen next to the numbers, not in a docstring",
    )
    citation: str


# --- grid and region --------------------------------------------------------


class QCGridCell(APIModel):
    """One patch of the grid."""

    col: int
    row: int
    dominant: str | None = Field(default=None, description="Class key that dominates the tissue")
    tissue_share: float
    metrics: dict[str, float] | None = None


class QCGrid(APIModel):
    """The full patch grid, for drawing heatmaps client-side."""

    upload_id: str
    grid: QCGridMeta
    metric_keys: list[str]
    cells: list[QCGridCell]


class QCRegionMetric(APIModel):
    """One metric for one inspected region, next to the slide's own baseline."""

    key: str
    label: str
    unit: str
    description: str
    low_is_bad: bool
    value: float
    clean_mean: float | None = None
    ratio_to_clean: float | None = None


class QCRegionExplain(APIModel):
    """Why one region was called what it was called."""

    upload_id: str
    x: int
    y: int
    size_px: int = Field(description="Region extent in level-0 pixels")
    measured_at_mpp: float
    dominant: str | None = None
    class_shares: dict[str, float] = Field(default_factory=dict)
    metrics: list[QCRegionMetric] = Field(default_factory=list)
    verdict: str = Field(description="One sentence, generated from the numbers above")
