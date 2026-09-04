"""Schemas for step 8 - tissue-type segmentation.

Step 8 reports four things, and they are in this order because that is the order they
have to be read in:

  the model     which checkpoint ran, what it was trained on, **what licence that
                leaves the result under**, and what it scored on held-out data. First,
                because every number below is a claim made by a specific 45 MB file
                and a reader who does not know which file cannot check anything.
  the classes   how the three classes divide the tissue, in windows and in mm2. The
                step's product.
  the geometry  the window grid the model actually ran on, and how it relates to step
                7's tiles, since the two are deliberately different.
  the caveats   what this model cannot do, from its own manifest rather than from
                anyone's memory. The in-situ class is the one that matters and it is
                the one with almost no human-labelled test data behind it.

**No probability array is on the wire.** The report carries counts, areas, shares and
a confidence summary; the full per-window probability grid is step 10's input and stays
server-side, because it is `rows x cols x 3` floats over a whole slide and a browser
has no use for it that a picture does not serve better.

See `pipeline/step08_tissue_type_segmentation/`.
"""

from enum import Enum

from pydantic import Field

from app.schemas.common import APIModel


class TissueTypePanel(str, Enum):
    """The four panels the step is meant to be read as.

    An enum rather than a free string so a typo is rejected at the API boundary with
    the valid names, rather than reaching the service and coming back as a conflict -
    which is not what a typo is.
    """

    MAP = "map"
    FLAT = "flat"
    SCORED = "scored"
    CONFIDENCE = "confidence"


class TissueTypeModelInfo(APIModel):
    """One published checkpoint, described from its manifest.

    Listed even when it is not the one running, so the choice is visible: the
    difference between the two checkpoints in this project is not accuracy, it is
    whether the result may be sold.
    """

    name: str
    found: bool
    path: str | None = None
    selected: bool = Field(description="Whether this is the checkpoint that will run")

    init: str | None = Field(default=None, description="Backbone initialisation: imagenet or simclr")
    tile_px: int | None = Field(default=None, description="Window side the model was fitted at")
    mpp: float | None = Field(default=None, description="Resolution it was fitted at")
    standardise: bool | None = Field(
        default=None,
        description=(
            "Whether each window is divided by its own 99th-percentile density before "
            "the clip. The property that makes an H&E-fitted model's input distribution "
            "contain the IHC one it is served here."
        ),
    )
    created: str | None = None
    sha256: str | None = None
    bytes: int | None = None

    licence_track: str = Field(
        description=(
            "'permissive', 'research-only' or 'unknown', read from the manifest. "
            "Restrictive wins: a checkpoint fitted on any non-commercial source "
            "inherits that term, and so does every number computed from it."
        )
    )
    licences: dict[str, str] = Field(
        default_factory=dict, description="Every upstream term, so the report can name which one binds"
    )
    training_source: str | None = None

    held_out_accuracy: float | None = None
    dice_invasive: float | None = None
    dice_non_invasive: float | None = None
    non_invasive_test_tiles: int | None = Field(
        default=None,
        description=(
            "Human-labelled in-situ tiles in the held-out set. A Dice figure ships with "
            "its tile count or it does not ship, and on BCSS alone this number is nine."
        ),
    )

    problem: str | None = None


class TissueTypeCapability(APIModel):
    """What step 8 can do right now, and what is missing if it cannot."""

    ready: bool = Field(description="A verified checkpoint is on disk and torch can run it")
    reason: str

    torch_installed: bool
    torch_version: str | None = None
    device: str = Field(description="Device inference would run on: cuda, mps or cpu")
    device_name: str | None = None
    threads: int | None = Field(default=None, description="CPU threads torch will use")

    models_root: str | None = None
    models: list[TissueTypeModelInfo] = Field(default_factory=list)
    default_model: str

    licence_track: str = Field(description="The selected checkpoint's effective licence track")
    licence_note: str

    citation: str


class TissueTypeParams(APIModel):
    """The settings a run actually used, resolved rather than requested.

    The geometry is the checkpoint's and not this app's, which is the one thing to
    notice here: `windowPx` and `mpp` come out of the manifest, so a different
    checkpoint changes them and the report says so rather than quietly rescaling.
    """

    model: str
    model_sha256: str | None = None
    licence_track: str

    window_px: int = Field(description="Window side in its own pixels. The checkpoint's")
    window_um: float = Field(
        description=(
            "The window's field of view in microns - the physical scale the model reads "
            "architecture at. 112 um is about eleven nuclei across, which is what it "
            "takes to see whether abnormal cells are still inside a duct."
        )
    )
    mpp: float = Field(description="Resolution the windows are read at. The checkpoint's")
    overlap: float = Field(
        description=(
            "Fraction of its field of view a window shares with its neighbour. Costs "
            "compute quadratically and every window is a forward pass."
        )
    )
    standardise: bool
    gamma: float
    invert_polarity: bool

    span: int = Field(description="Level-0 pixels one window covers")
    stride: int = Field(description="Level-0 pixels the grid steps between windows")

    tile_overlap: float = Field(description="Step 7's overlap, which set the tiles this ran over")
    tissue_threshold: int = Field(description="The step 3 cut behind step 7's index")
    tissue_threshold_source: str
    qc_gated: bool
    qc_source: str | None = None

    min_tissue_share: float = Field(
        default=0.0,
        description=(
            "Share of a window that must be tissue before the model is shown it. This "
            "step's own gate, not step 7's: step 7's is on a 512 px tile and is set low "
            "on purpose, so a 224 px window whose centre sits in a kept tile can itself "
            "be almost entirely glass - which is where a false in-situ rim around the "
            "section came from."
        ),
    )
    block_windows: int
    batch_size: int
    device: str


class TissueTypeRun(APIModel):
    """Live state of one pass, or the state of the cached one."""

    upload_id: str
    state: str = Field(
        description=(
            "'idle', 'queued', 'running', 'ready', 'cancelled' or 'failed'. "
            "'cancelled' is its own state rather than a failure: it is a decision "
            "someone made, and collapsing the two would make the screen apologise for "
            "what the viewer just asked for."
        )
    )
    phase: str | None = Field(
        default=None, description="'grid', 'classifying' or 'rendering' while running"
    )
    message: str | None = None
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    done: int = Field(default=0, description="Windows classified so far")
    total: int = Field(default=0, description="Windows the grid marked")
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    error: str | None = None
    params: TissueTypeParams | None = None


class TissueTypeClass(APIModel):
    """One class, and how much of the tissue it claimed.

    `label` and `blurb` are the plain-language pair; `meaning` is the rigorous version
    for the notes. The screen has to read for someone with basic knowledge, so
    "in-situ epithelium" is not the default wording.
    """

    id: int
    key: str
    label: str
    blurb: str
    meaning: str
    colour: str
    scored: bool = Field(description="Whether this class enters the final score. Exactly one does")

    windows: int
    share: float = Field(description="Share of the classified windows")
    area_mm2: float = Field(
        description=(
            "Unique slide area, counted once per stride-sized cell rather than once per "
            "window - so it does not move when the overlap does."
        )
    )
    mean_confidence: float = Field(description="Mean top-class probability where this class won")


class TissueTypeGrid(APIModel):
    """The window grid, and how it relates to step 7's tiles.

    Both are reported because they are deliberately different grids doing different
    jobs, and a reader who thinks step 8 ran once per step-7 tile will misread every
    count on the screen.
    """

    cols: int
    rows: int
    every: int = Field(description="Windows in the grid over the whole canvas")
    classified: int = Field(description="Windows whose centre landed on a tile step 7 kept")

    tiles_kept: int = Field(description="Step 7's kept tiles - the region this ran over")
    tile_px: int = Field(description="Step 7's tile side, which is not this step's window")
    tile_um: float

    min_tissue_share: float = Field(
        default=0.0, description="This step's own window-level tissue gate"
    )
    gated_out: int = Field(
        default=0,
        description=(
            "Windows whose centre was on tissue step 7 kept but which held too little "
            "tissue of their own. Reported rather than absorbed: 'the model did not look "
            "there' and 'there was nothing there to look at' are different facts."
        ),
    )
    blocks_read: int = Field(description="Pyramid reads it took, one per block of windows")
    batches: int = Field(description="Forward passes, each of up to `batchSize` windows")
    seconds: float


class TissueTypeCaveat(APIModel):
    """Something this model cannot do, taken from its own manifest.

    Carried as structured data rather than prose because the important one is a
    *number* - how many human-labelled in-situ tiles stood behind the in-situ score -
    and a sentence can be skimmed past in a way a labelled figure cannot.
    """

    key: str
    severity: str = Field(description="'blocking', 'warning' or 'note'")
    headline: str
    detail: str


class TissueTypeReport(APIModel):
    """Everything step 8 produced for one slide."""

    upload_id: str
    filename: str
    generated_at: str

    params: TissueTypeParams
    run: TissueTypeRun

    classes: list[TissueTypeClass]
    grid: TissueTypeGrid

    tumour_content: float = Field(
        description=(
            "Share of the classified tissue that is invasive carcinoma - the only class "
            "the score is measured on. Not 'how much tumour': in-situ disease and fat "
            "are already out of this denominator."
        )
    )
    scored_mm2: float
    tissue_mm2: float = Field(description="Area step 7's kept tiles covered")
    mean_confidence: float

    model: TissueTypeModelInfo
    caveats: list[TissueTypeCaveat] = Field(default_factory=list)

    notes: list[str] = Field(
        default_factory=list,
        description="Caveats that belong on screen next to the numbers, not in a docstring",
    )
    citation: str
