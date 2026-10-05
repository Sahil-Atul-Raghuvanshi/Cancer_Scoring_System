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
    """The five panels the step is meant to be read as.

    An enum rather than a free string so a typo is rejected at the API boundary with
    the valid names, rather than reaching the service and coming back as a conflict -
    which is not what a typo is.
    """

    MAP = "map"
    FLAT = "flat"
    SCORED = "scored"
    CONFIDENCE = "confidence"
    UNCERTAINTY = "uncertainty"


class TissueTypeUncertaintyParams(APIModel):
    """The uncertainty layer's settings, as the run actually applied them.

    Carried on the report rather than left in the server's configuration because they
    decide which tissue is drawn as undetermined, and a flagged region shown without the
    bar it cleared asks to be trusted rather than read. Every one of them is a tuning
    knob: they are not measured constants and the two area figures are the softest of
    them - see the module docstring of `step08.../uncertainty.py`.
    """

    sigma_um: float = Field(
        description=(
            "Neighbourhood scale in microns. A physical distance and not a count of "
            "windows, so a changed overlap moves how densely the tissue is sampled "
            "without moving which tissue gets flagged."
        )
    )
    area_ref_mm2: float = Field(
        description="Component area at which the extent term starts to rise"
    )
    area_max_mm2: float = Field(description="Component area at which the extent term saturates")
    fill_lo: float
    fill_hi: float
    min_shape_mm2: float = Field(
        description=(
            "Area a component needs before its shape is read at all. A duct cut across "
            "is a solid disc, so solidity means nothing below the size of a duct."
        )
    )
    min_shape_windows: int
    threshold: float = Field(description="Score at or above which a window is called undetermined")


class TissueTypeUncertainty(APIModel):
    """What the post-pass flagged, and on what evidence.

    Absent on a per-pixel pass, which never derives one, and on a class map written
    before the layer existed.
    """

    params: TissueTypeUncertaintyParams

    sigma_cells: float = Field(
        description="The neighbourhood in grid cells, after the conversion from microns"
    )

    windows: int = Field(description="In-situ windows the layer would not stand behind")
    share: float = Field(
        description=(
            "Share of the classified windows. Note the denominator is unchanged by this "
            "layer - a flagged window is still a classified one."
        )
    )
    area_mm2: float

    components: int = Field(description="8-connected components of in-situ windows on the grid")
    flagged_components: int = Field(
        description="Of those, how many the morphology term flagged on its own"
    )
    largest_component_mm2: float = Field(
        description=(
            "Largest in-situ component. The figure the extent term reads, and the first "
            "one to look at when a whole field comes back undetermined."
        )
    )
    mean_score: float = Field(
        description="Mean score over the in-situ windows, flagged or not"
    )


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

    arch: str = Field(
        default="resnet18",
        description="Architecture the checkpoint records: resnet18 or "
                    "concat_resnet18_mlp. Every served checkpoint answers once per "
                    "window; the per-pixel U-Net arm was removed.",
    )
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

    # --- the BEETLE option, which is not a checkpoint of ours ---------------
    #
    # Everything above is read from a manifest this project wrote. BEETLE has none: it is
    # a downloaded release, so it declares no initialisation, no held-out accuracy on our
    # data and no Dice against our labels, and those stay null rather than being filled
    # with numbers from somebody else's paper that were measured on somebody else's
    # split. What it does have is below.

    per_pixel: bool = Field(
        default=False,
        description=(
            "Whether this model answers once per pixel rather than once per window. "
            "True only for BEETLE, and it is the reason its report counts pixels."
        ),
    )
    classes: list[str] | None = Field(
        default=None,
        description=(
            "The class names this model emits, in code order - five for BEETLE, whose "
            "own label set is reported rather than collapsed. Null for a checkpoint of "
            "ours, which always emits the fixed three."
        ),
    )
    folds: list[int] | None = Field(
        default=None, description="Which of BEETLE's five released folds are averaged"
    )
    folds_available: int | None = Field(
        default=None, description="How many folds the release ships - five"
    )
    patch_px: int | None = Field(
        default=None, description="BEETLE's own patch side, from its plans.json"
    )
    citation: str | None = Field(
        default=None,
        description=(
            "How to cite this model, where it is somebody else's. The report's own "
            "`citation` covers our checkpoints' training sources."
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
    #: Which of step 7's options this pass ran on. Recorded because it, and not the
    #: field of view alone, decides which checkpoint serves: the two branches share
    #: all four geometries, so `windowUm` no longer names one head.
    branch: str = "h_channel"
    #: Step 7's committed field of view, in microns. Explicit rather than left implicit
    #: in `windowUm` - a run should record the choice it was made under, not a number
    #: somebody has to divide back out.
    field_of_view_um: float | None = None
    #: What the checkpoint declares it is shown: `haematoxylin` for a deconvolved
    #: density plane, `rgb_he` for the colour photograph. Read from the manifest that
    #: passed verification, so it is what was actually fed.
    input_channel: str | None = None
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
    # --- the ResNet branches' input contract, and null on BEETLE -------------
    #
    # These three are lifted from a checkpoint's verified manifest, so they exist only
    # where there is a manifest. BEETLE's preprocessing is division by 255 and nothing
    # else - no per-tile standardisation, no shape term, no polarity - so reporting them
    # at their identity values would assert an input contract nobody published. Null is
    # the honest value and the screen reads it as "not applicable" rather than "off".
    standardise: bool | None = None
    gamma: float | None = None
    invert_polarity: bool | None = None

    #: The uncertainty layer's settings. Null on the per-pixel option, which does not
    #: derive one - reporting them at their defaults there would assert a rule nothing
    #: applied, the same argument the three fields above make.
    uncertainty: TissueTypeUncertaintyParams | None = None

    span: int = Field(description="Level-0 pixels one window covers")
    stride: int = Field(description="Level-0 pixels the grid steps between windows")

    # --- the BEETLE option, and null on the ResNet branches -----------------

    per_pixel: bool = Field(
        default=False,
        description=(
            "Whether this pass answered per pixel. False for the two ResNet branches, "
            "which answer once per window; true for BEETLE. It decides how every count "
            "on the report should be read - `pixels` rather than `windows`."
        ),
    )
    pixel_classes: list[str] | None = Field(
        default=None,
        description=(
            "The class names this pass emitted, in code order. Five for BEETLE, whose "
            "own label set is reported rather than collapsed onto the three the ResNet "
            "branches share; null on those branches, where `classes` is the fixed three."
        ),
    )
    folds: list[int] | None = Field(
        default=None,
        description=(
            "Which of BEETLE's five released folds were averaged. One by default: a "
            "fold is a full pass over every patch. A single fold's spread is zero, "
            "which is one fold and not consensus."
        ),
    )
    patch_step: float | None = Field(
        default=None,
        description=(
            "BEETLE's sliding window step, as a fraction of its 512 px patch. 0.5 is "
            "the release's own inference default; a full step makes the patch grid "
            "visible as a checkerboard that flips the in-situ/invasive call across it."
        ),
    )
    patch_px: int | None = Field(
        default=None, description="BEETLE's own patch side - 512, from its plans.json"
    )
    mask_mpp: float | None = Field(
        default=None,
        description=(
            "Microns per pixel the per-pixel mask is stored at. BEETLE answers at 0.5, "
            "and a whole section at that spacing is a gigapixel, so the mask is reduced "
            "onto this canvas - area-averaging probabilities, never resampling labels."
        ),
    )
    patches: int | None = Field(
        default=None, description="Forward passes the pass actually made, across all folds"
    )

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


class TissueTypePaint(APIModel):
    """Everything a screen needs to draw this pass onto a picture of the slide.

    Step 8 is tens of minutes long, so it is watched rather than waited for, and a bar
    with a number beside it does not say *where* on the section the model has got to.
    This is what lets the progress screen paint each window onto a slide thumbnail as
    its class comes back - the pass drawing itself, in the same colours the finished
    class map uses.

    The geometry is the window grid's own, in **level-0 pixels**, because that is the
    one frame of reference that does not move when a pyramid level or a thumbnail size
    is chosen. A cell's core is `stride` wide and centred in its `span`, so a client
    turns `(row, col)` into a fraction of the slide the way `overlay.py` does it in
    reverse - and `stride`, not `span`, is what it draws: the cores partition the
    tissue, the spans overlap, and painting spans would overprint every neighbour at
    50% overlap.

    It appears once the grid exists, which is a second or two into a run - before that
    there is no grid and so nothing to paint.
    """

    cols: int = Field(description="Window columns across the whole canvas")
    rows: int = Field(description="Window rows down the whole canvas")
    span: int = Field(description="Level-0 pixels one window covers")
    stride: int = Field(
        description=(
            "Level-0 pixels between windows - and the side of the cell a window is "
            "responsible for, which is what a client paints"
        )
    )
    slide_width: int = Field(description="Level-0 slide width, what the fractions are of")
    slide_height: int = Field(description="Level-0 slide height")
    colours: list[str] = Field(
        description=(
            "Hex colour per class, indexed by class id - the same palette the finished "
            "map is drawn in, so the live paint and the result are one picture"
        )
    )
    labels: list[str] = Field(
        description=(
            "Plain-language name per class, indexed by class id. Sent with the palette "
            "rather than written into the client, for the same reason the palette is: a "
            "legend on the live paint and a legend on the finished report that could "
            "drift apart would eventually say two different things about one colour"
        )
    )

    per_pixel: bool = Field(
        default=False,
        description=(
            "Whether each painted window carries a pixel mask rather than one class. "
            "False on the ResNet branches, where a window is a flat colour; true on "
            "BEETLE, where `paintedMasks` holds the shapes it found inside each window "
            "and the client draws them instead of filling the cell."
        ),
    )
    mask_px: int = Field(
        default=0,
        description=(
            "Side of each window's mask in `paintedMasks`, in pixels - so a client "
            "knows the shape of the bytes it decoded without inferring it from their "
            "length. Zero when `perPixel` is false and no masks are sent."
        ),
    )


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

    paint: TissueTypePaint | None = Field(
        default=None,
        description=(
            "Geometry for painting this pass onto the slide, once its grid exists. "
            "Null before the grid is laid, and on a pass read back from disk - a "
            "finished pass has four rendered panels and needs no live paint."
        ),
    )
    painted_cells: list[int] = Field(
        default_factory=list,
        description=(
            "Windows classified since `paintedSince`, as flat triples: row, column, "
            "class, row, column, class. Flat rather than nested objects because this "
            "is polled for the length of the pass and every window travels through it "
            "exactly once - the whole feed for a slide is one pass over the grid. "
            "Empty unless `paintedSince` was asked for."
        ),
    )
    painted_masks: list[str] = Field(
        default_factory=list,
        description=(
            "One base64 pixel mask per window in `paintedCells`, same order, on the "
            "BEETLE option only. Each decodes to `paint.maskPx` squared bytes of class "
            "ids - raw ids rather than a PNG, because the client draws them straight "
            "into an ImageData and a PNG would be an encode and a decode to move a "
            "kilobyte. Empty on the ResNet branches, where a window has one class and "
            "`paintedCells` already carries it."
        ),
    )
    painted_cursor: int = Field(
        default=0,
        description=(
            "How far into the pass's paint log `paintedCells` reaches. Pass it back as "
            "the next `paintedSince` - it is not the same as `done`, which counts "
            "finished reads, and a response is capped so a client rejoining a pass "
            "part-way catches up over a few polls rather than in one large reply. It "
            "counts windows, so it indexes `paintedMasks` directly and `paintedCells` "
            "in threes."
        ),
    )


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

    windows: int = Field(
        description=(
            "Windows this class won. On the BEETLE option a window has no single class, "
            "so this is the number of windows whose *dominant* class it was and `pixels` "
            "is the figure the share is actually computed from."
        )
    )
    pixels: int | None = Field(
        default=None,
        description=(
            "Mask pixels this class claimed, on the per-pixel option only. Null on the "
            "ResNet branches, which have no pixels to count."
        ),
    )
    share: float = Field(
        description=(
            "Share of the classified windows, or - on the per-pixel option - share of "
            "the tissue pixels. Glass is out of that denominator: a slide with more "
            "empty space around its section must not report a smaller tumour content "
            "for that reason alone."
        )
    )
    area_mm2: float = Field(
        description=(
            "Unique slide area. Once per stride-sized cell on the ResNet branches, so it "
            "does not move when the overlap does; on the per-pixel option it is the mask "
            "pixels times their own area, which is a real area rather than a cell count."
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
    batches: int = Field(
        description=(
            "Forward passes, each of up to `batchSize` windows. On the per-pixel option "
            "a window is many forward passes rather than a fraction of one, so "
            "`params.patches` is the figure that describes the work there."
        )
    )
    seconds: float

    mask_height: int | None = Field(
        default=None, description="Rows in the per-pixel mask. Null on the ResNet branches"
    )
    mask_width: int | None = Field(default=None, description="Columns in the per-pixel mask")
    tissue_pixels: int | None = Field(
        default=None,
        description=(
            "Mask pixels BEETLE called tissue - the denominator the shares are over, "
            "which is every class but `unannotated`."
        ),
    )


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

    uncertainty: TissueTypeUncertainty | None = Field(
        default=None,
        description=(
            "What the post-pass would not stand behind. Null on the per-pixel option and "
            "on a class map written before the layer existed. It can only ever qualify "
            "in-situ windows, which Rule 5 excludes anyway, so `tumourContent` and "
            "`scoredMm2` are identical with it and without it."
        ),
    )

    model: TissueTypeModelInfo
    caveats: list[TissueTypeCaveat] = Field(default_factory=list)

    notes: list[str] = Field(
        default_factory=list,
        description="Caveats that belong on screen next to the numbers, not in a docstring",
    )
    citation: str
