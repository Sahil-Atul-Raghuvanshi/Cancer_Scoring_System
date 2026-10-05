"""Step 7 - tiling, orchestrated.

Not a background job: one pass over a few tens of thousands of grid cells against
a mask step 3 has already cached, so it answers in the request that asked for it.

**Its input is step 3's mask and step 2's artefact map, and it asks the services
that own them.** The grid is geometry over the slide's own dimensions, and the two
gates are questions about pixels two earlier steps have already decided about - so
this service re-derives neither. That is the same argument
`tissue_service.footprint` was written for, and here it has an extra edge: the
whole justification for tiling running at this position in the pipeline is that
steps 2 and 3 have already narrowed the slide down, and a step that recomputed
their answers would not be depending on them, it would be duplicating them.

**Where the haematoxylin comes from.** The catalogue calls this step's input the
haematoxylin-channel tissue region, and the sample panel honours that literally:
it draws one tile of this index as step 6's H channel, on step 6's own scale and
through step 6's own ramp. The guide is explicit that train and inference must
call the same deconvolution function or the IHC slides will score nothing like
the H&E, and this is the first place downstream of the fork where that rule can be
kept or broken. It is kept by importing `separate` and `RUIFROK_HDAB` from step 6
rather than by having a second copy - see `_sample_png`, which also explains why
it does not go through step 6's *service*.

**The index holds addresses, never pixels.** A tile is a level-0 origin and an
extent; the pixels it resolves to are computed on demand. That keeps a whole
slide's index in kilobytes, and more importantly keeps exactly one answer to
"what does the model see" in the codebase.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path

from app.common.imaging import optical_density
from app.common.stains import RUIFROK_HDAB
from app.core.config import settings
from app.core.logging import get_logger
from app.ingestion.slide_reader import open_slide
from app.pipeline.step05_optical_density.tiles import read_tile
from app.pipeline.step06_colour_deconvolution.deconvolution import channel_range, separate
from app.pipeline.step07_tiling import index as tiling
from app.pipeline.step07_tiling import overlay
from app.pipeline.step07_tiling.index import TileIndex, TilingError
from app.pipeline.step08_tissue_type_segmentation.branches import (
    DEFAULT_BRANCH,
    SERVED_BRANCHES,
    SERVED_HEADS,
    ModelBranch,
    parse_branch,
)
from app.pipeline.step08_tissue_type_segmentation import model as region_model

# Torch-free: `beetle` reads two JSON members out of a zip and imports torch only inside
# `load_fold`, so step 7 keeps working - and keeps offering this branch with an honest
# reason - on a machine where step 8 cannot run at all.
from app.pipeline.step08_tissue_type_segmentation import beetle
from app.schemas.tiling import (
    TileOut,
    TilingBranchOut,
    TilingBranches,
    TilingCoverage,
    TilingFieldOfView,
    TilingFunnel,
    TilingParams,
    TilingReport,
    TilingSample,
    TilingSelection,
    TilingStaining,
)
from app.services.calibration_service import calibration_service
from app.services.density_service import density_service
from app.services.tissue_service import TissueFootprint, tissue_service
from app.services.upload_service import get_record, resolve_ready_path

logger = get_logger(__name__)

CITATION = (
    "Dolezal JM, Kochanny S, Dyer E, et al. Slideflow: deep learning for digital "
    "histopathology with real-time whole-slide visualization. BMC Bioinformatics "
    "25:134 (2024). arXiv:2304.04142 - the tile-index shape and the tissue-fraction "
    "filter. "
    "Tellez D, Litjens G, Bandi P, et al. Quantifying the effects of data "
    "augmentation and stain color normalization in convolutional neural networks for "
    "computational pathology. Medical Image Analysis 58:101544 (2019) - why the tiles "
    "carry the haematoxylin channel rather than RGB."
)

#: The panels the API serves, in the order they are meant to be read.
PANELS: tuple[str, ...] = overlay.PANELS

#: Coverage above this is worth remarking on: the kept tiles cover noticeably more
#: ground than step 3 called tissue. Expected - tiles are squares and a section is
#: not, so a tile clipping the edge brings glass with it - but past this the
#: tissue gate is admitting tiles that are mostly background.
COVERAGE_LIMIT = 1.6


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


#: The two fields of view step 7 offers, in microns of slide across one square, each
#: with the geometry it means: `(tile_px, mpp)`.
#:
#: **`tile_px` is 224 in both and only the resolution moves.** That is deliberate and it
#: is the convention `FIX1_FIX2_PLAN.md` fixed for the two exports: holding the pixel
#: side constant leaves the backbone untouched, so the two heads differ in what they can
#: see and in nothing else. A 448 um window cut at 224 px is 896x896 level-0 pixels at
#: 0.5 um/px, downsampled by four on the way in.
#:
#: **Why the choice exists at all.** A 112 um window - the served checkpoint's - cannot
#: contain the wall of a 300-1500 um duct, so solid and comedo DCIS is undecidable at
#: that scale rather than merely difficult. Both entries here are wider than that.
FIELDS_OF_VIEW: dict[float, tuple[int, float]] = {
    112.0: (224, 0.5),
    224.0: (224, 1.0),
    448.0: (224, 2.0),
    672.0: (224, 3.0),
}

#: Where two published checkpoints declare the *same* field of view, which one serves it.
#:
#: **This exists because the geometry rule ran out.** `model_for` matches a checkpoint on
#: `tile_px * mpp` and never on its name, which is right when one head is fitted per field
#: of view. It is ambiguous the moment two are: `invasive_tile_v3_concat_bach` and
#: `invasive_tile_fov112_fix1_concat` are both 224 px at 0.5 um/px, and which of them
#: `discover()` returns first depends on whether one happens to be the configured default
#: and otherwise on alphabetical order. An alphabetical accident is not a deployment
#: decision, so the choice is stated here instead.
#:
#: The two 112 um heads differ in their **label rule**, not their geometry - v3 was fitted
#: where a DCIS region the teacher misread was deleted, and `fov112_fix1` where it is
#: relabelled. That is exactly the comparison `FIX1_FIX2_PLAN.md` sets out to make, so
#: both must remain loadable by name; only the default needs deciding.
#:
#: A name here that is absent or unusable falls through to the geometry scan, so removing
#: a checkpoint degrades to the old behaviour rather than breaking the picker.
#:
#: Keyed on `(branch, fov)` rather than the field of view alone, because "which of two
#: heads at 112 um" is now a question per branch: each branch has its own head at each
#: geometry, and a preferred name is only meaningful inside the branch it belongs to.
#: The H&E branch has exactly one head per field of view, so it needs no entry - and if
#: a second is ever published there (a fix1/fix2 pair, say), one line here is the fix,
#: which is what this table is for.
FIELD_OF_VIEW_PREFERENCE: dict[tuple[ModelBranch, float], str] = {
    (ModelBranch.H_CHANNEL, 112.0): "invasive_tile_fov112_fix1_concat",
}

#: What each branch is called on screen and what it actually does, in words a reader who
#: is not a pathologist can act on. Kept beside the selection rule rather than in the
#: frontend because the disabled-option reason has to come from the same place as the
#: decision that disabled it - a label and an explanation that can drift apart will.
BRANCH_COPY: dict[ModelBranch, tuple[str, str]] = {
    ModelBranch.H_CHANNEL: (
        "Blue stain only",
        "Throw the colour away and show the model the blue nuclear stain on its own. "
        "This works on every slide in the panel, because the blue stain is the one "
        "thing all of them share.",
    ),
    ModelBranch.HE: (
        "Full colour (H&E)",
        "Show the model the colour photograph as it is, pink and blue together. The "
        "pink is what makes a duct wall look different from a band of scar tissue, so "
        "on an H&E slide it is half the evidence - but only an H&E slide has it.",
    ),
    ModelBranch.BEETLE: (
        "Run by BEETLE",
        "Hand the slide to a published network from another research group instead of "
        "ours. It answers for every pixel rather than once per square, so it draws the "
        "shape of each duct instead of colouring in a grid - and it can say 'there is "
        "nothing here', which ours cannot. It is much slower, and its licence is "
        "research-only.",
    ),
}

#: How close a checkpoint's own field of view must be to an offered one to be its model.
#: A micron of slack, because `tile_px * mpp` is a float product and a manifest may
#: record 1.0 as 0.9999.
FIELD_OF_VIEW_TOLERANCE_UM = 1.0


def committed_branch() -> ModelBranch:
    """The branch this pipeline runs, from `settings.tiling_branch`.

    Falls back to `DEFAULT_BRANCH` on an unrecognised value rather than raising: a typo
    in a deployment setting should not make the service refuse to start, and the branch
    it falls back to is the one every legacy checkpoint belongs to. The fallback is
    logged, because silently running a different model from the one configured is
    exactly the confusion this whole pair of constants exists to prevent.
    """
    try:
        return ModelBranch(settings.tiling_branch)
    except ValueError:
        logger.warning(
            "tiling.unknown_committed_branch",
            extra={"configured": settings.tiling_branch, "using": DEFAULT_BRANCH.value},
        )
        return DEFAULT_BRANCH


def field_of_view(requested: float | None) -> float:
    """The offered field of view a request resolves to, refusing anything else.

    A silent snap to the nearest offered value would be worse than a refusal here: the
    field of view decides which checkpoint runs, so a caller who asked for 300 um and
    quietly got 224 um would be reading one model's numbers under another's name.
    """
    share = settings.tiling_field_of_view_um if requested is None else float(requested)
    for offered in FIELDS_OF_VIEW:
        if abs(offered - share) <= FIELD_OF_VIEW_TOLERANCE_UM:
            return offered
    raise TilingError(
        f"{share:g} um is not one of the fields of view this pipeline has a model for. "
        f"Offered: {', '.join(f'{value:g} um' for value in FIELDS_OF_VIEW)}."
    )


def model_for(
    fov_um: float, branch: ModelBranch = DEFAULT_BRANCH
) -> region_model.Candidate | None:
    """The published checkpoint fitted at this field of view and branch, or `None`.

    **Matched on the manifest's own geometry, never on the file's name.** `tile_px * mpp`
    is what the model actually saw; a name is a label somebody typed. So a head trained
    by `FIX1_FIX2_PLAN`'s S5 starts serving the moment it is published into
    `models/tissue_type/`, under whatever name it was given and with no code change here.

    The first usable match wins, in `discover`'s order - the configured default first,
    then by name - so publishing a second head at the same field of view does not
    silently displace the one already serving.

    **The branch is part of the match, and it has to be.** Geometry alone stopped being
    a unique key the moment an H&E head was published: the two branches share the four
    geometries by construction - that identity is what makes them comparable at all - so
    `tile_px * mpp` now names two checkpoints at every field of view. Matching on it
    alone would let `discover()`'s ordering decide which one serves, and a colour model
    handed optical density does not fail loudly; it scores badly, and the bad score looks
    like a modelling result for a week. The branch comes from the manifest's own
    `input.channel`, so this is the same "read the fact, never the filename" rule the
    geometry match already follows.
    """
    # BEETLE resolves to no checkpoint of ours - it is a downloaded release with no
    # manifest in `models/tissue_type/` - so a scan over our own published heads is the
    # wrong question to ask about it and is answered here rather than searched. `window`
    # and `_offered_fields_of_view` route that branch through `beetle` instead.
    if branch not in SERVED_HEADS:
        return None

    def matches(candidate: region_model.Candidate) -> bool:
        if not candidate.usable or not candidate.tile_px or not candidate.mpp:
            return False
        if candidate.branch is not branch:
            return False
        return abs(candidate.tile_px * candidate.mpp - fov_um) <= FIELD_OF_VIEW_TOLERANCE_UM

    # An explicit preference first, where one is recorded - see
    # `FIELD_OF_VIEW_PREFERENCE` for why a geometry match alone cannot decide this. The
    # preferred name still has to *be* this field of view: a table entry naming a
    # checkpoint of the wrong geometry is a mistake in the table, and honouring it would
    # feed the model a differently-scaled field, which is the failure the whole
    # arrangement exists to prevent.
    preferred = FIELD_OF_VIEW_PREFERENCE.get((branch, fov_um))
    if preferred is not None:
        candidate = region_model.find(preferred)
        if candidate is not None and matches(candidate):
            return candidate

    for candidate in region_model.discover():
        if matches(candidate):
            return candidate
    return None


def window(
    fov_um: float, branch: ModelBranch = DEFAULT_BRANCH
) -> tuple[int, float, str, str | None]:
    """The square this step lays down at a field of view: `(px, mpp, source, model)`.

    **Why step 7 asks step 8 for its geometry rather than owning one.** The two used
    different squares - 512 px at the pipeline's working resolution here, 224 px at
    the checkpoint's there - and only step 8's is a real constraint: the model's field
    of view is a property of the checkpoint, recorded in its manifest, and feeding it
    a differently-scaled field is the same class of failure as feeding it the wrong
    channel. So the choosable one has to give way, and the grid this step prices is
    the grid that actually runs. Before this, a reader was shown "4,952 squares to
    process" and then watched step 8 make 98,262 passes over the same slide.

    **What is new is which checkpoint is asked.** It is no longer whichever one the
    settings name: it is the one fitted at `fov_um`, because that is what step 7's
    choice means. Step 8 inherits the same answer, so the two still run one grid.

    **The planned geometry stands in when no such checkpoint is published.** Then the
    square is `FIELDS_OF_VIEW`' entry rather than a manifest's, `model` is `None`, and
    the grid this step prices is the grid that head *will* run on once it is trained.
    That is worth laying: the square count is the compute bill, and knowing it before
    committing to a field of view is most of the point of this screen. Step 8 refuses
    separately and says which file it wants, so nothing scores a slide on a guess.

    **Read from the manifest, without torch.** `Candidate` is described from the
    manifest alone - that is what lets the capability endpoint list every published
    model on a machine with no torch installed - so this costs a small JSON read and
    keeps step 7 working when step 8 cannot run at all.
    """
    # **BEETLE's square is a different shape of answer to the same question.** The two
    # ResNet branches hold the *pixel* side at 224 and vary the resolution, so a field
    # of view selects one of four checkpoints. BEETLE holds the *resolution* at its
    # trained 0.5 um/px - it has no choice, feeding it another scale is the one
    # preprocessing error that looks like a bad model - and varies the pixel side
    # instead, so one release covers all four and 672 um is a 1344 px window. Both
    # report `px * mpp == fov_um`, which is what lets this screen price either with the
    # same arithmetic.
    if branch is ModelBranch.BEETLE:
        usable, _problem = beetle.available()
        return (
            beetle.window_px(fov_um),
            beetle.SPACING,
            "BEETLE's released nnU-Net" if usable else "BEETLE's geometry, weights absent",
            beetle.MODEL_NAME if usable else None,
        )

    candidate = model_for(fov_um, branch)
    if candidate is not None:
        return int(candidate.tile_px), float(candidate.mpp), candidate.name, candidate.name

    tile_px, mpp = FIELDS_OF_VIEW[fov_um]
    return tile_px, mpp, f"the planned {fov_um:g} um geometry", None


def _offered_fields_of_view(
    branch: ModelBranch = DEFAULT_BRANCH,
) -> list[TilingFieldOfView]:
    """Every offered field of view with the checkpoint that serves it, if any.

    One `discover()` pass per offered value, and that is cheap on purpose: a candidate is
    described from its manifest without loading torch, so listing what is available costs
    a couple of small JSON reads rather than a checkpoint load.
    """
    # BEETLE covers all four with one release, so availability is a single `stat` on the
    # archive rather than four scans for four heads. Every field of view it offers is
    # available together or none is, which is the honest shape of that branch: nothing
    # is being trained for it.
    if branch is ModelBranch.BEETLE:
        usable, _problem = beetle.available()
        return [
            TilingFieldOfView(
                um=um,
                tile_px=beetle.window_px(um),
                mpp=beetle.SPACING,
                model=beetle.MODEL_NAME if usable else None,
                available=usable,
            )
            for um in FIELDS_OF_VIEW
        ]

    offered: list[TilingFieldOfView] = []
    for um, (tile_px, mpp) in FIELDS_OF_VIEW.items():
        candidate = model_for(um, branch)
        offered.append(
            TilingFieldOfView(
                um=um,
                # The published checkpoint's own geometry when there is one, because that
                # is what will actually run; the planned geometry only stands in for a
                # head that does not exist yet.
                tile_px=int(candidate.tile_px) if candidate else tile_px,
                mpp=float(candidate.mpp) if candidate else mpp,
                model=candidate.name if candidate else None,
                available=candidate is not None,
            )
        )
    return offered


#: Both colour branches need a second dye, and this is the sentence that says so for
#: BEETLE specifically. Step 5's own `reason` is appended after it: that sentence names
#: the arms and the angle this slide actually produced, and restating it would be a
#: second version of a measurement that has one.
#:
#: The claim is made from the **input contract and not from the paper**. BEETLE's
#: `dataset.json` names all three channels `rgb_to_0_1` and `to_model_input` is a
#: division by 255 - no deconvolution, no white point, no stain normalisation - so what
#: the network sees is the colour of the section as it was scanned. A DAB section is
#: brown where that distribution is pink. That is checkable here; "it was trained on
#: H&E" is not, because the archive records channels and never a stain.
_BEETLE_NEEDS_COLOUR = (
    "BEETLE reads the colour photograph with no deconvolution and no stain "
    "normalisation - a division by 255 and nothing else - so the dyes on the section "
    "are its input distribution rather than something it is robust to. On an "
    "immunostained slide it would be reading brown where it expects pink, and unlike "
    "our own heads it cannot be retrained or reweighted for that. "
)


def _offered_branches(staining) -> list[TilingBranchOut]:
    """All three options, each with whether this slide can take it and why not.

    Every option is returned in every case. An unavailable branch is rendered disabled
    with its reason rather than hidden, for `TilingFieldOfView`'s reason one level up:
    a choice that silently shrinks makes the pipeline look smaller than it is, and the
    reason a branch is unavailable is often the most informative thing on the screen.

    `he` stays `implemented=True` even before its checkpoints exist, with every field of
    view reporting `available=False`. That is the informative state - an H&E slide can
    select the branch and see that the heads are being trained - where disabling the
    whole branch would hide it.

    **Both colour branches are gated on step 5's verdict, and BEETLE needs it at least
    as much as `he` does.** `he` is gated because a head fitted on pink and blue would
    be reading a colour that is not there; BEETLE is gated for the same reason with less
    recourse, because it is somebody else's released weights and there is no version of
    it fitted on anything else. Its unavailability is therefore two separate questions
    asked in order - *can this slide take it* before *is it installed* - because the
    archive being absent is a fixable fact about this machine and the section carrying
    one dye is not, and telling a reader to fetch 1.9 GB for a slide that could never
    use it is the least useful thing this screen could say.
    """
    offered: list[TilingBranchOut] = []
    for branch in ModelBranch:
        label, blurb = BRANCH_COPY[branch]
        implemented = branch in SERVED_BRANCHES
        enabled = implemented
        reason: str | None = None

        if not implemented:
            reason = "yet to be developed"
        elif branch is ModelBranch.HE and not staining.is_he:
            enabled = False
            reason = staining.reason
        elif branch is ModelBranch.BEETLE:
            if not staining.is_he:
                # The slide first. Nothing that can be downloaded makes an
                # immunostained section readable by these weights.
                enabled = False
                reason = _BEETLE_NEEDS_COLOUR + staining.reason
            else:
                # Then the machine. Unavailable for a reason nobody can guess from the
                # screen - a 1.9 GB download that is not committed - so the reason is
                # the message, verbatim from the module that knows where it looked.
                usable, problem = beetle.available()
                if not usable:
                    enabled = False
                    reason = problem

        offered.append(
            TilingBranchOut(
                id=branch.value,
                label=label,
                blurb=blurb,
                implemented=implemented,
                enabled=enabled,
                reason=reason,
                fields_of_view=(
                    _offered_fields_of_view(branch) if implemented else []
                ),
            )
        )
    return offered


def _describe_staining(verdict) -> TilingStaining:
    """Step 5's verdict on the wire. One place, so step 7 quotes it rather than
    restating it."""
    return TilingStaining(
        staining=verdict.staining.value,
        is_he=verdict.is_he,
        reason=verdict.reason,
        source=verdict.source,
        two_armed=verdict.two_armed,
        separation=verdict.separation,
        reference_separation=verdict.reference_separation,
        eosin_arm_degrees=verdict.eosin_arm_degrees,
        tolerance_deg=verdict.tolerance_deg,
        tile_x=verdict.tile[0] if verdict.tile else None,
        tile_y=verdict.tile[1] if verdict.tile else None,
    )


class TilingService:
    """Builds the tile index for a slide, and says what each gate cost.

    One memo entry, keyed on step 3's mask identity plus this step's own two gates
    and its overlap. The mask key already folds in the slide, the threshold, the
    resolution and which QC run was subtracted, so what is left to key on is only
    what this step itself decides.
    """

    #: How many runs are kept. One was enough while the screen had a single control;
    #: it is not enough now that the screen has two levels and the viewer bounces
    #: between a branch, a field of view and two panel requests. With one entry the
    #: report and its own panels evict each other, which is what made step 8's
    #: untargeted `report()` call rebuild step 7 on every click.
    MEMO_ENTRIES = 4

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._memo: OrderedDict[tuple[object, ...], _Run] = OrderedDict()

    # --- the committed choice ----------------------------------------------

    def _selection_path(self, upload_id: str) -> Path:
        return settings.tiling_dir / upload_id / "selection.json"

    def selection(self, upload_id: str) -> TilingSelection:
        """What this slide's viewer committed on step 7, or the configured defaults.

        The defaults are returned rather than raising, because every consumer wants an
        answer and "nobody has chosen yet" and "somebody chose the default" produce the
        same grid. What matters is that a caller who *did* choose gets their choice back
        without having to pass it.
        """
        path = self._selection_path(upload_id)
        if path.exists():
            try:
                return TilingSelection.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                logger.warning(
                    "tiling.selection_unreadable", extra={"upload_id": upload_id}
                )

        # The pipeline's committed branch, not the manifest fallback - see
        # `settings.tiling_branch` for why those are two different questions.
        return TilingSelection(
            branch=committed_branch().value,
            field_of_view_um=settings.tiling_field_of_view_um,
            overlap=settings.tiling_overlap,
            tissue_threshold=None,
        )

    def commit_selection(
        self,
        upload_id: str,
        *,
        branch: str | None,
        fov: float | None,
        overlap: float | None = None,
        threshold: int | None = None,
    ) -> TilingSelection:
        """Record the choice, and return the grid it produces.

        **Called from a POST and never as a side effect of a GET.** A GET that wrote
        this would make "the viewer glanced at 672 um" indistinguishable from "the
        viewer chose 672 um", and every panel request would rewrite it.
        """
        chosen_branch = parse_branch(branch)
        if chosen_branch not in SERVED_BRANCHES:
            raise TilingError(
                f"`{BRANCH_COPY[chosen_branch][0]}` is not developed yet, so there is "
                "no grid to lay for it."
            )

        chosen_fov = field_of_view(fov)
        size, mpp, _source, model_name = window(chosen_fov, chosen_branch)
        share = settings.tiling_overlap if overlap is None else float(overlap)

        record = TilingSelection(
            branch=chosen_branch.value,
            field_of_view_um=chosen_fov,
            overlap=share,
            tissue_threshold=threshold,
            model=model_name,
            tile_px=size,
            mpp=mpp,
            chosen_at=_now(),
        )

        path = self._selection_path(upload_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.with_name(path.name + ".tmp")
        staging.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        os.replace(staging, path)

        logger.info(
            "tiling.selection_committed",
            extra={
                "upload_id": upload_id,
                "branch": record.branch,
                "field_of_view_um": record.field_of_view_um,
                "overlap": record.overlap,
                "model": record.model,
            },
        )
        return record

    def forget_selection(self, upload_id: str) -> None:
        """Drop the record, when the slide is released."""
        path = self._selection_path(upload_id)
        try:
            path.unlink(missing_ok=True)
            if path.parent.is_dir() and not any(path.parent.iterdir()):
                path.parent.rmdir()
        except OSError:
            pass
        with self._lock:
            for key in [k for k in self._memo if k[0] == upload_id]:
                self._memo.pop(key, None)

    def _resolve(
        self,
        upload_id: str,
        *,
        branch: str | None,
        fov: float | None,
        overlap: float | None,
        threshold: int | None,
    ) -> tuple[ModelBranch, float, float | None, int | None]:
        """Explicit argument, then the committed record, then the settings default.

        **This is the whole fix for the diverging grids.** Step 8 asks step 7 for the
        index with no arguments; before this it therefore got the *default* field of
        view even when the caller had named the 448 um head, the two grids stopped
        coinciding, and step 8's window gate silently fell out of its identity fast
        path. Now the no-argument call resolves to what the viewer committed, so the
        two steps run one grid again without step 8 having to know the question.
        """
        record = self.selection(upload_id)
        chosen_branch = parse_branch(branch if branch is not None else record.branch)
        chosen_fov = field_of_view(
            fov if fov is not None else record.field_of_view_um
        )
        chosen_overlap = overlap if overlap is not None else record.overlap
        chosen_threshold = (
            threshold if threshold is not None else record.tissue_threshold
        )
        return chosen_branch, chosen_fov, chosen_overlap, chosen_threshold

    # --- running ------------------------------------------------------------

    def _run(
        self,
        upload_id: str,
        *,
        threshold: int | None,
        overlap: float | None,
        fov: float | None,
        branch: str | None = None,
    ) -> _Run:
        """Build the index, or return the one already built for these inputs.

        `threshold` belongs to step 3 and is passed straight through; step 7 owns
        it no more than steps 4, 5 and 6 do. This step's own controls are `fov` and
        `overlap`, and they are different in kind. The overlap moves only the cost.
        The **field of view** moves which checkpoint step 8 runs, because a field of
        view is a property of the weights - so it is the one control on this screen
        whose effect is a different model rather than a different price. The two gates
        are thresholds whose values are arguments rather than preferences.
        """
        chosen_branch, chosen_fov, overlap, threshold = self._resolve(
            upload_id, branch=branch, fov=fov, overlap=overlap, threshold=threshold
        )

        footprint = tissue_service.footprint(upload_id, threshold=threshold)
        share = settings.tiling_overlap if overlap is None else float(overlap)
        candidate = model_for(chosen_fov, chosen_branch)
        size, mpp, source, model_name = window(chosen_fov, chosen_branch)

        # The window's geometry joins the memo key. A republished checkpoint with a
        # different field of view is a different grid, and a memo that ignored it
        # would hand back the previous model's tiles under the new model's name.
        #
        # The **branch** joins it too, and not because the geometry differs - it does
        # not, by construction - but because the sample panel does: one branch shows a
        # deconvolved density plane and the other the photograph, and a memo that
        # conflated them would caption one picture with the other's transform.
        key = (upload_id, footprint.key, share, size, mpp, chosen_branch.value)
        with self._lock:
            found = self._memo.get(key)
            if found is not None:
                self._memo.move_to_end(key)
                return found

        index = tiling.build_index(
            tissue=footprint.mask,
            considered=footprint.considered,
            mask_mpp=footprint.mpp,
            base_mpp=footprint.base_mpp,
            slide_size=(footprint.slide_width, footprint.slide_height),
            target_mpp=mpp,
            size=size,
            overlap=share,
            min_tissue_share=settings.tiling_min_tissue_share,
            min_clean_share=settings.tiling_min_clean_share,
            max_tiles=settings.tiling_max_tiles,
        )

        logger.info(
            "tiling.index",
            extra={
                "upload_id": upload_id,
                "every": index.funnel.every,
                "clean": index.funnel.clean,
                "overlap": share,
                "window_px": size,
                "window_mpp": mpp,
                "window_from": source,
                "field_of_view_um": chosen_fov,
                "branch": chosen_branch.value,
                "model": model_name,
            },
        )

        run = _Run(
            index=index,
            footprint=footprint,
            overlap=share,
            field_of_view_um=chosen_fov,
            model=model_name,
            branch=chosen_branch,
            input_channel=candidate.channel if candidate is not None else None,
        )
        with self._lock:
            self._memo[key] = run
            self._memo.move_to_end(key)
            while len(self._memo) > self.MEMO_ENTRIES:
                self._memo.popitem(last=False)
        return run

    def report(
        self,
        upload_id: str,
        *,
        threshold: int | None = None,
        overlap: float | None = None,
        fov: float | None = None,
        branch: str | None = None,
    ) -> TilingReport:
        """Build the tile index and describe it.

        Every argument defaults to `None` and resolves through `_resolve`, so a caller
        who knows nothing - step 8's worker, the runner - gets the viewer's committed
        grid rather than the configured default.
        """
        run = self._run(
            upload_id, threshold=threshold, overlap=overlap, fov=fov, branch=branch
        )
        record = get_record(upload_id=upload_id)
        return self._describe(run, upload_id=upload_id, filename=record.filename)

    def branches(self, upload_id: str) -> TilingBranches:
        """What step 7 offers, before any grid has been laid.

        Manifests and step 5's cached verdict. No slide pixels, no grid, no torch -
        this is the payload the branch screen renders, and it must be answerable on a
        machine where step 8 cannot run at all.
        """
        record = get_record(upload_id=upload_id)
        verdict = density_service.staining(upload_id)
        return TilingBranches(
            upload_id=upload_id,
            filename=record.filename,
            generated_at=_now(),
            staining=_describe_staining(verdict),
            branches=_offered_branches(verdict),
            default_branch=committed_branch().value,
            notes=[
                "The two options of ours show the model the same squares at the same "
                "scale. The only difference is how much of the picture it is allowed "
                "to see, which is what makes those two comparable.",
                "BEETLE is not comparable to them window for window, and the report "
                "says so rather than putting the numbers side by side: it reads at its "
                "own fixed resolution and answers per pixel in five classes, so a "
                "field of view changes how much slide one window holds rather than how "
                "finely it is read.",
                "Whichever you pick, step 8 runs on it - and if you come back and "
                "change it, step 8's previous answer is thrown away rather than left "
                "on screen beside numbers that no longer describe it.",
            ],
            citation=CITATION,
        )

    def panel(
        self,
        upload_id: str,
        name: str,
        *,
        threshold: int | None = None,
        overlap: float | None = None,
        fov: float | None = None,
        branch: str | None = None,
    ) -> bytes:
        """`grid` or `sample`, as PNG."""
        if name not in PANELS:
            raise TilingError(f"unknown panel {name!r}; expected one of {sorted(PANELS)}")

        run = self._run(
            upload_id, threshold=threshold, overlap=overlap, fov=fov, branch=branch
        )

        if name == "grid":
            thumbnail, _ = calibration_service.thumbnail(upload_id, threshold=threshold)
            return overlay.grid_png(thumbnail, run.index)

        return self._sample_png(upload_id, run, threshold=threshold)

    def _sample_png(
        self, upload_id: str, run: _Run, *, threshold: int | None
    ) -> bytes:
        """One tile *of this index*, as the model will receive it.

        Assembled from the steps that own each piece rather than routed through
        step 6's service, and the distinction matters both ways round.

        **Not through `deconvolution_service`.** That service snaps a requested
        position to the nearest block step 5 *scored*, which is the right behaviour
        for step 5's own tile chooser and the wrong behaviour here: step 5's blocks
        are a different grid, so the panel would draw a neighbouring field while
        the report named this one. A picture captioned with another tile's
        coordinates is worse than no picture.

        **On the H&E branch there is no deconvolution to do.** That branch's model
        reads the colour image, so the honest panel is the colour image, and the
        function returns before any of the arithmetic below. Drawing it through the
        haematoxylin ramp would produce a perfectly plausible picture of an input that
        model never sees, which is worse than no picture at all.

        **On the haematoxylin branch, still the same deconvolution.** The rule the guide is emphatic about
        is that everything downstream calls the one deconvolution function, and it
        is kept exactly - `RUIFROK_HDAB` and `separate` are step 6's, `read_tile` is
        step 5's, `optical_density` is `app.common.imaging`'s, and the stretch is
        step 6's `channel_range`. Nothing here is reimplemented; the composition is
        new, the arithmetic is not.
        """
        sample = self._sample_tile(run)
        if sample is None:
            return overlay.empty_png(run.index.size)

        white = calibration_service.white_point(upload_id, threshold=threshold)

        path = resolve_ready_path(upload_id=upload_id)
        with open_slide(path) as reader:
            tile = read_tile(
                reader,
                x=sample.x,
                y=sample.y,
                # The grid's own geometry and not the pipeline's working one: this
                # panel is "what one of these squares looks like", so reading it at a
                # different size or resolution would illustrate a square that is not
                # on the map beside it.
                target_mpp=run.index.mpp,
                size=run.index.size,
                base_mpp=run.footprint.base_mpp,
            )

        # The H&E branch stops here. Its model is handed the photograph, so the
        # panel is the photograph - and every step below this line is a statement
        # about dye concentration that branch deliberately does not make.
        if run.branch is ModelBranch.HE:
            return overlay.rgb_sample_png(tile.rgb)

        od = optical_density(
            tile.rgb.astype("float32"),
            white.field_for(x=tile.x, y=tile.y, size=tile.size, mpp=tile.mpp),
            floor=white.od_floor,
        )
        channels = separate(od, RUIFROK_HDAB)

        # Step 5's own "this pixel carries stain" rule, so the stretch below is
        # computed over the same population step 6 computes it over.
        stained = od.mean(axis=-1) >= settings.density_beta
        high = (
            channel_range(channels, stained)[0]
            if bool(stained.any())
            # A tile with no stain in it has no percentile to stretch against. It
            # cannot normally reach here - it would have failed the tissue gate -
            # but a blank panel is a better answer than a division by nothing.
            else 1.0
        )

        return overlay.sample_png(channels.haematoxylin, high=high)

    @staticmethod
    def _sample_tile(run: _Run) -> tiling.Tile | None:
        """The kept tile with the most tissue on it, as the one worth showing.

        Not the first kept tile, which is the top-left corner of the section and
        therefore usually its thinnest edge. A reader being shown "what a tile looks
        like" should be shown a representative one, and the most solidly tissue-
        covered tile is the honest choice of representative - ties broken by
        position so the pick is stable across runs rather than dependent on sort
        order.
        """
        kept = [tile for tile in run.index.tiles if tile.kept]
        if not kept:
            return None
        return max(kept, key=lambda tile: (tile.tissue_share, -tile.row, -tile.col))

    # --- handing on to step 8 -----------------------------------------------

    def tile_index(
        self,
        upload_id: str,
        *,
        threshold: int | None = None,
        overlap: float | None = None,
        fov: float | None = None,
        branch: str | None = None,
    ) -> TileIndex:
        """The index itself, for the step that consumes it rather than displays it.

        The whole index and not the sampled slice: the report ships a couple of
        hundred tiles because that is what a browser and a reader can hold, and
        step 8 needs all of them. Two different audiences, two different answers,
        and the sampling belongs to the one that is a display concern.
        """
        return self._run(
            upload_id, threshold=threshold, overlap=overlap, fov=fov, branch=branch
        ).index

    # --- describing ---------------------------------------------------------

    def _describe(self, run: _Run, *, upload_id: str, filename: str) -> TilingReport:
        index, footprint = run.index, run.footprint
        listed = tiling.sample(index)
        sample = self._sample_tile(run)

        # One mask pixel as a fraction of a tile - the honest precision of the two
        # gates, since both shares are read off step 3's coarser grid.
        quantisation = (index.mask_mpp / max(index.mpp, 1e-9)) ** 2 / max(
            index.size**2, 1
        )

        return TilingReport(
            upload_id=upload_id,
            filename=filename,
            generated_at=_now(),
            params=TilingParams(
                target_mpp=index.mpp,
                tile_size=index.size,
                tile_um=round(index.tile_um, 2),
                overlap=round(index.overlap, 4),
                span=index.span,
                stride=index.stride,
                min_tissue_share=settings.tiling_min_tissue_share,
                min_clean_share=settings.tiling_min_clean_share,
                mask_mpp=round(index.mask_mpp, 4),
                share_quantisation=round(quantisation, 6),
                tissue_threshold=footprint.threshold,
                tissue_threshold_source=footprint.threshold_source,
                qc_gated=footprint.qc_gated,
                qc_source=footprint.qc_source,
                field_of_view_um=run.field_of_view_um,
                branch=run.branch.value,
                input_channel=run.input_channel,
                model=run.model,
            ),
            fields_of_view=_offered_fields_of_view(run.branch),
            staining=_describe_staining(density_service.staining(upload_id)),
            branches=_offered_branches(density_service.staining(upload_id)),
            funnel=TilingFunnel(
                every=index.funnel.every,
                on_tissue=index.funnel.on_tissue,
                clean=index.funnel.clean,
                reduction=round(index.funnel.reduction, 2),
            ),
            coverage=TilingCoverage(
                covered_mm2=round(index.covered_mm2, 3),
                tissue_mm2=round(index.tissue_mm2, 3),
                coverage=round(index.covered_mm2 / max(index.tissue_mm2, 1e-9), 3),
            ),
            cols=index.cols,
            rows=index.rows,
            tiles=[
                TileOut(
                    col=tile.col,
                    row=tile.row,
                    x=tile.x,
                    y=tile.y,
                    span=tile.span,
                    fx=round(tile.fx, 6),
                    fy=round(tile.fy, 6),
                    fw=round(tile.fw, 6),
                    fh=round(tile.fh, 6),
                    tissue_share=round(tile.tissue_share, 4),
                    clean_share=round(tile.clean_share, 4),
                    kept=tile.kept,
                    rejected_by=tile.rejected_by,
                )
                for tile in listed
            ],
            listed=len(listed),
            sample=(
                TilingSample(
                    x=sample.x,
                    y=sample.y,
                    span=sample.span,
                    size=index.size,
                    mpp=round(index.mpp, 5),
                    tissue_share=round(sample.tissue_share, 4),
                )
                if sample is not None
                else None
            ),
            notes=self._notes(run, density_service.staining(upload_id)),
            citation=CITATION,
        )

    @staticmethod
    def _notes(run: _Run, staining=None) -> list[str]:
        """The caveats that belong beside the numbers, built from the numbers."""
        index, footprint = run.index, run.footprint
        funnel = index.funnel
        notes: list[str] = []

        # **The branch is committed rather than chosen now, so nothing upstream can
        # refuse it on this slide's behalf.** The branch payload still marks the colour
        # option unavailable when step 5 has not found two dyes, but that is advice to
        # a picker that no longer exists: `commit_selection` validates that a branch is
        # *served*, not that it suits the section in front of it. On a DAB slide the
        # colour model is reading an eosin direction that points at nothing - it does
        # not fail, it scores badly - so if the pipeline is ever pointed at one, this
        # is the sentence that says so.
        if staining is not None and run.branch is ModelBranch.HE and not staining.is_he:
            notes.append(
                "THIS IS THE COLOUR MODEL ON A SECTION THAT DOES NOT LOOK LIKE H&E. "
                f"{staining.reason} The colour branch was fitted on haematoxylin and "
                "eosin, and half of what it reads - the pink - is a direction pointing "
                "at nothing here. It will still return a class for every square, and "
                "those classes should not be trusted. Steps 7 to 9 are meant to run on "
                "the case's H&E slide."
            )

        notes.append(
            f"A model cannot look at a whole slide - this one is "
            f"{footprint.slide_width:,} by {footprint.slide_height:,} pixels - so the tissue "
            f"is cut into patches it can take one at a time. The grid over the whole canvas "
            f"holds {funnel.every:,} tiles at {index.size} px and {index.mpp:g} um/px, with "
            f"{index.overlap:.0%} overlap between neighbours."
        )

        notes.append(
            f"That is the point of this step running where it does. Of those "
            f"{funnel.every:,} tiles, {funnel.on_tissue:,} hold enough tissue to be worth "
            f"looking at and {funnel.clean:,} of those are also clear enough of step 2's "
            f"artefacts - {funnel.reduction:.0f} times fewer than the grid. Every tile that "
            "survives is one forward pass through the region model at step 8, which is the "
            "most expensive thing in this pipeline, so this number *is* the compute bill for "
            "everything downstream. Tiling before the tissue mask would have paid it in full."
        )

        notes.append(
            f"Nothing here is stored as pixels. What leaves this step is "
            f"{funnel.clean:,} addresses - a level-0 origin and an extent each - and the "
            "pixels a tile resolves to are computed when they are needed, as step 6's "
            "haematoxylin channel rather than as colour. That is not a saving, it is the "
            "guide's one hard rule about the fork: training and inference have to call the "
            "same deconvolution function, and materialising tiles here would put a second, "
            "silent answer to 'what does the model see' on disk. The sample panel is drawn "
            "through step 6's own function and its own scale for exactly that reason."
        )

        notes.append(
            f"Each tile covers {index.tile_um:g} um of slide - roughly "
            f"{index.tile_um / 10:.0f} nuclei across, taking a breast epithelial nucleus at "
            "about 10 um. That field of view is the parameter that matters most here, and it "
            "is chosen for a specific job: step 8 has to tell ductal carcinoma in situ from "
            "invasive carcinoma, and the difference between them is *architecture* - whether "
            "the abnormal cells are still inside a duct. A duct is 300-1500 um across, so a "
            "window narrower than its wall cannot show the wall, and no model can read a "
            "distinction its input does not contain. At 40x you see the cells and lose the "
            "architecture; at 5x the reverse."
        )

        notes.append(
            f"This field of view is also which model runs: "
            + (
                f"{run.model}, the checkpoint whose manifest records a "
                f"{run.field_of_view_um:g} um window."
                if run.model
                else (
                    f"no checkpoint fitted at {run.field_of_view_um:g} um is published yet, "
                    "so the grid above is the one that head will run on rather than one it "
                    "has run on. The counts are real - they are what this choice would cost "
                    "- but step 8 cannot start until the head exists."
                )
            )
            + " A field of view is a property of the weights, not a setting, so the two "
            "cannot be chosen separately: a model shown squares at a scale it was never "
            "trained on reads them wrongly and reports no difficulty doing so."
        )

        if index.overlap <= 0:
            notes.append(
                "Neighbouring tiles do not overlap - they meet edge to edge, so every piece "
                "of tissue is one forward pass and no more. A model has no context past a "
                "tile's edge, so its predictions there are its worst, and overlapping is what "
                "would let those edges be averaged away instead of being stitched into "
                "visible seams across the class map. It is the one setting on this step that "
                "costs compute and it costs it quadratically - 25% overlap is 1.8 times these "
                "tiles, 50% is 4 times - which is the trade this pipeline declines. Expect "
                "faint square edges in step 8's map where two neighbours disagreed on their "
                "shared border."
            )
        else:
            notes.append(
                f"Neighbouring tiles overlap by {index.overlap:.0%}, which is why the grid "
                f"holds more tiles than the section needs to be covered once. A model has no "
                "context past a tile's edge, so its predictions there are its worst - "
                "overlapping lets those edges be averaged away instead of being stitched into "
                "visible seams across the class map. It is the one setting on this step that "
                "costs compute, and it costs it quadratically: 25% overlap is 1.8 times the "
                "tiles of none, 50% is 4 times."
            )

        coverage = index.covered_mm2 / max(index.tissue_mm2, 1e-9)
        notes.append(
            f"The kept tiles cover {index.covered_mm2:.1f} mm2 of slide against "
            f"{index.tissue_mm2:.1f} mm2 step 3 called tissue - {coverage:.2f} times as much. "
            "That is counted once per area rather than once per tile, so it does not move "
            "when the overlap does; multiplying the tile count by the tile area would report "
            "four times the ground at 50% overlap and make it look as though overlapping "
            "tiles see more of the slide. They do not - they see the same tissue more often."
            + (
                ""
                if coverage <= COVERAGE_LIMIT
                else " At this ratio the tiles are bringing a good deal of background with "
                "them, which happens on a fragmented specimen where the section's perimeter "
                "is long relative to its area. Nothing is wrong, but step 8 will be "
                "classifying more glass than usual."
            )
        )

        notes.append(
            f"Both gates were measured on step 3's mask at {index.mask_mpp:g} um/px rather "
            f"than on each tile's own pixels, so a share here is precise to about "
            f"{(index.mask_mpp / index.mpp) ** 2 / index.size**2:.1%} of a tile. Measuring "
            "them properly would mean reading the whole slide at working magnification - the "
            "exact cost this step exists to avoid - for a number whose only use is a "
            f"comparison against {settings.tiling_min_tissue_share:.0%}. The coarse "
            "measurement is the right one and its precision is stated rather than implied."
        )

        notes.append(
            f"The tissue gate is {settings.tiling_min_tissue_share:.0%}, which is much lower "
            "than the 85% step 5 demanded of its tile, and the two are not inconsistent. "
            "Step 5 needed one field it could trust every pixel of, because it measures a "
            "density there. Step 8 needs *all* the tissue: a tile it never sees is a region "
            "it cannot classify, and the invasive front - the part the whole score is gated "
            "on - often sits at the section's edge, where tiles are half glass."
        )

        if not footprint.qc_gated:
            notes.append(
                "Step 2 has not run, so the artefact gate passed everything and the middle "
                "and last numbers of the funnel are the same. Run quality control and open "
                "this step again: a blurred or folded tile does not produce no answer at "
                "step 8, it produces a confident wrong one."
            )
        elif funnel.on_tissue == funnel.clean:
            notes.append(
                "Step 2 ran and flagged nothing inside any tissue tile, so the artefact gate "
                "removed none. That is a clean scan rather than a gate that did not work."
            )

        if footprint.capped:
            notes.append(
                "Step 3's mask hit its own size cap on this slide, so it is coarser than the "
                f"{settings.tissue_mask_mpp:g} um/px it asked for. The grid is unaffected - "
                "it is laid out in level-0 coordinates - but the two shares each tile was "
                "judged on were read off that coarser mask, so they are correspondingly "
                "blunter."
            )

        return notes


class _Run:
    """One complete run of step 7, memoised as a unit.

    A plain class rather than a dataclass because it holds arrays and exists only
    to keep the report and both panels looking at the same index - if either panel
    rebuilt it, the picture on screen and the counts beside it could disagree.
    """

    __slots__ = (
        "branch",
        "field_of_view_um",
        "footprint",
        "index",
        "input_channel",
        "model",
        "overlap",
    )

    def __init__(
        self,
        *,
        index: TileIndex,
        footprint: TissueFootprint,
        overlap: float,
        field_of_view_um: float,
        branch: ModelBranch,
        #: What the resolved checkpoint declares it is shown. `None` when no head is
        #: published at this choice - the sample panel then falls back to the branch's
        #: own convention, because there is no manifest to read.
        input_channel: str | None,
        #: The checkpoint fitted at this field of view, or `None` when none is published
        #: yet and the grid was laid on the planned geometry instead.
        model: str | None,
    ) -> None:
        self.index = index
        self.footprint = footprint
        self.overlap = overlap
        self.field_of_view_um = field_of_view_um
        self.model = model
        self.branch = branch
        self.input_channel = input_channel


tiling_service = TilingService()

__all__ = ["TilingError", "tiling_service"]
