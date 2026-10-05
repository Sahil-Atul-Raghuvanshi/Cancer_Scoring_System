"""Application configuration, loaded from the environment via pydantic-settings."""

import sys
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve storage relative to the repository root, so the paths are the same no
# matter which directory the process was started from.
REPO_ROOT = Path(__file__).resolve().parents[3]

# The workspace holding this demo beside the two training folders.
WORKSPACE_ROOT = REPO_ROOT.parent

# Storage is versioned: `<workspace>/data/` holds the inputs every version shares
# (the source slides and datasets), and `<workspace>/v<N>_data/` holds what one
# version of the pipeline produced. `data_versions.py` at the workspace root decides
# which version this process is - read once at import, so switching version means
# restarting the backend (the version picker does that through --reload).
sys.path.append(str(WORKSPACE_ROOT))
import data_versions  # noqa: E402

#: Which version runs, and - when the one asked for cannot be opened by this code -
#: why the latest openable one runs instead. Its data is left exactly where it is.
DATA_RESOLUTION = data_versions.resolve()
DATA_VERSION = DATA_RESOLUTION.active
VERSION_ROOT = data_versions.ensure_version(DATA_VERSION)

# The shared, read-only inputs: the OncoStem slide library and the training sources.
ORIGINAL_ROOT = data_versions.ORIGINAL_ROOT

# Model checkpoints, shared by the demo and both training folders. Per version, so a
# retrained model is a new version rather than an overwrite; a version without its
# own `models/` inherits the newest earlier one's.
MODELS_ROOT = data_versions.models_root(DATA_VERSION)

# Every byte this app writes lives under the active version's `data/`, and this
# demo's share of it is `v<N>_data/data/demo/`. It is deliberately OUTSIDE the
# project folder, so it can be moved to another disk or deleted wholesale without
# touching any code. `data/demo/` is per-slide runtime state - uploads, QC reports,
# masks, tiles - and is rebuilt by re-running the pipeline on a slide.
#
# Model checkpoints are NOT here. They stay under `v<N>_data/models/`, because they
# are versioned by `models.lock.json` and fetched by `setup.py` rather than
# produced by a run.
DATA_ROOT = data_versions.data_root(DATA_VERSION) / "demo"


class Settings(BaseSettings):
    """Runtime settings for the API.

    Values are read from environment variables or a local ``.env`` file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "Breast Cancer IHC Tissue Scoring API"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = True

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # API
    api_v1_prefix: str = "/api/v1"

    # Storage. Chunked slide upload stages parts under uploads_dir and publishes
    # the reassembled slide to slides_dir.
    data_dir: Path = DATA_ROOT
    uploads_dir: Path = DATA_ROOT / "uploads"
    slides_dir: Path = DATA_ROOT / "slides"

    # Upload limits
    max_upload_bytes: int = 4 * 1024 * 1024 * 1024  # 4 GiB per slide
    upload_chunk_size: int = 8 * 1024 * 1024  # 8 MiB; the client may request smaller

    # Slide formats the reader can actually open, so anything else is rejected at
    # init rather than after a multi-gigabyte transfer. tif/tiff/svs go through
    # tiffslide; jpg/jpeg/png through the synthetic-pyramid image reader.
    #
    # NOT allowed: .ndpi, .mrxs, .scn, .bif - tiffslide cannot open them and there
    # is no native OpenSlide backend here yet, so they would upload then fail
    # validation. Add them back alongside an OpenSlide backend.
    allowed_slide_ext: set[str] = {".tif", ".tiff", ".svs", ".jpg", ".jpeg", ".png"}

    # Step 1 defaults: the magnification the pipeline works at, expressed in
    # microns per pixel because level indices mean different things per scanner.
    target_mpp: float = 0.5
    tile_size: int = 512

    # --- Step 2: quality control ------------------------------------------
    #
    # QC output is cached per upload because a run costs minutes on CPU, and
    # the walkthrough asks for the same report every time the step is opened.
    qc_dir: Path = DATA_ROOT / "qc"

    # Where the GrandQC checkpoints live. Empty means "search the usual
    # places" - see app/qc/models.py, which also looks in a sibling clone of
    # the GrandQC repository, so an existing checkout works without config.
    qc_models_dir: Path | None = None

    # Which GrandQC artefact model to run. The checkpoints are named by the
    # resolution they were trained at, not by magnification:
    #   2.0 um/px = 5x   1.5 um/px = 7x   1.0 um/px = 10x
    # The authors recommend 7x as the general-purpose choice; 5x is roughly
    # twice as fast per slide, which matters on a CPU-only machine.
    qc_model_mpp: float = 1.5

    # The tissue detector always runs at 10 um/px - it is what the checkpoint
    # was trained at, and at that scale a whole slide is a few thousand pixels.
    qc_tissue_model_mpp: float = 10.0

    # Patch size the models were trained on. Not a tuning knob.
    qc_patch_size: int = 512

    # Refuse a run that would need more forward passes than this, rather than
    # occupying the only worker thread for an hour. At 1.5 um/px a 30 mm slide
    # is ~1,400 patches, so this leaves real headroom.
    qc_max_patches: int = 6000

    # Read patches from the nearest at-or-finer pyramid level instead of always
    # from level 0. Upstream GrandQC reads level 0 and downsamples in software;
    # on a 40x slide that is 15x more pixels off disk per patch for a result
    # that differs only in resampling. Set true to match upstream exactly.
    qc_read_from_level_0: bool = False

    # --- Step 3: tissue mask ----------------------------------------------
    #
    # Cached per upload like QC's, though for a different reason: a run is
    # seconds, not minutes, but the saturation channel it is computed from is
    # what makes re-thresholding on a slider drag instant instead of a fresh
    # read of the slide each time the viewer moves it.
    tissue_dir: Path = DATA_ROOT / "tissue"

    # The guide's recommendation. Tissue-versus-glass is a question about
    # millimetres, so working finer than this buys nothing.
    tissue_mask_mpp: float = 2.0

    # Cap on the mask's longest edge, and it binds on real slides: a 28 mm
    # specimen at 2 um/px is 14,000 px across, and the morphology below is
    # iterative. Small biopsies land at the requested resolution; whole
    # excisions land coarser, and the readout says which happened.
    tissue_mask_max_px: int = 4096

    # Morphology, in microns rather than pixels so a change of scanner cannot
    # silently change what they mean.
    #   closing  bridges gaps narrower than this, so adjacent strands of one
    #            specimen stop being separate islands. Roughly three cells.
    #   opening  drops specks smaller than this: dust, stain precipitate.
    tissue_close_um: float = 30.0
    tissue_open_um: float = 12.0

    # Smallest connected region kept, in mm^2. 0.02 mm^2 is a ~140 um square -
    # below a genuine tissue fragment, above debris.
    tissue_min_component_mm2: float = 0.02

    # Enclosed pale regions under this area are filled back in, because fat and
    # gland lumina are pale and fat is tissue (Rule 1). Above it, a void is a
    # real gap between two pieces of tissue and is left alone.
    tissue_fill_hole_max_mm2: float = 1.0

    # When one saturation level holds at least this share of the histogram, the
    # histogram is a spike and a tail rather than two humps, and Otsu - which is
    # derived for two humps - is replaced by Zack's triangle rule. Both answers
    # are always computed and reported; this only decides which one is used.
    #
    # 0.35 is well clear of both cases seen so far: a weakly counterstained IHC
    # slide puts ~0.82 of its pixels on one level, while an H&E slide's glass
    # spreads over many and lands far below this.
    tissue_spike_share: float = 0.35

    # --- Step 4: white calibration ----------------------------------------
    #
    # Cached per upload like step 3's, and for the same reason: the RGB
    # thumbnail this step measures I0 from is the expensive part, and the step
    # is re-run whenever step 3's threshold moves - a different tissue mask is
    # a different set of glass pixels.
    calibration_dir: Path = DATA_ROOT / "calibration"

    # Worked at step 3's resolution, and it is not a coincidence: sharing the grid
    # means the tissue mask needs no resampling at all, so the two steps agree
    # pixel-for-pixel about where the tissue is - and step 4 decides what counts
    # as glass *by position*, so a half-pixel disagreement at the tissue boundary
    # would be a real error rather than a rounding detail.
    #
    # There is deliberately no size cap of this step's own. The grid is step 3's,
    # so step 3's cap is the one that binds, and a second cap here could only ever
    # disagree with it.
    calibration_mpp: float = 2.0

    # Which percentile of the glass is I0. Not the maximum - a single hot pixel
    # or a specular reflection off the coverslip would set it, and I0 sits in a
    # denominator. Not the mean either, which any tissue leaking past the mask
    # drags down. The 95th is above the dust and below the outliers, and the
    # report carries the whole percentile ladder so the plateau is visible
    # rather than asserted.
    calibration_percentile: float = 95.0

    # Glass to discard before sampling, all physical:
    #   border     the outermost frame of the scan. Scanners vignette hardest
    #              at the edge of the field and the coverslip edge lives there.
    #   clearance  a halo around the tissue. The pixels just outside a section
    #              carry its edge, its mounting medium and the mask's own
    #              resolution error, and every one of those is darker than glass.
    calibration_border_um: float = 500.0
    calibration_tissue_clearance_um: float = 60.0

    # A scanner writes a constant into the part of the canvas it never imaged, and
    # on the demo's Aperio scans that constant is two thirds of the frame. It is
    # not glass and it is darker than glass, so it is excluded - but only when it
    # is unmistakably a digital constant rather than merely quantised glass.
    #
    #   fill_min_share      how much of the candidate glass a level must hold
    #                       before it is worth testing at all.
    #   fill_max_shoulder   the colours within one level of the candidate in
    #                       every channel, over the candidate's own count. Real
    #                       glass carries sensor noise so it always has such
    #                       neighbours; a digital constant has none. Measured on
    #                       the demo slide: the fill scores 0.0017 and the real
    #                       glass 0.24, and on an uncompressed scan the glass
    #                       scores 7 and up. 0.05 is thirty times above the fill
    #                       and five times below the tightest glass seen.
    calibration_fill_min_share: float = 0.05
    calibration_fill_max_shoulder: float = 0.05

    # And a halo around that fill, for the same reason there is one around the
    # tissue. The fill is excluded by matching its exact colour, which is what
    # makes the test safe - but a boundary between a digital constant and real
    # glass does not land on a pixel edge, and the thumbnail these exclusions run
    # on is a downsample, so the seam between the two is a blend of both. Those
    # blended pixels are not the constant, so the colour match keeps them, and
    # they are darker than glass, so they land in the dark tail the noise floor
    # is read from - which is the one place a handful of pixels decides the
    # answer.
    #
    # Measured on CAN_00270_26_H&E at 6.9 um/px: 7% of the sampled glass sits at
    # 174-178 against a body at 190-196, and that 7% lands exactly on the 5th
    # percentile. Those pixels have a median distance of 1.4 px to the fill and
    # 17 px to the tissue, so they are the fill's seam and not the section's
    # halo. Standing back 1 px takes the floor from 0.044 to 0.030 OD, 2 px to
    # 0.020, and 5 px to 0.018, where it stops moving. 60 um is 9 px on that
    # slide - past the plateau with margin, and the same distance step 4 already
    # stands back from the tissue, for a seam that is the same kind of edge.
    # I0 itself is a high percentile and does not move at all: 196 throughout.
    calibration_fill_clearance_um: float = 60.0

    # The illumination surface is fitted from patch samples rather than from raw
    # pixels: one percentile per patch is what makes the fit robust, and the
    # patches are also what the demo draws on the thumbnail.
    calibration_patch_um: float = 1000.0

    # A patch is only sampled when this share of it is glass. Below that its
    # percentile is measured from too few pixels to be a white point.
    calibration_patch_min_glass: float = 0.35

    # Fewest usable patches before a surface is fitted at all. Six terms in a
    # quadratic need six points to be determined and far more than six to be
    # trusted, so the flat white point stands until there are this many.
    calibration_min_patches: int = 12

    # The fitted surface is used instead of one flat value only when its swing
    # across the slide is at least this many times its own residual scatter.
    # Below that the surface is fitting the noise in the samples rather than the
    # lamp, and a curved white point that follows noise is worse than a flat one
    # that does not. Both are always computed and always reported.
    calibration_vignette_snr: float = 4.0

    # And the surface is only used when it is actually pinned down over the tissue,
    # which is where every optical density downstream is computed. Glass samples
    # necessarily ring the section, so the frame's corners are always extrapolated
    # and always irrelevant; what matters is whether the ring encloses the tissue.
    # This is the 99th percentile of the fit's extrapolation leverage over the
    # tissue - around 1 at the edge of the samples' spread, growing without bound
    # beyond it. Measured on the demo's slides: 0.88, so the ring does enclose them.
    calibration_max_leverage: float = 1.0

    # Intensity floor for the Beer-Lambert transform, so a black pixel yields a
    # large finite optical density instead of an infinity. Applied to the
    # observed intensity only, never to I0.
    calibration_od_floor: float = 1.0

    # --- Step 5: optical density ------------------------------------------
    #
    # Nothing is cached on disk here, and that is a decision rather than an
    # omission. Steps 3 and 4 cache because their input is the whole slide
    # re-read; step 5's input is one 512 px tile, which is a thousandth of the
    # pixels and a single `read_region` off the pyramid. The white point it
    # divides by comes from step 4's service, which is already cached, and the
    # tile chooser scores its candidates on the thumbnail that service holds. A
    # disk cache would have to be keyed on step 3's threshold, step 4's
    # percentile and the tile position at once, and it would save less time than
    # invalidating it correctly would cost.
    #
    # The resolution and tile size are step 1's - `target_mpp` and `tile_size`
    # above - and deliberately not step 5's own. The working magnification is one
    # decision for the whole pipeline, made and reported at step 1; a second copy
    # here would be a second answer to the same question.

    # Share of a candidate block that must be tissue before step 5 will stand on
    # it. High, because a tile straddling the section's edge has glass in it, and
    # glass in the point cloud is a spray of near-origin noise pointing in every
    # direction - which looks exactly like a mixture of the two stains.
    density_min_tissue_share: float = 0.85

    # Mean optical density below which a pixel is treated as carrying no stain and
    # kept out of the cloud. Macenko's beta at his own value. Not cosmetic: the
    # cloud is about direction, and the direction of a near-zero vector is set by
    # whichever way the last quantisation step rounded.
    density_beta: float = 0.15

    # A block is only eligible to be the tile once its mean optical density clears
    # this many times the noise floor step 4 measured on this slide's own glass. A
    # multiple of a measured quantity rather than a constant, because "enough stain
    # to have a direction" is a statement about this acquisition.
    #
    # The gate exists because the tile chooser's `mixing` term is a ratio of two
    # eigenvalues of a point cloud, and a ratio means nothing until the cloud is
    # larger than the noise around it. Without it, a faint patch of false colour -
    # a scanning artefact, or chroma ringing along a high-contrast edge - outscores
    # real haematoxylin and DAB, because two wrong colours are still two colours.
    #
    # The 3 was first set against a noise floor of 0.048, on the reading that it
    # put the gate at 0.144 - between an artefact patch at 0.11 mean OD and a
    # stained duct at 0.21. That floor was wrong: `calibration_fill_clearance_um`
    # names the seam pixels that were inflating it, and the same slides now measure
    # 0.005 to 0.018. So the gate is 3 to 10 times lower than it was, and the
    # separation it appeared to provide against faint false colour was an artefact
    # of a contaminated measurement rather than a property of the multiple.
    #
    # It stays at 3, and the claim it makes is now only the one it can support:
    # the cloud has to be several times this acquisition's own noise before the
    # ratio of its eigenvalues means anything. No multiple of an honest floor can
    # also separate false colour from real stain - on the corrected floors the
    # artefact patch sits at 17x and a genuine H&E block at 7x, the wrong way
    # round - so that job is step 2's, and where step 2 misses, the candidate list
    # is on screen for the viewer to move off. Measured after the correction, on
    # four slides: the chosen tile is unchanged on every one.
    density_min_stain_multiple: float = 3.0

    # Which tail of the angular distribution is taken as a stain arm. Macenko's
    # alpha. The extremes of the wedge are the pure stains, but the literal
    # extremes are one pixel each, so a robust percentile stands in for them.
    density_arm_percentile: float = 1.0

    # How far 8-bit quantisation may move a pixel's direction before it is kept out
    # of the point cloud, in degrees. The mirror image of beta: beta excludes pixels
    # with too little stain to have a direction, this excludes the very dark ones,
    # where a stain has absorbed almost everything in a channel and one level of
    # intensity moves that channel's density by 1/(I ln10) - 0.014 at I=32, 0.43 at
    # I=1. Those pixels are also long, so they land at the angular extremes, which is
    # exactly where the arms are read from.
    #
    # Measured on the demo's slide: without this test the pixels beyond the low arm
    # have a median green channel of 1, and the arm they set lands nearest *eosin* on
    # a haematoxylin-and-DAB slide. With it, that arm lands on haematoxylin at 11-20
    # degrees on every tile tried and the wedge narrows from 68 degrees to 63, for 3%
    # of the tile - about a tenth of what would otherwise have been plotted.
    #
    # 1.5 is what 8-bit data supports, not a preference: at the beta boundary it asks
    # for a weakest channel of about 64 levels and at 1.5 OD for about 11. Half a
    # degree empties the cloud, which is the sign that the limit is the data's.
    density_angular_tolerance_deg: float = 1.5

    # Candidate tiles returned with the report. Enough that the viewer can try a
    # few alternatives and watch the arms close up on a worse one; far short of
    # the hundreds scored, because a list of hundreds is not a choice.
    density_candidates: int = 12

    # --- Step 6: colour deconvolution -------------------------------------
    #
    # Nothing cached on disk, and for a stronger reason than step 5's. Step 6's
    # input *is* step 5's output - it asks `density_service.transformed_tile` for
    # the density array rather than recomputing a logarithm - so its own work is
    # two 3x3 inverses and two matrix multiplies over a quarter of a million
    # pixels. A cache keyed on step 3's threshold, step 4's percentile and the
    # tile position would cost more to invalidate correctly than the arithmetic
    # costs to redo.
    #
    # There is deliberately no stain-vector setting here. The vectors are Ruifrok
    # & Johnston's published constants and they live in `app.common.stains`, which
    # step 5 already reads its arms against. Making them configurable would make
    # the DAB scale a deployment choice, and a measurement whose units depend on a
    # config file is not comparable across the study it was built to compare.
    # Calibrating them once per scanner on control slides and freezing them, as
    # the guide suggests, is a change to that constant - not a knob.

    # The absolute DAB optical density a pixel must clear to count as positive in
    # step 6's preview score. Absolute, and that is the entire point: it is applied
    # identically to both bases, so the gap between the two scores on screen is
    # caused by the choice of stain vectors and by nothing else. A per-slide
    # percentile would return the same number under either basis and under any
    # staining whatsoever - which is exactly how a pipeline normalises the
    # diagnosis away without noticing (primer item 79).
    #
    # 0.25 sits well above the optical-density noise floor a working scan leaves
    # (0.05 or so on the demo's slides) and well below a genuinely strong 3+
    # membrane, so a tile of counterstain alone scores near zero and a positive
    # tile does not saturate. It is a demo cut and says so: step 14 sets the real
    # cut-points, five sets of them, one per antibody and against clinical
    # guidance rather than against a round number.
    deconvolution_positive_cut: float = 0.25

    # --- Step 7: tiling ---------------------------------------------------
    #
    # Plumbing, and nothing here is cached: the index is derived from step 3's
    # mask, which is already cached on disk, and building it is one pass over a
    # few tens of thousands of grid cells. A cache would have to be keyed on the
    # mask, the QC state, the overlap and the two gates at once, and would save
    # less than invalidating it correctly would cost.
    #
    # The tile size and the working resolution are deliberately *not* here, and no
    # longer step 1's either: this step lays the **region model's** square, read from
    # the published checkpoint's manifest by `tiling_service.window`. The model's
    # field of view is a property of its checkpoint and cannot be overridden, so a
    # tile size chosen here could only ever disagree with the grid that actually
    # runs - which is what let step 7 price "4,952 squares" and step 8 then make
    # 98,262 passes over the same slide. `tile_size` and `target_mpp` above remain
    # step 1's and steps 4-6 still work at them; they are this step's fallback only
    # when no checkpoint is published, where there is no window to agree with.

    # Fraction of a tile's own extent that it shares with its neighbour. **None.**
    #
    # Overlap exists because a model has no context beyond a tile's edge, so its
    # predictions there are its worst; overlapping lets those edges be averaged
    # away instead of being stitched into visible seams across the class map. It
    # is not free - it is the one parameter on this step that costs compute, and
    # it costs it quadratically. At 25% the grid holds 1.8x the tiles of a
    # non-overlapping one; at 50%, 4x. Every one of those is a forward pass at
    # step 8 - which runs this exact grid, so the count this step prices is the
    # bill rather than a proxy for it. Measured on `CAN_00251_26_A`, 0 / 0.25 / 0.5
    # is about 10 / 15 / 29 minutes at step 8.
    #
    # This used to be 0.25 - the guide's 25-50% band at its lower end - with step 7
    # offering all three as buttons. The band is still what the guide asks for and
    # the seams are still real; the demo trades them for the ten-minute run. Step 7
    # now offers this setting alone, so changing this value changes what that one
    # button means, and the endpoint's `overlap` query parameter remains the way to
    # run the other geometries.
    tiling_overlap: float = 0.0

    # Step 7's other control, and the one that decides which checkpoint step 8 runs:
    # **the field of view**, in microns of slide inside one square.
    #
    # The served 112 um window (224 px at 0.5 um/px) is the reason this exists. A
    # high-grade solid or comedo duct is 300-1500 um across, so a 112 um window centred
    # inside one holds a sheet of tumour cells with no basement membrane, no periductal
    # stroma and no rim - physically indistinguishable from invasive. That is not a
    # training deficiency, it is arithmetic, and it is why the served checkpoint records
    # `dcis_called_invasive = 0.224` and why a pathologist's DCIS contour on
    # `CAN_00270_26_H&E` came back filled with invasive. See `FIX1_FIX2_PLAN.md`.
    #
    # So the field of view is a choice, offered as the four entries in
    # `tiling_service.FIELDS_OF_VIEW` - 112, 224, 448 and 672 um, each with a published
    # head. It is not the same kind of choice as the overlap: overlap moves only the
    # cost, while this moves **which model runs**, because a field of view is a property
    # of the weights. Step 7 resolves the checkpoint whose manifest declares this field
    # of view and step 8 runs that one - see `tiling_service.window`.
    #
    # **Widening it is not free, and the sweep measured both prices.** A wider window
    # covers more tissue per decision, so the corpus yields fewer independent windows:
    # 6,907 tiles at 224 um, 1,157 at 448, 268 at 672, where BACH contributes none at
    # all because its images are shorter than 672 um in one axis. And a coarse window
    # inflates whatever it detects - on `CAN_00270_26_H&E` the 448 um head reported
    # 77.5 mm2 of invasive against the 112 um head's 42.0 mm2 over identical tissue,
    # which is an 86 % swing in the tumour content this pipeline exists to report. So a
    # fine window measures area honestly and a wide one sees architecture, and no single
    # value does both. `RESULTS_224_VS_448.md` carries the numbers.
    #
    # 224 um is the default because it is the widest field that keeps a large tile
    # count; 112 um remains the honest choice for an area measurement.
    tiling_field_of_view_um: float = 224.0

    #: The branch the pipeline commits to. `he` - the colour photograph - at the 224 um
    #: field of view above, which resolves to `invasive_tile_fov224_he_concat`.
    #:
    #: **Deliberately not `DEFAULT_BRANCH`, which is a different question.** That
    #: constant answers "what does a manifest with no `input.channel` key mean", and
    #: the answer has to stay `h_channel` for ever, because that is what every
    #: checkpoint published before the H&E branch existed silently is. This setting
    #: answers "what does this pipeline run", and the two only looked like one question
    #: while there was one branch.
    #:
    #: **Why `he` rather than the h_channel head this started on.** The h_channel branch
    #: was built to serve all six slides of a case from one model, and measured it does
    #: not: 0% invasive on all five immunostained markers at 0.96-0.98 confidence. Since
    #: the region is found on the H&E and registered across (step 10), the generality
    #: that branch was paying for is generality the pipeline no longer uses - and on the
    #: slide it does run on, eosin is half the evidence. A duct wall and a collagen band
    #: are the same shade of nothing in the H channel.
    tiling_branch: str = "he"

    #: How close a step 5 cloud arm must sit to Ruifrok's published eosin direction
    #: before this pipeline calls a section haematoxylin-and-eosin and offers step 7's
    #: H&E branch.
    #:
    #: **Deliberately not `density_angular_tolerance_deg`.** That one is 1.5 degrees
    #: and bounds the *stability of a single pixel's direction* under resampling; this
    #: one compares a weighted percentile of a whole tile's point cloud against a
    #: literature vector measured on someone else's scanner with someone else's dyes.
    #: The two quantities live on entirely different scales, and 1.5 degrees here would
    #: reject every real slide.
    #:
    #: **Measured, not guessed.** Over 18 tissue tiles from this project's own slides,
    #: three tiles on each of six sections:
    #:
    #:   H&E (CAN_00270_26_H&E, CAN_00267_26_H&E) - an eosin-nearest arm every time,
    #:   at 5.0 to 9.4 degrees. Arm separation 18.8 to 38.1 degrees.
    #:
    #:   IHC (CAN_00270_26_A, _R, CAN_00267_26_W, _U) - haematoxylin at 14.0-22.3 and
    #:   DAB at 20.8-29.4 degrees. **No eosin-nearest arm on any of the twelve.**
    #:
    #: So the discriminating fact is *which* published vector an arm lands nearest, and
    #: on this panel that alone separates 18 of 18. This tolerance is the safety bound
    #: on top of it: 15 degrees sits comfortably above the 9.4 an H&E section actually
    #: measured, and refuses an eosin claim further out than any real one here was.
    #:
    #: Re-measure on a new scanner before trusting it - the vectors are Ruifrok's,
    #: measured on somebody else's dyes. `scripts/` has no runner for this; the
    #: measurement above came from reading tiles through `density.transform` and
    #: `staining.classify_staining` directly. Every verdict carries the angle it decided
    #: on, so a reader can see how close a call was rather than being handed a boolean.
    he_eosin_tolerance_deg: float = 15.0


    # Share of a tile that must be tissue before it is kept, and **the only place
    # this pipeline gates a square on how much tissue it holds.**
    #
    # It used to be 0.10 here and 0.50 again at step 8, on that step's own finer
    # grid. Once the two grids became one, two thresholds on the identical square
    # was simply two answers to one question - the screens disagreed by a factor of
    # 1.7 and only the second one was doing anything. The value kept is step 8's,
    # because that is the one with a measured consequence: at 0.1 the median window
    # called in-situ was 6% tissue, and the "structures" it found were a one-window
    # rim tracing the section's edge. Raising it to 0.5 removed two thirds of them
    # while dropping only 22% of windows - the glass-rim false positive, which is
    # what a mostly-empty window standardised against its own p99 produces.
    #
    # Still deliberately below step 5's 0.85, and the two steps still ask opposite
    # questions. Step 5 needs one tile whose every pixel it can trust, because it
    # measures a density there and glass in the point cloud is noise pointing in
    # every direction. Step 8 needs *all* the tissue, because a square it never sees
    # is a region it cannot classify - and a section's edge, where the invasive
    # front often sits, is exactly where squares are half glass. 0.5 excludes the
    # square that is mostly glass while keeping the one that is genuinely partial.
    tiling_min_tissue_share: float = 0.50

    # Share of a tile's *tissue* that step 2 must have left in play. Measured over
    # the tissue and not over the tile, because a tile that is half glass would
    # otherwise be credited for the glass being clean - and glass is always clean,
    # having nothing on it to be wrong.
    #
    # 0.5 rather than something stricter: an artefact mask is per-pixel and a tile
    # clipping the corner of one fold is still mostly usable tissue, while a tile
    # that is half fold is not usable at all. The tolerant reading here is safe
    # because it is not the last word - step 9's ROI mask and step 16's validation
    # both see the artefact map again.
    tiling_min_clean_share: float = 0.50

    # Refuse a grid larger than this rather than build it. Not a cap: a grid past
    # this size means either a corrupt slide dimension or a working resolution
    # finer than this pipeline was configured for, and truncating it would hand
    # step 8 an arbitrary two thirds of a section. At 0.5 um/px with 25% overlap
    # a 30 mm square scan is about 287,000 tiles on the region model's 224 px window,
    # so this leaves headroom for the scans this pipeline sees while still catching a
    # slide whose dimensions are nonsense. It is tighter than it was, because the
    # window is smaller than the 512 px tile this step used to lay.
    tiling_max_tiles: int = 400_000

    # --- Step 8: tissue-type segmentation ---------------------------------
    #
    # Cached on disk per upload, like step 2's and for the same reason: a pass is
    # tens of thousands of forward passes on a CPU, and the walkthrough asks for
    # the same class map every time someone steps onto step 8. Unlike step 2 the
    # cache is also what makes the step usable at all - nobody opens a screen that
    # takes half an hour twice.
    #: Where step 7 records the choice a viewer committed - the branch, the field of
    #: view, the overlap and the threshold behind them. Kilobytes per slide, and
    #: deliberately not the index: step 7 holds no pixels on disk and that stays true.
    #:
    #: **This exists because the choice had nowhere to live.** Step 8 called
    #: `tiling_service.report(upload_id)` with no arguments, so it rebuilt the grid at
    #: `tiling_field_of_view_um` no matter which head the caller had named - the two
    #: grids then diverged and step 8's window gate quietly stopped agreeing with step
    #: 7's tiles. Threading four parameters through every call site would have fixed
    #: the symptom while leaving the same failure available to the next caller who
    #: forgot one. A record the services read means forgetting is not possible.
    tiling_dir: Path = DATA_ROOT / "tiling"

    tissue_type_dir: Path = DATA_ROOT / "tissue_type"

    # Where published checkpoints live. Empty means `<workspace>/models/tissue_type`,
    # which is where `tissue_type_model_training`'s publish step writes - so the
    # trained model and the served model are the same file rather than two copies
    # that can drift.
    tissue_type_models_dir: Path | None = None

    # Which published checkpoint scores the slide.
    #
    # `_std` is the variant fitted with per-tile standardisation, which is the
    # property that makes the H&E-trained model's input distribution contain the
    # IHC one it is served (coverage of the served percentiles went 1/6 -> 6/6).
    # The non-`_std` checkpoint is the ablation baseline and lacks it.
    #
    # ⚠ **This default is RESEARCH ONLY.** It is fitted on BCSS (CC0) *plus* BRACS
    # regions of interest labelled by BEETLE's released nnU-Net - non-commercial
    # and ShareAlike respectively - because BCSS supplies nine non-invasive
    # epithelium tiles in its entire release and none at all in its held-out
    # institutions, so the in-situ class could be neither fitted nor measured from
    # it alone. The model inherits both restrictions and so does every number
    # computed from it. `invasive_tile_v1_imagenet` is the commercially clean
    # BCSS-only checkpoint; it cannot separate in-situ from invasive disease.
    #
    # The licence track is read from the chosen checkpoint's manifest and reported
    # on screen rather than trusted to this comment - see
    # `step08_tissue_type_segmentation/model.py`.
    # v2 differs from v1 in *which source taught which class*, not in architecture:
    # BRACS regions were selected for being DCIS-heavy, so its 270 invasive and 662
    # non-epithelium tiles are a side effect of that selection - BEETLE's guesses
    # inside DCIS ROIs - against BCSS's 5,282 and 5,435 drawn by hand. v2 drops them,
    # so BCSS teaches invasive and non-epithelium and BRACS teaches in-situ only.
    # Every held-out number improved and the fold spread tightened; see the training
    # folder's `datasets.SOURCE_CLASSES`.
    #
    # **v3 differs from v2 in three ways and each is load-bearing.**
    #
    # 1. *Two bodies.* ImageNet and SimCLR ResNet18 features concatenated to 1024. The
    #    two initialisations describe a tile differently and neither is ever updated, so
    #    this is two descriptions rather than one twice - the single largest measured
    #    gain of any change tried. **It costs two forward passes per tile**, so step 8's
    #    wall clock roughly doubles. That is the price of the accuracy.
    # 2. *A threshold, not an argmax.* v3's manifest carries `metrics.decision_rule.tau`
    #    and serving MUST apply it - `Pinned.tau` and `inference._decide` do. A three-way
    #    argmax lets a badly-conditioned in-situ head outvote invasive, which is exactly
    #    the failure v2 shows on CAN_00251_26_H&E.
    # 3. *A third laboratory.* BACH (Porto) joins BCSS and BRACS. This bought almost no
    #    accuracy on comparable data (+0.002) but broke the confound that made in-situ
    #    ~99.7% one laboratory, and measured the thing that matters: on a laboratory
    #    absent from training, in-situ recall is ~0.29 against invasive's ~0.77.
    #
    # **LICENCE, UNRESOLVED.** BACH is CC BY-NC-ND 4.0 - NoDerivatives - and whether a
    # model fitted on it is a derivative work has not been decided. v3 was published at
    # the project owner's explicit direction with that open. Set this back to
    # `invasive_tile_v2_imagenet_std` to serve the NC/SA-only model.
    #
    # **Superseded, and the default moved off it.** This is no longer what a run
    # resolves to: `tiling_service.model_for(fov, branch)` picks the head fitted at
    # step 7's committed choice, and at 112 um `FIELD_OF_VIEW_PREFERENCE` names v3's
    # Fix 1 successor. What this setting still does is order `discover()` and answer a
    # caller who names nothing directly, so it should point at a head that matches the
    # default field of view and passes its own gate - and v3 does neither.
    #
    # v3 fails G6: `scripts/check_tissue_model.py` replays the twelve probe tiles its
    # manifest records and gets a worst logit difference of 1.0e+01, where
    # `invasive_tile_fov112_fix1_concat` - the same 224 px at 0.5 um/px, the same tile
    # store - reproduces to 3.6e-06. So the store is not the problem; v3's recorded
    # logits were written either against a differently resampled export or by a publish
    # path that no longer exists. Its weights may well be fine. They cannot be checked,
    # which for the *default* is reason enough to move.
    #
    # `invasive_tile_fov224_concat` matches `tiling_field_of_view_um` above, so the two
    # defaults now describe one grid instead of two.
    tissue_type_model: str = "invasive_tile_fov224_concat"

    # Fraction of its own field of view a window shares with its neighbour. The
    # guide's stride 112 for a 224 px window: a model has no context past a window's
    # edge, so overlap is what lets step 10 average those edges away instead of
    # stitching them into visible seams across the class map.
    #
    # **This is a fallback, not the value a run normally uses.** Step 8 takes step
    # 7's overlap - see `tissue_type_service.resolve_params`. The two grids differ in
    # size and resolution because the model's field of view is a property of its
    # checkpoint, but overlap is the one parameter they share, and step 7's picker
    # prices the choice in squares on screen. Honouring it there and ignoring it one
    # step later made that price a fiction. This value is what a caller gets when
    # there is no explicit choice to inherit.
    #
    # **It costs roughly linearly in windows, and that is measured.** On
    # `CAN_00251_26_A` (199 mm2 kept tissue, 4 threads) overlap 0 / 0.25 / 0.5 is
    # 15,713 / 27,880 / 62,962 windows and about 10 / 15 / 29 minutes; the 0.5 run is
    # the measured one at 1,757.7 s. Forward passes are 65-86% of the time and scale
    # with the window count, while reads are nearly flat in overlap - 3.5 to 4.1 min -
    # because a block spans its own windows' extremes either way. An earlier note here
    # claimed 4x the windows cost only a fifth more time; that came from an overlap-0
    # run on a contended box, where the same geometry has been clocked at both 1,757.7
    # and 3,816.5 s. Always check machine load before trusting a step-8 timing.
    tissue_type_overlap: float = 0.5

    # Windows per axis in one read. A window is 224 px and a `read_region` off the
    # pyramid costs far more than 224 px of pixels, so windows are read in blocks
    # and cut up in memory - identical predictions, a fraction of the reads. 8 is
    # 64 windows and about a megapixel per read at the model's own resolution.
    # Share of a *window* that must be tissue before the model is shown it.
    #
    # **Normally a pass-through, and kept anyway.** Step 7 now lays this step's own
    # square and gates it at this same 0.5, so in an ordinary run every window has
    # already cleared this and `grid.gated_out` is zero. What it still covers is the
    # case where the two grids do not coincide - a caller naming an overlap of its own
    # on this step's route, which is the one geometry parameter that route may still
    # move. Deleting it would leave that case ungated, and the measurement below is
    # what that costs.
    #
    # The measurement, from when this was the only gate at 0.5 and step 7's was 0.10:
    #
    # measured on CAN_00251_26_A with the v2 checkpoint, the median window classified
    # as in-situ epithelium was **11% tissue and 89% glass**, and 69% of them were under
    # half tissue - against a median of 100% for non-epithelium. Those windows drew a
    # one-window rim around the whole section that looked like duct structures and was
    # not. Raising this to 0.5 removed two thirds of the in-situ calls and cost only 22%
    # of windows, because the ones it drops were nearly empty.
    #
    # Why they misfire: a mostly-glass window has a low 99th-percentile density, so the
    # per-tile standardisation multiplies its little signal up to the target and invents
    # texture the slide does not have. `STANDARDISE_FLOOR` guards the extreme case; a
    # window with a tenth of a tile of real tissue clears it easily.
    tissue_type_min_tissue_share: float = 0.5

    tissue_type_block_windows: int = 8

    # Windows per forward pass. Small on purpose: the whole batch is held as
    # float32 3x224x224 (600 kB each), and on a CPU the throughput curve is flat
    # past about this point while the memory is not.
    tissue_type_batch_size: int = 32

    # Refuse a grid larger than this rather than build it, for step 7's reason: a
    # grid past this size means a corrupt slide dimension or an overlap nobody
    # meant to ask for, and truncating would hand step 9 an arbitrary part of a
    # section with no record of which part.
    tissue_type_max_windows: int = 400_000

    # --- Step 8, the uncertainty layer -------------------------------------
    #
    # The post-pass that gives the step a fourth answer - "cannot determine" - over the
    # windows the model called in-situ. See
    # `app/pipeline/step08_tissue_type_segmentation/uncertainty.py` for what each of
    # these does and, more usefully, for why the obvious version of this rule does not
    # work. Every one of them is a tuning knob rather than a measured constant.
    #
    # Neighbourhood scale, in **microns**. Not a count of windows: the grid's stride
    # halves when the overlap goes from 0 to 0.5, so a neighbourhood fixed in cells
    # would change size with a parameter that is only meant to change how densely the
    # tissue is sampled. About one duct system, which is the scale at which "does the
    # surrounding tissue corroborate this call" is a question worth asking.
    tissue_type_uncertainty_sigma_um: float = 250.0

    # Where a connected in-situ component stops being a plausible duct system. The ramp
    # is logarithmic between these two, because component areas span orders of
    # magnitude. These are the softest numbers here - extensive DCIS is genuinely a
    # multi-millimetre thing - and they are deliberately permissive: the measured false
    # in-situ field on CAN_00251_26_H&E was 141 mm2, which clears the ceiling seven times
    # over, so nothing subtle rests on where between 2 and 20 the ramp sits.
    tissue_type_uncertainty_area_ref_mm2: float = 2.0
    tissue_type_uncertainty_area_max_mm2: float = 20.0

    # Bounding-box fill at which a component starts and stops reading as a solid sheet
    # rather than as ducts. The measured real duct component on CAN_00270 filled 0.28 of
    # its box.
    tissue_type_uncertainty_fill_lo: float = 0.35
    tissue_type_uncertainty_fill_hi: float = 0.75

    # And the size below which solidity says nothing at all. A duct traced along its
    # length is hollow, but the same duct cut across is a solid disc - a solid 4x4 block
    # of windows is 0.2 mm2, a small comedo-type DCIS focus, and an earlier version of
    # this gated on a window count and flagged it outright. So the gate is an area, with
    # a window floor beside it for the coarsest geometries where one window is already a
    # fifth of a square millimetre.
    tissue_type_uncertainty_min_shape_mm2: float = 1.0
    tissue_type_uncertainty_min_shape_windows: int = 12

    # The one number that decides how much purple appears. Tune this on held-out slides
    # before either of the two above.
    tissue_type_uncertainty_threshold: float = 0.5

    # --- Step 8, the BEETLE option ----------------------------------------
    #
    # The third option on step 7 runs a published nnU-Net from another group, per pixel,
    # instead of one of our fitted ResNet18 heads. Nothing above applies to it except
    # `tissue_type_min_tissue_share` and the grid limits, because it shares step 7's
    # geometry and nothing else. See
    # `app/pipeline/step08_tissue_type_segmentation/beetle.py`.

    # How many of the five released folds to average.
    #
    # **One, and the reason is arithmetic.** A fold is a full forward pass over every
    # patch, so five folds is five times a pass that is already the longest step in the
    # pipeline. What the extra four buy is a disagreement map, which is the only
    # uncertainty signal available without a pathologist - genuinely useful, and nothing
    # downstream currently reads it. Raise this when something does.
    #
    # A single fold's spread is zero everywhere, and that is *one fold* rather than
    # consensus. The report says which folds ran so the number cannot be misread as
    # agreement.
    tissue_type_beetle_folds: int = 1

    # The sliding window's step inside one window, as a fraction of BEETLE's 512 px patch.
    #
    # **0.5, which is nnU-Net's own inference default, and this is not a place to
    # economise.** The step is a quadratic cost - 0.5 needs four times the forward passes
    # 1.0 does - so 1.0 was tried first and it fails, visibly and in the worst possible
    # way.
    #
    # Measured on a 672 um window over a duct/invasive boundary on CAN_00270_26_H&E at
    # (49357, 22425), with the patch origins already spread evenly by
    # `beetle.patch_grid` rather than packed flush:
    #
    #   step 1.0   9 patches, 14 s   the 3x3 patch grid is plainly visible as a
    #                                CHECKERBOARD, and neighbouring squares disagree
    #                                between in-situ and invasive - the seam does not
    #                                blur the boundary, it inverts the class across it
    #   step 0.75 16 patches, 23 s   mostly coherent, some blocking remains
    #   step 0.5  25 patches, 29 s   coherent regions, duct outlines continuous
    #
    # Why it fails so badly rather than merely looking rough: BEETLE's in-situ/invasive
    # call is genuinely context-dependent at its 256 um patch, so adjacent patches given
    # different context reach different answers, and with only a 96 px overlap the
    # Gaussian has almost nothing to blend across. Averaging in probability space over a
    # half-step is what reconciles them, and it is also what the released model's own
    # reported numbers were produced with.
    #
    # This is the single largest cost in the pass. Raising it to 1.0 makes a whole-slide
    # run about four times faster and the in-situ/invasive boundary - the one thing this
    # step exists to draw - unreliable. Do not.
    tissue_type_beetle_patch_step: float = 0.5

    # Patches per forward pass, within one window.
    #
    # Measured on this machine: one 512x512 patch is 1.26 s, four together are 1.12 s
    # each. The curve is nearly flat, and each patch in flight costs 3x512x512 float32
    # in plus 5x512x512 out, so a large batch buys little and costs memory that the
    # accumulators already want.
    tissue_type_beetle_batch_size: int = 4

    # Microns per pixel the slide-level pixel mask is kept at.
    #
    # BEETLE answers at 0.5 um/px, and a whole section at that resolution is a gigapixel
    # - which is not a file this step can write, let alone one a browser can be handed.
    # So each window's mask is reduced onto a slide-level canvas at this resolution.
    #
    # 4.0 um/px is a 64x area reduction from the network's own spacing, and it is chosen
    # against what has to stay readable rather than for a round number: a duct is 300 to
    # 1500 um across, which is 75 to 375 pixels here, so duct *shape* - the thing that
    # separates in-situ from invasive, and the whole reason a per-pixel model is worth
    # running - survives intact. It is also 2x finer than step 3's own mask, so the
    # pixel map is never the coarsest thing in the picture.
    tissue_type_beetle_mask_mpp: float = 4.0

    # Side of the per-window mask sent to the browser while a pass runs, in pixels.
    #
    # The progress screen paints each window as it comes back, and what it needs is only
    # what its canvas can show: that canvas is 1200 px for the whole slide, so a window
    # occupies roughly 10 to 30 px of it. 32 is comfortably past that and costs 1 kB a
    # window before base64 - a few megabytes over a whole-slide pass, each window
    # travelling exactly once.
    tissue_type_beetle_paint_px: int = 32

    # --- Storage housekeeping ---------------------------------------------
    #
    # Measured on this machine: `data/` is 1,542 MB and **one uploaded slide is 1,524 MB
    # of it**. Every per-step cache together is 19 MB, about 1 %. So clearing caches is
    # not a storage strategy - it saves a hundredth of the space and throws away the
    # expensive part (step 8's class map is half an hour of CPU; the tissue mask under it
    # is seconds). Deleting a *slide* is what frees space, and it costs a re-upload.
    #
    # Nothing here deletes a slide on its own. See app/services/maintenance_service.py.

    # Sweep unreachable data at start-up: caches whose slide no longer exists, and upload
    # parts that were already reassembled. Both are waste by construction - nothing can
    # read them again - so this needs no confirmation and is on by default.
    cleanup_orphans_on_start: bool = True

    # Free a slide's derived caches when the browser goes away.
    #
    # **Off by default, and it should usually stay off.** A backend cannot tell a closed
    # tab from a refresh - `pagehide` fires for both - so this is a guess about intent,
    # and the thing it guesses away is step 8's class map, which is half an hour of CPU
    # to rebuild against about 19 MB of disk saved.
    #
    # What makes it survivable at all is the grace period below rather than the flag:
    # the disconnect only *schedules* a release, and the page re-claims the slide as
    # soon as it loads again, which a refresh does within a second. So a refresh
    # cancels it and a genuine close does not.
    cleanup_on_disconnect: bool = False

    # How long a slide stays claimable after the browser goes away. Long enough that a
    # refresh, a crash-and-reopen or a moment on another tab all get there first.
    cleanup_disconnect_grace_seconds: float = 120.0

    # Which scope a disconnect releases. **Deliberately not settable to `slide`**: the
    # trigger is a guess, and a wrong guess that deletes a 1.5 GB upload is a different
    # order of mistake from one that costs a re-run. See `maintenance_service.release`.
    cleanup_disconnect_scope: str = "derived"

    # Slides untouched for this many days are *reported* as stale by
    # `maintenance_service.stale_uploads`. Zero disables it, and nothing deletes them
    # automatically at any value: "old" is a judgement about how this machine is used,
    # and a demo slide somebody returns to next month is not waste.
    data_retention_days: float = 0.0

    # CORS
    backend_cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )

    @field_validator("backend_cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: str | list[str]) -> list[str]:
        """Allow the origins list to be supplied as a comma-separated string."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


    # --- step 9, the ROI mask ------------------------------------------------
    #
    # Cached per upload, but not for step 8's reason. The region itself takes
    # about a tenth of a second - a blur and a closing over a few hundred cells a
    # side - and the ~20 s a cold build costs is the thumbnail and the panels.
    # The cache exists because the ROI is the denominator every later step
    # measures inside, and steps 10 to 15 must all see the same one.
    roi_dir: Path = DATA_ROOT / "roi"

    # Gaussian blur applied to P(invasive) before thresholding, in grid cells.
    # Smoothing runs on probabilities rather than labels because averaging
    # argmaxes is a vote, and a vote has already discarded the margin.
    roi_sigma: float = 1.0

    # Where the smoothed probability becomes region. Deliberately a server
    # setting rather than a per-slide control: a threshold moved per slide is a
    # score moved per slide.
    roi_threshold: float = 0.5

    # Closing radius in grid cells - how far apart two patches of tumour may be
    # and still be called one focus. At step 8's 112 um window, 3 cells is about
    # a third of a millimetre.
    roi_close_cells: int = 3

    # In-situ probability at which a window is carved back out of the closed
    # region. This is Rule 5 defended against the closing above: a radius large
    # enough to merge neighbouring tumour is large enough to swallow the in-situ
    # ducts lying between them, and in-situ carcinoma is not scored. Measured on
    # CAN_00270_26_H&E, a plain close at radius 3 annexed 20.3 % of the in-situ
    # inside the pathologist's DCIS contour; carving it back cost 1.4 mm2 of
    # 37.6 and returned that to 1.6 %. Set to 0.0 for the guide's plain recipe.
    roi_protect_in_situ: float = 0.5

    # Components below this are speckle, not foci. In mm2 rather than pixels so
    # the cutoff survives a change of scanner, which is the guide's own
    # instruction for this parameter.
    roi_min_area_mm2: float = 0.25

    # Keep only the N largest foci, matching how a pathologist circles one or
    # two. None keeps every component that cleared the area cutoff.
    roi_keep_largest: int | None = None

    # Margin added around a top-3 region's bounding box before it is cropped from
    # the slide, in microns rather than pixels so it survives a change of scanner.
    # A crop tight to the boundary shows nothing beyond it; this is what puts
    # surrounding tissue in the picture.
    roi_crop_pad_um: float = 250.0

    # --- step 10, reviewing the candidate regions ----------------------------
    #
    # Step 9's `class_regions` already traces every patch of invasive tile the
    # model drew, 13 to 20 of them on a typical section. Until now they were a
    # picture; this step makes them a *choice*, because what follows is BEETLE
    # and BEETLE is the most expensive thing in the pipeline per square
    # millimetre. Nothing here segments anything - it crops, ranks and records
    # a selection - so the cost is one slide read per candidate.
    roi_selection_dir: Path = DATA_ROOT / "roi_selection"

    # Candidates below this are never offered. `borders.class_regions` keeps
    # every component with no cutoff on purpose (it is showing the model's raw
    # call), so a busy slide carries dozens of single-window specks; offering
    # them would bury the regions a reader is actually choosing between under
    # noise, and a single window of tile is too little for BEETLE's context to
    # mean anything anyway. In mm2, like `roi_min_area_mm2`, so it survives a
    # change of scanner.
    roi_selection_min_area_mm2: float = 0.10

    # Hard cap on how many candidates are offered, largest first. A refusal
    # would be wrong here - a finely dispersed tumour is a property of the
    # disease - but a grid of 200 cards is not a review, it is a wall.
    roi_selection_max_candidates: int = 40

    # Which candidates arrive pre-ticked: enough of the largest to cover this
    # share of the invasive area. The same rule step 12 uses to decide what to
    # carry across, applied one step earlier so the default selection and the
    # default carry agree. A person may tick or untick anything on top of it.
    roi_selection_default_coverage: float = 0.95

    # Longest edge of a candidate's card image. Small, because a card is a
    # thumbnail in a grid of twenty and the full-size crop is one click away.
    roi_selection_crop_px: int = 512

    # --- step 11, refining each chosen region per pixel ----------------------
    #
    # BEETLE, but not over the slide. Step 8's tile head answers per window -
    # 224 um of slide gets one label - and this step replaces that square
    # staircase with the actual tumour boundary inside the regions a person
    # chose. The whole architecture of the step is that the candidate regions
    # are a *computational filter*: a section is 800 mm2 and its invasive
    # carcinoma is 40, so running BEETLE on the chosen regions alone is the
    # difference between a pass measured in hours and one measured in minutes.
    roi_refinement_dir: Path = DATA_ROOT / "roi_refinement"

    # Context added around a candidate's bounding box before BEETLE sees it.
    # Not cosmetic and not the same as `roi_crop_pad_um`, which pads a picture:
    # this pads the *input*, and a crop tight to the tile boundary would truncate
    # tumour at the edge of the box and ask a segmentation network to decide a
    # boundary with nothing beyond it. One window's worth is the smallest
    # padding at which every pixel of the candidate is seen by a window whose
    # centre is inside it.
    roi_refinement_pad_um: float = 224.0

    # The field of view BEETLE reads each window at. Fixed here rather than
    # inherited from step 7: step 7's choice belongs to the tile head whose
    # checkpoint was fitted at it, and BEETLE's weights are identical at all
    # four fields of view (only the extent of the array changes). 224 um is the
    # middle of the four and the one this project measured the branch on.
    roi_refinement_fov_um: float = 224.0

    # Resolution the refined mask is kept at, in microns per pixel. BEETLE
    # answers at 0.5; the slide-wide pass reduces to 4.0 because a section at
    # 0.5 um/px is 3.2 gigapixels. A region is not a section - a 5 mm2 focus at
    # 1 um/px is a few megapixels - so this step can afford four times the
    # linear precision the slide-wide pass could, which is the point of only
    # running on regions.
    roi_refinement_mask_mpp: float = 1.0

    # Share of a window that must be tissue before BEETLE is shown it. The same
    # gate step 8 applies and for the same measured reason - a mostly-glass
    # window returns a confident wrong answer - but lower, because a window here
    # has already been chosen by a person as part of a tumour region and the
    # cost of missing its rim is a truncated boundary.
    roi_refinement_min_tissue_share: float = 0.25

    # Specks below this are dropped from the refined mask, and holes below it
    # are filled. Both in mm2. A per-pixel network run at 1 um/px produces
    # single-pixel noise that a per-window one cannot; leaving it in would turn
    # one tumour boundary into four hundred rings, every one of which would be
    # densified, warped by VALIS and drawn.
    roi_refinement_min_component_mm2: float = 0.005
    roi_refinement_min_hole_mm2: float = 0.005

    # Douglas-Peucker tolerance applied to a traced boundary, in microns. The
    # trace is exact and rectilinear - it runs along mask-pixel edges - so at
    # 1 um/px a millimetre of boundary is a thousand vertices of staircase. This
    # is what turns that into a polygon; it is set at twice the mask pixel so it
    # can remove the staircase and nothing larger.
    roi_refinement_simplify_um: float = 2.0

    # Longest edge of a region's before/after images. Larger than the selection
    # card: this is the picture the whole step exists to show - the square tile
    # candidate beside the boundary BEETLE drew inside it - and it is shown one
    # region at a time rather than twenty to a grid.
    roi_refinement_crop_px: int = 900

    # --- step 12, aligning the ROI onto the IHC slide ------------------------
    #
    # Cached per case-and-marker, and this is the strongest cache in the
    # pipeline: a registration is minutes of CPU, the transform that comes out
    # of it is a few hundred kilobytes, and it cannot change unless one of the
    # two slides does. VALIS writes its own registrar pickle into the same
    # directory, so a second pass over the same pair reloads instead of
    # re-registering.
    ihc_alignment_dir: Path = DATA_ROOT / "ihc_alignment"

    #: Where the isolated registration environment lives. None means the default
    #: location beside the backend, `tissue_scoring_demo/valis_service/`.
    #:
    #: It is a separate Python because it has to be: its dependencies pin
    #: `numpy<2.0`, which has no build for the Python 3.13 this backend runs
    #: on. Step 12 applies its stored SimpleITK transform there as a subprocess.
    #: VALIS itself, and the settings that tuned it, were retired to
    #: `storage/decrecated_code/valis_registration/`.
    valis_service_dir: Path | None = None

    #: Longest gap between consecutive ring vertices before warping, in microns.
    #:
    #: Step 9's rings are traced on step 8's tile grid, so two neighbouring
    #: vertices can be a whole window apart - hundreds of microns. A non-rigid
    #: warp is not affine, so the straight line between two warped vertices is
    #: not the warp of the straight line between them: every deformation along
    #: that edge would be dropped. Densifying to this spacing first, and warping
    #: every vertex, is what makes the moved border follow the tissue instead of
    #: merely starting and ending in the right place.
    registration_vertex_spacing_um: float = 40.0

    # --- step 10's confidence gate -------------------------------------------
    #
    # VALIS returns a transform for a pair it could not really register, and it
    # will be wrong silently - the failure mode
    # `docs/segmentation_research/segmentation-approaches-implementation.md` Part 7
    # names explicitly.
    # These four thresholds are what turn that into a visible refusal. They are
    # deliberately permissive rather than tight: the step also puts the overlay
    # in front of a person, and a gate that blocks usable registrations would
    # just teach people to bypass it.

    #: Fewest matched features before a registration is believable at all.
    registration_min_keypoints: int = 50

    #: Largest median residual between matched features, in microns, after the
    #: non-rigid pass. Tens of microns is within the blockiness the ROI already
    #: carries from step 8's tile grid; hundreds is a different region.
    registration_max_residual_um: float = 250.0

    #: How different the two sections' tissue areas may be before the pair is
    #: refused. The near-blank IHC section in the docs' own worked example sits
    #: at 17.5 mm2 against the H&E's 84.8 - a ratio of 0.21, which this refuses.
    registration_tissue_ratio_min: float = 0.5
    registration_tissue_ratio_max: float = 2.0

    #: Largest median round-trip error, in microns: warp H&E -> IHC -> H&E and
    #: see how far the points came back from where they started. Unlike the
    #: residual above, this needs no matched features to be meaningful - it
    #: measures whether the transform is self-consistent over the actual region
    #: being moved, rather than over the keypoints it was fitted on.
    registration_max_round_trip_um: float = 100.0

    #: How many invasive-carcinoma regions are carried onto the IHC slide.
    #: A cap, not a target - see `ihc_alignment_area_coverage` below, which is
    #: what actually decides. It exists because a hundred speckle regions is not
    #: a scoring plan, and because every ring is warped and drawn.
    ihc_alignment_region_count: int = 24

    #: Share of the invasive area the carried regions must reach before the
    #: selection stops. OncoStem's procedure (SOP 4.2) is that the entire slide
    #: is scanned and every field averaged, so measuring inside the three
    #: largest regions - 63% of the invasive area on CAN_00270 - leaves more
    #: than a third of the tumour unread with nothing said about it.
    #:
    #: Not 1.0: the tail of step 9's region list is single-tile speckle whose
    #: rings cost more to carry than the area they add, and `registration`
    #: warps every vertex. 0.9 reaches the same tumour a reader would scan
    #: without chasing fragments smaller than the sampling tile.
    ihc_alignment_area_coverage: float = 0.9

    #: Regions smaller than this are never carried, however much coverage is
    #: still wanted. At the 0.0538 mm2 sampling field, a region below this holds
    #: about four fields - too few for its own percentage to mean anything, and
    #: a region that cannot carry a percentage cannot contribute one to an
    #: average.
    ihc_alignment_min_region_mm2: float = 0.25

    # --- step 11, nuclei ------------------------------------------------------

    nuclei_dir: Path = DATA_ROOT / "nuclei"

    #: Where the InstanSeg checkpoint lives. None means `<workspace>/models/nuclei`,
    #: beside step 2's and step 8's, for the same reason they are there: model
    #: weights are fetched by `setup.py` against `models.lock.json` and are not
    #: per-run state, so they do not belong under `data/`.
    nuclei_models_dir: Path | None = None

    #: Field of view handed to the segmenter, in pixels of its own input scale.
    #:
    #: Measured on this machine at 4 threads: 256 px runs at 79 s/mm2, 512 at
    #: **60**, 1024 at 78 and 2048 at 89. The dip in the middle is the usual
    #: shape - below it the per-call overhead dominates, above it the
    #: post-processing works on a larger label map than fits comfortably in
    #: cache. 512 px at 0.5 um/px is a 256 um square, which is also close
    #: enough to step 8's 224 um window that the two screens show fields of
    #: roughly the same size.
    nuclei_tile_px: int = 512

    #: Nuclei whose centroid falls within this margin of a tile edge are found
    #: but not *counted*, and the margin is excluded from the counted area too.
    #:
    #: A nucleus straddling a tile boundary is seen by the model as a truncated
    #: object, so its shape measurements are wrong and it may be split in two.
    #: Dropping a border band and shrinking the denominator to match is the
    #: honest repair: it costs tile area rather than biasing the density, which
    #: is the metric the cross-marker QC check below depends on. 24 px at
    #: 0.5 um/px is 12 um - larger than a breast epithelial nucleus, so a
    #: counted nucleus was always fully inside the field the model saw.
    nuclei_border_margin_px: int = 24

    #: Fallback field count for a caller that asks for one region's fields
    #: without saying how many it wants - `fields_for_region(count=None)`.
    #:
    #: **It no longer decides the sample.** It used to: every carried region got
    #: this many fields regardless of size, which made the measured cells a
    #: sample of the regions rather than of the tumour. `nuclei_field_budget`
    #: below is what decides now, split by area. This remains only for the
    #: H&E reference pass and for direct callers, both of which want "some
    #: fields from this one region" rather than a share of a slide-wide budget.
    #:
    #: **The step samples rather than exhausts, and that is still the design.**
    #: The carried regions total tens of mm2, which at the measured 60 s/mm2 is
    #: hours; the density is an estimate with a stated sample size rather than a
    #: census, and the report says so in those words.
    nuclei_tiles_per_region: int = 12

    #: Total sampling fields across ALL carried regions, split between them in
    #: proportion to their area. This replaces `nuclei_tiles_per_region` as the
    #: thing that decides the sample; that setting is now the per-region cap.
    #:
    #: **Proportional, because the alternative silently reweights the tumour.**
    #: A fixed count per region gives a 2 mm2 region the same say as a 22 mm2
    #: one, so the measured cells are not a sample of the tumour and no
    #: after-the-fact weighting can fully repair that - the small region's cells
    #: are over-represented and the large region's under-sampled at the same
    #: time. With a proportional sample the plain mean over fields equals the
    #: area-weighted mean, which is also what makes open question Q3 stop
    #: changing the answer.
    nuclei_field_budget: int = 96

    #: Fields a carried region gets even when its proportional share rounds to
    #: zero. **One, deliberately, and not more.**
    #:
    #: A larger floor looks safer and is not. Covering 90% of a real tumour takes
    #: about twenty regions, so a floor of four spends eighty of a ninety-six
    #: field budget before proportionality is considered at all - and on
    #: CAN_00270 that gave the region holding 63% of the invasive area 18% of the
    #: sample. The floor had undone the thing it was added alongside.
    #:
    #: It can be one because a region does not need a readable percentage of its
    #: own. Step 16 combines the regions by area, so a 0.25 mm2 region carries
    #: 0.7% of the weight however noisily it was measured; what the allocation
    #: decides is precision, not the answer. Spending fields to make a
    #: negligible region's percentage tidy takes them from the region that
    #: actually determines the score.
    nuclei_min_tiles_per_region: int = 1

    #: Minimum share of a candidate tile that must be tissue before it is worth
    #: segmenting. Tiles are ranked by haematoxylin content and the best are
    #: taken, so this only rejects the case where a region's bounding box is
    #: mostly glass - a warped ring can be very non-convex.
    nuclei_min_tile_tissue: float = 0.35

    #: Smallest object kept, in square microns. A breast epithelial nucleus is
    #: 7-10 um across, so about 40-80 um2; lymphocytes run to about 20. Ten is
    #: below anything real and removes the single-cell debris the model
    #: occasionally finds at a tissue edge. In um2 rather than pixels so it
    #: survives a change of scanner, per the guide's own instruction.
    nuclei_min_area_um2: float = 10.0

    #: Detection runs on the counterstain with the DAB removed, not on RGB.
    #:
    #: The guide is emphatic and it is right: the brown is what is being
    #: measured, so it must not influence where cells are thought to be, or
    #: strongly stained cells become more findable and the percentage inflates
    #: itself. Set this false only to demonstrate that effect - the UI has a
    #: toggle that does exactly that, side by side, with both counts on screen.
    nuclei_remove_dab: bool = True

    #: Estimate the stain basis from the slide's own pixels (Macenko) for the
    #: detection path, rather than using Ruifrok's fixed matrix.
    #:
    #: **Off, and this reverses the guide's own recommendation on a measurement.**
    #: The guide argues for a per-slide estimate precisely because a fixed matrix
    #: is at its worst when one stain dominates, which is this panel. Measured on
    #: CAN_00270's CD44 slide over six fields: fixed Ruifrok finds **695** nuclei,
    #: Macenko **581**, a hybrid taking only DAB from Macenko **574**.
    #:
    #: The estimated vectors say why. Macenko recovers DAB well - 3.3 degrees off
    #: the published direction, because DAB is what dominates - and returns an
    #: **achromatic** second arm, (0.577, 0.577, 0.577), 18.7 degrees off
    #: haematoxylin. Under heavy DAB with a weak counterstain there is no second
    #: colour in the cloud, so the method finds the darkness axis instead. The
    #: condition that is supposed to justify estimating is the condition that
    #: breaks the estimate of the stain we actually need.
    #:
    #: It is still computed and reported either way, because the drift is the
    #: evidence for this comment and a marker with a stronger counterstain may
    #: well flip the result back.
    nuclei_macenko_per_slide: bool = False

    #: Contrast applied to the haematoxylin amount when it is drawn back as an
    #: image for the segmenter. A rendering term, not a correction.
    #:
    #: It exists because this panel's counterstain is weak under heavy DAB, and
    #: it is set **on a plateau rather than at a peak**, which is the only way a
    #: knob like this can be honest. Nuclei found over six fields of the CD44
    #: slide: gain 1.0 -> 325, 1.5 -> 523, 2.0 -> 640, 2.5 -> 695, 3.0 -> 699,
    #: 4.0 -> 692. From 2.5 the answer stops depending on the setting, and above
    #: 3 individual fields start losing nuclei to over-amplification. Anything in
    #: 2.5-3.0 gives the same result, which is what makes 2.5 a defensible
    #: default instead of a tuned one.
    nuclei_haematoxylin_gain: float = 2.5

    nuclei_macenko_percentile: float = 1.0

    #: Fields segmented on the **H&E** slide to establish the reference density
    #: the IHC figure is read against.
    #:
    #: Six, not twelve: this is a level to compare with, not a result to publish,
    #: and it costs about ten seconds. Measured on CAN_00270 it is also the single
    #: most informative number the step produces - the H&E gives 1,245-2,825
    #: nuclei per mm2 with a median nuclear area of 43 um2, and the CD44 section
    #: of the same block gives 549 and 15. Same code, same regions, different
    #: slide: that is the counterstain failing under dense DAB, exactly as the
    #: guide predicts, and it is a 73% loss from the denominator.
    nuclei_reference_fields: int = 6

    # --- step 12, cell typing -------------------------------------------------
    #
    # Tiny by comparison with every cache above it: a class per nucleus id and a
    # report. It is still a directory rather than a field on step 11's report,
    # because the thresholds are live - a viewer can move one and get a new
    # answer - and overwriting step 11's result to record a step 12 choice would
    # make the segmentation look like it had been redone.
    cell_typing_dir: Path = DATA_ROOT / "cell_typing"

    # --- step 13, compartments ------------------------------------------------
    #
    # The widths are NOT here, and that is deliberate: which compartment a marker
    # gets and how wide it is are properties of the antibody, so they live in
    # `app/panel.py` beside everything else that forks by marker. A width in
    # this file would be a second place to change it, and the two could differ
    # while both looked right.
    compartments_dir: Path = DATA_ROOT / "compartments"

    # --- steps 14, 15 and 16, the measurement ---------------------------------
    #
    # Three directories rather than one, because the three steps are genuinely
    # separate results a viewer can be looking at: step 14's rows survive a
    # change to the cut points, step 15's bins do not, and step 16's score is
    # recomputed from both. Collapsing them would mean re-measuring every cell -
    # which means reopening the slide - every time a threshold moved.
    #
    # The cut points themselves are NOT here. They are five independent sets of
    # numbers with provenance attached, so they live in a versioned data file,
    # `backend/config/marker_cuts.v1.json`, read through `app/scoring/cuts.py`.
    # A threshold in this file would be a threshold with no record of which
    # slides it was anchored on.
    per_cell_dir: Path = DATA_ROOT / "per_cell"
    binning_dir: Path = DATA_ROOT / "binning"
    scores_dir: Path = DATA_ROOT / "scores"

    # Where a reader sheet is looked for by step 17. Absent by default: the
    # 120 pathologist readings are not in this repository, and step 17 reports
    # that it has nothing to compare against rather than inventing a comparison.
    reader_scores_path: Path | None = None

    def ensure_dirs(self) -> None:
        """Create the storage directories the upload flow writes into."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.slides_dir.mkdir(parents=True, exist_ok=True)
        self.qc_dir.mkdir(parents=True, exist_ok=True)
        self.tissue_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        self.tiling_dir.mkdir(parents=True, exist_ok=True)
        self.tissue_type_dir.mkdir(parents=True, exist_ok=True)
        self.roi_dir.mkdir(parents=True, exist_ok=True)
        self.roi_selection_dir.mkdir(parents=True, exist_ok=True)
        self.roi_refinement_dir.mkdir(parents=True, exist_ok=True)
        self.ihc_alignment_dir.mkdir(parents=True, exist_ok=True)
        self.nuclei_dir.mkdir(parents=True, exist_ok=True)
        self.cell_typing_dir.mkdir(parents=True, exist_ok=True)
        self.compartments_dir.mkdir(parents=True, exist_ok=True)
        self.per_cell_dir.mkdir(parents=True, exist_ok=True)
        self.binning_dir.mkdir(parents=True, exist_ok=True)
        self.scores_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()


settings = get_settings()
