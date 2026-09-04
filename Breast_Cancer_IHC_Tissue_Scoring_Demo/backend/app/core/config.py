"""Application configuration, loaded from the environment via pydantic-settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve storage relative to the repository root, so the paths are the same no
# matter which directory the process was started from.
REPO_ROOT = Path(__file__).resolve().parents[3]


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
    data_dir: Path = REPO_ROOT / "data"
    uploads_dir: Path = REPO_ROOT / "data" / "uploads"
    slides_dir: Path = REPO_ROOT / "data" / "slides"

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
    qc_dir: Path = REPO_ROOT / "data" / "qc"

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
    tissue_dir: Path = REPO_ROOT / "data" / "tissue"

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
    calibration_dir: Path = REPO_ROOT / "data" / "calibration"

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
    # Measured on the demo's slide: the artefact patch sits at 0.11 mean OD against
    # 0.21 for a stained duct, and a noise floor of 0.048, so 3x separates them
    # cleanly while leaving every genuinely stained block eligible.
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

    # Fraction of a tile's own extent that it shares with its neighbour. The
    # guide's 25-50% band for segmentation work, at its lower end.
    #
    # Overlap exists because a model has no context beyond a tile's edge, so its
    # predictions there are its worst; overlapping lets those edges be averaged
    # away instead of being stitched into visible seams across the class map. It
    # is not free - it is the one parameter on this step that costs compute, and
    # it costs it quadratically. At 25% the grid holds 1.8x the tiles of a
    # non-overlapping one; at 50%, 4x. Every one of those is a forward pass at
    # step 8 - which now runs this exact grid, so the count this step prices is the
    # bill rather than a proxy for it - and 0.25 is the setting that buys clean
    # seams without quadrupling it. Measured on `CAN_00251_26_A`, 0 / 0.25 / 0.5 is
    # about 10 / 15 / 29 minutes at step 8.
    tiling_overlap: float = 0.25

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
    tissue_type_dir: Path = REPO_ROOT / "data" / "tissue_type"

    # Where published checkpoints live. Empty means `<repo>/models/tissue_type`,
    # which is where `bcss_hchannel_resnet18`'s publish step writes - so the
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
    tissue_type_model: str = "invasive_tile_v2_imagenet_std"

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


    def ensure_dirs(self) -> None:
        """Create the storage directories the upload flow writes into."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.slides_dir.mkdir(parents=True, exist_ok=True)
        self.qc_dir.mkdir(parents=True, exist_ok=True)
        self.tissue_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        self.tissue_type_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()


settings = get_settings()
