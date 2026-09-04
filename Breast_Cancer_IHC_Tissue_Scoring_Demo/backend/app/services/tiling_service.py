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

import threading
from datetime import UTC, datetime

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
from app.pipeline.step08_tissue_type_segmentation import model as region_model
from app.schemas.tiling import (
    TileOut,
    TilingCoverage,
    TilingFunnel,
    TilingParams,
    TilingReport,
    TilingSample,
)
from app.services.calibration_service import calibration_service
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


def window() -> tuple[int, float, str]:
    """The square this step lays down: the region model's own, in px and um/px.

    **Why step 7 asks step 8 for its geometry rather than owning one.** The two used
    different squares - 512 px at the pipeline's working resolution here, 224 px at
    the checkpoint's there - and only step 8's is a real constraint: the model's field
    of view is a property of the checkpoint, recorded in its manifest, and feeding it
    a differently-scaled field is the same class of failure as feeding it the wrong
    channel. So the choosable one has to give way, and the grid this step prices is
    the grid that actually runs. Before this, a reader was shown "4,952 squares to
    process" and then watched step 8 make 98,262 passes over the same slide.

    **Read from the manifest, without torch.** `Candidate` is described from the
    manifest alone - that is what lets the capability endpoint list every published
    model on a machine with no torch installed - so this costs a small JSON read and
    keeps step 7 working when step 8 cannot run at all.

    The fallback is `settings.tile_size` at the pipeline's working resolution, for the
    case where nothing is published or a manifest is unreadable. It is honest rather
    than useful: with no checkpoint there is no window to agree with, and step 8 is
    not going to run anyway.
    """
    candidate = region_model.find(settings.tissue_type_model)
    if candidate is None or not candidate.usable:
        return settings.tile_size, settings.target_mpp, "the pipeline's own default"
    if not candidate.tile_px or not candidate.mpp:
        return settings.tile_size, settings.target_mpp, "the pipeline's own default"
    return int(candidate.tile_px), float(candidate.mpp), candidate.name


class TilingService:
    """Builds the tile index for a slide, and says what each gate cost.

    One memo entry, keyed on step 3's mask identity plus this step's own two gates
    and its overlap. The mask key already folds in the slide, the threshold, the
    resolution and which QC run was subtracted, so what is left to key on is only
    what this step itself decides.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._memo: tuple[tuple[object, ...], _Run] | None = None

    # --- running ------------------------------------------------------------

    def _run(self, upload_id: str, *, threshold: int | None, overlap: float | None) -> _Run:
        """Build the index, or return the one already built for these inputs.

        `threshold` belongs to step 3 and is passed straight through; step 7 owns
        it no more than steps 4, 5 and 6 do. `overlap` is this step's own control
        and the only one, because it is the only parameter here a viewer can move
        and watch something change: the square's size and resolution are the region
        model's - see `window` - and the two gates are thresholds whose values are
        arguments rather than preferences.
        """
        footprint = tissue_service.footprint(upload_id, threshold=threshold)
        share = settings.tiling_overlap if overlap is None else float(overlap)
        size, mpp, source = window()

        # The window's geometry joins the memo key. A republished checkpoint with a
        # different field of view is a different grid, and a memo that ignored it
        # would hand back the previous model's tiles under the new model's name.
        key = (upload_id, footprint.key, share, size, mpp)
        with self._lock:
            if self._memo is not None and self._memo[0] == key:
                return self._memo[1]

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
            },
        )

        run = _Run(index=index, footprint=footprint, overlap=share)
        with self._lock:
            self._memo = (key, run)
        return run

    def report(
        self,
        upload_id: str,
        *,
        threshold: int | None = None,
        overlap: float | None = None,
    ) -> TilingReport:
        """Build the tile index and describe it."""
        run = self._run(upload_id, threshold=threshold, overlap=overlap)
        record = get_record(upload_id=upload_id)
        return self._describe(run, upload_id=upload_id, filename=record.filename)

    def panel(
        self,
        upload_id: str,
        name: str,
        *,
        threshold: int | None = None,
        overlap: float | None = None,
    ) -> bytes:
        """`grid` or `sample`, as PNG."""
        if name not in PANELS:
            raise TilingError(f"unknown panel {name!r}; expected one of {sorted(PANELS)}")

        run = self._run(upload_id, threshold=threshold, overlap=overlap)

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

        **But still the same deconvolution.** The rule the guide is emphatic about
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
    ) -> TileIndex:
        """The index itself, for the step that consumes it rather than displays it.

        The whole index and not the sampled slice: the report ships a couple of
        hundred tiles because that is what a browser and a reader can hold, and
        step 8 needs all of them. Two different audiences, two different answers,
        and the sampling belongs to the one that is a display concern.
        """
        return self._run(upload_id, threshold=threshold, overlap=overlap).index

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
            ),
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
            notes=self._notes(run),
            citation=CITATION,
        )

    @staticmethod
    def _notes(run: _Run) -> list[str]:
        """The caveats that belong beside the numbers, built from the numbers."""
        index, footprint = run.index, run.footprint
        funnel = index.funnel
        notes: list[str] = []

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
            "about 10 um. That field of view is the parameter that matters most here and it "
            "is chosen for a specific job: step 8 has to tell ductal carcinoma in situ from "
            "invasive carcinoma, and the difference between them is *architecture* - whether "
            "the abnormal cells are still inside a duct. At 40x you see the cells and lose "
            "the architecture; at 5x the reverse. This is the scale that holds both."
        )

        notes.append(
            f"Neighbouring tiles overlap by {index.overlap:.0%}, which is why the grid holds "
            f"more tiles than the section needs to be covered once. A model has no context "
            "past a tile's edge, so its predictions there are its worst - overlapping lets "
            "those edges be averaged away instead of being stitched into visible seams "
            "across the class map. It is the one setting on this screen that costs compute, "
            "and it costs it quadratically: 25% overlap is 1.8 times the tiles of none, 50% "
            "is 4 times."
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

    __slots__ = ("footprint", "index", "overlap")

    def __init__(
        self, *, index: TileIndex, footprint: TissueFootprint, overlap: float
    ) -> None:
        self.index = index
        self.footprint = footprint
        self.overlap = overlap


tiling_service = TilingService()

__all__ = ["TilingError", "tiling_service"]
