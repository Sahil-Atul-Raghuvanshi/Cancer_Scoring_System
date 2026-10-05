"""Step 4 - white calibration, orchestrated.

Like step 3 and unlike step 2, this is not a background job. Given step 3's mask
it is one thumbnail read and a handful of numpy passes, so it answers in the
request that asked for it and there is nothing to poll.

What is cached is the RGB thumbnail, and only that:

    basis.json      the resolution, the shape, and which tissue mask this was
                    built against
    slide.png       the RGB thumbnail, unresampled and lossless

**Why the thumbnail and nothing else.** Everything past the thumbnail is cheap -
a percentile, a patch sweep, a six-coefficient least squares - and all of it
depends on which pixels are glass, which depends on step 3's threshold, which the
viewer can still move. Caching a white point would mean either invalidating it on
every slider move or serving one that belongs to a mask nobody is looking at.
Caching the pixels it is measured *from* has neither problem: they do not depend
on the threshold at all.

**Why lossless and unresampled.** I0 is a high percentile of these values. JPEG
would move it by a few levels; resampling averages neighbouring pixels, which
pulls any percentile towards the mean. Either would leave the cached slide
measuring a different white from the one the report quotes. This is the same rule
step 3 applies to its saturation channel, and for the same reason.

**Why the grid is step 3's and not its own.** The service asks
`tissue_service.footprint()` for the mask *and the resolution it was built at*,
then reads the RGB at exactly that size. Reading at some other resolution would
mean resampling one of the two, and step 4 is a question about position - which
pixels are glass - so a half-pixel disagreement between the mask and the image is
a real error at the tissue boundary rather than a rounding detail. Sharing the
grid removes the question. The 60 um clearance around the tissue then covers
whatever is left.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from app.common.memo import BoundedMemo
from app.core.config import settings
from app.core.logging import get_logger
from app.ingestion.slide_reader import open_slide
from app.pipeline.step03_tissue_mask import mask as tissue_mask
from app.pipeline.step04_white_calibration import calibration as calib
from app.pipeline.step04_white_calibration import overlay
from app.pipeline.step04_white_calibration.calibration import CalibrationError
from app.schemas.calibration import (
    BackgroundFill,
    CalibrationComparison,
    CalibrationDifference,
    CalibrationExclusion,
    CalibrationParams,
    CalibrationPatch,
    CalibrationReport,
    CalibrationSummary,
    Channels,
    FieldChoice,
    NoiseFloorOut,
    SurfaceOut,
    WhitePointOut,
)
from app.services.tissue_service import TissueFootprint, tissue_service
from app.services.upload_service import get_record, resolve_ready_path

logger = get_logger(__name__)

CITATION = (
    "Ruifrok AC, Johnston DA. Quantification of histochemical staining by colour "
    "deconvolution. Analytical and Quantitative Cytology and Histology 23(4):291-299 "
    "(2001). https://pubmed.ncbi.nlm.nih.gov/11531144/ · "
    "Beer A. Bestimmung der Absorption des rothen Lichts in farbigen Flussigkeiten. "
    "Annalen der Physik und Chemie 86:78-88 (1852)."
)

#: The panels the API serves. All four depend on step 3's threshold, because all
#: four depend on which pixels are glass - so unlike step 3, none of them can be
#: served from a cache that ignores the query string.
PANELS: tuple[str, ...] = ("thumbnail", "glass", "field", "corrected")

#: Percentiles the plateau is measured across. If I0 barely moves between these
#: two, the choice of the 95th is not doing the work; if it climbs, the glass is
#: contaminated and the notes say so.
PLATEAU_RANGE = (90.0, 99.0)

#: Spread across `PLATEAU_RANGE` at or above which the glass is called
#: contaminated. Measured rather than picked: on the demo's slides clean glass
#: moves about 1% across that range - quantisation in a compressed scan accounts
#: for most of it - while the same slide with the scanner's background fill left
#: in moves 32%. 3% sits well clear of both.
PLATEAU_LIMIT = 0.03

#: Channel spread below which the illuminant is reported as neutral rather than
#: as cast. Under a level out of 240, which is where quantisation lives, so
#: claiming a colour cast below this would be reading the rounding.
NEUTRAL_LIMIT = 0.004


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(
        f"{int(round(max(0.0, min(255.0, value)))):02x}" for value in rgb
    )


def _channels(values: tuple[float, float, float], *, digits: int = 3) -> Channels:
    return Channels(
        r=round(values[0], digits), g=round(values[1], digits), b=round(values[2], digits)
    )


@dataclass
class Basis:
    """The RGB thumbnail a run is measured from, loaded from cache or the slide.

    Threshold-independent by construction, which is the whole reason it is the
    thing cached: step 3's slider changes which of these pixels are glass, never
    what the pixels are.
    """

    rgb: np.ndarray
    mpp: float
    target_mpp: float
    capped: bool


@dataclass(frozen=True)
class WhiteReference:
    """Step 4's white point handed to a later step, with everything needed to key a cache.

    The analogue of `tissue_service.footprint()` one step along, and it exists for
    the same reason: step 5 needs step 4's *output* and must get it through the
    service that owns it, rather than by re-deriving a white point of its own. Two
    places that decide what zero stain measures as are two places that can
    disagree, and every optical density in the pipeline is divided by this number.

    `field_for` is the reason this is a class and not a triple. Step 4 may have
    justified a white point that varies across the slide, and step 5 works on one
    tile at a time - so the reference a tile is divided by is that surface
    evaluated over the patch of the slide the tile occupies. Doing that here keeps
    step 5's package from importing step 4's, which is the rule the pipeline
    packages keep: a step reads another step's result, never its code.

    `key` is the reference's *identity*, for a caller memoising on it. Built from
    the tissue cut, the percentile and the field mode rather than from a
    timestamp, because those are what actually determine the number.
    """

    #: I0 as one triple - the flat value, in force unless `mode` is "surface".
    rgb: tuple[float, float, float]
    mode: str
    percentile: float
    saturated: bool
    #: Worst-channel optical density of the dimmest sampled glass. The floor every
    #: density step 5 computes has to be read against.
    noise_floor: float
    #: The intensity floor step 4 applied, so step 5 clips the same way.
    od_floor: float

    tissue_threshold: int
    tissue_threshold_source: str
    qc_gated: bool
    qc_source: str | None
    key: str

    #: The fitted surface, when one is in force, plus the two grids needed to
    #: evaluate it somewhere other than the thumbnail it was fitted on.
    surface: calib.Surface | None
    #: (height, width) of the thumbnail the surface's normalised coordinates span.
    frame: tuple[int, int]
    #: Level-0 (width, height) of the slide, and its microns per pixel.
    slide: tuple[int, int]
    base_mpp: float

    def field_for(
        self, *, x: int, y: int, size: int, mpp: float
    ) -> np.ndarray | tuple[float, float, float]:
        """The reference a tile at `(x, y)` should be divided by.

        Returns the flat triple when that is what step 4 justified, and an
        HxWx3 field otherwise - the shapes `optical_density` broadcasts, so the
        caller never branches on which.

        The coordinate conversion goes level-0 pixel -> thumbnail column ->
        normalised, and each hop is there for a reason. The tile's origin is a
        level-0 coordinate because that is the only frame of reference that does
        not move when a resolution changes. The surface's coordinates are
        normalised to the *thumbnail's* extent, so the middle hop is what makes
        the field evaluated here the same function step 4 fitted rather than one
        shifted by however much the thumbnail's rounding lost.
        """
        if self.mode != "surface" or self.surface is None:
            return self.rgb

        frame_height, frame_width = self.frame
        slide_width, slide_height = self.slide
        scale = mpp / self.base_mpp

        # Pixel centres, not corners: a tile's first pixel covers level-0 x from
        # `x` to `x + scale`, and the field belongs at the middle of it.
        columns = (x + (np.arange(size, dtype=np.float64) + 0.5) * scale) * (
            frame_width / max(1, slide_width)
        )
        rows = (y + (np.arange(size, dtype=np.float64) + 0.5) * scale) * (
            frame_height / max(1, slide_height)
        )

        return calib.evaluate_surface_on(
            self.surface,
            2.0 * columns / max(1, frame_width - 1) - 1.0,
            2.0 * rows / max(1, frame_height - 1) - 1.0,
        )


class CalibrationService:
    """Runs step 4, caches the thumbnail, and recomputes everything else per request.

    Two in-process memos, each holding the last few entries. A single view of this
    step fetches the report and up to four panels, which without them would be five
    identical runs of the same arithmetic.

    **They hold several entries rather than one because this step now runs on two
    slides.** I0 is measured per slide - that is the whole of step 4's argument - so
    a case puts the H&E's white point and the immunostained slide's on screen
    together, and the viewer flips between them. With a single slot each flip evicted
    the other slide and re-ran step 3's morphology and this step's glass selection
    from scratch. See `app.common.memo` for how the capacity is reasoned about.

    The comparison endpoint, which genuinely holds several slides at once, still goes
    through `summary()` and still keeps nothing.
    """

    def __init__(self) -> None:
        self._basis_memo: BoundedMemo[Basis] = BoundedMemo()
        self._result_memo: BoundedMemo[calib.Calibration] = BoundedMemo()

    # --- storage ------------------------------------------------------------

    def _dir(self, upload_id: str) -> Path:
        return settings.calibration_dir / upload_id

    def _path(self, upload_id: str, name: str) -> Path:
        return self._dir(upload_id) / name

    # --- the basis ----------------------------------------------------------

    def _basis(self, upload_id: str, *, footprint: TissueFootprint) -> Basis:
        """The RGB thumbnail on step 3's grid, from memo, disk, or the slide.

        Keyed on the mask's shape rather than on the requested resolution,
        because the shape is what the image has to match. Step 3's own size cap
        can make its achieved resolution coarser than anything requested here,
        and the grid that matters is the one the mask actually landed on.
        """
        shape = (int(footprint.mask.shape[0]), int(footprint.mask.shape[1]))
        key = (upload_id, shape, round(footprint.mpp, 6))

        cached = self._basis_memo.get(key)
        if cached is not None:
            return cached

        basis = self._basis_from_disk(upload_id, footprint=footprint, shape=shape)
        self._basis_memo.put(key, basis)
        return basis

    def _basis_from_disk(
        self, upload_id: str, *, footprint: TissueFootprint, shape: tuple[int, int]
    ) -> Basis:
        meta_path = self._path(upload_id, "basis.json")

        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if (
                    int(meta["height"]) == shape[0]
                    and int(meta["width"]) == shape[1]
                    and abs(float(meta["mpp"]) - footprint.mpp) < 1e-6
                ):
                    return self._load_basis(upload_id, meta, shape=shape)
            except (OSError, KeyError, ValueError, json.JSONDecodeError):
                logger.warning(
                    "rebuilding unreadable calibration basis for %s", upload_id, exc_info=True
                )

        return self._compute_basis(upload_id, footprint=footprint, shape=shape)

    def _load_basis(self, upload_id: str, meta: dict, *, shape: tuple[int, int]) -> Basis:
        """Reconstitute a cached thumbnail, or raise so the caller rebuilds it.

        The shape check is load-bearing, not defensive padding: step 4 decides
        which pixels are glass by indexing this array with step 3's mask, and two
        arrays that disagree about their own size would sample the wrong pixels
        rather than fail. Better to re-read the slide than to report a white
        point measured somewhere else.
        """
        rgb = np.asarray(Image.open(self._path(upload_id, "slide.png")).convert("RGB"))
        if rgb.shape[:2] != shape:
            raise ValueError("cached calibration thumbnail does not match the tissue mask")

        return Basis(
            rgb=rgb,
            mpp=float(meta["mpp"]),
            target_mpp=float(meta["target_mpp"]),
            capped=bool(meta["capped"]),
        )

    def _compute_basis(
        self, upload_id: str, *, footprint: TissueFootprint, shape: tuple[int, int]
    ) -> Basis:
        path = resolve_ready_path(upload_id=upload_id)

        with open_slide(path) as reader:
            base_mpp = reader.mpp
            if not base_mpp:
                raise CalibrationError(
                    "this slide records no microns-per-pixel, so the border and clearance "
                    "distances this step discards have no physical meaning. Supply a scale "
                    "on step 1 first - stating them in pixels instead would make them "
                    "change meaning on the next scanner."
                )

            # The same reader call step 3 used, asked for step 3's own achieved
            # resolution and capped at the mask's own longest edge. That is what
            # makes the two grids identical rather than merely similar.
            rgb, achieved = tissue_mask.thumbnail_at_mpp(
                reader,
                base_mpp=base_mpp,
                target_mpp=footprint.mpp,
                max_px=max(shape),
            )

        if rgb.shape[:2] != shape:
            # `thumbnail_pil` honours an aspect ratio rather than an exact size,
            # so a pyramid level can land a pixel out. Nearest-neighbour onto the
            # mask's grid: this is the image, not the mask, and one pixel of
            # resampling at the boundary is covered many times over by the
            # clearance ring the algorithm drops anyway.
            rgb = np.asarray(
                Image.fromarray(rgb, mode="RGB").resize(
                    (shape[1], shape[0]), Image.Resampling.NEAREST
                )
            )

        longest_um = max(footprint.slide_width, footprint.slide_height) * base_mpp
        wanted = round(longest_um / settings.calibration_mpp)
        basis = Basis(
            rgb=rgb,
            mpp=achieved,
            target_mpp=settings.calibration_mpp,
            capped=wanted > max(shape),
        )

        directory = self._dir(upload_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "slide.png").write_bytes(overlay.store_rgb(rgb))
        (directory / "basis.json").write_text(
            json.dumps(
                {
                    "mpp": basis.mpp,
                    "target_mpp": basis.target_mpp,
                    "capped": basis.capped,
                    "width": int(shape[1]),
                    "height": int(shape[0]),
                    "base_mpp": base_mpp,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return basis

    # --- running ------------------------------------------------------------

    def _run(
        self, upload_id: str, *, threshold: int | None, percentile: float | None
    ) -> tuple[calib.Calibration, Basis, TissueFootprint]:
        """Step 3's mask, step 4's thumbnail, and the calibration over both.

        `threshold` is passed straight through to step 3. It is not step 4's
        parameter and step 4 does not interpret it: the viewer moves one slider,
        on step 3's screen, and this step re-derives its glass from whatever mask
        that produced. Re-implementing the threshold here would be a second place
        that decides what tissue is.
        """
        footprint = tissue_service.footprint(upload_id, threshold=threshold)
        basis = self._basis(upload_id, footprint=footprint)

        resolved = percentile if percentile is not None else settings.calibration_percentile
        key = (upload_id, footprint.key, round(resolved, 4))

        cached = self._result_memo.get(key)
        if cached is not None:
            return cached, basis, footprint

        result = calib.calibrate(
            rgb=basis.rgb,
            tissue=footprint.mask,
            considered=footprint.considered,
            mpp=basis.mpp,
            percentile=resolved,
            border_um=settings.calibration_border_um,
            clearance_um=settings.calibration_tissue_clearance_um,
            fill_clearance_um=settings.calibration_fill_clearance_um,
            fill_min_share=settings.calibration_fill_min_share,
            fill_max_shoulder=settings.calibration_fill_max_shoulder,
            patch_um=settings.calibration_patch_um,
            patch_min_glass=settings.calibration_patch_min_glass,
            min_patches=settings.calibration_min_patches,
            required_snr=settings.calibration_vignette_snr,
            leverage_limit=settings.calibration_max_leverage,
            od_floor=settings.calibration_od_floor,
        )

        self._result_memo.put(key, result)
        return result, basis, footprint

    def thumbnail(
        self, upload_id: str, *, threshold: int | None = None
    ) -> tuple[np.ndarray, float]:
        """The RGB thumbnail I0 was measured from, on step 3's grid, and its resolution.

        Public for step 5, which needs these exact pixels rather than a second
        copy: it screens the slide for a tile to stand on, and the screening is a
        question about position - which block holds two stains - so it has to be
        asked on the same grid the tissue mask and the white point live on. A
        second read at a second resolution would answer it half a pixel to one
        side, and would also re-open a multi-gigabyte scan for pixels already on
        disk here.
        """
        footprint = tissue_service.footprint(upload_id, threshold=threshold)
        basis = self._basis(upload_id, footprint=footprint)
        return basis.rgb, basis.mpp

    def white_point(
        self,
        upload_id: str,
        *,
        threshold: int | None = None,
        percentile: float | None = None,
    ) -> WhiteReference:
        """Step 4's I0, for a later step that needs the number and not the prose.

        Exists so step 5 can ask step 4 for its output through the service that
        owns it - the same reason `tissue_service.footprint` exists, and the same
        argument one step along. Optical density is *defined* against this value,
        so a second estimate of it in step 5's package would not be duplication,
        it would be a second definition.
        """
        result, basis, footprint = self._run(
            upload_id, threshold=threshold, percentile=percentile
        )

        return WhiteReference(
            rgb=result.white.rgb,
            mode=result.choice.mode,
            percentile=result.white.percentile,
            saturated=result.white.saturated,
            noise_floor=result.noise.worst,
            od_floor=settings.calibration_od_floor,
            tissue_threshold=footprint.threshold,
            tissue_threshold_source=footprint.threshold_source,
            qc_gated=footprint.qc_gated,
            qc_source=footprint.qc_source,
            key=(
                f"{footprint.key}/p{result.white.percentile:g}"
                f"/{result.choice.mode}"
            ),
            surface=result.surface if result.uses_surface else None,
            frame=result.shape,
            slide=(footprint.slide_width, footprint.slide_height),
            base_mpp=footprint.base_mpp,
        )

    def report(
        self,
        upload_id: str,
        *,
        threshold: int | None = None,
        percentile: float | None = None,
    ) -> CalibrationReport:
        """Run step 4 and describe the result."""
        result, basis, footprint = self._run(
            upload_id, threshold=threshold, percentile=percentile
        )
        record = get_record(upload_id=upload_id)
        return self._describe(
            result, basis, footprint, upload_id=upload_id, filename=record.filename
        )

    def panel(
        self,
        upload_id: str,
        name: str,
        *,
        threshold: int | None = None,
        percentile: float | None = None,
    ) -> bytes:
        """One of the four panels as PNG.

        Only the thumbnail is threshold-independent, and even it is served from
        the run rather than from disk: the cached file is the unresampled one, and
        serving a 4,096 px PNG to a browser that will draw it at 500 would be
        sending sixteen times the bytes for the same picture.
        """
        if name not in PANELS:
            raise CalibrationError(
                f"unknown panel {name!r}; expected one of {sorted(PANELS)}"
            )

        result, basis, _ = self._run(upload_id, threshold=threshold, percentile=percentile)

        if name == "thumbnail":
            return overlay.thumbnail_png(basis.rgb)
        if name == "glass":
            return overlay.glass_png(basis.rgb, result)
        if name == "field":
            return overlay.field_png(result)
        return overlay.corrected_png(basis.rgb, result)

    def swatch(
        self,
        upload_id: str,
        *,
        threshold: int | None = None,
        percentile: float | None = None,
    ) -> bytes:
        """I0 as a flat square. Trivial to render, and the point of the whole step."""
        result, _, _ = self._run(upload_id, threshold=threshold, percentile=percentile)
        return overlay.swatch_png(result.white.rgb)

    # --- the cross-slide argument -------------------------------------------

    def summary(self, upload_id: str) -> CalibrationSummary:
        """One slide's white point, at default settings, for the comparison."""
        result, _, _ = self._run(upload_id, threshold=None, percentile=None)
        record = get_record(upload_id=upload_id)

        pixel_mm2 = (result.mpp / 1000.0) ** 2
        total = float(result.glass.glass.size) * pixel_mm2

        return CalibrationSummary(
            upload_id=upload_id,
            filename=record.filename,
            rgb=_channels(result.white.rgb, digits=2),
            hex=_hex(result.white.rgb),
            mode=result.choice.mode,
            saturated=result.white.saturated,
            noise_floor=round(result.noise.worst, 4),
            glass_share=round(
                result.glass.pixels / max(1, result.glass.glass.size), 6
            )
            if total
            else 0.0,
        )

    def compare(self, upload_ids: list[str]) -> CalibrationComparison:
        """Two or more slides' white points, and the OD cost of confusing them.

        The demo's "why per slide" argument, as one response. Each pair's shift is
        reported rather than only the worst, because the interesting case is often
        a pair that agrees sitting next to a pair that does not - one scanner
        drifting, or one batch stained differently.

        Deliberately keeps nothing: two slides at once would evict each other from
        a one-entry memo anyway, and the summaries are small.
        """
        if len(upload_ids) < 2:
            raise CalibrationError(
                "comparing white points needs at least two slides - one slide's I0 in "
                "isolation is exactly the thing this comparison exists to argue against"
            )

        slides = [self.summary(upload_id) for upload_id in upload_ids]

        differences: list[CalibrationDifference] = []
        worst = 0.0
        for index, left in enumerate(slides):
            for right in slides[index + 1 :]:
                shift = calib.od_difference(
                    (left.rgb.r, left.rgb.g, left.rgb.b),
                    (right.rgb.r, right.rgb.g, right.rgb.b),
                )
                magnitude = max(abs(value) for value in shift)
                worst = max(worst, magnitude)
                differences.append(
                    CalibrationDifference(
                        upload_id=left.upload_id,
                        other_upload_id=right.upload_id,
                        od_shift=_channels(shift, digits=4),
                        worst=round(magnitude, 4),
                    )
                )

        return CalibrationComparison(
            slides=slides,
            differences=differences,
            worst_shift=round(worst, 4),
            note=(
                f"Using one of these slides' I0 on another would shift every optical density "
                f"by up to {worst:.3f} - and as a constant *added* to every pixel, because "
                "density is a logarithm of a ratio. That is not a scaling a later step could "
                "absorb: on the wrong slide it is indistinguishable from more stain. This is "
                "the whole reason calibration is per slide and not once per study."
            ),
        )

    # --- describing ---------------------------------------------------------

    def _describe(
        self,
        result: calib.Calibration,
        basis: Basis,
        footprint: TissueFootprint,
        *,
        upload_id: str,
        filename: str,
    ) -> CalibrationReport:
        mpp = result.mpp
        pixel_mm2 = (mpp / 1000.0) ** 2

        def area(pixels: int) -> float:
            return round(pixels * pixel_mm2, 4)

        exclusions: list[CalibrationExclusion] = []
        previous = 0
        for index, step in enumerate(result.glass.steps):
            delta = step.pixels - previous if index else 0
            exclusions.append(
                CalibrationExclusion(
                    key=step.key,
                    label=step.label,
                    what=step.what,
                    extent_um=step.extent_um,
                    pixels=step.pixels,
                    area_mm2=area(step.pixels),
                    delta_pixels=delta,
                    delta_area_mm2=area(delta),
                    delta_share=round(delta / previous, 6) if previous else 0.0,
                )
            )
            previous = step.pixels

        white = result.white
        low, high = PLATEAU_RANGE
        plateau = 0.0
        if low in white.ladder and high in white.ladder:
            for channel in range(3):
                base = white.ladder[low][channel]
                if base > 1e-6:
                    plateau = max(
                        plateau, abs(white.ladder[high][channel] - base) / base
                    )

        patches = [
            CalibrationPatch(
                col=patch.col,
                row=patch.row,
                x=round(patch.x / result.shape[1], 6),
                y=round(patch.y / result.shape[0], 6),
                width=round(patch.width / result.shape[1], 6),
                height=round(patch.height / result.shape[0], 6),
                glass_pixels=patch.glass_pixels,
                glass_share=round(patch.glass_share, 4),
                used=patch.used,
                rgb=_channels(patch.rgb, digits=2) if patch.rgb else None,
                hex=_hex(patch.rgb) if patch.rgb else None,
            )
            for patch in result.patches
        ]

        surface = result.surface
        surface_out = (
            SurfaceOut(
                coefficients=[
                    [round(value, 6) for value in channel] for channel in surface.coefficients
                ],
                terms=list(calib.SURFACE_TERMS),
                r2=_channels(surface.r2, digits=4),
                residual_rms=_channels(surface.residual_rms, digits=3),
                swing=_channels(surface.swing, digits=3),
                mean=_channels(surface.mean, digits=2),
                patches_used=surface.patches_used,
                leverage=round(surface.leverage, 4),
                leverage_max=round(surface.leverage_max, 4),
                snr=round(surface.snr, 3),
                amplitude=round(surface.amplitude, 5),
                od_error=round(surface.od_error, 4),
            )
            if surface is not None
            else None
        )

        params = CalibrationParams(
            target_mpp=basis.target_mpp,
            calibration_mpp=round(mpp, 4),
            width=int(result.shape[1]),
            height=int(result.shape[0]),
            capped=basis.capped or footprint.capped,
            percentile=white.percentile,
            border_um=settings.calibration_border_um,
            fill_min_share=settings.calibration_fill_min_share,
            fill_max_shoulder=settings.calibration_fill_max_shoulder,
            clearance_um=settings.calibration_tissue_clearance_um,
            fill_clearance_um=settings.calibration_fill_clearance_um,
            border_px=max(1, round(settings.calibration_border_um / mpp)),
            clearance_px=max(1, round(settings.calibration_tissue_clearance_um / mpp)),
            fill_clearance_px=max(1, round(settings.calibration_fill_clearance_um / mpp)),
            patch_um=settings.calibration_patch_um,
            patch_px=max(8, round(settings.calibration_patch_um / mpp)),
            patch_min_glass=settings.calibration_patch_min_glass,
            min_patches=settings.calibration_min_patches,
            vignette_snr=settings.calibration_vignette_snr,
            max_leverage=settings.calibration_max_leverage,
            od_floor=settings.calibration_od_floor,
            tissue_threshold=footprint.threshold,
            tissue_threshold_source=footprint.threshold_source,
            qc_gated=footprint.qc_gated,
            qc_source=footprint.qc_source,
        )

        glass_pixels = result.glass.pixels
        total_pixels = int(result.glass.glass.size)

        return CalibrationReport(
            upload_id=upload_id,
            filename=filename,
            generated_at=_now(),
            params=params,
            white=WhitePointOut(
                rgb=_channels(white.rgb, digits=2),
                hex=_hex(white.rgb),
                percentile=white.percentile,
                ladder={
                    # Keyed by the percentile as written, so the frontend can order
                    # the ladder without re-deriving which levels were sampled.
                    f"{level:g}": _channels(values, digits=2)
                    for level, values in white.ladder.items()
                },
                plateau=round(plateau, 5),
                clipped=_channels(white.clipped, digits=6),
                saturated=white.saturated,
                cast=round(white.cast, 5),
                sampled_pixels=white.sampled_pixels,
                sampled_area_mm2=area(white.sampled_pixels),
            ),
            exclusions=exclusions,
            fills=[
                BackgroundFill(
                    rgb=_channels(
                        (float(found.rgb[0]), float(found.rgb[1]), float(found.rgb[2])),
                        digits=0,
                    ),
                    hex=_hex((float(found.rgb[0]), float(found.rgb[1]), float(found.rgb[2]))),
                    share=round(found.share, 5),
                    shoulder=round(found.shoulder, 5),
                    pixels=found.pixels,
                )
                for found in result.glass.fills
            ],
            patches=patches,
            surface=surface_out,
            choice=FieldChoice(
                mode=result.choice.mode,
                reason=result.choice.reason,
                fitted=result.choice.fitted,
                patches_used=result.choice.patches_used,
                min_patches=result.choice.min_patches,
                snr=round(result.choice.snr, 3),
                snr_required=result.choice.snr_required,
                leverage=round(result.choice.leverage, 4),
                leverage_limit=result.choice.leverage_limit,
            ),
            noise=NoiseFloorOut(
                floor=_channels(result.noise.floor, digits=4),
                median=_channels(result.noise.median, digits=4),
                percentile=result.noise.percentile,
                worst=round(result.noise.worst, 4),
            ),
            glass_pixels=glass_pixels,
            glass_area_mm2=area(glass_pixels),
            glass_share=round(glass_pixels / total_pixels, 6) if total_pixels else 0.0,
            notes=self._notes(result, params, plateau=plateau),
            citation=CITATION,
        )

    @staticmethod
    def _notes(
        result: calib.Calibration, params: CalibrationParams, *, plateau: float
    ) -> list[str]:
        """The caveats that belong beside the numbers, built from the numbers."""
        white = result.white
        notes = [
            "This is calibration, not normalisation, and Rule 2 turns on the difference. "
            "Nothing here rewrites a pixel. What is recorded is what 'zero stain' measures "
            "as on this slide, so every optical density downstream lands on an absolute "
            "scale that another slide's densities can be compared against. Rescaling each "
            "slide's colours to look alike would remove the very differences being measured.",
            f"I0 is the {white.percentile:g}th percentile of the glass, not its maximum and "
            "not its mean. I0 sits in a denominator, so the maximum - which one hot pixel or "
            "one glint off the coverslip sets - would darken the whole slide, and the mean is "
            "dragged down by anything that leaked past step 3's mask. The ladder above shows "
            "the value at every percentile so this choice can be checked rather than trusted.",
            f"The glass was arrived at by five exclusions, not one: not tissue, not the outer "
            f"{params.border_um:g} um frame, not step 2's artefacts, not the scanner's "
            f"background fill, and not the {params.clearance_um:g} um ring just outside the "
            "section. The last three are the ones a naive inversion of the tissue mask "
            "misses, and every one of them is darker than glass, so each would bias I0 down "
            "and compress every density downstream - faint stain and no stain measuring "
            "closer together than they should.",
        ]

        if result.glass.fills:
            listed = ", ".join(
                f"({found.rgb[0]}, {found.rgb[1]}, {found.rgb[2]}) covering "
                f"{found.share:.1%} of the candidate glass"
                for found in result.glass.fills
            )
            notes.append(
                f"This scan carries a digital background fill - {listed} - and it was "
                "excluded before I0 was measured. That fill is the part of the slide canvas "
                "the scanner never imaged: nothing was measured there, and it is darker than "
                "the real glass, so leaving it in would drag I0 down and report apparent "
                "stain over a region that does not exist. It is told apart from glass by the "
                "one property a digital constant cannot fake - real glass carries sensor "
                "noise, so colours sit either side of its commonest colour. This fill scores "
                f"{max(found.shoulder for found in result.glass.fills):.4f} on that measure "
                f"against {params.fill_max_shoulder:g} required, where genuine glass scores "
                "0.24 and up."
            )

        if plateau < PLATEAU_LIMIT:
            notes.append(
                f"I0 moves {plateau:.2%} between the 90th and 99th percentiles, so the "
                "choice of percentile is not doing the work here - there is a genuine "
                "plateau of clean glass, and any cut through it would give much the same "
                "answer."
            )
        else:
            notes.append(
                f"I0 moves {plateau:.1%} between the 90th and 99th percentiles, over the "
                f"{PLATEAU_LIMIT:.0%} a plateau of clean glass should stay inside. Something "
                "bright and non-glass is in the sample, or the glass genuinely varies across "
                "the slide - check the field panel and step 3's threshold before trusting "
                "this white point to three significant figures."
            )

        if white.saturated:
            notes.append(
                f"A channel is clipped: {max(white.clipped):.1%} of the sampled glass sits at "
                "255. The sensor ran out of range before the glass did, so the true incident "
                "intensity is higher than I0 says and every optical density downstream is "
                "compressed - faint stain and no stain measure closer together than they "
                "should. That is a scanning fault and no arithmetic here recovers it; rescan "
                "with a lower exposure if these numbers have to carry weight."
            )

        triple = f"({white.rgb[0]:.0f}, {white.rgb[1]:.0f}, {white.rgb[2]:.0f})"
        if white.cast >= NEUTRAL_LIMIT:
            notes.append(
                f"The illuminant is not neutral: R, G and B differ by {white.cast:.1%} of "
                f"their mean at {triple}. That is why I0 is three numbers and not one - a "
                "single grey white point would leave the scanner's own colour cast in the "
                "deconvolution at step 6, where nothing can tell it from a difference in "
                "stain."
            )
        else:
            notes.append(
                f"This slide's illuminant is very nearly neutral - R, G and B agree to "
                f"{white.cast:.2%} of their mean at {triple} - so here the three channels of "
                "I0 happen to carry almost the same number. That is a fact about this "
                "acquisition and not a general one: a lamp, a coverslip and a white balance "
                "are all coloured, and the next slide's I0 may not be grey at all. I0 stays "
                "three numbers so that when it is not, nothing has to change."
            )

        notes.append(result.choice.reason)

        if result.choice.mode == "surface" and result.surface is not None:
            notes.append(
                f"The field is supported where it is used. Over 99% of the tissue the fit "
                f"reaches {result.choice.leverage:.2f} outside its own samples, under the "
                f"{result.choice.leverage_limit:g} allowed - so the ring of glass around the "
                "section encloses it well enough to pin the illumination down across it. "
                "That check is deliberately made over the tissue and not the whole frame: "
                "glass samples always ring the section, so the frame's corners are always "
                f"extrapolated (here up to {result.surface.leverage_max:.1f}) and no density "
                "is ever computed there."
            )

        if result.choice.mode == "flat" and result.choice.fitted:
            notes.append(
                "Both fields are reported either way. The flat value is the one in force; the "
                "fitted surface's coefficients and diagnostics are above, and the field and "
                "corrected panels draw it - so a reader who disagrees with the test can see "
                "exactly what was declined and how much it would have changed."
            )

        floor = result.noise
        notes.append(
            f"Applied back to the glass it came from, this calibration leaves the dimmest "
            f"{floor.percentile:g}% of that glass measuring {floor.worst:.3f} OD of apparent "
            f"stain, and the median measuring {max(floor.median):.3f}. That is the noise "
            "floor of every measurement in this pipeline. No positivity threshold at step 14 "
            "can sit below it, which is why the number is reported here rather than "
            "discovered there."
        )

        notes.append(
            f"The glass was taken from the mask step 3 built at a cut of "
            f"{params.tissue_threshold} by the {params.tissue_threshold_source} rule. Move "
            "that cut and this white point moves with it: a threshold that calls pale tissue "
            "glass puts tissue into the sample, and tissue is darker than glass. Step 3's "
            "slider is the control for this step too."
        )

        if not params.qc_gated:
            notes.append(
                "Step 2 has not run, so no artefacts were excluded from the glass. Pen ink "
                "sits on the glass and is far darker than it, so a pen mark anywhere outside "
                "the tissue is pulling I0 down right now. Run quality control and open this "
                "step again."
            )
        elif params.qc_source != "grandqc":
            notes.append(
                "Step 2 found its tissue with the saturation/Otsu fallback rather than "
                "GrandQC's model, so the artefact map excluded from this glass is weaker than "
                "it should be. Download the tissue checkpoint before trusting I0 to three "
                "significant figures."
            )

        if params.capped:
            notes.append(
                f"Worked at {params.calibration_mpp:g} um/px rather than the "
                f"{params.target_mpp:g} um/px asked for: this step shares step 3's grid so "
                "the mask needs no resampling, and step 3's size cap bound on this slide. For "
                "a white point that is immaterial - illumination is smooth over millimetres "
                "and this is still far finer than it needs to be - but it is a different "
                "number from the one requested, so it is stated."
            )

        return notes


calibration_service = CalibrationService()
