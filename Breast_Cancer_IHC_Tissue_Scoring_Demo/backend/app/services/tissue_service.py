"""Step 3 - the tissue mask, orchestrated.

Unlike step 2 this is not a background job. A cold run is a slide read plus a
handful of numpy passes - ten seconds or so on a whole-slide scan - so it answers
in the request that asked for it and there is nothing to poll.

What *is* cached is the *basis*: the thumbnail, the saturation channel, and the
set of pixels step 2 left in play. None of the three depends on the threshold,
and caching them is what makes the demo's threshold slider honest. Each move
re-runs the real thresholding and the real morphology - the same code path Otsu
drives - instead of approximating it in the browser. It is not instant: the
morphology over a 4,096 px mask is a second and a half, and that is the genuine
cost of the answer rather than overhead. What the cache removes is the slide
read, which is the part that would make it ten.

    basis.json      the mpp, the shape, and which QC run this was built against
    saturation.png  the channel the decision is made on
    considered.png  pixels step 2 left in play, when step 2 has run
    display.png     the thumbnail, already fitted for the browser

The first two are written unresampled. They are re-read on every threshold
change, and `mask_mpp` - the number every physical cutoff in this step is
divided by - was measured against the arrays they were written from, so a
display-sized copy would leave a re-threshold running at one resolution while
the report quoted another. `display.png` is the deliberate exception: nothing is
measured from it, and fitting it once here rather than per request is the largest
single saving in the overlay path.

The cache is keyed on the target resolution and on step 2's `generatedAt`. A QC
re-run with a different artefact model changes which pixels are in play, so a
basis built against the previous run is stale and is rebuilt rather than reused.

`footprint()` is the handoff to step 4, which needs the mask's pixels rather than
its report: the glass step 4 samples I0 from is this mask's complement.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.common.imaging import class_means, saturation_channel
from app.core.config import settings
from app.core.logging import get_logger
from app.ingestion.slide_reader import open_slide
from app.pipeline.step02_quality_control import classes as qc_classes
from app.pipeline.step03_tissue_mask import mask as tissue_mask
from app.pipeline.step03_tissue_mask import overlay
from app.pipeline.step03_tissue_mask.mask import TissueMaskError
from app.schemas.tissue import (
    TissueComponents,
    TissueHistogram,
    TissueParams,
    TissueReport,
    TissueStage,
    TissueThreshold,
)
from app.services.qc_service import qc_service
from app.services.upload_service import get_record, resolve_ready_path

logger = get_logger(__name__)

CITATION = (
    "Otsu N. A threshold selection method from gray-level histograms. IEEE Transactions on "
    "Systems, Man, and Cybernetics 9(1):62-66 (1979). "
    "https://doi.org/10.1109/TSMC.1979.4310076 · "
    "Zack GW, Rogers WE, Latt SA. Automatic measurement of sister chromatid exchange "
    "frequency. Journal of Histochemistry & Cytochemistry 25(7):741-753 (1977). "
    "https://doi.org/10.1177/25.7.70454"
)

#: The panels the API serves, and whether the threshold changes them.
PANELS: dict[str, bool] = {
    "thumbnail": False,
    "saturation": False,
    "mask": True,
    "overlay": True,
}

#: Artefact classes to subtract before thresholding. Every class step 2 flags,
#: not just pen: a fold reads as darker and denser than the tissue it doubles,
#: and an out-of-focus region has no business setting a colour threshold.
_ARTEFACT_IDS: tuple[int, ...] = tuple(item.id for item in qc_classes.ARTEFACT_CLASSES)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class Basis:
    """The threshold-independent half of a run, loaded from cache or computed.

    The thumbnail is deliberately not here. Only the overlay panel needs it, and
    decoding a 4,096 px RGB PNG costs more than the report it would be decoded
    for; `TissueService._thumbnail` fetches it for the one caller that does.
    """

    saturation: np.ndarray
    considered: np.ndarray
    mpp: float
    target_mpp: float
    capped: bool
    qc_gated: bool
    qc_source: str | None
    qc_generated_at: str | None
    slide_width: int
    slide_height: int
    base_mpp: float


@dataclass(frozen=True)
class TissueFootprint:
    """Step 3's mask handed to a later step, with everything needed to key a cache.

    A dataclass rather than the tuple `qc_service.artefact_footprint` returns:
    that one carries four values and this one carries a dozen, which is well past
    the point where positional unpacking at the call site stays readable.

    `key` is the mask's *identity*, and it is deliberately not the report's
    `generatedAt`. That timestamp is stamped fresh on every call, so a consumer
    keying a cache on it would rebuild on every request. What actually determines
    these pixels is the cut, the resolution and which QC run was subtracted
    first, so that is what the key is made of.
    """

    mask: np.ndarray
    considered: np.ndarray | None
    mpp: float
    threshold: int
    threshold_source: str
    qc_gated: bool
    qc_source: str | None
    slide_width: int
    slide_height: int
    base_mpp: float
    capped: bool
    key: str


class TissueService:
    """Runs step 3, caches what does not depend on the threshold, reports the rest.

    Two caches, and they are different in kind. The disk cache above survives a
    restart and removes the slide read. The two in-process memos below survive
    only the process and remove *repeated* work within one interaction: a single
    move of the threshold slider fetches the report, the mask panel and the
    overlay panel, which without them would be three identical runs of the same
    second-and-a-half of morphology.

    Both hold one entry. More would only help someone comparing two thresholds
    at once, and each entry is tens of megabytes of boolean array.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._basis_memo: tuple[tuple[Any, ...], Basis] | None = None
        self._result_memo: tuple[tuple[Any, ...], tissue_mask.TissueMask] | None = None

    # --- storage ------------------------------------------------------------

    def _dir(self, upload_id: str) -> Path:
        return settings.tissue_dir / upload_id

    def _path(self, upload_id: str, name: str) -> Path:
        return self._dir(upload_id) / name

    # --- the basis ----------------------------------------------------------

    def _qc_state(self, upload_id: str) -> tuple[Any, float, str, str] | None:
        try:
            return qc_service.artefact_footprint(upload_id)
        except (OSError, KeyError, ValueError):
            # A half-written QC cache should degrade step 3 to ungated, not
            # break it. The report says which happened either way.
            logger.warning("unreadable QC output for %s", upload_id, exc_info=True)
            return None

    def _basis(self, upload_id: str, *, target_mpp: float) -> Basis:
        """The basis for this slide and resolution, from memo, disk, or the slide.

        The memo key carries step 2's `generatedAt`, so a QC re-run invalidates
        it for the same reason it invalidates the files on disk: it changes which
        pixels are in play, and therefore the histogram the threshold came from.
        """
        qc = self._qc_state(upload_id)
        qc_generated_at = qc[3] if qc else None
        key = (upload_id, target_mpp, qc_generated_at)

        with self._lock:
            if self._basis_memo is not None and self._basis_memo[0] == key:
                return self._basis_memo[1]

        basis = self._basis_from_disk(upload_id, target_mpp=target_mpp, qc=qc)

        with self._lock:
            self._basis_memo = (key, basis)
        return basis

    def _basis_from_disk(
        self, upload_id: str, *, target_mpp: float, qc: tuple[Any, float, str, str] | None
    ) -> Basis:
        import json

        qc_generated_at = qc[3] if qc else None
        meta_path = self._path(upload_id, "basis.json")

        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                fresh = (
                    abs(float(meta["target_mpp"]) - target_mpp) < 1e-6
                    and meta.get("qc_generated_at") == qc_generated_at
                )
                if fresh:
                    return self._load_basis(upload_id, meta)
            except (OSError, KeyError, ValueError, json.JSONDecodeError):
                logger.warning("rebuilding unreadable tissue basis for %s", upload_id)

        return self._compute_basis(upload_id, target_mpp=target_mpp, qc=qc)

    def _load_basis(self, upload_id: str, meta: dict[str, Any]) -> Basis:
        """Reconstitute a cached basis, or raise so the caller rebuilds it.

        The shape checks are not defensive padding. `mask_mpp` is what every
        physical cutoff in this step is divided by, and it was measured against
        the array these files were written from - so three arrays that disagree
        about their own size are three arrays that disagree about what 30 um
        means. Better to rebuild than to report a threshold in the wrong units.
        """
        saturation = np.asarray(
            Image.open(self._path(upload_id, "saturation.png")).convert("L")
        )

        expected = (int(meta["mask_height"]), int(meta["mask_width"]))
        if saturation.shape != expected:
            raise ValueError("cached tissue basis does not match its recorded shape")

        considered_path = self._path(upload_id, "considered.png")
        if considered_path.is_file():
            considered = np.asarray(Image.open(considered_path).convert("L")) > 0
            if considered.shape != expected:
                raise ValueError("cached artefact footprint does not match the mask shape")
        else:
            considered = np.ones(saturation.shape, dtype=bool)

        return Basis(
            saturation=saturation,
            considered=considered,
            mpp=float(meta["mask_mpp"]),
            target_mpp=float(meta["target_mpp"]),
            capped=bool(meta["capped"]),
            qc_gated=bool(meta["qc_gated"]),
            qc_source=meta.get("qc_source"),
            qc_generated_at=meta.get("qc_generated_at"),
            slide_width=int(meta["slide_width"]),
            slide_height=int(meta["slide_height"]),
            base_mpp=float(meta["base_mpp"]),
        )

    def _compute_basis(
        self, upload_id: str, *, target_mpp: float, qc: tuple[Any, float, str, str] | None
    ) -> Basis:
        import json

        path = resolve_ready_path(upload_id=upload_id)

        with open_slide(path) as reader:
            base_mpp = reader.mpp
            if not base_mpp:
                raise TissueMaskError(
                    "this slide records no microns-per-pixel, so the closing radius and the "
                    "minimum component area have no physical meaning. Supply a scale on "
                    "step 1 first - stating those cutoffs in pixels instead would make them "
                    "change meaning on the next scanner."
                )

            slide_width, slide_height = reader.dimensions
            thumbnail, achieved = tissue_mask.thumbnail_at_mpp(
                reader,
                base_mpp=base_mpp,
                target_mpp=target_mpp,
                max_px=settings.tissue_mask_max_px,
            )

        saturation = saturation_channel(thumbnail)
        shape = (saturation.shape[0], saturation.shape[1])

        if qc is not None:
            artefacts = tissue_mask.artefact_footprint(
                qc[0], artefact_ids=_ARTEFACT_IDS, shape=shape
            )
            considered = ~artefacts
        else:
            considered = np.ones(shape, dtype=bool)

        wanted_px = round(max(slide_width, slide_height) * base_mpp / target_mpp)
        capped = wanted_px > settings.tissue_mask_max_px

        basis = Basis(
            saturation=saturation,
            considered=considered,
            mpp=achieved,
            target_mpp=target_mpp,
            capped=capped,
            qc_gated=qc is not None,
            qc_source=qc[2] if qc else None,
            qc_generated_at=qc[3] if qc else None,
            slide_width=slide_width,
            slide_height=slide_height,
            base_mpp=base_mpp,
        )

        # `store_*` for the saturation channel and the footprint: they are
        # re-read on every threshold change and a resampled copy would silently
        # move the resolution every physical cutoff is divided by. The thumbnail
        # is the one exception - nothing is measured from it - so it is cached
        # already fitted, which is the single largest saving in the overlay path.
        directory = self._dir(upload_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "display.png").write_bytes(overlay.thumbnail_png(thumbnail))
        (directory / "saturation.png").write_bytes(overlay.store_grey(saturation))
        if qc is not None:
            (directory / "considered.png").write_bytes(overlay.store_binary(considered))
        else:
            (directory / "considered.png").unlink(missing_ok=True)

        (directory / "basis.json").write_text(
            json.dumps(
                {
                    "target_mpp": basis.target_mpp,
                    "mask_mpp": basis.mpp,
                    "capped": basis.capped,
                    "qc_gated": basis.qc_gated,
                    "qc_source": basis.qc_source,
                    "qc_generated_at": basis.qc_generated_at,
                    "mask_width": int(saturation.shape[1]),
                    "mask_height": int(saturation.shape[0]),
                    "slide_width": slide_width,
                    "slide_height": slide_height,
                    "base_mpp": base_mpp,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return basis

    def _display(self, upload_id: str) -> np.ndarray:
        """The cached display-sized thumbnail, for the panel that composites over it."""
        return np.asarray(Image.open(self._path(upload_id, "display.png")).convert("RGB"))

    # --- running ------------------------------------------------------------

    def _run(
        self, upload_id: str, *, threshold: int | None, target_mpp: float | None
    ) -> tuple[tissue_mask.TissueMask, Basis]:
        resolved = target_mpp or settings.tissue_mask_mpp
        basis = self._basis(upload_id, target_mpp=resolved)

        # The QC stamp is in the key as well as in the basis key: without it, a
        # QC re-run would hand back a fresh basis and a stale mask built from the
        # previous one.
        key = (upload_id, resolved, threshold, basis.qc_generated_at)
        with self._lock:
            if self._result_memo is not None and self._result_memo[0] == key:
                return self._result_memo[1], basis

        result = tissue_mask.build_mask(
            saturation=basis.saturation,
            considered=basis.considered,
            mpp=basis.mpp,
            threshold=threshold,
            close_um=settings.tissue_close_um,
            open_um=settings.tissue_open_um,
            min_component_mm2=settings.tissue_min_component_mm2,
            fill_hole_max_mm2=settings.tissue_fill_hole_max_mm2,
            spike_share=settings.tissue_spike_share,
        )

        with self._lock:
            self._result_memo = (key, result)
        return result, basis

    def footprint(
        self,
        upload_id: str,
        *,
        threshold: int | None = None,
        target_mpp: float | None = None,
    ) -> TissueFootprint:
        """Step 3's mask, for a later step that needs the pixels and not the prose.

        Exists so step 4 can ask step 3 for its output through the service that
        owns it, rather than reading `data/tissue/` and re-deriving what the files
        mean - the same reason `qc_service.artefact_footprint` exists, and the
        same argument one step along: two places that decide what a tissue mask
        is are two places that can disagree about it.

        `considered` travels with the mask because step 4 needs both and they mean
        different things. The mask says where the tissue is; `considered` says
        which pixels step 2 left in play, and step 4 has to exclude the artefacts
        from its *glass* as well - pen ink sits on the glass, and it is darker
        than glass. None means step 2 never ran.
        """
        result, basis = self._run(upload_id, threshold=threshold, target_mpp=target_mpp)

        return TissueFootprint(
            mask=result.mask,
            considered=basis.considered if basis.qc_gated else None,
            mpp=result.mpp,
            threshold=result.choice.value,
            threshold_source=result.choice.rule,
            qc_gated=basis.qc_gated,
            qc_source=basis.qc_source,
            slide_width=basis.slide_width,
            slide_height=basis.slide_height,
            base_mpp=basis.base_mpp,
            capped=basis.capped,
            key=(
                f"{result.choice.value}@{result.mpp:.4f}"
                f"/{basis.qc_generated_at or 'ungated'}"
            ),
        )

    def report(
        self, upload_id: str, *, threshold: int | None = None, target_mpp: float | None = None
    ) -> TissueReport:
        """Run step 3 and describe the result.

        `threshold` of None means Otsu picks. A value makes this a manual
        comparison run; Otsu's own answer is reported regardless, so the screen
        always carries both.
        """
        result, basis = self._run(upload_id, threshold=threshold, target_mpp=target_mpp)
        record = get_record(upload_id=upload_id)
        return self._describe(result, basis, upload_id=upload_id, filename=record.filename)

    def panel(
        self,
        upload_id: str,
        name: str,
        *,
        threshold: int | None = None,
        target_mpp: float | None = None,
    ) -> bytes:
        """One of the four panels as PNG.

        The two threshold-independent panels are served straight from the cached
        basis. The two that depend on the cut are rendered per request, because
        caching a picture per threshold would mean the slider either lagged or
        filled the disk.
        """
        if name not in PANELS:
            raise TissueMaskError(
                f"unknown panel {name!r}; expected one of {sorted(PANELS)}"
            )

        if not PANELS[name]:
            # Builds the basis if this is the first request for the slide, so a
            # panel is never served from a cache that does not exist yet.
            basis = self._basis(
                upload_id, target_mpp=target_mpp or settings.tissue_mask_mpp
            )
            if name == "saturation":
                return overlay.saturation_png(basis.saturation)
            # Already rendered at cache time, so this is a file read.
            return self._path(upload_id, "display.png").read_bytes()

        result, _ = self._run(upload_id, threshold=threshold, target_mpp=target_mpp)
        if name == "mask":
            return overlay.mask_png(result.mask)
        return overlay.overlay_png(self._display(upload_id), result.mask)

    # --- describing ---------------------------------------------------------

    def _describe(
        self,
        result: tissue_mask.TissueMask,
        basis: Basis,
        *,
        upload_id: str,
        filename: str,
    ) -> TissueReport:
        mpp = result.mpp
        pixel_mm2 = (mpp / 1000.0) ** 2

        def area(pixels: int) -> float:
            return round(pixels * pixel_mm2, 4)

        histogram = result.histogram
        counted = int(histogram.sum())

        choice = result.choice
        criterion = result.criterion
        peak = float(criterion.max()) if criterion.size else 0.0
        at_cut = float(criterion[choice.value]) if peak > 0 else 0.0

        below, above = class_means(histogram, choice.value)

        threshold = TissueThreshold(
            value=choice.value,
            source=choice.rule,
            otsu=choice.otsu,
            triangle=choice.triangle,
            modal_level=choice.modal_level,
            modal_share=round(choice.modal_share, 6),
            spike_share=choice.spike_share,
            bimodal=choice.bimodal,
            triangle_from=choice.triangle_from,
            triangle_to=choice.triangle_to,
            mean_below=round(below, 3) if below is not None else None,
            mean_above=round(above, 3) if above is not None else None,
            separation=(
                round(above - below, 3) if below is not None and above is not None else None
            ),
            # Negative entries mark thresholds that leave one side empty; clamping
            # keeps a slider dragged to either extreme reporting 0, not a
            # nonsensical negative ratio.
            variance_ratio=round(max(0.0, at_cut) / peak, 4) if peak > 0 else 0.0,
        )

        stages: list[TissueStage] = []
        previous = 0
        for index, stage in enumerate(result.stages):
            delta = stage.pixels - previous if index else 0
            stages.append(
                TissueStage(
                    key=stage.key,
                    label=stage.label,
                    what=stage.what,
                    extent_um=stage.extent_um,
                    pixels=stage.pixels,
                    area_mm2=area(stage.pixels),
                    delta_pixels=delta,
                    delta_area_mm2=area(delta),
                    delta_share=round(delta / previous, 6) if previous else 0.0,
                )
            )
            previous = stage.pixels

        assert result.components is not None  # build_mask always sets it
        parts = result.components
        tissue_pixels = result.tissue_pixels

        components = TissueComponents(
            found=parts.found,
            kept=parts.kept,
            dropped=parts.dropped,
            kept_pixels=parts.kept_pixels,
            dropped_pixels=parts.dropped_pixels,
            dropped_area_mm2=area(parts.dropped_pixels),
            largest_pixels=parts.largest_pixels,
            largest_area_mm2=area(parts.largest_pixels),
            largest_share=(
                round(parts.largest_pixels / parts.kept_pixels, 4) if parts.kept_pixels else 0.0
            ),
            min_area_mm2=parts.min_area_mm2,
            min_area_px=parts.min_area_px,
        )

        total_pixels = int(result.mask.size)
        share = tissue_pixels / total_pixels if total_pixels else 0.0

        params = TissueParams(
            target_mpp=basis.target_mpp,
            mask_mpp=round(mpp, 4),
            mask_width=int(result.mask.shape[1]),
            mask_height=int(result.mask.shape[0]),
            capped=basis.capped,
            close_um=settings.tissue_close_um,
            open_um=settings.tissue_open_um,
            close_px=max(1, round(settings.tissue_close_um / mpp)),
            open_px=max(1, round(settings.tissue_open_um / mpp)),
            min_component_mm2=settings.tissue_min_component_mm2,
            fill_hole_max_mm2=settings.tissue_fill_hole_max_mm2,
            qc_gated=basis.qc_gated,
            qc_source=basis.qc_source,
        )

        return TissueReport(
            upload_id=upload_id,
            filename=filename,
            generated_at=_now(),
            params=params,
            threshold=threshold,
            histogram=TissueHistogram(
                bins=[int(value) for value in histogram],
                counted_pixels=counted,
                excluded_pixels=result.artefact_pixels,
                criterion=[
                    round(max(0.0, float(value)) / peak, 6) if peak > 0 else 0.0
                    for value in criterion
                ],
            ),
            stages=stages,
            components=components,
            tissue_pixels=tissue_pixels,
            tissue_area_mm2=area(tissue_pixels),
            slide_area_mm2=area(total_pixels),
            tissue_share=round(share, 6),
            glass_share=round(1.0 - share, 6),
            holes_filled_pixels=result.holes_filled_pixels,
            holes_filled_area_mm2=area(result.holes_filled_pixels),
            notes=self._notes(
                basis=basis,
                params=params,
                threshold=threshold,
                components=components,
                share=share,
                holes_filled_area_mm2=area(result.holes_filled_pixels),
            ),
            citation=CITATION,
        )

    @staticmethod
    def _notes(
        *,
        basis: Basis,
        params: TissueParams,
        threshold: TissueThreshold,
        components: TissueComponents,
        share: float,
        holes_filled_area_mm2: float,
    ) -> list[str]:
        """The caveats that belong beside the numbers, built from the numbers."""
        notes = [
            "Fat is tissue and stays in this mask. Nothing here removes a region for "
            "being pale - fat leaves the analysis at step 8, as a class the model names "
            "and you can toggle. That is what makes its removal auditable.",
            f"Every distance is physical: closing {params.close_um:g} um "
            f"({params.close_px} px at this resolution), opening {params.open_um:g} um "
            f"({params.open_px} px), smallest component {params.min_component_mm2} mm2 "
            f"({components.min_area_px:,} px). None of these is a pixel count, so none of "
            "them changes meaning on a different scanner.",
            f"{share:.1%} of the slide is tissue, so steps 4 onward have "
            f"{1 - share:.1%} less area to visit. That reduction is the reason this step "
            "runs before anything expensive.",
        ]

        if holes_filled_area_mm2 > 0:
            notes.append(
                f"{holes_filled_area_mm2:.2f} mm2 of enclosed pale region - anything under "
                f"{params.fill_hole_max_mm2} mm2 - was filled back in. That is mostly fat "
                "lobules and gland lumina, which a saturation threshold cuts out and Rule 1 "
                "says must stay. Fat at the cut edge of the section is still lost; no "
                "morphology recovers it, which is precisely why fat is a learned class "
                "later and not a threshold here."
            )

        if basis.qc_gated:
            notes.append(
                "Step 2's artefacts were subtracted before the histogram was built, not "
                "after the mask was made. Pen ink is more saturated than any stain, so "
                "leaving it in does not merely add a pen-shaped blob - it drags the cut "
                "towards the ink, whichever rule picks it."
            )
            if basis.qc_source != "grandqc":
                notes.append(
                    "Step 2 found its tissue with the saturation/Otsu fallback rather than "
                    "GrandQC's tissue model, so the artefact map it handed over is weaker "
                    "than it should be. Download the tissue checkpoint before trusting the "
                    "area below."
                )
        else:
            notes.append(
                "Step 2 has not run for this slide, so nothing was subtracted before "
                "thresholding. Any pen mark on the glass is in this mask and is pulling the "
                "threshold. Run quality control and open this step again."
            )

        if params.capped:
            notes.append(
                f"Asked for {params.target_mpp:g} um/px, worked at "
                f"{params.mask_mpp:g} um/px: the mask's longest edge is capped at "
                f"{settings.tissue_mask_max_px:,} px and this slide is large enough for the "
                "cap to bind. Every physical threshold above is divided by the resolution "
                "actually achieved, not the one requested. For this question that is still "
                "ample - GrandQC's own tissue detector runs at 10 um/px, coarser than this - "
                "but it is a different number from the one asked for, so it is stated."
            )

        # The rule choice, stated with the statistic it turned on. Both answers
        # appear either way - a threshold is the most consequential number in
        # this step, so it never appears without its alternative.
        if threshold.bimodal:
            notes.append(
                f"The histogram is two humps - its busiest level holds "
                f"{threshold.modal_share:.1%} of the pixels, under the "
                f"{threshold.spike_share:.0%} mark - so Otsu's assumption holds and Otsu "
                f"chose the cut at {threshold.otsu}. Zack's triangle rule would have said "
                f"{threshold.triangle}; on this shape it is the weaker of the two, because "
                "a straight chord is a poor model of a histogram with a valley in it."
            )
        else:
            notes.append(
                f"Otsu was set aside here, and this is the one place this step overrides the "
                f"pipeline guide. {threshold.modal_share:.1%} of the pixels sit on a single "
                f"saturation level ({threshold.modal_level}), over the "
                f"{threshold.spike_share:.0%} mark, so the histogram is a spike and a tail "
                "rather than two humps. Otsu's criterion is a ratio of variances and a spike "
                f"has none, so it is maximised far out in the tail: it says {threshold.otsu}, "
                "which on this slide keeps about a third of the section. Zack's triangle "
                f"rule is derived for this shape and says {threshold.triangle}, which is what "
                "was used. Both numbers are above; drag the cut and judge for yourself."
            )

        if threshold.source == "manual":
            note = (
                f"This is a hand-set threshold of {threshold.value}, against Otsu's "
                f"{threshold.otsu} and the triangle rule's {threshold.triangle}."
            )
            # Otsu's criterion only means something as a score when the shape it
            # assumes is actually present. Quoting the ratio on a spike-and-tail
            # histogram would dress up "disagrees with Otsu" as "is worse".
            if threshold.bimodal:
                note += (
                    f" Otsu's criterion scores it at {threshold.variance_ratio:.1%} of its "
                    "value at Otsu's own cut, which is the honest measure of what you gave "
                    "up by picking the number yourself."
                )
            else:
                note += (
                    " No criterion score is quoted for it: on a spike-and-tail histogram "
                    "Otsu's criterion is the wrong yardstick, and a low score against it "
                    "would say your cut disagrees with Otsu rather than that it is worse."
                )
            notes.append(note)

        return notes


tissue_service = TissueService()
