"""Step 5 - optical density, orchestrated.

Like steps 3 and 4 and unlike step 2, this is not a background job: one tile read
off the pyramid and a handful of numpy passes over a quarter of a million pixels,
so it answers in the request that asked for it and there is nothing to poll.

**Nothing is cached on disk, and that is a decision.** Steps 3 and 4 cache because
their input is the whole slide re-read at 2 um/px. Step 5's input is one 512 px
tile - a thousandth of those pixels, one `read_region` - and everything else it
needs already sits in another service's cache: the white point comes from
`calibration_service.white_point()`, and the tile chooser scores its candidates on
the RGB thumbnail that same service holds. A disk cache here would have to be keyed
on step 3's threshold, step 4's percentile *and* the tile position at once, and
would save less time than invalidating it correctly would cost. One in-process
memo, holding one entry, covers the case that actually happens: a single view of
this step fetches the report and five panels, which without it would be six runs of
the same arithmetic.

**Why this step asks two earlier services rather than reading their files.** Step 5
needs three things from upstream - the tissue mask, the white point, and the
thumbnail both were built on - and asks for each through the service that owns it.
That is the same argument `tissue_service.footprint` and
`calibration_service.white_point` were written for, one step further along: optical
density is *defined* against I0, so a second estimate of I0 in this package would
not be duplication but a second definition, and the two could differ while both
looked right.

**Why the tile is chosen and not taken.** A density is per pixel, so unlike the
three steps before it this one has to pick a field of view. That pick is not
neutral. Two thirds of a stained section is counterstain and stroma, and a tile of
that has *one* arm in its point cloud - so a demo that picks at random will,
often, put a one-armed scatter on screen underneath a caption explaining that the
two arms are the two stains. `tiles.rank_tiles` scores every block instead and
hands back its reasons, and the report carries the alternatives so the choice can
be checked rather than trusted.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from app.common.stains import direction_is_stable
from app.core.config import settings
from app.core.logging import get_logger
from app.ingestion.slide_reader import open_slide
from app.pipeline.step05_optical_density import density as od
from app.pipeline.step05_optical_density import overlay, tiles
from app.pipeline.step05_optical_density.density import DensityError
from app.pipeline.step05_optical_density.tiles import Candidate, Tile, TileError
from app.schemas.calibration import Channels
from app.schemas.density import (
    AdditivityOut,
    ArmOut,
    CloudOut,
    DensityParams,
    DensityReport,
    DensityStatsOut,
    DriftBin,
    LimitsOut,
    ReferenceOut,
    TileCandidate,
    TileOut,
    WhiteUsed,
)
from app.services.calibration_service import WhiteReference, calibration_service
from app.services.tissue_service import TissueFootprint, tissue_service
from app.services.upload_service import get_record, resolve_ready_path

logger = get_logger(__name__)

CITATION = (
    "Ruifrok AC, Johnston DA. Quantification of histochemical staining by colour "
    "deconvolution. Analytical and Quantitative Cytology and Histology 23(4):291-299 "
    "(2001). https://pubmed.ncbi.nlm.nih.gov/11531144/ · "
    "Macenko M, Niethammer M, Marron JS, et al. A method for normalizing histology "
    "slides for quantitative analysis. ISBI 2009:1107-1110. · "
    "Beer A. Bestimmung der Absorption des rothen Lichts in farbigen Flussigkeiten. "
    "Annalen der Physik und Chemie 86:78-88 (1852)."
)

#: The panels the API serves, in the order they are meant to be read.
PANELS: tuple[str, ...] = ("map", "tile", "density", "scatter", "limits")

#: Angular drift, in degrees, at or below which density space is called linear -
#: the property step 6's matrix inverse depends on. Set where 8-bit quantisation
#: and the sensor's own noise land: a stain vector recovered from a few thousand
#: pixels wobbles by about a degree even when the physics is exact, so anything
#: under two is indistinguishable from perfect.
LINEAR_DRIFT = 2.0

#: How many times the intensity-space drift has to exceed the density-space drift
#: before the additivity check is called a demonstration rather than a wash. Two:
#: below that the two spaces are behaving similarly on this tile, which happens
#: when the tile is faint - `I = I0 * 10^(-cv)` is very nearly linear for small c,
#: so the curvature the check looks for is genuinely not there to find.
DRIFT_RATIO = 2.0

#: How the reference stains are written in prose. "dab" is an acronym and reads as
#: a typo in a sentence; the API keys stay lower case so the frontend can switch on
#: them, and the presentation happens here and in the frontend's own label map.
STAIN_LABELS = {"haematoxylin": "haematoxylin", "dab": "DAB", "eosin": "eosin"}


def _stain(name: str) -> str:
    return STAIN_LABELS.get(name, name)


#: Share of a tile that may be brighter than I0 before it is worth remarking on.
#: Some is expected and benign - a tile inside a section still contains gland
#: lumina and tears - but a large share means I0 is low for this magnification.
NEGATIVE_SHARE_LIMIT = 0.02


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


class DensityService:
    """Runs step 5 on one tile, and says which tile and why.

    One memo entry, like step 4's, and keyed on everything that can change the
    answer: the slide, step 3's mask, step 4's white point and the tile position.
    Anything less would serve a density computed against a white point nobody is
    looking at any more.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._memo: tuple[tuple[object, ...], _Run] | None = None
        self._screen_memo: tuple[tuple[object, ...], _Screen] | None = None
        self._sheet_memo: tuple[tuple[object, ...], dict[tuple[int, int], bytes]] | None = None

    # --- running ------------------------------------------------------------

    def _screen(
        self,
        upload_id: str,
        *,
        threshold: int | None,
        percentile: float | None,
    ) -> _Screen:
        """Everything that is true before a tile is chosen, memoised on its own.

        Split from `_run` because it is the half that does not depend on the tile.
        Twelve contact-sheet thumbnails and the report behind them all need the
        same candidate list, and a single memo keyed on the chosen tile would be
        evicted by every one of those requests in turn - so the screening would be
        re-run twelve times to serve twelve crops of a picture it had already
        made. Two memos, each keyed on what its own answer depends on.
        """
        footprint = tissue_service.footprint(upload_id, threshold=threshold)
        white = calibration_service.white_point(
            upload_id, threshold=threshold, percentile=percentile
        )

        key = (upload_id, footprint.key, white.key)
        with self._lock:
            if self._screen_memo is not None and self._screen_memo[0] == key:
                return self._screen_memo[1]

        thumbnail, screening_mpp = calibration_service.thumbnail(
            upload_id, threshold=threshold
        )

        tile_um = settings.tile_size * settings.target_mpp
        candidates = tiles.rank_tiles(
            rgb=thumbnail,
            tissue=footprint.mask,
            considered=footprint.considered,
            # The flat triple even when a surface is in force, and only here. This
            # pass ranks blocks against each other; a vignette of a few percent
            # cannot reorder scores that differ by tens of percent, and evaluating
            # the field over the whole thumbnail to find that out would cost more
            # than the screening itself. The chosen tile is divided by the real
            # field, in `_run`.
            white=white.rgb,
            mpp=screening_mpp,
            base_mpp=footprint.base_mpp,
            slide_size=(footprint.slide_width, footprint.slide_height),
            tile_um=tile_um,
            min_tissue_share=settings.density_min_tissue_share,
            min_stain=settings.density_min_stain_multiple * white.noise_floor,
            od_floor=white.od_floor,
            limit=settings.density_candidates,
        )

        screen = _Screen(
            footprint=footprint,
            white=white,
            thumbnail=thumbnail,
            screening_mpp=screening_mpp,
            candidates=candidates,
            tile_um=tile_um,
        )

        with self._lock:
            self._screen_memo = (key, screen)
        return screen

    def _run(
        self,
        upload_id: str,
        *,
        threshold: int | None,
        percentile: float | None,
        x: int | None,
        y: int | None,
    ) -> _Run:
        """Choose a tile, read it, and transform it.

        `threshold` and `percentile` belong to steps 3 and 4 and are passed
        straight through to them. Step 5 owns neither and interprets neither: the
        viewer moves those controls on those steps' screens, and this step
        re-derives its density from whatever white point that produced. Its own
        control is the tile, and only the tile.
        """
        screen = self._screen(upload_id, threshold=threshold, percentile=percentile)
        footprint, white = screen.footprint, screen.white
        thumbnail, screening_mpp = screen.thumbnail, screen.screening_mpp
        candidates, tile_um = screen.candidates, screen.tile_um

        requested = x is not None and y is not None
        chosen = (
            tiles.nearest(candidates, x=int(x or 0), y=int(y or 0))
            if requested
            else candidates[0]
        )

        key = (upload_id, footprint.key, white.key, chosen.x, chosen.y)
        with self._lock:
            if self._memo is not None and self._memo[0] == key:
                return self._memo[1]

        path = resolve_ready_path(upload_id=upload_id)
        with open_slide(path) as reader:
            tile = tiles.read_tile(
                reader,
                x=chosen.x,
                y=chosen.y,
                target_mpp=settings.target_mpp,
                size=settings.tile_size,
                base_mpp=footprint.base_mpp,
            )

        result = od.transform(
            rgb=tile.rgb,
            # The real field here, flat or fitted. `field_for` evaluates step 4's
            # surface over the patch of normalised slide coordinates this tile
            # occupies, which is exact - the surface is an analytic quadratic in
            # those coordinates, so sampling it on the tile's grid is the same
            # function and not an interpolation of it.
            white=white.field_for(x=tile.x, y=tile.y, size=tile.size, mpp=tile.mpp),
            mpp=tile.mpp,
            floor=white.od_floor,
            beta=settings.density_beta,
            alpha=settings.density_arm_percentile,
            tolerance_deg=settings.density_angular_tolerance_deg,
        )

        run = _Run(
            density=result,
            tile=tile,
            chosen=chosen,
            candidates=candidates,
            white=white,
            thumbnail=thumbnail,
            screening_mpp=screening_mpp,
            requested=requested,
            tile_um=tile_um,
        )

        with self._lock:
            self._memo = (key, run)
        return run

    def report(
        self,
        upload_id: str,
        *,
        threshold: int | None = None,
        percentile: float | None = None,
        x: int | None = None,
        y: int | None = None,
    ) -> DensityReport:
        """Run step 5 and describe the result."""
        run = self._run(
            upload_id, threshold=threshold, percentile=percentile, x=x, y=y
        )
        record = get_record(upload_id=upload_id)
        return self._describe(run, upload_id=upload_id, filename=record.filename)

    def panel(
        self,
        upload_id: str,
        name: str,
        *,
        threshold: int | None = None,
        percentile: float | None = None,
        x: int | None = None,
        y: int | None = None,
    ) -> bytes:
        """One of the five panels as PNG."""
        if name not in PANELS:
            raise DensityError(f"unknown panel {name!r}; expected one of {sorted(PANELS)}")

        run = self._run(
            upload_id, threshold=threshold, percentile=percentile, x=x, y=y
        )

        if name == "map":
            return overlay.map_png(
                run.thumbnail, candidates=run.candidates, chosen=run.chosen
            )
        if name == "tile":
            return overlay.tile_png(run.density.rgb)
        if name == "density":
            return overlay.density_png(run.density)
        if name == "scatter":
            return overlay.scatter_png(run.density)
        return overlay.limits_png(run.density)

    def _contact_sheet(
        self,
        upload_id: str,
        *,
        threshold: int | None,
        percentile: float | None,
    ) -> dict[tuple[int, int], bytes]:
        """Every candidate as a thumbnail, rendered together and memoised together.

        All twelve in one slide open, and that is the whole reason this is a batch
        rather than twelve independent renders. Opening a scan costs about a
        second and reading a 192 px block off a coarse level costs twenty
        milliseconds, so twelve reads behind one open is a fifth of a second while
        twelve opens is twelve seconds - and the browser asks for all twelve at
        once, because that is what a contact sheet is.

        Memoised beside the screening rather than inside it: the screening is
        needed by every request this step serves and this is needed only by the
        chooser, so a viewer who never looks at the alternatives never pays for
        them.
        """
        screen = self._screen(upload_id, threshold=threshold, percentile=percentile)

        key = (upload_id, screen.footprint.key, screen.white.key)
        with self._lock:
            if self._sheet_memo is not None and self._sheet_memo[0] == key:
                return self._sheet_memo[1]

        # The resolution follows from the display size and the block's physical
        # extent, so a thumbnail is the same field of view as the tile that would
        # be read if it were clicked - just coarser. Anything else would show the
        # reader one field and transform another.
        target_mpp = screen.tile_um / overlay.CANDIDATE_SIZE

        sheet: dict[tuple[int, int], bytes] = {}
        path = resolve_ready_path(upload_id=upload_id)
        with open_slide(path) as reader:
            for candidate in screen.candidates:
                block = tiles.read_tile(
                    reader,
                    x=candidate.x,
                    y=candidate.y,
                    target_mpp=target_mpp,
                    size=overlay.CANDIDATE_SIZE,
                    base_mpp=screen.footprint.base_mpp,
                )
                sheet[(candidate.col, candidate.row)] = overlay.candidate_png(block.rgb)

        logger.info(
            "density.contact_sheet",
            extra={"upload_id": upload_id, "blocks": len(sheet), "mpp": target_mpp},
        )

        with self._lock:
            self._sheet_memo = (key, sheet)
        return sheet

    def candidate(
        self,
        upload_id: str,
        *,
        col: int,
        row: int,
        threshold: int | None = None,
        percentile: float | None = None,
    ) -> bytes:
        """One scored block as a thumbnail, for the contact sheet.

        Addressed by `col` and `row` rather than by level-0 x and y, because this
        is a picture of a *scored block* and not of an arbitrary position. A
        position would have to be snapped to the nearest block before it meant
        anything, and a caller who mistyped one by a hundred pixels would silently
        be shown its neighbour; a column and row either name a block that was
        scored or name nothing, and the second is a 404.
        """
        sheet = self._contact_sheet(
            upload_id, threshold=threshold, percentile=percentile
        )

        png = sheet.get((col, row))
        if png is None:
            raise DensityError(
                f"no scored block at column {col}, row {row}; "
                f"this slide has {len(sheet)} candidates"
            )
        return png

    # --- handing on to step 6 -----------------------------------------------

    def transformed_tile(
        self,
        upload_id: str,
        *,
        threshold: int | None = None,
        percentile: float | None = None,
        x: int | None = None,
        y: int | None = None,
    ) -> DensityHandoff:
        """Step 5's output, for the step that consumes it rather than displays it.

        The analogue of `tissue_service.footprint` and
        `calibration_service.white_point` one step further along, and it exists for
        exactly the reason those do: step 6's input is the optical density tile, and
        the only defensible way for it to get one is to ask the step that owns the
        transform. A second `-log10(I / I0)` inside step 6's package would not be
        duplication, it would be a second answer to the question step 4 and step 5
        were built to answer once.

        There is a stronger version of that argument here than at either earlier
        hand-off, and it is the fork itself. Rule 2 says the pipeline becomes a Y at
        step 6: one deconvolution, the haematoxylin channel to the model and the DAB
        channel to the measurement. A Y is only meaningful if both arms start from
        the *same* point, so step 6 receiving anything other than step 5's own array
        would make the fork a drawing rather than a fact about the code.

        Free, in the common case. `_run` is memoised on the slide, the mask, the
        white point and the tile, so a reader who has just looked at step 5 and
        moved to step 6 gets the identical `Density` object back without a second
        read off the pyramid or a second logarithm.
        """
        run = self._run(upload_id, threshold=threshold, percentile=percentile, x=x, y=y)
        result = run.density

        flat = result.od.reshape(-1, 3).astype(np.float64)
        observed = np.maximum(
            result.rgb.reshape(-1, 3).astype(np.float64), float(result.floor)
        )
        mean_od = flat.mean(axis=1)

        # The same two admission tests step 5's cloud uses, recomputed here rather
        # than carried on `Cloud` - because `Cloud` is None on a tile with no arms
        # and step 6 still has to be able to say *why* it cannot un-mix that tile.
        stained = mean_od >= result.limits.beta
        admitted = stained & direction_is_stable(
            flat, observed, tolerance_deg=result.tolerance_deg
        )

        height, width = result.shape

        # Broadcast rather than tiled: `np.broadcast_to` returns a read-only view, so
        # a flat white point costs three floats here and not three megabytes, and the
        # caller never has to branch on which kind of I0 step 4 justified.
        field = np.asarray(result.white, dtype=np.float64)
        white_field = (
            field if field.ndim == 3 else np.broadcast_to(field, (height, width, 3))
        )

        arms = (
            tuple(
                (arm.nearest, np.asarray(arm.vector, dtype=np.float64))
                for arm in result.cloud.arms
            )
            if result.cloud is not None
            else ()
        )

        return DensityHandoff(
            od=result.od,
            rgb=result.rgb,
            white_field=white_field,
            admitted=admitted,
            stained=stained.reshape(height, width),
            arms=arms,
            cloud_explained=result.cloud.explained if result.cloud is not None else None,
            tile=run.tile,
            chosen=run.chosen,
            candidates=run.candidates,
            white=run.white,
            floor=result.floor,
            beta=result.limits.beta,
            alpha=settings.density_arm_percentile,
            tolerance_deg=result.tolerance_deg,
            tile_um=run.tile_um,
            screening_mpp=run.screening_mpp,
            requested=run.requested,
            # The upload id leads, and it is not redundant. Neither
            # `footprint.key` nor `white.key` names the slide - the services that
            # build them always pair them with an upload id in their own memo
            # tuples - so a key made of those two alone would be identical for two
            # different slides that happened to share a threshold rule, a
            # resolution, a QC state and a tile position. A consumer memoising on
            # this string would then serve one slide's work for another's.
            key=f"{upload_id}|{run.white.key}|{run.tile.x},{run.tile.y}",
        )

    # --- describing ---------------------------------------------------------

    def _describe(
        self, run: _Run, *, upload_id: str, filename: str
    ) -> DensityReport:
        result, tile, white = run.density, run.tile, run.white
        stats, limits, cloud = result.stats, result.limits, result.cloud

        params = DensityParams(
            target_mpp=settings.target_mpp,
            tile_mpp=round(tile.mpp, 5),
            tile_size=tile.size,
            tile_um=round(run.tile_um, 2),
            level=tile.level,
            downsample=round(tile.downsample, 4),
            resampled=tile.resampled,
            od_floor=white.od_floor,
            beta=settings.density_beta,
            arm_percentile=settings.density_arm_percentile,
            angular_tolerance_deg=settings.density_angular_tolerance_deg,
            screening_mpp=round(run.screening_mpp, 4),
            screening_block_px=max(4, round(run.tile_um / run.screening_mpp)),
            min_tissue_share=settings.density_min_tissue_share,
            min_stain=round(settings.density_min_stain_multiple * white.noise_floor, 4),
            min_stain_multiple=settings.density_min_stain_multiple,
            tissue_threshold=white.tissue_threshold,
            tissue_threshold_source=white.tissue_threshold_source,
            qc_gated=white.qc_gated,
            qc_source=white.qc_source,
        )

        return DensityReport(
            upload_id=upload_id,
            filename=filename,
            generated_at=_now(),
            params=params,
            white=WhiteUsed(
                rgb=_channels(white.rgb, digits=2),
                hex=_hex(white.rgb),
                mode=white.mode,
                percentile=white.percentile,
                saturated=white.saturated,
                noise_floor=round(white.noise_floor, 4),
            ),
            tile=TileOut(
                x=tile.x,
                y=tile.y,
                span=tile.span,
                size=tile.size,
                mpp=round(tile.mpp, 5),
                level=tile.level,
                downsample=round(tile.downsample, 4),
                resampled=tile.resampled,
                tissue_share=round(run.chosen.tissue_share, 4),
                considered_share=round(run.chosen.considered_share, 4),
                stain=round(run.chosen.stain, 4),
                mixing=round(run.chosen.mixing, 4),
                score=round(run.chosen.score, 5),
                rank=run.candidates.index(run.chosen) + 1,
                requested=run.requested,
                candidates_scored=len(run.candidates),
            ),
            candidates=[
                TileCandidate(
                    col=candidate.col,
                    row=candidate.row,
                    x=candidate.x,
                    y=candidate.y,
                    fx=round(candidate.fx, 6),
                    fy=round(candidate.fy, 6),
                    fw=round(candidate.fw, 6),
                    fh=round(candidate.fh, 6),
                    tissue_share=round(candidate.tissue_share, 4),
                    considered_share=round(candidate.considered_share, 4),
                    stain=round(candidate.stain, 4),
                    mixing=round(candidate.mixing, 4),
                    score=round(candidate.score, 5),
                    chosen=candidate is run.chosen,
                )
                for candidate in run.candidates
            ],
            stats=DensityStatsOut(
                median=_channels(stats.median, digits=4),
                p99=_channels(stats.p99, digits=4),
                maximum=_channels(stats.maximum, digits=4),
                mean_median=round(stats.mean_median, 4),
                mean_p99=round(stats.mean_p99, 4),
                mean_maximum=round(stats.mean_maximum, 4),
                mean_minimum=round(stats.mean_minimum, 4),
                histogram=list(stats.histogram),
                histogram_low=round(stats.histogram_low, 4),
                histogram_high=round(stats.histogram_high, 4),
            ),
            limits=LimitsOut(
                floor_share=round(limits.floor_share, 6),
                negative_share=round(limits.negative_share, 6),
                negative_worst=round(limits.negative_worst, 4),
                transparent_share=round(limits.transparent_share, 5),
                beta=limits.beta,
                roundtrip_max=round(limits.roundtrip_max, 5),
                roundtrip_mean=round(limits.roundtrip_mean, 6),
                roundtrip_exact_share=round(limits.roundtrip_exact_share, 6),
            ),
            cloud=self._describe_cloud(cloud) if cloud is not None else None,
            additivity=self._describe_additivity(result.additivity),
            notes=self._notes(run),
            citation=CITATION,
        )

    @staticmethod
    def _describe_cloud(cloud: od.Cloud) -> CloudOut:
        return CloudOut(
            basis=[_channels(vector, digits=5) for vector in cloud.basis],
            explained=round(cloud.explained, 5),
            plotted=cloud.plotted,
            admitted_share=round(cloud.admitted_share, 5),
            faint_share=round(cloud.faint_share, 5),
            unstable_share=round(cloud.unstable_share, 5),
            tolerance_deg=cloud.tolerance_deg,
            x_low=round(cloud.x_low, 5),
            x_high=round(cloud.x_high, 5),
            y_low=round(cloud.y_low, 5),
            y_high=round(cloud.y_high, 5),
            arms=[
                ArmOut(
                    angle=round(arm.angle, 5),
                    vector=_channels(arm.vector, digits=4),
                    share=round(arm.share, 4),
                    plot_x=round(arm.plot[0], 5),
                    plot_y=round(arm.plot[1], 5),
                    nearest=arm.nearest,
                    degrees_from_nearest=round(arm.degrees_from_nearest, 2),
                )
                for arm in cloud.arms
            ],
            references=[
                ReferenceOut(
                    name=reference.name,
                    vector=_channels(reference.vector, digits=4),
                    plot_x=round(reference.plot[0], 5),
                    plot_y=round(reference.plot[1], 5),
                    out_of_plane=round(reference.out_of_plane, 4),
                    degrees_from_arm=round(reference.degrees_from_arm, 2),
                )
                for reference in cloud.references
            ],
            separation=round(cloud.separation, 2),
            two_armed=cloud.two_armed,
            angles=list(cloud.angles),
            angle_low=round(cloud.angle_low, 5),
            angle_high=round(cloud.angle_high, 5),
            angle_centre=round(cloud.angle_centre, 5),
            reference_separation=round(cloud.reference_separation, 2),
        )

    @staticmethod
    def _describe_additivity(additivity: od.Additivity | None) -> AdditivityOut | None:
        if additivity is None:
            return None
        return AdditivityOut(
            arm=additivity.arm,
            bins=[
                DriftBin(
                    density=round(entry.density, 4),
                    pixels=entry.pixels,
                    od_degrees=round(entry.od_degrees, 3),
                    intensity_degrees=round(entry.intensity_degrees, 3),
                )
                for entry in additivity.bins
            ],
            od_drift=round(additivity.od_drift, 3),
            intensity_drift=round(additivity.intensity_drift, 3),
            pixels=additivity.pixels,
        )

    @staticmethod
    def _notes(run: _Run) -> list[str]:
        """The caveats that belong beside the numbers, built from the numbers."""
        result, tile, white = run.density, run.tile, run.white
        stats, limits, cloud = result.stats, result.limits, result.cloud
        chosen = run.chosen

        notes = [
            "Optical density is not a colour transform with a logarithm in it, it is a "
            "change of units into the quantity Beer-Lambert says is proportional to "
            "concentration. Transmitted light is multiplicative - two stains stacked "
            "multiply their transmissions - so no matrix can un-mix an RGB image. The "
            "logarithm turns that product into a sum, and a sum is what linear algebra "
            "inverts. That single fact is what makes step 6 valid, and it is measured "
            "rather than asserted below.",
            "This is the trunk the fork leaves from, and step 6 is where the Y is drawn. "
            "One deconvolution runs on these densities and serves both arms: the model is "
            "handed the haematoxylin channel in place of RGB, which is what lets a single "
            "detector work on the H&E and on every IHC slide, and the measurement is "
            "handed the DAB channel on an absolute optical density scale. Nothing between "
            "here and the measurement rewrites a pixel - doing so would overwrite the very "
            "intensity being measured.",
            f"The transform is reversible, and that is the honest test that this step is "
            f"not a filter. Applying the exact inverse I = I0 x 10^(-OD) back to these "
            f"densities returns "
            f"{limits.roundtrip_exact_share:.1%} of the tile to within half an intensity "
            f"level, worst error {limits.roundtrip_max:.4f} levels - which is float32 "
            "rounding and nothing else. No pixel was enhanced, smoothed or clipped on the "
            "way through.",
        ]

        # --- which tile, and why -------------------------------------------
        if run.requested:
            notes.append(
                f"This tile was asked for. It sits at ({tile.x}, {tile.y}) in level-0 "
                f"coordinates and ranks {run.candidates.index(chosen) + 1} of "
                f"{len(run.candidates)} on the screening score - so if its point cloud "
                "shows less than the best tile's, that is the comparison working rather "
                "than the step failing."
            )
        else:
            notes.append(
                f"This tile was chosen, not taken, and the choice matters more than it "
                f"looks. Two thirds of a stained section is counterstain, and a tile of "
                f"counterstain has one arm in its point cloud rather than two - so picking "
                f"at random would regularly put a one-armed scatter under a caption about "
                f"two stains. Every block of the slide was scored on stain "
                f"({chosen.stain:.3f} mean OD here) times mixing ({chosen.mixing:.3f} - how "
                f"far its cloud spreads off a single ray), and this one won at "
                f"{chosen.score:.4f}. The next {len(run.candidates) - 1} are listed: open "
                "one and watch the arms close up."
            )

        notes.append(
            f"Read at {tile.mpp:g} um/px from pyramid level {tile.level}, covering "
            f"{run.tile_um:g} um of slide - roughly {run.tile_um / 10:.0f} nuclei across, "
            "taking a breast epithelial nucleus at about 10 um. The level was chosen by "
            "microns and not by index, because the same level number is a different "
            "resolution on two scanners."
        )

        if tile.resampled:
            notes.append(
                f"The pyramid's nearest level was finer than {settings.target_mpp:g} um/px, "
                f"so the read was area-averaged down from {tile.span} level-0 pixels to "
                f"{tile.size}. The averaging happens in *intensity* space, before the "
                "logarithm, and that ordering is not a detail: a coarser sensor averages "
                "the light arriving over a larger area, so averaging transmissions is what "
                "a coarser scan physically is. Averaging densities instead would take the "
                "mean of logarithms - the logarithm of a geometric mean of transmission - "
                "which is a different number, biased low, and one no instrument records."
            )

        # --- the white point it was divided by -----------------------------
        triple = f"({white.rgb[0]:.0f}, {white.rgb[1]:.0f}, {white.rgb[2]:.0f})"
        if white.mode == "surface":
            notes.append(
                f"I0 varies across this slide, so the tile was divided by step 4's fitted "
                f"field evaluated over its own position rather than by the flat {triple}. "
                "That evaluation is exact and not an interpolation: the field is an "
                "analytic quadratic in coordinates normalised to the slide, so sampling it "
                "on this tile's grid returns the same function step 4 fitted."
            )
        else:
            notes.append(
                f"Divided by step 4's flat I0, {triple}, measured from this slide's own "
                f"glass at the {white.percentile:g}th percentile. Step 4 declined to fit a "
                "varying field on this slide and said why; that decision is in force here."
            )

        notes.append(
            f"Step 4 left a noise floor of {white.noise_floor:.3f} OD - what empty glass "
            f"measures as under this calibration. The median pixel of this tile measures "
            f"{stats.mean_median:.3f} and its 99th percentile {stats.mean_p99:.3f}, so the "
            f"signal here sits "
            f"{stats.mean_p99 / max(white.noise_floor, 1e-6):.0f}x above that floor at the "
            "top end. No positivity threshold at step 14 can be set below the floor, which "
            "is why the two numbers belong on one screen."
        )

        if white.saturated:
            notes.append(
                "A channel of I0 is clipped at 255, so every density on this screen is "
                "compressed: the sensor ran out of range before the glass did, the true "
                "incident intensity is higher than I0 says, and faint stain and no stain "
                "therefore measure closer together than they should. That is a scanning "
                "fault and no arithmetic here recovers it."
            )

        # --- the arms ------------------------------------------------------
        if cloud is None:
            notes.append(
                f"No point cloud was built: fewer than the minimum number of pixels in this "
                f"tile carry a mean density above beta = {limits.beta:g}, so there is "
                "nothing whose *direction* could be measured. That is a real state, not a "
                "failure - a tile of pale stroma has no stain vectors in it - and the "
                "honest response is to say so rather than to draw two arms through noise."
            )
        else:
            arms = ", ".join(
                f"{_stain(arm.nearest)} at {arm.degrees_from_nearest:.1f} degrees"
                for arm in cloud.arms
            )
            found = {arm.nearest for arm in cloud.arms}
            expected = {"haematoxylin", "dab"}
            notes.append(
                f"The two arms of the cloud land {arms} from Ruifrok & Johnston's published "
                "vectors, measured in three dimensions rather than in the projection. "
                + (
                    "Those are the two stains this assay uses, and nothing in this step was "
                    "told what either looks like - the wedge's edges came out pointing at "
                    "them. That is the claim 'the arms are the stains', as a number rather "
                    "than as an assertion."
                    if found == expected
                    else "One of those is not a stain this assay uses, and that is the "
                    "finding rather than a rounding error: an arm nearest eosin on a "
                    "haematoxylin-and-DAB slide means the wedge's edge points somewhere no "
                    "dye on this section absorbs. Read it as a question about the scan - "
                    "chroma artefacts in the tile, a third absorber, or a counterstain "
                    "whose colour has shifted - and try another tile before drawing a "
                    "conclusion about the stains."
                )
            )
            notes.append(
                "Those estimated directions are for looking at and not for measuring with, "
                "and the difference is step 6's central choice. Estimating stain vectors per "
                "image - which is what has just been done here, and is Macenko's method - "
                "gives every slide its own scale, so '0.4 DAB' would mean a different amount "
                "of stain on every slide in a study. Step 6 measures with Ruifrok's fixed "
                "vectors instead, and these arms are the evidence that the fixed ones fit "
                "this slide."
            )

            if cloud.two_armed:
                notes.append(
                    f"The two arms sit {cloud.separation:.1f} degrees apart against "
                    f"{cloud.reference_separation:.0f} between Ruifrok's own haematoxylin and "
                    f"DAB, and the plane drawn holds {cloud.explained:.1%} of the cloud's "
                    "energy - so the picture is a fair flat view of a genuinely two-armed "
                    "cone. The angular histogram under the scatter is the same shape counted "
                    "rather than drawn, and what it shows is how this tile's pixels "
                    "distribute *between* the arms: one broad hump means most pixels are "
                    "mixtures of both stains, a mode at each end means the tile holds regions "
                    "of each stain nearly alone. Either way the arms are the edges, and the "
                    "edges are what step 6 needs."
                )
            else:
                notes.append(
                    f"The arms are only {cloud.separation:.1f} degrees apart, under the "
                    f"{od.TWO_ARM_SEPARATION:g} this step calls two-armed. What is on screen "
                    "is the two tails of a *single* lobe, which is what a tile carrying only "
                    "a counterstain looks like - the maths did not fail, the tile has one "
                    "stain in it. Try another tile before reading anything into the arm "
                    "directions."
                )

            if cloud.two_armed and cloud.separation > 1.4 * cloud.reference_separation:
                notes.append(
                    f"The wedge is wide - {cloud.separation:.0f} degrees against "
                    f"{cloud.reference_separation:.0f} for the published pair - and that is "
                    "worth reading carefully rather than as a better result. Three things "
                    "widen a wedge: a genuine third absorber in the tile, the 1st and 99th "
                    "percentile extremes reaching into pixels that are mostly noise, and DAB "
                    "itself, whose colour shifts with concentration because it is a "
                    "scattering precipitate rather than a clean chromophore. Only the last of "
                    "those is benign, and none of them changes what step 6 does - it measures "
                    "with the published vectors either way."
                )

            if cloud.explained < 0.98:
                notes.append(
                    f"The plotted plane holds {cloud.explained:.1%} of the cloud's energy, "
                    "so a noticeable part of it points out of the page. Two absorbers make a "
                    "planar cloud; three or more do not. Read the reference vectors' "
                    "out-of-plane figures before concluding that one lies on an arm - a "
                    "shadow can land anywhere."
                )

        # --- Beer-Lambert, measured ----------------------------------------
        additivity = result.additivity
        if additivity is None:
            notes.append(
                "The additivity check did not run: the dominant arm has too few pixels to "
                "split into concentration bands. It is the one test on this screen that "
                "needs a range of stain intensities in one place, and this tile does not "
                "have one."
            )
        else:
            # The verdict turns on the *ratio*, not on the density drift being near
            # zero. Both numbers are measured on the same pixels in the same bins, so
            # the confound they share - darker pixels in a section are also purer -
            # inflates both and cancels in the comparison. Requiring the density
            # drift to be near zero in absolute terms would report that shared
            # confound as a failure of Beer-Lambert, which it is not.
            convincing = additivity.intensity_drift >= DRIFT_RATIO * max(
                additivity.od_drift, 0.05
            )
            clean = additivity.od_drift <= LINEAR_DRIFT
            notes.append(
                f"Beer-Lambert, tested on this tile rather than quoted. Taking the "
                f"{_stain(additivity.arm)} arm's pixels and sorting them from faintest to "
                f"darkest, "
                f"the *direction* of the density vector turns "
                f"{additivity.od_drift:.2f} degrees from the faintest band to the darkest, "
                f"while the direction of the light removed - I0 minus I, the intensity-space "
                f"equivalent - turns {additivity.intensity_drift:.2f} degrees over the same "
                f"pixels. "
                + (
                    f"That factor of "
                    f"{additivity.intensity_drift / max(additivity.od_drift, 1e-6):.1f} is the "
                    "whole argument for the logarithm: one stain holds its direction in "
                    "density space as its concentration rises, so a fixed matrix can un-mix "
                    "it, and loses it in intensity space, so no matrix can."
                    if convincing
                    else "The gap is small on this tile, and the honest reading is that there "
                    "is little curvature here to find: I = I0 x 10^(-cv) is very nearly "
                    "linear for small c, so a faint tile genuinely cannot show the effect. "
                    "Try a tile with more stain in it."
                )
                + (
                    ""
                    if clean
                    else " The density drift is not zero either, and it is worth knowing why: "
                    "the pixels were chosen by their direction in intensity space, on purpose "
                    "- choosing them by density direction would have capped the density drift "
                    "and made this test prove itself - and darker pixels in a section also "
                    "tend to be purer, so a little of both drifts is composition rather than "
                    "physics. That confound is identical in both spaces, measured on the same "
                    "pixels in the same bins, so it cancels in the ratio above and only the "
                    "two absolute figures carry it."
                )
            )

        # --- where it is not a measurement ---------------------------------
        if limits.floor_share > 0:
            notes.append(
                f"{limits.floor_share:.2%} of this tile hit the intensity floor: a channel "
                f"recorded no light at all, so its density is a *lower bound* rather than a "
                "reading - the true value is larger and unknowable. Without the floor those "
                "pixels would be infinities, and an infinity propagates through every "
                "average downstream. They are marked in red on the limits panel."
            )

        if limits.negative_share > NEGATIVE_SHARE_LIMIT:
            notes.append(
                f"{limits.negative_share:.1%} of this tile is *brighter* than I0 and so has "
                f"negative density, down to {limits.negative_worst:.3f}. Negative stain does "
                "not exist, so this is a statement about I0: it is slightly low for this "
                "field of view. Some is expected - step 4 measured I0 at "
                f"{run.screening_mpp:g} um/px and this tile is at {tile.mpp:g}, and a "
                "percentile of averaged pixels sits lower than the same percentile of the "
                "pixels that were averaged - but a large share means the glass step 4 "
                "sampled is not representative of the light reaching here. Not clamped, "
                "because clamping would make an impossible reading look like a faint one."
            )
        elif limits.negative_share > 0:
            notes.append(
                f"{limits.negative_share:.2%} of this tile is brighter than I0 - negative "
                "density, which cannot happen physically. At this share it is the expected "
                "residue of measuring I0 on a coarser grid than this tile is read at, plus "
                "whatever gland lumen or tear the tile contains. It is reported rather than "
                "clamped, and marked in amber on the limits panel."
            )

        if cloud is not None:
            notes.append(
                f"Only {cloud.admitted_share:.0%} of the tile reached the point cloud, and "
                f"the two ways a pixel was turned away are opposite ends of one problem. "
                f"{cloud.faint_share:.1%} carried less than beta = {limits.beta:g} mean OD: "
                "too little stain to have a direction at all, since dividing a near-zero "
                "vector by its own near-zero length amplifies the last bit of quantisation "
                f"into an angle. {cloud.unstable_share:.1%} were the other case - so dark "
                "that a channel had almost nothing left, where one 8-bit level moves that "
                f"channel's density by 1/(I ln10), which is 0.43 at an intensity of 1. Both "
                "populations sit at the angular *extremes*, which is exactly where the arms "
                "are read from, so admitting either would hand the two most important "
                f"numbers on this screen to the least reliable pixels in the tile. The "
                f"tolerance is {cloud.tolerance_deg:g} degrees of direction, and it is what "
                "8-bit data supports rather than a preference."
            )
        else:
            notes.append(
                f"{limits.transparent_share:.1%} of the tile sits below beta = "
                f"{limits.beta:g} mean OD, which is too little stain for a direction to mean "
                "anything - dividing a near-zero vector by its own near-zero length "
                "amplifies the last bit of quantisation into an angle."
            )

        # --- what upstream did or did not do -------------------------------
        notes.append(
            f"Everything here descends from step 3's cut at {white.tissue_threshold} by the "
            f"{white.tissue_threshold_source} rule: it set the glass step 4 measured I0 "
            "from, and I0 is the denominator of every number on this screen. It also set "
            "which blocks were eligible to be this tile. Step 3's slider is upstream of all "
            "of it."
        )

        if not white.qc_gated:
            notes.append(
                "Step 2 has not run, so no artefacts were excluded - not from the glass I0 "
                "was measured on, and not from the blocks this tile was chosen among. A "
                "fold or a pen mark is dark, and dark reads as stain here. Run quality "
                "control and open this step again."
            )

        return notes


@dataclass(frozen=True)
class DensityHandoff:
    """Everything step 6 needs from step 5, and nothing step 5 only needs to draw.

    Public where `_Run` is private, because this is the *contract* between two
    steps rather than one step's working state. Anything on `_Run` that exists to
    render a panel - the thumbnail, the angular histogram, the plotted point cloud -
    is deliberately absent: step 6 consumes the density, it does not redraw step 5.

    `white_field` is always an HxWx3 array, even when step 4 justified one flat
    triple, so a consumer never branches on which. `admitted` and `stained` are the
    two populations step 5 already decided about, carried across rather than
    re-derived, because a step that disagreed with the one above it about which
    pixels carry stain would be un-mixing a different tile than the one whose arms
    the reader was just shown.
    """

    #: The tile in optical density, HxWx3 float32 - step 5's own array.
    od: np.ndarray
    #: The tile as it was read, HxWx3 uint8.
    rgb: np.ndarray
    #: I0 evaluated over the tile, HxWx3. May be a broadcast view of one triple.
    white_field: np.ndarray

    #: Flat boolean over the tile: pixels with a direction worth estimating from -
    #: enough stain to have one, enough light for it to survive quantisation.
    admitted: np.ndarray
    #: HxW boolean: pixels carrying at least beta of stain. The tile's own "tissue",
    #: which is what colour statistics and stain measurements are taken over.
    stained: np.ndarray

    #: Step 5's two arms as (nearest published stain, unit vector), or empty when
    #: the tile had no cloud. Carried so step 6 can show its matrix agrees with the
    #: arms the reader has just been looking at.
    arms: tuple[tuple[str, np.ndarray], ...]
    cloud_explained: float | None

    tile: Tile
    chosen: Candidate
    candidates: list[Candidate]
    white: WhiteReference

    floor: float
    beta: float
    alpha: float
    tolerance_deg: float
    tile_um: float
    screening_mpp: float
    requested: bool
    #: Identity of this hand-off, for a caller memoising on it.
    key: str


class _Screen:
    """The screening pass: everything that is true before a tile is chosen.

    Held separately from `_Run` because it has a different lifetime. A run is one
    tile; a screening is the whole candidate list, and every tile in that list -
    plus the map and the contact sheet - is a view of this one object. Keeping the
    thumbnail here is what lets a candidate's thumbnail cost a crop instead of a
    read off the pyramid.
    """

    __slots__ = (
        "candidates",
        "footprint",
        "screening_mpp",
        "thumbnail",
        "tile_um",
        "white",
    )

    def __init__(
        self,
        *,
        footprint: TissueFootprint,
        white: WhiteReference,
        thumbnail: np.ndarray,
        screening_mpp: float,
        candidates: list[Candidate],
        tile_um: float,
    ) -> None:
        self.footprint = footprint
        self.white = white
        self.thumbnail = thumbnail
        self.screening_mpp = screening_mpp
        self.candidates = candidates
        self.tile_um = tile_um


class _Run:
    """One complete run of step 5, memoised as a unit.

    A plain class rather than a dataclass because it holds arrays and exists only
    to keep the report, the five panels and the memo looking at the same numbers -
    if any of them recomputed, the picture on screen and the figure beside it could
    disagree.
    """

    __slots__ = (
        "candidates",
        "chosen",
        "density",
        "requested",
        "screening_mpp",
        "thumbnail",
        "tile",
        "tile_um",
        "white",
    )

    def __init__(
        self,
        *,
        density: od.Density,
        tile: Tile,
        chosen: Candidate,
        candidates: list[Candidate],
        white: WhiteReference,
        thumbnail: np.ndarray,
        screening_mpp: float,
        requested: bool,
        tile_um: float,
    ) -> None:
        self.density = density
        self.tile = tile
        self.chosen = chosen
        self.candidates = candidates
        self.white = white
        self.thumbnail = thumbnail
        self.screening_mpp = screening_mpp
        self.requested = requested
        self.tile_um = tile_um


density_service = DensityService()

__all__ = ["DensityError", "DensityHandoff", "TileError", "density_service"]
