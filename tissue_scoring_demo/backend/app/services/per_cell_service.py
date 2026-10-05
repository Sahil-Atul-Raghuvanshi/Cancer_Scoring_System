"""Step 14 - measure the DAB in every tumour cell's own compartment.

    report.json                  the summary, the histogram and the scatter sample
    region{r}/cells.json         every cell's row, which steps 15 and 16 read

**This is the step Rule 2 exists to protect.** The DAB channel measured here is
the un-normalised, calibrated one: step 4's white point, step 6's deconvolution,
nothing in between that rewrites a pixel. The haematoxylin arm of that fork went
to the models - step 8's classifier, step 11's segmenter - and never comes back.
So a cell's intensity here is an optical density on a scale that means the same
thing on the next slide, which is the entire reason an absolute cut point is
possible at step 15.

**It re-reads the fields rather than carrying pixels forward, and reads them at
step 11's own scale.** The label maps step 11 stored are rasters at the model's
0.5 um/px, and a DAB channel read at any other resolution would not line up with
them pixel for pixel - it would measure a cell's stain slightly beside that
cell. So the mpp comes out of the stored `.npz` and is handed to `read_tile` as
the target, which makes the alignment exact rather than approximately right.

**Only tumour cells, and the same tumour cells step 13 used.** Step 12 decided;
this step asks rather than re-deriving, because two definitions of the
denominator is how a percentage ends up with a numerator that is not a subset of
it.

Request-shaped rather than job-shaped: 36 tiles of 512 px and two distance
transforms per field is tens of seconds, and everything expensive - the
segmentation, the registration - already happened upstream.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app import panel
from app.core.config import settings
from app.core.logging import get_logger
from app.ingestion.slide_reader import open_slide
from app.pipeline.step05_optical_density.tiles import read_tile
from app.pipeline.step13_nuclei_segmentation.stain_input import (
    DAB,
    concentrations,
    ruifrok_basis,
)
from app.pipeline.step14_cell_typing import types_map
from app.pipeline.step15_compartments import geometry
from app.pipeline.step16_per_cell_measurement.measure import (
    INTENSITY_STATISTIC,
    RING_BINS,
    CellMeasurement,
    measure_field,
)
from app.schemas.per_cell import (
    CellPoint,
    FieldMeasurement,
    MeasurementParams,
    ODHistogramBin,
    PerCellReport,
    RegionMeasurement,
)
from app.scoring import cuts as cut_points
from app.services.cell_typing_service import cell_typing_service
from app.services.compartment_service import compartment_service
from app.services.nuclei_service import NucleiError, nuclei_service
from app.services.upload_service import resolve_ready_path

logger = get_logger(__name__)

#: Dots drawn on the scatter. A hundred thousand is not a picture, and the real
#: count travels beside the sample so nobody reads the sample as the population.
MAX_SCATTER_POINTS = 4000

#: Bars in the optical-density histogram the cut lines are drawn on.
HISTOGRAM_BINS = 60
HISTOGRAM_MAX_OD = 1.5

#: A membrane cell with fewer occupied bins than this has had most of its
#: circumference taken by neighbours, so its completeness is a fact about the
#: packing rather than about the stain. Counted and reported, never dropped.
CROWDED_BIN_FLOOR = 12


def _now() -> str:
    return datetime.now(UTC).isoformat()


class PerCellError(ValueError):
    """A client-correctable problem: an earlier step has not run for this pair."""


class PerCellService:
    """Measures one (H&E, IHC) pair's tumour cells, once per cell."""

    # --- storage ------------------------------------------------------------

    def key(self, he_upload_id: str, ihc_upload_id: str) -> str:
        return f"{he_upload_id}__{ihc_upload_id}"

    def artifact(self, he_upload_id: str, ihc_upload_id: str, *parts: str) -> Path:
        return settings.per_cell_dir.joinpath(self.key(he_upload_id, ihc_upload_id), *parts)

    def _write(self, he_upload_id: str, ihc_upload_id: str, name: str, payload: str) -> None:
        path = self.artifact(he_upload_id, ihc_upload_id, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")

    # --- reading what steps 11 and 12 stored --------------------------------

    def _field_labels(
        self, he_upload_id: str, ihc_upload_id: str, rank: int, index: int
    ) -> tuple[np.ndarray, float, int, int, int] | None:
        path = nuclei_service.artifact(
            he_upload_id, ihc_upload_id, f"region{rank}", f"f{index}_labels.npz"
        )
        if not path.is_file():
            return None
        with np.load(path) as stored:
            return (
                stored["labels"].astype(np.int32),
                float(stored["mpp"]),
                int(stored["x"]),
                int(stored["y"]),
                int(stored["span"]),
            )

    def _tumour_ids(
        self, he_upload_id: str, ihc_upload_id: str, rank: int, field_index: int
    ) -> set[int]:
        """Which nuclei in ONE field step 12 called tumour.

        Per field, because a nucleus id only identifies a nucleus within the
        field it was segmented in - and because step 13 restricts the same way.
        The two have to agree about which cells these are: step 13 builds the
        compartment and this step measures it, so a disagreement would measure
        one set of cells in another set's geometry.
        """
        path = cell_typing_service.artifact(
            he_upload_id, ihc_upload_id, f"region{rank}", "types.json"
        )
        try:
            return types_map.tumour_ids(path, field_index)
        except types_map.TypesUnavailableError as exc:
            raise PerCellError(
                f"{exc} Only tumour cells are measured - measuring a lymphocyte's "
                "membrane would put a cell in the numerator that is not in the "
                "denominator."
            ) from exc

    def _stored_geometry(
        self,
        he_upload_id: str,
        ihc_upload_id: str,
        rank: int,
        index: int,
        *,
        width_um: float,
        is_membrane: bool,
        shape: tuple[int, int],
    ) -> tuple[geometry.Compartments | None, str | None]:
        """Step 13's compartments for one field, or None and the reason why.

        **Reading them is what makes step 13 an input rather than an illustration.**
        Until this existed, step 14 imported `geometry` and rebuilt the shapes from
        step 11's labels - so the screen a person approved and the geometry a score
        was measured in were two separate computations that merely happened to agree.

        **Refusing a mismatched width is the other half, and it is not pedantry.**
        Step 13's expansion is adjustable on screen; step 14's is fixed to the
        antibody's own figure in `app.panel`, because a ring width is a property of
        what CD44 does, not of what a viewer dragged a slider to. Simply loading
        whatever was on disk would mean an exploration left open on step 13 silently
        moved the deliverable, and the resulting percentage would look completely
        ordinary. So a stored geometry is used only when it was built at this
        marker's own width, for tumour cells, in the shape this field actually is;
        anything else is rebuilt here and the disagreement is reported.
        """
        path = compartment_service.artifact(
            he_upload_id, ihc_upload_id, f"region{rank}", f"f{index}_geometry.npz"
        )
        if not path.is_file():
            return None, None

        try:
            with np.load(path) as stored:
                stored_width = float(stored["width_um"])
                if abs(stored_width - width_um) > 1e-6:
                    return None, (
                        f"step 13 last drew region {rank} field {index} at "
                        f"{stored_width:g} um, not this marker's {width_um:g} um"
                    )
                if bool(stored["is_membrane"]) != is_membrane:
                    return None, (
                        f"step 13 last drew region {rank} field {index} as the other "
                        "compartment kind"
                    )
                if not bool(stored["tumour_only"]):
                    return None, (
                        f"step 13 last drew region {rank} field {index} over every "
                        "cell, not tumour cells only"
                    )
                nucleus = stored["nucleus"].astype(np.int32)
                if nucleus.shape != shape:
                    return None, (
                        f"step 13's geometry for region {rank} field {index} is "
                        f"{nucleus.shape}, against a label map of {shape}"
                    )
                return (
                    geometry.Compartments(
                        nucleus=nucleus,
                        cell=stored["cell"].astype(np.int32),
                        measured=stored["measured"].astype(np.int32),
                        expansion_um=float(stored["expansion_um"]),
                    ),
                    None,
                )
        except (OSError, ValueError, KeyError) as exc:
            # A truncated or older npz is a rebuild, not a failure: the arrays are a
            # cache of something this step can still derive.
            return None, f"step 13's stored geometry could not be read ({exc})"

    @staticmethod
    def _restrict(labels: np.ndarray, keep: set[int]) -> np.ndarray:
        """Blank every cell that is not tumour, before the expansion.

        Before rather than after, for the reason step 13 gives: a lymphocyte
        removed afterwards has already taken the pixels between it and the
        tumour cell beside it, so that tumour cell's ring is dented by a cell
        nobody is measuring.
        """
        if not keep:
            return np.zeros_like(labels)
        return np.where(np.isin(labels, list(keep)), labels, 0).astype(np.int32)

    # --- the work -----------------------------------------------------------

    def rows(self, he_upload_id: str, ihc_upload_id: str) -> list[dict]:
        """Every measured cell, as steps 15 and 16 read them back.

        Stored rather than recomputed, so the two later steps cannot disagree
        with this one about what was measured - and so neither of them ever
        needs to open a slide.
        """
        directory = self.artifact(he_upload_id, ihc_upload_id)
        if not directory.is_dir():
            raise PerCellError(
                "step 14 has not measured this pair yet, so there are no cells to score."
            )
        out: list[dict] = []
        for path in sorted(directory.glob("region*/cells.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            out.extend(payload.get("cells", []))
        if not out:
            raise PerCellError(
                "step 14 measured no cells for this pair. Either step 12 called none of "
                "them tumour, or step 11 found none inside the carried regions."
            )
        return out

    def report(self, he_upload_id: str, ihc_upload_id: str) -> PerCellReport:
        """Measure every tumour cell, and say what the measurement was made with."""
        try:
            nuclei_report = nuclei_service.report(he_upload_id, ihc_upload_id)
        except NucleiError as exc:
            raise PerCellError(
                "step 11 has found no nuclei for this pair, and a cell is measured in a "
                "compartment grown from its nucleus."
            ) from exc

        letter = nuclei_report.marker
        if not letter:
            raise PerCellError(
                "this pair records no antibody letter. Which part of the cell to measure, "
                "which second number to compute and which cut points to apply all resolve "
                "from it, so without it there is nothing defensible to measure."
            )

        spec = panel.spec(letter)
        if spec.compartment == panel.Compartment.NONE:
            raise PerCellError(f"{spec.full_name} is not a scored marker.")

        marker_cuts = cut_points.for_marker(letter)
        is_membrane = spec.compartment == panel.Compartment.MEMBRANE
        width = spec.compartment_width_um
        # Read from the panel, exactly as step 13 reads it. Step 14 measures the
        # geometry step 13 stored, so the only reason this is here at all is the
        # fallback rebuild below - and a second opinion about the shell would be a
        # second compartment.
        shell = spec.membrane_shell_um if is_membrane else None

        ihc_path = resolve_ready_path(upload_id=ihc_upload_id)
        reader = open_slide(ihc_path)
        try:
            base_mpp = reader.mpp
            if not base_mpp:
                raise PerCellError(
                    "this IHC slide records no microns-per-pixel, so a compartment "
                    "cannot be sized in microns."
                )

            # The same white point step 11 used, and the same fixed basis. The DAB
            # arm of a per-slide Macenko estimate is recovered well, but its
            # haematoxylin arm is not (see `stain_input`), and a basis that differed
            # between the detection and the measurement would be two different
            # un-mixings of one slide.
            #
            # That paragraph used to be an assertion with nothing behind it. This
            # step's basis is fixed here, while step 11's is a *setting* -
            # `nuclei_macenko_per_slide` - so turning that setting on made detection
            # and measurement disagree about what this slide's haematoxylin is, and
            # nothing said so: the cells would be found on one un-mixing and scored
            # on another, and the percentage that came out would look entirely
            # ordinary. Checking costs one string comparison, so the comment above
            # is now enforced rather than believed.
            from app.services.calibration_service import calibration_service

            white = calibration_service.white_point(ihc_upload_id)
            basis = ruifrok_basis()

            if nuclei_report.stain.basis != basis.source:
                raise PerCellError(
                    f"step 11 detected these nuclei on a '{nuclei_report.stain.basis}' "
                    f"basis, and this step measures on '{basis.source}'. Those are two "
                    "different un-mixings of one slide: the cells were found under one "
                    "and their stain would be measured under the other. Measurement is "
                    "fixed to the published vectors on purpose - a per-slide estimate "
                    "rescales itself to whatever slide it is given, so '0.4 DAB' would "
                    "mean a different amount of dye on every slide in a study. Re-run "
                    "step 11 with `nuclei_macenko_per_slide` off."
                )

            regions: list[RegionMeasurement] = []
            pooled: list[CellMeasurement] = []
            #: Fields where step 13's stored geometry was present but not usable, and
            #: fields where it was simply absent. Kept apart: the first is a
            #: disagreement a reader should see, the second only means step 13 has
            #: not been opened for this pair yet.
            rebuild_reasons: list[str] = []
            missing_geometry = 0

            for region in nuclei_report.regions:
                fields: list[FieldMeasurement] = []
                region_cells: list[CellMeasurement] = []

                for field in region.fields:
                    loaded = self._field_labels(
                        he_upload_id, ihc_upload_id, region.rank, field.index
                    )
                    if loaded is None:
                        continue
                    labels, mpp, x, y, span = loaded
                    labels = self._restrict(
                        labels,
                        self._tumour_ids(
                            he_upload_id, ihc_upload_id, region.rank, field.index
                        ),
                    )
                    if not labels.any():
                        continue

                    # Step 13's geometry when it is the geometry this marker calls
                    # for, and a rebuild when it is not. `_stored_geometry` says which.
                    built, rebuild_reason = self._stored_geometry(
                        he_upload_id,
                        ihc_upload_id,
                        region.rank,
                        field.index,
                        width_um=width,
                        is_membrane=is_membrane,
                        shape=(labels.shape[0], labels.shape[1]),
                    )
                    if built is None:
                        if rebuild_reason is not None:
                            rebuild_reasons.append(rebuild_reason)
                        else:
                            missing_geometry += 1
                        built = geometry.build(
                            labels,
                            mpp=mpp,
                            expansion_um=width,
                            ring=is_membrane,
                            ring_um=shell,
                        )

                    tile = read_tile(
                        reader,
                        x=x,
                        y=y,
                        target_mpp=mpp,
                        size=labels.shape[0],
                        base_mpp=base_mpp,
                    )
                    rgb = np.asarray(tile.rgb, dtype=np.uint8)
                    if rgb.shape[:2] != labels.shape:
                        raise PerCellError(
                            f"region {region.rank} field {field.index}: the DAB tile came "
                            f"back {rgb.shape[:2]} against a label map of {labels.shape}. "
                            "Measuring one on the other would read each cell's stain from "
                            "slightly beside that cell."
                        )

                    white_field = (
                        white.field_for(x=x, y=y, size=labels.shape[0], mpp=mpp)
                        if hasattr(white, "field_for")
                        else white
                    )
                    dab = concentrations(rgb, white_field, basis)[..., DAB]

                    measured = measure_field(
                        built.nucleus,
                        built.measured,
                        dab,
                        membrane=is_membrane,
                        positivity_od=marker_cuts.positivity_od,
                        mpp=mpp,
                        region_rank=region.rank,
                        field_index=field.index,
                        x0=float(x),
                        y0=float(y),
                        level0_scale=span / max(1, labels.shape[0]),
                    )
                    if not measured:
                        continue

                    region_cells.extend(measured)
                    fields.append(
                        FieldMeasurement(
                            region_rank=region.rank,
                            index=field.index,
                            cells=len(measured),
                            mean_od=round(
                                float(np.mean([c.intensity_od for c in measured])), 4
                            ),
                            mean_second=round(
                                float(np.mean([c.second for c in measured])), 4
                            ),
                        )
                    )

                self._store_region(he_upload_id, ihc_upload_id, region.rank, region_cells)
                pooled.extend(region_cells)

                od = np.array([c.intensity_od for c in region_cells], dtype=np.float64)
                second = np.array([c.second for c in region_cells], dtype=np.float64)
                regions.append(
                    RegionMeasurement(
                        rank=region.rank,
                        area_mm2=region.area_mm2,
                        cells=len(region_cells),
                        mean_od=round(float(od.mean()), 4) if od.size else 0.0,
                        median_od=round(float(np.median(od)), 4) if od.size else 0.0,
                        mean_second=round(float(second.mean()), 4) if second.size else 0.0,
                        fields=fields,
                    )
                )
        finally:
            reader.close()

        if not pooled:
            raise PerCellError(
                "no tumour cell in any carried region produced a compartment with "
                "pixels in it. Nothing can be measured, so nothing is reported."
            )

        od = np.array([c.intensity_od for c in pooled], dtype=np.float64)
        second = np.array([c.second for c in pooled], dtype=np.float64)
        crowded = (
            sum(
                1
                for c in pooled
                if c.occupied_bins is not None and c.occupied_bins < CROWDED_BIN_FLOOR
            )
            if is_membrane
            else None
        )

        report = PerCellReport(
            he_upload_id=he_upload_id,
            ihc_upload_id=ihc_upload_id,
            marker=letter,
            marker_name=spec.full_name,
            second_measure="ring_completeness" if is_membrane else "stained_fraction",
            generated_at=_now(),
            nuclei_generated_at=nuclei_report.generated_at,
            params=MeasurementParams(
                intensity_statistic=INTENSITY_STATISTIC,
                compartment="membrane" if is_membrane else "cytoplasm",
                expansion_um=width,
                ring_um=shell,
                positivity_od=marker_cuts.positivity_od,
                ring_bins=RING_BINS if is_membrane else None,
                second_min=marker_cuts.second_min,
                cuts_version=cut_points.cut_set().version,
                cuts_provisional=marker_cuts.provisional,
            ),
            regions=regions,
            cells=len(pooled),
            mean_od=round(float(od.mean()), 4),
            median_od=round(float(np.median(od)), 4),
            mean_second=round(float(second.mean()), 4),
            histogram=self._histogram(od),
            points=self._sample(pooled),
            sampled=min(len(pooled), MAX_SCATTER_POINTS),
            crowded_cells=crowded,
            notes=self._notes(
                spec,
                marker_cuts,
                is_membrane,
                crowded,
                len(pooled),
                rebuild_reasons,
                missing_geometry,
            ),
        )

        self._write(
            he_upload_id,
            ihc_upload_id,
            "report.json",
            json.dumps(report.model_dump(by_alias=True), indent=2),
        )
        return report

    # --- storing ------------------------------------------------------------

    def _store_region(
        self,
        he_upload_id: str,
        ihc_upload_id: str,
        rank: int,
        cells: list[CellMeasurement],
    ) -> None:
        """One region's rows, kept out of the report on purpose.

        The report is fetched every time the screen opens; the rows are what
        steps 15 and 16 read, once. Splitting them keeps the summary small and
        makes the two later steps' inputs a file on disk rather than a payload
        that had to survive a round trip.
        """
        payload = {
            "rank": rank,
            "cells": [
                {
                    "cellId": cell.cell_id,
                    "regionRank": cell.region_rank,
                    "fieldIndex": cell.field_index,
                    "x": round(cell.x, 1),
                    "y": round(cell.y, 1),
                    "intensityOd": round(cell.intensity_od, 5),
                    "maxOd": round(cell.max_od, 5),
                    "second": round(cell.second, 5),
                    "secondMeasure": cell.second_measure,
                    "pixels": cell.pixels,
                    "areaUm2": round(cell.area_um2, 3),
                    "occupiedBins": cell.occupied_bins,
                }
                for cell in cells
            ],
        }
        self._write(
            he_upload_id, ihc_upload_id, f"region{rank}/cells.json", json.dumps(payload)
        )

    # --- presentation -------------------------------------------------------

    @staticmethod
    def _histogram(od: np.ndarray) -> list[ODHistogramBin]:
        counts, edges = np.histogram(
            np.clip(od, 0.0, HISTOGRAM_MAX_OD),
            bins=HISTOGRAM_BINS,
            range=(0.0, HISTOGRAM_MAX_OD),
        )
        return [
            ODHistogramBin(
                lower=round(float(edges[index]), 4),
                upper=round(float(edges[index + 1]), 4),
                count=int(value),
            )
            for index, value in enumerate(counts)
        ]

    @staticmethod
    def _sample(cells: list[CellMeasurement]) -> list[CellPoint]:
        """Evenly spaced through the list, not the first N.

        The first N would all come from region 1's first fields, so the scatter
        would show one corner of one region and look like the slide.
        """
        if len(cells) <= MAX_SCATTER_POINTS:
            chosen = cells
        else:
            step = len(cells) / MAX_SCATTER_POINTS
            chosen = [cells[int(index * step)] for index in range(MAX_SCATTER_POINTS)]

        return [
            CellPoint(
                cell_id=cell.cell_id,
                region_rank=cell.region_rank,
                field_index=cell.field_index,
                x=round(cell.x, 1),
                y=round(cell.y, 1),
                intensity_od=round(cell.intensity_od, 4),
                max_od=round(cell.max_od, 4),
                second=round(cell.second, 4),
                second_measure=cell.second_measure,  # type: ignore[arg-type]
                pixels=cell.pixels,
                area_um2=round(cell.area_um2, 2),
                occupied_bins=cell.occupied_bins,
            )
            for cell in chosen
        ]

    @staticmethod
    def _notes(
        spec,
        marker_cuts,
        is_membrane: bool,
        crowded: int | None,
        total: int,
        rebuild_reasons: list[str] | None = None,
        missing_geometry: int = 0,
    ) -> list[str]:
        second = (
            "how complete the ring is, over 36 bins of 10 degrees"
            if is_membrane
            else "what share of the band's pixels are brown"
        )
        notes = [
            f"Every cell gets one number always - the mean DAB optical density in its "
            f"{spec.compartment.value} - and a second one: {second}. Mean rather than "
            "maximum, for every marker, so two markers' intensities can be compared; the "
            "maximum is reported per cell but never enters a score.",
            "The DAB read here is the calibrated, un-normalised channel: step 4's white "
            "point and step 6's deconvolution, with nothing in between that rewrites a "
            "pixel. That is what makes an absolute cut point possible at step 15.",
        ]
        if rebuild_reasons:
            shown = "; ".join(sorted(set(rebuild_reasons))[:3])
            notes.append(
                f"MEASURED IN ITS OWN GEOMETRY, NOT STEP 13'S, for "
                f"{len(rebuild_reasons)} field(s): {shown}. Step 13's expansion can be "
                f"moved on screen and this step's cannot - a ring width is a property "
                f"of {spec.name}, not of where a slider was left - so a geometry built "
                "at another setting is rebuilt here rather than measured. What you see "
                "on step 13 is that exploration; these numbers are the marker's own "
                "width."
            )
        elif missing_geometry:
            notes.append(
                f"Step 13 has not drawn {missing_geometry} of these fields yet, so their "
                "compartments were built here from step 11's nuclei by the same "
                "function step 13 uses. Open step 13 and they will be read from it "
                "instead - the shapes are identical either way, but one of them is a "
                "step's output and the other is a re-derivation."
            )
        if not is_membrane:
            notes.append(
                f"{spec.full_name} is cytoplasmic, so ring completeness is not computed "
                "for it - not computed and discarded, not computed at all. Cytoplasmic "
                "staining has no circumference, and a near-zero completeness fed into a "
                "positivity rule would make a cell negative for a geometry it never had."
            )
        if crowded:
            notes.append(
                f"{crowded} of {total} cells ({crowded / max(1, total):.0%}) had fewer than "
                f"{CROWDED_BIN_FLOOR} of 36 bins with enough pixels to judge - their rings "
                "were taken by neighbouring cells. They cannot show complete staining "
                "however strongly they are stained, so their completeness is a fact about "
                "how tightly the tissue is packed."
            )
        if marker_cuts.provisional:
            notes.append(
                "The positivity cut these second numbers are measured against is "
                "provisional: it has not been fitted against pathologist scores, because "
                "the reader sheet is not on disk. It comes from the optical-density scale "
                "itself, and every number computed under it is labelled accordingly."
            )
        return notes


per_cell_service = PerCellService()
