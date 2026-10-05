"""Step 13 - grow each cell outward into the region the marker lives in.

    report.json                  the geometry, and its sensitivity to the width
    region{r}/f{i}_compartments.png   nucleus / cell / measured band, drawn

**This is the first step that reads the antibody letter**, and the fork is
`app.panel`'s: A, F and R are membrane markers and get a ring; U and W are
cytoplasmic and get a band. The width comes from the same place. Nothing here
decides which a marker gets, and the request cannot override it - putting a
cytoplasmic marker through the membrane path is the specific regression the
guide warns about, so it is not reachable from the outside at all.

**Only tumour cells get compartments.** Step 12 decided which cells those are,
and measuring a lymphocyte's membrane would put a cell in the numerator that is
not in the denominator. The two steps have to agree about that, so this one asks
step 12 rather than re-deriving it.

Request-shaped like step 12: it reads the label maps step 11 stored and runs two
distance transforms per field, which is fast enough to answer while a viewer
drags a width slider.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app import panel
from app.core.config import settings
from app.core.logging import get_logger
from app.pipeline.step14_cell_typing import types_map
from app.pipeline.step15_compartments import geometry, overlay, rings
from app.schemas.compartments import (
    CompartmentParams,
    CompartmentsReport,
    FieldCompartments,
    RegionCompartments,
    WidthSensitivityPoint,
)
from app.services.cell_typing_service import cell_typing_service
from app.services.nuclei_service import NucleiError, nuclei_service

logger = get_logger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _reference_widths() -> tuple[float, float]:
    """The two shipped compartment widths, read from the panel, not typed here.

    The screen draws both bands on every cell whatever antibody is loaded, so
    that a reader can see what the *other* fork would have measured. That needs
    the other fork's width, and the only place a width may come from is
    `app.panel` - a second copy here is how a screen ends up illustrating a 5 um
    ring the pipeline has never used.
    """
    membrane = next(
        spec.compartment_width_um
        for spec in panel.PANEL.values()
        if spec.compartment is panel.Compartment.MEMBRANE
    )
    cytoplasm = next(
        spec.compartment_width_um
        for spec in panel.PANEL.values()
        if spec.compartment is panel.Compartment.CYTOPLASM
    )
    return membrane, cytoplasm


class CompartmentError(ValueError):
    """A client-correctable problem: usually that step 11 or 12 has not run."""


class CompartmentService:
    """Builds the measurement geometry for one pair, forked by antibody."""

    def key(self, he_upload_id: str, ihc_upload_id: str) -> str:
        return f"{he_upload_id}__{ihc_upload_id}"

    def artifact(self, he_upload_id: str, ihc_upload_id: str, *parts: str) -> Path:
        return settings.compartments_dir.joinpath(
            self.key(he_upload_id, ihc_upload_id), *parts
        )

    def _write(self, he_upload_id: str, ihc_upload_id: str, name: str, payload) -> None:
        path = self.artifact(he_upload_id, ihc_upload_id, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, bytes):
            path.write_bytes(payload)
        else:
            path.write_text(payload, encoding="utf-8")

    def _write_geometry(
        self,
        he_upload_id: str,
        ihc_upload_id: str,
        rank: int,
        index: int,
        built: geometry.Compartments,
        *,
        width_um: float,
        is_membrane: bool,
        tumour_only: bool,
        shell_um: float | None,
        nuclei_generated_at: str | None,
        typing_stamp: str | None,
        field_x: int,
        field_y: int,
    ) -> None:
        """Persist one field's compartments so step 14 measures what this step drew.

        **This step used to leave nothing but a picture.** It wrote one PNG per
        region and a report, and step 14 - whose entire input is "the region of each
        cell where the marker is supposed to be" - imported `geometry` and built the
        shapes again from step 11's labels. Two builds of the same geometry from the
        same inputs agree only for as long as nobody touches either call site, and
        the whole point of the pipeline order is that a step's input is the previous
        step's *output*, not a re-derivation of it.

        The parameters are stored next to the arrays rather than implied by the
        folder. Step 13's width is adjustable on screen and step 14's is not, so the
        reader has to be able to tell whether the geometry on disk is the marker's
        own or an exploration - see `PerCellService._stored_geometry`, which refuses
        the second kind rather than quietly measuring it.

        **And what it was built from (P-15).** Width, kind and shape were all the reader
        could check, so a field whose nuclei had been re-segmented or whose cells had
        been re-typed since was measured with the old cells. The nuclei run, the typing,
        the field's position and the shell thickness are now stored beside the arrays,
        and the reader refuses a geometry that disagrees with any of them.
        """
        path = self.artifact(
            he_upload_id, ihc_upload_id, f"region{rank}", f"f{index}_geometry.npz"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            nucleus=built.nucleus.astype(np.int32),
            cell=built.cell.astype(np.int32),
            measured=built.measured.astype(np.int32),
            expansion_um=np.float32(built.expansion_um),
            width_um=np.float32(width_um),
            is_membrane=np.bool_(is_membrane),
            tumour_only=np.bool_(tumour_only),
            shell_um=np.float32(shell_um if shell_um is not None else -1.0),
            nuclei_generated_at=np.array(nuclei_generated_at or ""),
            typing_stamp=np.array(typing_stamp or ""),
            field_x=np.int64(field_x),
            field_y=np.int64(field_y),
        )

    # --- reading what step 11 stored ---------------------------------------

    def _field_labels(
        self, he_upload_id: str, ihc_upload_id: str, rank: int, index: int
    ) -> tuple[np.ndarray, float, int, int] | None:
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
            )

    @staticmethod
    def _restrict(labels: np.ndarray, keep: set[int]) -> np.ndarray:
        """Blank every nucleus that is not in `keep`.

        Done *before* the expansion, not after, and that is not a detail: a
        lymphocyte that is removed afterwards has already taken its share of the
        pixels between it and the tumour cell beside it, so the tumour cell's
        compartment would be dented by a cell nobody is measuring. Removing it
        first lets the tumour cell grow into that space, which is what would
        have happened if the lymphocyte had never been detected.
        """
        if not keep:
            return np.zeros_like(labels)
        mask = np.isin(labels, list(keep))
        return np.where(mask, labels, 0).astype(np.int32)

    def _types_path(self, he_upload_id: str, ihc_upload_id: str, rank: int) -> Path:
        return cell_typing_service.artifact(
            he_upload_id, ihc_upload_id, f"region{rank}", "types.json"
        )

    def _tumour_ids(
        self, he_upload_id: str, ihc_upload_id: str, rank: int, field_index: int
    ) -> set[int] | None:
        """Which nuclei in ONE field step 12 called tumour.

        Per field, not per region: a nucleus id only identifies a nucleus within
        the field it was segmented in. See `step14_cell_typing.types_map`.
        """
        try:
            return types_map.tumour_ids(
                self._types_path(he_upload_id, ihc_upload_id, rank), field_index
            )
        except types_map.TypesUnavailableError:
            return None

    # --- the work -----------------------------------------------------------

    def _cached(
        self,
        he_upload_id: str,
        ihc_upload_id: str,
        *,
        width_um: float,
        shell_um: float | None,
        is_membrane: bool,
        voronoi: bool,
        tumour_only: bool,
        nuclei_generated_at: str | None,
        expected_fields: list[tuple[int, int]],
    ) -> CompartmentsReport | None:
        """The stored report, when it describes exactly this request. Else None.

        **This step had no cache at all, and it is one of the more expensive ones.**
        Every request rebuilt the geometry for every field, and then `_sweep` built
        it six more times on one field - and `contested_px` inside that is one binary
        dilation per nucleus. A screen that fetches a report and a panel paid for all
        of it twice.

        The key is every input that can change the answer, and nothing else: the two
        request parameters, the fork resolved from the antibody, and step 11's
        stamp - which is what moves when the nuclei underneath are re-segmented. The
        geometry files are checked for existence as well as the JSON, because step 14
        reads those rather than the report, and a cache that satisfied the screen
        while leaving the measurement without its input would be worse than none.
        """
        path = self.artifact(he_upload_id, ihc_upload_id, "report.json")
        if not path.is_file():
            return None
        try:
            stored = CompartmentsReport.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return None

        if stored.nuclei_generated_at != nuclei_generated_at:
            return None
        if stored.tumour_only != tumour_only:
            return None
        if stored.params.voronoi_constrained != voronoi:
            return None
        if abs(stored.params.expansion_um - width_um) > 1e-6:
            return None
        expected_ring = shell_um
        if (stored.params.ring_um is None) != (expected_ring is None):
            return None
        if expected_ring is not None and abs((stored.params.ring_um or 0.0) - expected_ring) > 1e-6:
            return None

        for rank, index in expected_fields:
            geometry_path = self.artifact(
                he_upload_id, ihc_upload_id, f"region{rank}", f"f{index}_geometry.npz"
            )
            if not geometry_path.is_file():
                return None

        # The outlines the screen draws, checked for the same reason as the
        # arrays step 14 measures: a cache that satisfied one consumer and left
        # the other without its input is worse than no cache. This is also what
        # makes a run stored before the outlines existed rebuild itself once,
        # rather than serving a report the screen cannot draw.
        for rank in {rank for rank, _ in expected_fields}:
            if not self.artifact(
                he_upload_id, ihc_upload_id, f"region{rank}", "rings.json"
            ).is_file():
                return None

        return stored

    def report(
        self,
        he_upload_id: str,
        ihc_upload_id: str,
        *,
        width_um: float | None = None,
        voronoi: bool = True,
        tumour_only: bool = True,
    ) -> CompartmentsReport:
        try:
            nuclei_report = nuclei_service.report(he_upload_id, ihc_upload_id)
        except NucleiError as exc:
            raise CompartmentError(
                "step 11 has not found any nuclei for this pair, and a compartment is "
                "grown from a nucleus."
            ) from exc

        letter = nuclei_report.marker
        if not letter:
            raise CompartmentError(
                "this pair records no antibody letter, and the compartment a cell gets "
                "depends entirely on which antibody was used - a membrane ring for "
                "CD44, ABCC4 and ABCC11, a cytoplasm band for the two cadherins. "
                "Without it there is no defensible shape to build."
            )

        spec = panel.spec(letter)
        if spec.compartment == panel.Compartment.NONE:
            raise CompartmentError(
                f"{spec.full_name} is not a scored marker, so it has no compartment."
            )

        is_membrane = spec.compartment == panel.Compartment.MEMBRANE
        width = spec.compartment_width_um if width_um is None else float(width_um)

        # How thick the measured shell is, for a membrane marker. Scaled with the
        # width when the width is swept, so the shell stays the same share of the
        # cell body rather than swallowing it whole at 2 um and becoming a hair at
        # 10 um. `None` for a cytoplasmic marker, which measures the whole body.
        shell = (
            spec.membrane_shell_um * (width / spec.compartment_width_um)
            if is_membrane and spec.compartment_width_um > 0
            else None
        )

        # What the OTHER fork would have measured on the same cell, drawn dashed
        # so the choice this step makes is visible rather than asserted. It stays
        # at its shipped width while the slider moves this marker's: a reference
        # that moved with the thing it is a reference for would not be one.
        reference_membrane, reference_cytoplasm = _reference_widths()
        if is_membrane:
            alternate_body = reference_cytoplasm
            alternate_shell = 0.0
        else:
            alternate_body = reference_membrane
            alternate_shell = panel.MEMBRANE_SHELL_UM

        # `ring_um` is the thickness of the measured shell at the cell's outer
        # edge, for a membrane marker. None for a cytoplasmic one, which measures
        # the whole body and so has no shell.
        params = CompartmentParams(
            expansion_um=width,
            ring_um=shell,
            voronoi_constrained=voronoi,
        )

        cached = self._cached(
            he_upload_id,
            ihc_upload_id,
            width_um=width,
            shell_um=shell,
            is_membrane=is_membrane,
            voronoi=voronoi,
            tumour_only=tumour_only,
            nuclei_generated_at=nuclei_report.generated_at,
            # Only the fields step 11 actually stored labels for. The build loop
            # skips the rest, so demanding geometry for them would mean the cache
            # never hit on any pair with a field step 11 declined to segment.
            expected_fields=[
                (region.rank, field.index)
                for region in nuclei_report.regions
                for field in region.fields
                if nuclei_service.artifact(
                    he_upload_id,
                    ihc_upload_id,
                    f"region{region.rank}",
                    f"f{field.index}_labels.npz",
                ).is_file()
            ],
        )
        if cached is not None:
            return cached

        regions: list[RegionCompartments] = []
        pooled_measured: list[float] = []
        pooled_cell: list[float] = []
        pooled_nucleus: list[float] = []
        typed_total = 0

        for region in nuclei_report.regions:
            if tumour_only and not self._types_path(
                he_upload_id, ihc_upload_id, region.rank
            ).is_file():
                raise CompartmentError(
                    "step 12 has not sorted the cells for this pair. Compartments are "
                    "built for tumour cells only - measuring a lymphocyte's membrane "
                    "would put a cell in the numerator that is not in the denominator."
                )

            fields: list[FieldCompartments] = []
            traced: list[dict] = []
            region_cells = 0
            region_contested = 0
            region_expanded = 0

            for field in region.fields:
                loaded = self._field_labels(
                    he_upload_id, ihc_upload_id, region.rank, field.index
                )
                if loaded is None:
                    continue
                labels, mpp, field_x, field_y = loaded
                if tumour_only:
                    keep = self._tumour_ids(
                        he_upload_id, ihc_upload_id, region.rank, field.index
                    )
                    if keep is None:
                        raise CompartmentError(
                            "step 12's cell types for this pair were written in an older "
                            "format that cannot say which cells in a given field are "
                            "tumour. Re-run step 12."
                        )
                    typed_total += int(len(keep))
                    labels = self._restrict(labels, keep)

                built = geometry.build(
                    labels,
                    mpp=mpp,
                    expansion_um=width,
                    ring=is_membrane,
                    ring_um=shell,
                )

                # Measured on the first field of each region only: it costs one
                # dilation per cell, which is the expensive operation in this
                # step, and it is a property of how crowded the tissue is rather
                # than a per-field result anybody reads.
                contested = (
                    geometry.contested_px(labels, mpp=mpp, expansion_um=width)
                    if field.index == region.fields[0].index
                    else 0
                )

                measured = geometry.areas_um2(built.measured, mpp=mpp)
                cell = geometry.areas_um2(built.cell, mpp=mpp)
                nucleus = geometry.areas_um2(built.nucleus, mpp=mpp)
                count = len(nucleus)
                if count == 0:
                    continue

                fields.append(
                    FieldCompartments(
                        region_rank=region.rank,
                        index=field.index,
                        cells=count,
                        mean_measured_um2=round(
                            float(np.mean(list(measured.values()))) if measured else 0.0, 2
                        ),
                        mean_cell_um2=round(float(np.mean(list(cell.values()))), 2),
                        mean_nucleus_um2=round(float(np.mean(list(nucleus.values()))), 2),
                        contested_px=contested,
                    )
                )

                region_cells += count
                region_contested += contested
                region_expanded += int(np.count_nonzero(built.cell))
                pooled_measured.extend(measured.values())
                pooled_cell.extend(cell.values())
                pooled_nucleus.extend(nucleus.values())

                # Every field, unlike the PNG below: step 14 measures all of them,
                # and a picture is for a person while the arrays are for the pipeline.
                self._write_geometry(
                    he_upload_id,
                    ihc_upload_id,
                    region.rank,
                    field.index,
                    built,
                    width_um=width,
                    is_membrane=is_membrane,
                    tumour_only=tumour_only,
                    shell_um=shell,
                    nuclei_generated_at=nuclei_report.generated_at,
                    typing_stamp=types_map.stamp(
                        self._types_path(he_upload_id, ihc_upload_id, region.rank)
                    ),
                    field_x=field_x,
                    field_y=field_y,
                )

                # Both bands of every cell, as outlines in the slide's own
                # coordinates. One extra distance transform per field, which is
                # what a screen that can pan the real slide costs; the label maps
                # above stay the measurement's input and these stay the picture.
                drawn = geometry.anatomy(
                    labels,
                    mpp=mpp,
                    body_um=width,
                    shell_um=shell or 0.0,
                    alternate_body_um=alternate_body,
                    alternate_shell_um=alternate_shell,
                )
                traced.append(
                    {
                        "index": field.index,
                        "x": field.x,
                        "y": field.y,
                        "span": field.span,
                        "cells": rings.trace_field(
                            drawn,
                            x0=float(field.x),
                            y0=float(field.y),
                            level0_scale=float(field.span) / float(max(1, field.size)),
                        ),
                    }
                )

                if field.index == region.fields[0].index:
                    self._write(
                        he_upload_id,
                        ihc_upload_id,
                        f"region{region.rank}/f{field.index}_compartments.png",
                        overlay.compartments_png(built, membrane=is_membrane),
                    )

            self._write(
                he_upload_id,
                ihc_upload_id,
                f"region{region.rank}/rings.json",
                json.dumps(
                    {
                        "rank": region.rank,
                        "measured": "membrane" if is_membrane else "cytoplasm",
                        "widthsUm": {
                            "body": round(width, 3),
                            "shell": round(shell or 0.0, 3),
                            "alternate": round(alternate_body, 3),
                        },
                        "fields": traced,
                    },
                    separators=(",", ":"),
                ),
            )

            regions.append(
                RegionCompartments(
                    rank=region.rank,
                    cells=region_cells,
                    mean_measured_um2=round(
                        float(np.mean(pooled_measured)) if pooled_measured else 0.0, 2
                    ),
                    mean_cell_um2=round(float(np.mean(pooled_cell)) if pooled_cell else 0.0, 2),
                    mean_nucleus_um2=round(
                        float(np.mean(pooled_nucleus)) if pooled_nucleus else 0.0, 2
                    ),
                    contested_share=round(
                        region_contested / max(1, region_expanded + region_contested), 4
                    ),
                    fields=fields,
                )
            )

        report = CompartmentsReport(
            he_upload_id=he_upload_id,
            ihc_upload_id=ihc_upload_id,
            marker=letter,
            marker_name=spec.full_name,
            compartment="membrane" if is_membrane else "cytoplasm",
            second_measure=spec.second_measure,
            generated_at=_now(),
            nuclei_generated_at=nuclei_report.generated_at,
            params=params,
            regions=regions,
            cells=sum(r.cells for r in regions),
            mean_measured_um2=round(
                float(np.mean(pooled_measured)) if pooled_measured else 0.0, 2
            ),
            mean_cell_um2=round(float(np.mean(pooled_cell)) if pooled_cell else 0.0, 2),
            width_sensitivity=self._sweep(
                he_upload_id,
                ihc_upload_id,
                nuclei_report,
                is_membrane,
                tumour_only,
                shell_ratio=(
                    spec.membrane_shell_um / spec.compartment_width_um
                    if is_membrane and spec.compartment_width_um > 0
                    else 0.0
                ),
            ),
            tumour_only=tumour_only,
            typed_cells=typed_total if tumour_only else None,
            notes=self._notes(spec, is_membrane),
        )

        self._write(
            he_upload_id,
            ihc_upload_id,
            "report.json",
            json.dumps(report.model_dump(by_alias=True), indent=2),
        )
        return report

    def _sweep(
        self,
        he_upload_id: str,
        ihc_upload_id: str,
        nuclei_report,
        is_membrane: bool,
        tumour_only: bool,
        shell_ratio: float,
    ) -> list[WidthSensitivityPoint]:
        """How the compartment area moves with the width, on one field.

        One field, not all of them: this is a shape of a curve, and paying for
        every field would make the width slider stop feeling live for a figure
        nobody reads to three significant figures.

        Published because **neither default is established fact.** Both come from
        QuPath's convention plus the physical size of a breast epithelial cell,
        and the guide is explicit that the sensitivity has to be reported - a
        score that swings fifteen points between a 3 um and a 5 um ring is a
        score with a hidden parameter in it.
        """
        if not nuclei_report.regions or not nuclei_report.regions[0].fields:
            return []

        region = nuclei_report.regions[0]
        field = region.fields[0]
        loaded = self._field_labels(he_upload_id, ihc_upload_id, region.rank, field.index)
        if loaded is None:
            return []

        labels, mpp = loaded
        if tumour_only:
            keep = self._tumour_ids(
                he_upload_id, ihc_upload_id, region.rank, field.index
            )
            if keep is not None:
                labels = self._restrict(labels, keep)

        widths = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0) if is_membrane else (4.0, 5.0, 6.0, 7.0, 8.0, 10.0)
        out: list[WidthSensitivityPoint] = []
        for width in widths:
            # The shell keeps its share of the body as the body grows, which is
            # the same rule `report` applies. A fixed shell would make the sweep
            # measure two things changing at once.
            built = geometry.build(
                labels,
                mpp=mpp,
                expansion_um=width,
                ring=is_membrane,
                ring_um=width * shell_ratio if is_membrane else None,
            )
            areas = geometry.areas_um2(built.measured, mpp=mpp)
            expanded = int(np.count_nonzero(built.cell))
            contested = geometry.contested_px(labels, mpp=mpp, expansion_um=width)
            out.append(
                WidthSensitivityPoint(
                    width_um=width,
                    mean_measured_um2=round(
                        float(np.mean(list(areas.values()))) if areas else 0.0, 2
                    ),
                    contested_share=round(contested / max(1, expanded + contested), 4),
                )
            )
        return out

    @staticmethod
    def _notes(spec, is_membrane: bool) -> list[str]:
        shape = "a ring around the outside of the cell" if is_membrane else "the cell body"
        other = "a cytoplasm band" if is_membrane else "a membrane ring"
        return [
            f"{spec.full_name} is a {spec.compartment.value} marker, so the brown is "
            f"measured in {shape}, {spec.compartment_width_um:g} um wide. Giving it "
            f"{other} instead would measure the wrong part of the cell.",
            "Each cell grows outward only as far as the midline between it and its "
            "neighbours, so no two cells can claim the same pixel. Without that, a "
            "strongly stained cell bleeds its signal into a negative neighbour and the "
            "score tracks how tightly the tissue is packed rather than the marker.",
            "Neither width is established fact - both come from QuPath's convention and "
            "the size of a breast epithelial cell. The sweep shows what the answer would "
            "have been at other widths.",
        ]


compartment_service = CompartmentService()
