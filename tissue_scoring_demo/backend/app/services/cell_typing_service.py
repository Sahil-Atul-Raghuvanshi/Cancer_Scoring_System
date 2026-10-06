"""Step 12 - sort step 11's nuclei into the ones that count and the ones that do not.

    report.json            the mix, the sensitivity sweep, and the trust verdict
    region{r}/types.json   a class per nucleus, keyed `field:id`, for the overlay

**Request-shaped, not job-shaped**, and that is the whole reason this step feels
different from the three before it. It reads numbers step 11 already wrote and
applies five thresholds to them: no slide is opened, no model runs, and the
answer arrives in well under a second on a hundred thousand nuclei. That also
makes the thresholds *live* - a viewer can drag one and watch the mix move,
which is the only honest way to present parameters nobody has fitted.

**It inherits step 11's faults and says so.** Typing is size and shape
arithmetic. If the segmentation under-called the nuclei, the size rules are
sorting fragments rather than cells, and a confident-looking stacked bar would be
the worst possible output. `trustworthy` is that check, and it is computed from
the same median nuclear area the step 11 report already publishes.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.core.config import settings
from app.core.logging import get_logger
from app.pipeline.step14_cell_typing.classify import (
    CELL_TYPE_COLOURS,
    CELL_TYPE_LABELS,
    CELL_TYPE_NAMES,
    CellType,
    Rules,
    classify,
    sensitivity,
)
from app.schemas.cell_typing import (
    CellTypingReport,
    RegionTyping,
    SensitivityPoint,
    SensitivitySweep,
    TypeCount,
    TypingRules,
)
from app.services.nuclei_service import NucleiError, nuclei_service

logger = get_logger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class CellTypingError(ValueError):
    """A client-correctable problem: usually that step 11 has not run."""


class CellTypingService:
    """Applies the typing rules to one pair's stored nuclei."""

    # --- storage ------------------------------------------------------------

    def key(self, he_upload_id: str, ihc_upload_id: str) -> str:
        return f"{he_upload_id}__{ihc_upload_id}"

    def artifact(self, he_upload_id: str, ihc_upload_id: str, *parts: str) -> Path:
        return settings.cell_typing_dir.joinpath(
            self.key(he_upload_id, ihc_upload_id), *parts
        )

    def _write(self, he_upload_id: str, ihc_upload_id: str, name: str, payload: str) -> None:
        path = self.artifact(he_upload_id, ihc_upload_id, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")

    # --- reading step 11 ----------------------------------------------------

    def _nuclei(self, he_upload_id: str, ihc_upload_id: str, rank: int) -> list[dict]:
        """One region's nuclei, flattened across its fields.

        Only the counted ones. A nucleus in the border band was seen truncated,
        so its area and shape are wrong - and those are exactly the two things
        this step decides on, which makes including them worse here than it is
        in a count.

        **Each nucleus carries the field it came from**, because its id does not
        identify it on its own. Step 11 segments every field separately, so every
        field's instance map starts again at 1 and a region holds a dozen
        nucleus 14s. Flattening without the field index and keying a map by id
        alone is how one field's lymphocyte came to be recorded as another
        field's tumour cell - see `_key`.
        """
        path = nuclei_service.artifact(he_upload_id, ihc_upload_id, f"region{rank}", "nuclei.json")
        if not path.is_file():
            raise CellTypingError(
                f"step 11 has stored no nuclei for region {rank} of this pair"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [
            {**nucleus, "fieldIndex": field.get("index")}
            for field in payload.get("fields", [])
            for nucleus in field.get("nuclei", [])
            if nucleus.get("counted")
        ]

    #: Bumped when the shape of `types.json` changes. Steps 13 and 14 refuse a
    #: map they do not recognise rather than reading it as if it were current -
    #: version 1 keyed on the nucleus id alone, which collided across fields.
    TYPES_FORMAT = 2

    @staticmethod
    def _key(nucleus: dict) -> str:
        """`field:id` - the pair that actually identifies a nucleus in a region."""
        return f"{nucleus.get('fieldIndex')}:{nucleus['id']}"

    @staticmethod
    def _arrays(nuclei: list[dict]) -> tuple[np.ndarray, ...]:
        return (
            np.array([n["areaUm2"] for n in nuclei], dtype=np.float64),
            np.array([n["circularity"] for n in nuclei], dtype=np.float64),
            np.array([n["eccentricity"] for n in nuclei], dtype=np.float64),
            np.array([n["haematoxylin"] for n in nuclei], dtype=np.float64),
        )

    # --- the work -----------------------------------------------------------

    def report(
        self,
        he_upload_id: str,
        ihc_upload_id: str,
        *,
        rules: TypingRules | None = None,
    ) -> CellTypingReport:
        """Type every nucleus of every region, and say what the answer rests on."""
        try:
            nuclei_report = nuclei_service.report(he_upload_id, ihc_upload_id)
        except NucleiError as exc:
            raise CellTypingError(
                "step 11 has not found any nuclei for this pair yet, and there is "
                "nothing to sort until it has."
            ) from exc

        wanted = rules or TypingRules()
        engine_rules = Rules(
            lymphocyte_max_area_um2=wanted.lymphocyte_max_area_um2,
            lymphocyte_min_circularity=wanted.lymphocyte_min_circularity,
            lymphocyte_min_darkness=wanted.lymphocyte_min_darkness,
            spindle_min_eccentricity=wanted.spindle_min_eccentricity,
            spindle_max_area_um2=wanted.spindle_max_area_um2,
        )

        regions: list[RegionTyping] = []
        pooled: dict[str, list[np.ndarray]] = {"area": [], "circ": [], "ecc": [], "haem": []}
        totals = {int(t): 0 for t in CellType}

        for region in nuclei_report.regions:
            nuclei = self._nuclei(he_upload_id, ihc_upload_id, region.rank)
            areas, circ, ecc, haem = self._arrays(nuclei)
            typed = classify(areas, circ, ecc, haem, rules=engine_rules)

            pooled["area"].append(areas)
            pooled["circ"].append(circ)
            pooled["ecc"].append(ecc)
            pooled["haem"].append(haem)
            for key, value in typed.counts.items():
                totals[key] += value

            # A class per nucleus id, so the overlay can colour by class without
            # re-deriving the rules in the browser - which would be a second
            # definition of what a lymphocyte is.
            self._write(
                he_upload_id,
                ihc_upload_id,
                f"region{region.rank}/types.json",
                json.dumps(
                    {
                        "rank": region.rank,
                        "format": self.TYPES_FORMAT,
                        "types": {
                            self._key(n): int(label)
                            for n, label in zip(nuclei, typed.labels, strict=True)
                        },
                    }
                ),
            )

            tumour_mask = typed.labels == int(CellType.TUMOUR)
            lymph_mask = typed.labels == int(CellType.LYMPHOCYTE)
            sampled = region.sampled_mm2

            regions.append(
                RegionTyping(
                    rank=region.rank,
                    area_mm2=region.area_mm2,
                    counted=int(areas.size),
                    counts=self._counts(typed.counts, int(areas.size)),
                    tumour_per_mm2=(
                        round(int(tumour_mask.sum()) / sampled, 1) if sampled > 0 else 0.0
                    ),
                    median_tumour_area_um2=(
                        round(float(np.median(areas[tumour_mask])), 2)
                        if tumour_mask.any()
                        else 0.0
                    ),
                    median_lymphocyte_area_um2=(
                        round(float(np.median(areas[lymph_mask])), 2)
                        if lymph_mask.any()
                        else 0.0
                    ),
                )
            )

        areas = np.concatenate(pooled["area"]) if pooled["area"] else np.empty(0)
        circ = np.concatenate(pooled["circ"]) if pooled["circ"] else np.empty(0)
        ecc = np.concatenate(pooled["ecc"]) if pooled["ecc"] else np.empty(0)
        haem = np.concatenate(pooled["haem"]) if pooled["haem"] else np.empty(0)

        counted = int(areas.size)
        median_area = float(np.median(areas)) if counted else 0.0
        trustworthy, reason = self._trust(median_area, nuclei_report.density_shortfall)

        report = CellTypingReport(
            he_upload_id=he_upload_id,
            ihc_upload_id=ihc_upload_id,
            marker=nuclei_report.marker,
            generated_at=_now(),
            nuclei_generated_at=nuclei_report.generated_at,
            rules=wanted,
            regions=regions,
            counted=counted,
            counts=self._counts(totals, counted),
            tumour_share=(
                round(totals[int(CellType.TUMOUR)] / counted, 4) if counted else 0.0
            ),
            sensitivity=[
                SensitivitySweep(
                    parameter=entry["parameter"],
                    baseline=entry["baseline"],
                    points=[
                        SensitivityPoint(value=p["value"], tumour_share=p["tumourShare"])
                        for p in entry["points"]
                    ],
                    swing=entry["swing"],
                )
                for entry in sensitivity(areas, circ, ecc, haem, rules=engine_rules)
            ],
            trustworthy=trustworthy,
            trust_reason=reason,
            median_area_um2=round(median_area, 2),
            notes=self._notes(),
        )

        self._write(
            he_upload_id,
            ihc_upload_id,
            "report.json",
            json.dumps(report.model_dump(by_alias=True), indent=2),
        )
        return report

    # --- the honesty terms --------------------------------------------------

    #: Smallest median nuclear area, in um2, at which sorting cells by size means
    #: anything. A breast epithelial nucleus is 7-10 um across, so 40-80 um2;
    #: measured on this project's own H&E the median is 43. Well below that, the
    #: objects being sorted are fragments and every size rule is arbitrary.
    MIN_CREDIBLE_MEDIAN_AREA_UM2 = 25.0

    @classmethod
    def _trust(
        cls, median_area: float, shortfall: float | None
    ) -> tuple[bool, str | None]:
        if median_area and median_area < cls.MIN_CREDIBLE_MEDIAN_AREA_UM2:
            detail = (
                f"The typical segmented nucleus here is {median_area:.0f} um2, against "
                f"{cls.MIN_CREDIBLE_MEDIAN_AREA_UM2:.0f} um2 as the floor for a real "
                "epithelial nucleus and about 43 um2 measured on this case's own H&E. "
                "These rules sort cells by size and shape, so on objects this small they "
                "are sorting fragments rather than cells."
            )
            if shortfall and shortfall >= 0.3:
                detail += (
                    f" Step 11 also reports {shortfall:.0%} fewer nuclei per mm2 of tissue than the "
                    "H&E of the same block, which is the same finding from the other side."
                )
            return False, detail
        return True, None

    @staticmethod
    def _counts(counts: dict[int, int], total: int) -> list[TypeCount]:
        return [
            TypeCount(
                type=CELL_TYPE_NAMES[int(cell_type)],
                label=CELL_TYPE_LABELS[int(cell_type)],
                count=counts.get(int(cell_type), 0),
                share=round(counts.get(int(cell_type), 0) / total, 4) if total else 0.0,
                colour=CELL_TYPE_COLOURS[int(cell_type)],
            )
            for cell_type in CellType
        ]

    @staticmethod
    def _notes() -> list[str]:
        return [
            "These classes come from thresholds on size, roundness, elongation and "
            "stain darkness - not from a model fitted on labelled cells, because none "
            "have been labelled. The sensitivity table shows how much each threshold "
            "is deciding the answer.",
            "Only cells counted by step 11 are sorted here. One in the edge band was "
            "seen cut off, so its size and shape are wrong - and those are the two "
            "things this step decides on.",
        ]


cell_typing_service = CellTypingService()
