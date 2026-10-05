"""Step 15 - turn each cell's optical density into a level, on absolute cuts.

    report.json   the bins, the cuts drawn on the histogram, and the schemes rejected

Reads step 14's stored rows and applies five numbers to them. No slide is
opened, nothing is segmented, and the answer arrives in milliseconds on a
hundred thousand cells - which is what makes the cut points *live* on screen: a
viewer can drag one and watch the bins move. That is the only honest way to
present thresholds that have not been fitted.

**The cuts come from this antibody's own set, and the request cannot supply
them.** A `?odCuts=` parameter would be a way to reach the reported number by
choosing the threshold that produces it. The width at step 13 is a parameter
because neither default is established fact and the sweep is the point; a cut
point is different, because it is the thing being calibrated.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app import panel
from app.core.config import settings
from app.core.logging import get_logger
from app.pipeline.step17_intensity_binning.binning import (
    BIN_LABELS,
    bin_cells,
    compare,
    positive_mask,
)
from app.schemas.binning import (
    BandTableRow,
    BinCount,
    BinningParams,
    BinningReport,
    CutLine,
    ODHistogramBin,
    SchemeComparison,
)
from app.scoring import cuts as cut_points
from app.services.per_cell_service import PerCellError, per_cell_service

logger = get_logger(__name__)

#: The marker whose cuts stand in for "one shared table across the panel" in the
#: comparison. CD44 - the membrane markers' set - because applying it to the two
#: cadherins is the concrete version of the mistake, and CD44 is the panel's
#: hero marker so a reader already has its numbers in mind.
SHARED_TABLE_MARKER = "A"


def _now() -> str:
    return datetime.now(UTC).isoformat()


class BinningError(ValueError):
    """A client-correctable problem: step 14 has not measured this pair."""


class BinningService:
    """Applies one antibody's cut points to one pair's measured cells."""

    def key(self, he_upload_id: str, ihc_upload_id: str) -> str:
        return f"{he_upload_id}__{ihc_upload_id}"

    def artifact(self, he_upload_id: str, ihc_upload_id: str, *parts: str) -> Path:
        return settings.binning_dir.joinpath(self.key(he_upload_id, ihc_upload_id), *parts)

    def cells(self, he_upload_id: str, ihc_upload_id: str) -> list[dict]:
        """One row per cell: where it is, what it measured, and what bin it fell in.

        **The bin is computed here and not in the browser**, even though it is
        one `searchsorted` and the cut points are already on the report. Step 15's
        screen colours every cell on the slide by its level and says whether it
        counts as positive, and positivity is two conditions plus a partial-staining
        rule that is an open question with three implemented answers. A copy of
        that rule in the client would be a second place for it to be decided, and
        the one thing this step must not do is report a percentage whose rule the
        picture beside it disagrees with.

        Cheap: it is the same arrays `report` already builds, kept per cell instead
        of counted.
        """
        try:
            rows = per_cell_service.rows(he_upload_id, ihc_upload_id)
            measured = per_cell_service.report(he_upload_id, ihc_upload_id)
        except PerCellError as exc:
            raise BinningError(
                "step 14 has not measured this pair's cells yet, and there is nothing "
                "to bin until it has."
            ) from exc

        marker_cuts = cut_points.for_marker(measured.marker)
        cut_file = cut_points.cut_set()

        od = np.array([float(row["intensityOd"]) for row in rows], dtype=np.float64)
        second = np.array([float(row["second"]) for row in rows], dtype=np.float64)

        binned = bin_cells(od, marker_cuts)
        positive, weight = positive_mask(
            od, second, marker_cuts, rule=cut_file.partial_membrane_rule
        )

        return [
            {
                "cellId": row["cellId"],
                "regionRank": row["regionRank"],
                "fieldIndex": row["fieldIndex"],
                "intensityOd": row["intensityOd"],
                "second": row["second"],
                "bin": int(binned.bins[index]),
                "label": BIN_LABELS[int(binned.bins[index])],
                "positive": bool(positive[index]),
                "weight": round(float(weight[index]), 3),
            }
            for index, row in enumerate(rows)
        ]

    def report(self, he_upload_id: str, ihc_upload_id: str) -> BinningReport:
        try:
            measured = per_cell_service.report(he_upload_id, ihc_upload_id)
            rows = per_cell_service.rows(he_upload_id, ihc_upload_id)
        except PerCellError as exc:
            raise BinningError(
                "step 14 has not measured this pair's cells yet, and there is nothing "
                "to bin until it has."
            ) from exc

        letter = measured.marker
        spec = panel.spec(letter)
        marker_cuts = cut_points.for_marker(letter)
        cut_file = cut_points.cut_set()

        od = np.array([float(row["intensityOd"]) for row in rows], dtype=np.float64)
        second = np.array([float(row["second"]) for row in rows], dtype=np.float64)

        binned = bin_cells(od, marker_cuts)
        positive, weight = positive_mask(
            od, second, marker_cuts, rule=cut_file.partial_membrane_rule
        )

        shared = (
            cut_points.for_marker(SHARED_TABLE_MARKER)
            if letter != SHARED_TABLE_MARKER
            else None
        )

        report = BinningReport(
            he_upload_id=he_upload_id,
            ihc_upload_id=ihc_upload_id,
            marker=letter,
            marker_name=spec.full_name,
            generated_at=_now(),
            measured_at=measured.generated_at,
            params=BinningParams(
                marker=letter,
                marker_name=spec.full_name,
                scheme="absolute",
                od_cuts=list(marker_cuts.od),
                second_measure=marker_cuts.second_measure,
                second_min=marker_cuts.second_min,
                partial_rule=cut_file.partial_membrane_rule,
                cuts_version=cut_file.version,
                cuts_provisional=marker_cuts.provisional,
                cuts_source=cut_file.source_path,
            ),
            cells=int(od.size),
            bins=self._bins(binned, marker_cuts),
            over_od_cut=int(np.count_nonzero(od >= marker_cuts.positivity_od)),
            positive_cells=round(float(weight.sum()), 2),
            positive_share=round(float(weight.sum() / max(1, od.size)), 4),
            histogram=[
                ODHistogramBin(lower=entry.lower, upper=entry.upper, count=entry.count)
                for entry in measured.histogram
            ],
            cut_lines=self._cut_lines(marker_cuts),
            valley_od=self._valley(measured.histogram),
            band_table=self._band_table(marker_cuts, cut_file),
            comparisons=[
                SchemeComparison(
                    scheme=entry.scheme,
                    label=entry.label,
                    od_cuts=list(entry.od_cuts),
                    counts=list(entry.counts),
                    positive_share=entry.positive_share,
                    delta_points=entry.delta_points,
                    note=entry.note,
                )
                for entry in compare(
                    od,
                    second,
                    marker_cuts,
                    shared=shared,
                    rule=cut_file.partial_membrane_rule,
                )
            ],
            notes=self._notes(marker_cuts, cut_file, positive, od),
        )

        path = self.artifact(he_upload_id, ihc_upload_id, "report.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report.model_dump(by_alias=True), indent=2), encoding="utf-8"
        )
        return report

    # --- presentation -------------------------------------------------------

    @staticmethod
    def _bins(binned, marker_cuts) -> list[BinCount]:
        edges = (0.0, *marker_cuts.od)
        return [
            BinCount(
                bin=level,
                label=BIN_LABELS[level],
                count=binned.counts[level],
                share=binned.shares[level],
                od_from=round(edges[level], 4),
                od_to=(round(marker_cuts.od[level], 4) if level < 3 else None),
            )
            for level in range(4)
        ]

    @staticmethod
    def _cut_lines(marker_cuts) -> list[CutLine]:
        return [
            CutLine(od=marker_cuts.od[0], label="positive", separates="0 from 1+"),
            CutLine(od=marker_cuts.od[1], label="moderate", separates="1+ from 2+"),
            CutLine(od=marker_cuts.od[2], label="strong", separates="2+ from 3+"),
        ]

    @staticmethod
    def _band_table(marker_cuts, cut_file) -> list[BandTableRow]:
        return [
            BandTableRow(
                od_below=ceiling,
                band=band,
                label=cut_file.label_for(band),
            )
            for ceiling, band in marker_cuts.od_to_band
        ]

    @staticmethod
    def _valley(histogram) -> float | None:
        """The lowest bar between the two tallest humps, if there are two.

        A well-placed cut sits in the valley between the unstained and stained
        populations. Finding it here means the screen can show whether the cut
        landed there rather than asking the viewer to judge by eye - and `None`
        is an honest answer: a unimodal histogram has no valley, which is itself
        worth knowing, because it means the two populations are not separated on
        this slide at all.
        """
        counts = np.array([entry.count for entry in histogram], dtype=np.float64)
        if counts.size < 5 or counts.sum() == 0:
            return None

        # Smooth over three bars so a single noisy bar is not a hump.
        smooth = np.convolve(counts, np.ones(3) / 3.0, mode="same")
        first = int(np.argmax(smooth))
        # The second hump must be a genuine local maximum away from the first.
        gap = max(3, smooth.size // 10)
        candidates = [
            index
            for index in range(1, smooth.size - 1)
            if abs(index - first) > gap
            and smooth[index] >= smooth[index - 1]
            and smooth[index] >= smooth[index + 1]
        ]
        if not candidates:
            return None
        second = max(candidates, key=lambda index: smooth[index])
        low, high = sorted((first, second))
        if high - low < 2:
            return None

        trough = low + int(np.argmin(smooth[low : high + 1]))
        entry = histogram[trough]
        return round((entry.lower + entry.upper) / 2.0, 4)

    @staticmethod
    def _notes(marker_cuts, cut_file, positive, od) -> list[str]:
        notes = [
            "Two different things happen at this step and they are kept apart. Each cell "
            "gets 0 / 1+ / 2+ / 3+, which is internal machinery for counting positives "
            "and for the H-score. The case's reported intensity is a different scale - "
            "OncoStem's 0 to 2 - and it is decided once, at step 16, over the positive "
            "cells' mean.",
            "The cuts are absolute optical densities on the calibrated scale, not "
            "percentiles of this slide. The comparison below shows what per-slide "
            "percentiles would have reported on these same cells: a weak slide and a "
            "strong one come out alike, because the scale moves underneath the number.",
            f"These cuts belong to {marker_cuts.name} alone. Each antibody has its own "
            "concentration, incubation time and detection chemistry, so one table across "
            "five markers cuts at least three of them in the wrong place.",
        ]
        if marker_cuts.provisional:
            notes.append(
                "PROVISIONAL. These cut points have not been fitted against pathologist "
                "scores - the reader sheet with the 120 (percent, intensity) pairs is not "
                "on disk - so they come from the optical-density scale itself. Every "
                "number computed under them carries this."
            )
        if cut_file.band_conflict:
            notes.append(cut_file.band_conflict)
        share = float(positive.sum()) / max(1, od.size)
        if share in (0.0, 1.0):
            notes.append(
                f"Every cell fell on the same side of the cut ({share:.0%} positive). That "
                "is a cut in the wrong place for this slide far more often than it is a "
                "slide that is uniformly one thing, and it is the first thing to check "
                "before reading the score."
            )
        return notes


binning_service = BinningService()
