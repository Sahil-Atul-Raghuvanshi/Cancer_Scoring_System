"""Step 16 - apply the rulebook and produce the two numbers that leave the system.

    report.json   the pair, the arithmetic that produced it, and every caveat

This service is thin on purpose. The arithmetic is in
`step18_aggregate/score.py`, which imports nothing from this layer and can be
run over a JSON file by anybody wanting to check a number. What is added here is
the things a *report* needs and a formula does not: the region areas, the
heterogeneity tiles, the cascade written out with the numbers substituted, and
the caveats that have to travel with a result - a provisional cut point, a
machine-confirmed alignment, a nuclei shortfall that inflated the percentage.

**The caveats are part of the output, not a footnote on a screen.** A percentage
computed over a denominator that is missing 70 % of its cells is not a
percentage with a caveat; it is a different number. So the caveat is on the
record beside it, and it is what the Excel prints in the row.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from app import panel
from app.core.config import settings
from app.core.logging import get_logger
from app.pipeline.step18_aggregate.score import MarkerScore, score
from app.schemas.scores import (
    CascadeStep,
    HeterogeneityTile,
    MarkerScoreOut,
    RegionScoreOut,
    ScoreReport,
)
from app.core import provenance
from app.scoring import cuts as cut_points
from app.services.ihc_alignment_service import ihc_alignment_service
from app.services.nuclei_service import NucleiError, nuclei_service
from app.services.per_cell_service import PerCellError, per_cell_service

logger = get_logger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ScoreError(ValueError):
    """A client-correctable problem: an earlier step has not run for this pair."""


class ScoreService:
    """Produces one marker's reported pair, and everything shown beside it."""

    def key(self, he_upload_id: str, ihc_upload_id: str) -> str:
        return f"{he_upload_id}__{ihc_upload_id}"

    def artifact(self, he_upload_id: str, ihc_upload_id: str, *parts: str) -> Path:
        return settings.scores_dir.joinpath(self.key(he_upload_id, ihc_upload_id), *parts)

    def report(self, he_upload_id: str, ihc_upload_id: str) -> ScoreReport:
        try:
            measured = per_cell_service.report(he_upload_id, ihc_upload_id)
            rows = per_cell_service.rows(he_upload_id, ihc_upload_id)
        except PerCellError as exc:
            raise ScoreError(
                "step 14 has not measured this pair's cells, and every number this step "
                "produces is a count or a mean over those rows."
            ) from exc

        letter = measured.marker
        spec = panel.spec(letter)
        marker_cuts = cut_points.for_marker(letter)
        cut_file = cut_points.cut_set()

        areas = {region.rank: region.area_mm2 for region in measured.regions}

        result = score(
            letter,
            rows,
            marker_cuts,
            marker_name=spec.full_name,
            compartment=spec.compartment.value,
            second_measure=measured.second_measure,
            region_areas=areas,
            partial_rule=cut_file.partial_membrane_rule,
        )

        out = MarkerScoreOut(
            marker=result.marker,
            marker_name=result.marker_name,
            percent=result.percent,
            intensity=result.intensity,
            intensity_label=cut_file.label_for(result.intensity),
            percent_raw=result.percent_raw,
            intensity_raw=result.intensity_raw,
            percent_pooled=result.percent_pooled,
            intensity_pooled=result.intensity_pooled,
            compartment=result.compartment,
            second_measure=result.second_measure,
            cells=result.cells,
            positive_cells=result.positive_cells,
            bin_counts=list(result.bin_counts),
            bin_shares=list(result.bin_shares),
            h_score=result.h_score,
            allred_proportion=result.allred_proportion,
            allred_intensity=result.allred_intensity,
            allred_total=result.allred_total,
            her2_call=result.her2_call,
            her2_note=result.her2_note,
            percent_area_weighted=result.percent_area_weighted,
            percent_plain_mean=result.percent_plain_mean,
            averaging_gap_points=result.averaging_gap_points,
            averaging_used="area_weighted",
            partial_rule=result.partial_rule,
            percent_by_partial_rule=result.percent_by_partial_rule,
            regions=[
                RegionScoreOut(
                    rank=region.rank,
                    area_mm2=region.area_mm2,
                    cells=region.cells,
                    positive_cells=region.positive_cells,
                    percent_raw=region.percent_raw,
                    intensity_raw=region.intensity_raw,
                )
                for region in result.regions
            ],
            heterogeneity=self._heterogeneity(
                he_upload_id, ihc_upload_id, rows, marker_cuts, cut_file
            ),
            cascade=self._cascade(result, cut_file),
            od_cuts=list(result.od_cuts),
            second_min=result.second_min,
            cuts_provisional=result.cuts_provisional,
            caveats=self._caveats(he_upload_id, ihc_upload_id, result, measured),
        )

        report = ScoreReport(
            he_upload_id=he_upload_id,
            ihc_upload_id=ihc_upload_id,
            generated_at=_now(),
            measured_at=measured.generated_at,
            score=out,
            notes=self._notes(),
            provenance=provenance.for_pair(he_upload_id, ihc_upload_id),
        )

        path = self.artifact(he_upload_id, ihc_upload_id, "report.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report.model_dump(by_alias=True), indent=2), encoding="utf-8"
        )
        return report

    # --- the screens --------------------------------------------------------

    def _heterogeneity(
        self, he_upload_id: str, ihc_upload_id: str, rows: list[dict], marker_cuts, cut_file
    ) -> list[HeterogeneityTile]:
        """Percent positive inside each sampled field, at that field's position.

        The ROI tiled by local percent positive, which is what makes "50 %"
        legible as an average over a very non-uniform field rather than a
        property the slide has evenly. The tiles are step 11's own sampled
        fields, so their positions are already on record and this invents no
        geometry of its own.
        """
        import numpy as np

        from app.pipeline.step17_intensity_binning.binning import positive_mask

        try:
            nuclei_report = nuclei_service.report(he_upload_id, ihc_upload_id)
        except NucleiError:
            return []

        placement = {
            (region.rank, field.index): (field.x, field.y, field.span)
            for region in nuclei_report.regions
            for field in region.fields
        }

        grouped: dict[tuple[int, int], list[dict]] = {}
        for row in rows:
            grouped.setdefault(
                (int(row["regionRank"]), int(row["fieldIndex"])), []
            ).append(row)

        tiles: list[HeterogeneityTile] = []
        for (rank, index), cells in sorted(grouped.items()):
            where = placement.get((rank, index))
            if where is None:
                continue
            od = np.array([float(c["intensityOd"]) for c in cells])
            second = np.array([float(c["second"]) for c in cells])
            _, weight = positive_mask(
                od, second, marker_cuts, rule=cut_file.partial_membrane_rule
            )
            x, y, span = where
            tiles.append(
                HeterogeneityTile(
                    region_rank=rank,
                    field_index=index,
                    x=float(x),
                    y=float(y),
                    span=int(span),
                    cells=len(cells),
                    percent=round(100.0 * float(weight.sum()) / max(1, len(cells)), 1),
                )
            )
        return tiles

    @staticmethod
    def _cascade(result: MarkerScore, cut_file) -> list[CascadeStep]:
        """The arithmetic written out with the actual numbers substituted in.

        One screen, the whole cascade visible at once, no hidden step between
        the cell counts and the pair. The rounding appears as its own line
        because it changes the answer and a number that changed silently is the
        one a reader stops trusting.
        """
        label = cut_file.label_for(result.intensity)
        return [
            CascadeStep(
                label="Tumour cells measured",
                expression="cells inside the invasive regions, typed as tumour",
                value=f"{result.cells:,}",
            ),
            CascadeStep(
                label="Positive by both conditions",
                expression=(
                    f"mean DAB OD >= {result.od_cuts[0]:.2f}  AND  "
                    f"{result.second_measure.replace('_', ' ')} >= {result.second_min:.2f}"
                ),
                value=f"{result.positive_cells:,.1f}",
            ),
            CascadeStep(
                label="Percent positive, pooled",
                expression=(
                    f"100 x {result.positive_cells:,.1f} / {result.cells:,} = "
                    f"{result.percent_pooled:.2f} - every measured cell counted once"
                ),
                value=f"{result.percent_pooled:.2f} %",
            ),
            CascadeStep(
                label="Percent positive, over the ROI",
                expression=(
                    "each invasive region measured on its own sample, then combined by "
                    "area: "
                    + "  +  ".join(
                        f"{region.percent_raw:.1f}% x {region.area_mm2:.2f}mm2"
                        for region in result.regions
                    )
                    + f"  /  {sum(r.area_mm2 for r in result.regions):.2f}mm2"
                ),
                value=f"{result.percent_raw:.2f} %",
            ),
            CascadeStep(
                label="Percent positive, reported",
                expression=(
                    f"round {result.percent_raw:.2f} to the nearest {5} "
                    "(real reported percentages are multiples of 5)"
                ),
                value=f"{result.percent} %",
            ),
            CascadeStep(
                label="Intensity, raw",
                expression=(
                    "mean DAB optical density over the positive cells only, combined "
                    f"across the regions by area = {result.intensity_raw:.4f} "
                    f"(pooled across all cells it would be {result.intensity_pooled:.4f})"
                ),
                value=f"{result.intensity_raw:.4f} OD",
            ),
            CascadeStep(
                label="Intensity, reported",
                expression=(
                    f"{result.intensity_raw:.4f} OD through {result.marker_name}'s band "
                    "table -> nearest permitted band of 0 / 0.5 / 1 / 1.5 / 1.75 / 2"
                ),
                value=f"{result.intensity:g}" + (f" ({label})" if label else ""),
            ),
            CascadeStep(
                label="The pair OncoStem receives",
                expression=f"{result.marker_name}: percent positive, intensity",
                value=f"{result.percent} %, {result.intensity:g}",
            ),
        ]

    def _caveats(
        self, he_upload_id: str, ihc_upload_id: str, result: MarkerScore, measured
    ) -> list[str]:
        """What a reader has to know before using these two numbers."""
        caveats: list[str] = []

        spec = panel.spec(result.marker)
        low, high = spec.expected_percent
        if high > 0 and not (low <= result.percent <= high):
            # `expected_percent` is the spread OncoStem has actually seen for this
            # antibody across six cases. It is explicitly NOT a threshold and not an
            # error - a real case can sit outside it - but a first run with
            # unfitted cut points landing outside it is far more likely to be the
            # cut points than the biology, so it is flagged for a person to look at.
            caveats.append(
                f"OUTSIDE THE EXPECTED RANGE. {result.marker_name} has been reported "
                f"between {low} % and {high} % across the cases OncoStem has read; this "
                f"slide scores {result.percent} %. That is a flag for human review, not "
                "an error - but with cut points that have not been fitted, a result this "
                "far out is more likely to be the cut points than the tissue."
            )

        # Where this slide sits on the band table, and how much of the table it
        # could ever have reached.
        #
        # The reported intensity is one of six values, but which one a slide gets
        # is decided by where its optical densities fall against fixed
        # breakpoints. A slide whose cells all sit inside one band's range is
        # reported on a six-value scale with one value of resolution - and that is
        # measurable without any reference to what the right answer is. A reader
        # comparing our intensity against a pathologist's should know it before
        # drawing a conclusion from the gap.
        table = cut_points.cut_set().for_marker(result.marker).od_to_band
        ceilings = [ceiling for ceiling, _ in table if ceiling is not None]
        if result.intensity_raw > 0 and len(ceilings) > 1:
            below = sum(1 for ceiling in ceilings if ceiling <= result.intensity_raw)
            if below <= 1 or below >= len(ceilings) - 1:
                where = "bottom" if below <= 1 else "top"
                caveats.append(
                    f"INTENSITY NEAR THE EDGE OF THE BAND TABLE. This slide's positive "
                    f"cells average {result.intensity_raw:.3f} OD, at the {where} of the "
                    f"mapping onto the 0-2 scale (breakpoints {ceilings}). A slide whose "
                    "densities cluster inside one or two bands is reported with far less "
                    "resolution than six values imply - and unlike the percentage, the "
                    "band table has no calibration path until reader intensities can be "
                    "fitted against it."
                )

        # **A percentage is only as real as the cells it was counted over.**
        #
        # CAN_00267's CD44 pair scored "0 % positive, Negative" over a denominator of
        # **one cell**: BEETLE kept 0.01 mm2 of a 1.96 mm2 region, step 11 found three
        # nuclei in it and step 12 typed one as tumour. Nothing else in the row says so -
        # `0 %` and `Negative` read exactly like a measured absence of staining, which is
        # the most misleading thing this pipeline can print.
        #
        # The bands are chosen so the warning gets stronger as the count gets absurd,
        # rather than one threshold that either fires or does not. 400 is the same figure
        # `compare_scores.py` uses for a denominator too thin to quote to the nearest
        # percent; below 50 the percentage carries no information at all.
        if result.cells < 400:
            per_cell = 100.0 / result.cells if result.cells else 0.0
            if result.cells < 50:
                caveats.append(
                    f"NOT A MEASUREMENT. This score was counted over {result.cells:,} "
                    f"cell(s), so each one moves the percentage by {per_cell:.0f} points "
                    "and the result can only be 0, 100, or a step in between. Whatever "
                    "the figures below say, this pair did not yield enough cells to "
                    "measure anything - read it as 'no usable tissue', never as a "
                    "negative or positive result."
                )
            else:
                caveats.append(
                    f"THIN DENOMINATOR. This score was counted over {result.cells:,} "
                    f"cells, so a single cell is worth {per_cell:.2f} percentage points. "
                    "The figure is quoted to the nearest percent, which implies a "
                    "precision this many cells cannot support - treat differences "
                    "smaller than a few points as noise."
                )

        if result.cuts_provisional:
            caveats.append(
                "PROVISIONAL CUT POINTS. They have not been fitted against the 120 "
                "pathologist readings, because that sheet is not on disk. The percentage "
                "and the intensity both move with these numbers, so treat the pair as a "
                "demonstration that the pipeline computes the contract, not as a "
                "measurement to act on."
            )

        try:
            nuclei_report = nuclei_service.report(he_upload_id, ihc_upload_id)
        except NucleiError:
            nuclei_report = None

        if nuclei_report is not None and nuclei_report.density_shortfall:
            shortfall = nuclei_report.density_shortfall
            if shortfall >= 0.3:
                caveats.append(
                    f"DENOMINATOR INCOMPLETE. This slide yielded {shortfall:.0%} fewer "
                    "nuclei per mm2 than the case's own H&E inside the same regions. "
                    "Serial sections of one block hold the same cells, so that gap is a "
                    "segmentation failure rather than biology - under heavy DAB the "
                    "counterstain is too weak for nuclear boundaries to survive "
                    "deconvolution. Every nucleus missed is a cell out of the "
                    "denominator, and missed cells are disproportionately the strongly "
                    "stained ones, so this inflates the percentage."
                )

        try:
            alignment = ihc_alignment_service.report(he_upload_id, ihc_upload_id)

            # A registration that failed the gate and was allowed through anyway. The
            # reasons travel with every number the pair produces: this is the difference
            # between a flagged result and an unflagged one, and it is the whole basis on
            # which the pair was allowed past the check at all.
            overridden = list(getattr(alignment.diagnostics, "gate_overridden", []) or [])
            if overridden:
                caveats.append(
                    "ALIGNMENT FAILED ITS OWN CHECK AND WAS ALLOWED THROUGH. "
                    + " ".join(f"({n}) {reason}" for n, reason in enumerate(overridden, 1))
                    + " The regions were carried onto this slide regardless, by explicit "
                    "instruction, because a refusal would drop the marker from the results "
                    "entirely. Treat every figure below as indicative only: if the "
                    "registration is wrong, these cells are not the cells the regions were "
                    "drawn around, and no measurement here can detect that."
                )

            if not alignment.confirmed:
                caveats.append(
                    "ALIGNMENT NOT CONFIRMED BY A PERSON. Step 10 refuses to approve its "
                    "own registration; somebody is meant to look at the two panels and "
                    "say the regions landed on the same tissue. These cells were measured "
                    "inside regions that check has not passed."
                )
            elif getattr(alignment, "confirmed_by", None) == "machine":
                caveats.append(
                    "ALIGNMENT MACHINE-CONFIRMED. The batch run confirmed step 10 "
                    "programmatically so it could proceed unattended. No person has "
                    "looked at the two panels."
                )
            elif getattr(alignment, "confirmed_by", None) is None:
                # Confirmed before the stamp existed, so who confirmed it is not on
                # record. Reported as unknown rather than assumed to be a person:
                # the whole value of the gate is knowing that somebody looked, and an
                # unrecorded sign-off is not evidence that they did.
                caveats.append(
                    "ALIGNMENT CONFIRMED, BY WHOM UNRECORDED. This pair was signed off "
                    "before the confirmation started recording whether a person or a "
                    "batch run did it. Re-confirm it on step 10 to put a person's "
                    "judgement on the record."
                )
        except Exception:  # noqa: BLE001 - a missing alignment is reported by earlier steps
            pass

        if measured.crowded_cells:
            share = measured.crowded_cells / max(1, measured.cells)
            if share >= 0.2:
                caveats.append(
                    f"{share:.0%} of cells had most of their membrane ring taken by "
                    "neighbours, so their completeness reflects how tightly the tissue is "
                    "packed as much as how they stained."
                )

        if abs(result.percent_area_weighted - result.percent_pooled) >= 5:
            caveats.append(
                f"REGION WEIGHTING MATTERS HERE. Combined by area the answer is "
                f"{result.percent_area_weighted:.1f} %; pooling every measured cell "
                f"regardless of which region it came from gives "
                f"{result.percent_pooled:.1f} %; a plain mean of the regions gives "
                f"{result.percent_plain_mean:.1f} %. Step 11 now allocates its fields in "
                "proportion to region area, so those first two figures are close where "
                "the tumour is one dominant mass and drift apart as the guaranteed "
                "minimum field per region lifts small regions above their share. The "
                "plain mean is the outlier by design: it gives a 0.25 mm2 fragment the "
                "same say as a 22 mm2 mass. The area-weighted figure is reported, which "
                "is the reading OncoStem's own procedure implies - the entire slide "
                "scanned, every field averaged (SOP 4.2) - though whether they weight by "
                "area or by cell count is still unanswered (Q3)."
            )
        elif result.averaging_gap_points >= 5:
            caveats.append(
                f"Combining the invasive regions by area and averaging them plainly "
                f"differ by {result.averaging_gap_points:.1f} points. Area-weighted is "
                "reported; which OncoStem uses is unanswered (Q3)."
            )

        spread = max(result.percent_by_partial_rule.values()) - min(
            result.percent_by_partial_rule.values()
        )
        if spread >= 10:
            caveats.append(
                f"How partial staining is counted moves this percentage by {spread} "
                f"points ({result.percent_by_partial_rule}). That is Q1, unanswered, and "
                "on this slide it is the largest single source of uncertainty in the "
                "number."
            )

        return caveats

    @staticmethod
    def _notes() -> list[str]:
        return [
            "The deliverable is the pair: percent positive and intensity. The H-score, "
            "the Allred score and the ASCO/CAP category are the field's standard "
            "vocabulary and are shown because they make the pair legible - none of them "
            "is an output of this system. OncoStem's sheet has ten columns, five "
            "percent-and-intensity pairs, and no combined score anywhere.",
            "Both roundings are load-bearing. Real reported percentages are multiples of "
            "5 and 119 of 120 real intensities land exactly on a band value, so an "
            "unrounded 61.7 % and 1.34 would not read as more precise - they would read "
            "as a different measurement from the one that was asked for.",
            "This step contains no image processing. Every number on it is a count or a "
            "mean over step 14's stored rows, which is what makes it separately "
            "checkable: the same arithmetic can be run over the JSON with a calculator.",
        ]


score_service = ScoreService()
