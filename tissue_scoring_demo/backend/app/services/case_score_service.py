"""The case's ten numbers: five markers, two each, as the grid OncoStem receives.

Every step below this one works on a *pair* - one H&E slide and one IHC slide -
because that is the unit everything from step 10 onward is keyed on. But the
deliverable is a **case**: five markers of the same block, scored separately and
reported together. This module is the join, and it is the only place that knows
a case is five pairs.

It gathers rather than computes. Each marker's pair was scored by
`score_service`; this reads those results back through the case sidecars
`case_service` writes, and fills in a row per marker saying what happened where
one is missing. A partial grid is the honest picture of a run in progress, and
far more useful than refusing to answer until all five have finished.

**Column order is the panel's - A, F, R, U, W - and not the reader sheet's.**
That sheet runs A, W, U, R, F, which silently swaps ABCC4 with ABCC11 for anyone
joining on position. Both are plausible pump markers with overlapping ranges, so
the mistake would not announce itself. Anything joining the two orders converts
explicitly, by letter.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from app import panel
from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.scores import CaseScoreReport, CaseScoreRow
from app.services.score_service import ScoreError, score_service

logger = get_logger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class CaseScoreService:
    """Reads back whatever of a case's five markers has been scored."""

    def _cases_dir(self) -> Path:
        return settings.data_dir / "cases"

    def pairs(self, case_id: str) -> dict[str, tuple[str, str]]:
        """Marker letter -> (he_upload_id, ihc_upload_id), from the sidecars.

        The sidecar is `case_service`'s record of which two slides a marker was
        loaded as, written when the biomarker was chosen before step 1. Reading
        it here is what carries that choice all the way to the reported number:
        nothing downstream re-infers the marker from a filename.
        """
        out: dict[str, tuple[str, str]] = {}
        directory = self._cases_dir()
        if not directory.is_dir():
            return out

        for path in sorted(directory.glob(f"{case_id}_*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                logger.warning("unreadable case sidecar %s", path)
                continue
            letter = str(payload.get("marker", "")).upper()
            he = payload.get("he_upload_id")
            ihc = payload.get("ihc_upload_id")
            if letter in panel.PANEL and he and ihc:
                out[letter] = (he, ihc)
        return out

    def report(self, case_id: str) -> CaseScoreReport:
        pairs = self.pairs(case_id)
        rows: list[CaseScoreRow] = []

        for letter in panel.SCORED_MARKERS:
            spec = panel.spec(letter)
            found = pairs.get(letter)
            if found is None:
                rows.append(
                    CaseScoreRow(
                        marker=letter,
                        marker_name=spec.full_name,
                        state="not_loaded",
                        detail=(
                            f"no {spec.name} slide has been loaded for this case. Load it "
                            "with POST /cases/load and run the pipeline on it."
                        ),
                    )
                )
                continue

            he, ihc = found
            try:
                result = score_service.report(he, ihc).score
            except ScoreError as exc:
                rows.append(
                    CaseScoreRow(
                        marker=letter,
                        marker_name=spec.full_name,
                        state="not_scored",
                        detail=str(exc),
                    )
                )
                continue
            except Exception as exc:  # noqa: BLE001 - one marker must not fail the grid
                logger.exception("scoring %s for case %s failed", letter, case_id)
                rows.append(
                    CaseScoreRow(
                        marker=letter,
                        marker_name=spec.full_name,
                        state="failed",
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue

            rows.append(
                CaseScoreRow(
                    marker=letter,
                    marker_name=spec.full_name,
                    percent=result.percent,
                    intensity=result.intensity,
                    intensity_label=result.intensity_label,
                    cells=result.cells,
                    state="scored",
                    detail="; ".join(result.caveats),
                )
            )

        complete = all(row.state == "scored" for row in rows)
        return CaseScoreReport(
            case_id=case_id,
            generated_at=_now(),
            rows=rows,
            complete=complete,
            notes=[
                "Ten numbers: five markers, a percent positive and an intensity each. "
                "That is the whole deliverable - no combined score, because OncoStem's "
                "own sheet has no column for one and collapsing the pair would throw away "
                "information their model uses.",
                "Rows are in panel order - A, F, R, U, W. OncoStem's sheet runs A, W, U, "
                "R, F; joining on position rather than on the letter swaps ABCC4 with "
                "ABCC11, and both are plausible pump markers with overlapping ranges, so "
                "the error would not announce itself.",
            ],
        )


case_score_service = CaseScoreService()
