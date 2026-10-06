"""Step 17 - compare our numbers with pathologists', or say we cannot.

Reads a reader sheet if one has been pointed at (`settings.reader_scores_path`),
joins it to whatever this system has scored, and hands both to
`step19_validation.agreement`. The sheet is not in this repository, so the
normal answer today is the honest one: no reader data, nothing validated.

**The join is by letter, never by position.** The sheet's columns run A, W, U,
R, F - not panel order - so anything reading them positionally swaps ABCC4 with
ABCC11. Both are plausible pump markers with overlapping ranges, so the error
produces entirely believable agreement statistics for the wrong pairing. The
column headers carry the letter (`A%-M`, `WI-C`), so the letter is what is read.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger
from app.pipeline.step19_validation.agreement import AgreementReport, agreement
from app.scoring.cuts import PERMITTED_BANDS, cut_set
from app.services.case_score_service import case_score_service

logger = get_logger(__name__)

#: `A%-M` / `AI-C`: the leading letter is the marker, `%` or `I` says which of
#: the two numbers it is. The trailing letter is a compartment on the percent
#: columns and always `-C` on the intensity ones, so it is deliberately not read.
_COLUMN = re.compile(r"^(?P<letter>[AFRUW])(?P<kind>%|I)-", re.IGNORECASE)

#: `PS_CAN/00251/26_H26-364-B1`: reader initials, case id, block. Punctuation in
#: the case id is inconsistent across rows, so it is normalised on both sides.
_ROW_ID = re.compile(r"^(?P<reader>[A-Z]{2})[_/](?P<case>.+?)_(?P<block>[^_]+)$")


def _normalise_case(value: str) -> str:
    """`CAN/00251/26`, `CAN00259/26` and `CAN_00251_26` all become one string."""
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def _same_case(ours: str, theirs: str) -> bool:
    """Whether two normalised case ids name the same case.

    Not equality, because the two sides carry different amounts of the name. A
    case folder is `CAN_00270`; the reader sheet writes `CAN00270/26`, with the
    accession year on the end. Normalised those are `CAN00270` and `CAN0027026`,
    and an equality join quietly finds nothing in common - which is
    indistinguishable, in the report, from having no reader data at all.

    A prefix test rather than a fuzzy match: the shorter id has to be a complete
    leading portion of the longer one. `CAN00270` matches `CAN0027026`; it does
    not match `CAN00271`, and nothing here would match two different cases unless
    one case id were a literal prefix of another, which this numbering scheme
    does not produce.
    """
    if not ours or not theirs:
        return False
    return ours.startswith(theirs) or theirs.startswith(ours)


class ValidationService:
    """Joins this system's scores to the readers' and reports the agreement."""

    def _sheet_path(self) -> Path | None:
        configured = settings.reader_scores_path
        if configured is not None and Path(configured).is_file():
            return Path(configured)
        # Conventional locations, searched in order of authority. The client's own
        # workbook wins wherever it is found; the transcription is a fallback that
        # was read off a screenshot and carries only six of the eleven columns, so
        # using it while the real file sits on disk would silently discard four
        # markers' intensity ground truth.
        #
        # `data/oncostem_requiremnet_docs/` is where the client's material actually
        # lives in this repository. It is searched first for that reason - the copy
        # under `data/original/` is a drop-box, and a stale file there should not
        # outrank the delivered one.
        #
        # The client's material is shared by every data version, so it lives under
        # `storage/data/`, not beside this version's runs. Looking only next to
        # `settings.data_dir` found nothing after the versioned layout landed, and
        # step 19 silently had no readers to compare against.
        from app.core.config import ORIGINAL_ROOT, data_versions

        SHARED_DATA_ROOT = data_versions.SHARED_DATA_ROOT

        sheet = "6Slide Reports2.xlsx"
        root = settings.data_dir.parent
        for candidate in (
            SHARED_DATA_ROOT / "oncostem_requiremnet_docs" / "client" / "OncoStem" / sheet,
            SHARED_DATA_ROOT / "oncostem_docs" / "client" / "OncoStem" / sheet,
            ORIGINAL_ROOT / "oncostem_scores" / sheet,
            root / "oncostem_requiremnet_docs" / "client" / "OncoStem" / sheet,
            root / "original" / sheet,
            root / "original" / "reader_scores_transcribed.xlsx",
        ):
            if candidate.is_file():
                return candidate
        return None

    def readers(self) -> dict[str, dict[str, list[tuple[float, float]]]]:
        """`{marker: {case: [(percent, intensity) per reader]}}`, or empty.

        `AVG_` rows are skipped: they are a plain arithmetic mean of the four
        readers, verified on all 60 averages, so including them would weight
        every case's consensus into its own comparison twice.
        """
        path = self._sheet_path()
        if path is None:
            return {}

        try:
            from openpyxl import load_workbook
        except ImportError:
            logger.warning("openpyxl is not installed; the reader sheet cannot be read")
            return {}

        try:
            sheet = load_workbook(path, data_only=True).active
        except Exception:  # noqa: BLE001 - a bad sheet must not take the step down
            logger.exception("could not read the reader sheet at %s", path)
            return {}

        rows = list(sheet.iter_rows(values_only=True))
        if len(rows) < 2:
            return {}

        header = [str(value or "") for value in rows[0]]
        columns: dict[int, tuple[str, str]] = {}
        for index, name in enumerate(header):
            match = _COLUMN.match(name.strip())
            if match:
                columns[index] = (
                    match.group("letter").upper(),
                    "percent" if match.group("kind") == "%" else "intensity",
                )

        out: dict[str, dict[str, list[tuple[float, float]]]] = {}
        for row in rows[1:]:
            label = str(row[0] or "")
            if not label or label.upper().startswith("AVG"):
                continue
            match = _ROW_ID.match(label.strip())
            case = _normalise_case(match.group("case") if match else label)

            pairs: dict[str, dict[str, float]] = {}
            for index, (letter, kind) in columns.items():
                if index >= len(row) or row[index] is None:
                    continue
                try:
                    pairs.setdefault(letter, {})[kind] = float(row[index])
                except (TypeError, ValueError):
                    continue

            for letter, values in pairs.items():
                # Percent is required; intensity is not. The sheet we have carries
                # only one of the five intensity columns, and refusing a marker for
                # a missing column would discard the percent comparison it CAN
                # support. A missing intensity travels as None and is skipped in
                # the intensity statistics rather than being filled with a zero,
                # which would read as "the readers called it negative".
                if "percent" in values:
                    out.setdefault(letter, {}).setdefault(case, []).append(
                        (values["percent"], values.get("intensity"))
                    )
        return out

    def ours(self, case_id: str | None) -> dict[str, dict[str, tuple[float, float]]]:
        """`{marker: {case: (percent, intensity)}}` for what has been scored."""
        cases = [case_id] if case_id else self._known_cases()
        out: dict[str, dict[str, tuple[float, float]]] = {}
        for case in cases:
            report = case_score_service.report(case)
            for row in report.rows:
                if row.state != "scored" or row.percent is None or row.intensity is None:
                    continue
                out.setdefault(row.marker, {})[_normalise_case(case)] = (
                    float(row.percent),
                    float(row.intensity),
                )
        return out

    @staticmethod
    def _align(
        ours: dict[str, dict[str, tuple[float, float]]],
        readers: dict[str, dict[str, list[tuple[float, float | None]]]],
    ) -> dict[str, dict[str, list[tuple[float, float | None]]]]:
        """Re-key the readers' cases onto the ids this system uses.

        Done here rather than inside `agreement` so that the comparison itself
        stays a pure function over two dicts with matching keys - it has enough
        to be responsible for without also knowing how two organisations spell a
        case number.
        """
        aligned: dict[str, dict[str, list[tuple[float, float | None]]]] = {}
        for marker, cases in readers.items():
            mine = ours.get(marker, {})
            for their_case, readings in cases.items():
                match = next(
                    (case for case in mine if _same_case(case, their_case)), their_case
                )
                aligned.setdefault(marker, {})[match] = readings
        return aligned

    def _known_cases(self) -> list[str]:
        directory = settings.data_dir / "cases"
        if not directory.is_dir():
            return []
        return sorted({path.stem.rsplit("_", 1)[0] for path in directory.glob("*.json")})

    def report(self, case_id: str | None = None) -> dict:
        mine = self.ours(case_id)
        result: AgreementReport = agreement(
            mine,
            self._align(mine, self.readers()),
            bands=list(PERMITTED_BANDS),
        )
        return {
            "generatedAt": datetime.now(UTC).isoformat(),
            "caseId": case_id,
            "readerSheet": str(self._sheet_path() or ""),
            "cutsProvisional": cut_set().status != "calibrated",
            "available": result.available,
            "reason": result.reason,
            "markers": [
                {
                    "marker": entry.marker,
                    "cases": entry.cases,
                    "percentBias": entry.percent_bias,
                    "percentLimits": list(entry.percent_limits),
                    "percentWithinTolerance": entry.percent_within_tolerance,
                    "percentMaxError": entry.percent_max_error,
                    "intensityBias": entry.intensity_bias,
                    "intensityLimits": list(entry.intensity_limits),
                    "intensityKappa": entry.intensity_kappa,
                    "intensityCases": entry.intensity_cases,
                    "readerSpread": entry.reader_spread,
                }
                for entry in result.markers
            ],
            "notes": result.notes,
        }


validation_service = ValidationService()
