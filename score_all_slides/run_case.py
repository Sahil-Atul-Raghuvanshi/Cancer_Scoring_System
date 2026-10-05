"""One case, end to end: five markers scored, recorded, and filed into history.

    python run_case.py CAN_00251

Run as its own process by `run_all.py`, one case at a time, and that isolation is the
point. Steps 8 and 11 hold gigabytes and call into torch; a case that runs out of memory
or dies inside a native library takes its process with it. Keeping the orchestrator in a
different process means such a death is a case marked `crashed` and retried rather than
an overnight run that ended at 1am.

**The scoring itself is the demo's own runner, not a copy of it.** `run_marker` from
`backend/scripts/run_case_scores.py` is what the pipeline already uses: it shares one
registered H&E across the five markers, skips whatever is on disk, confirms step 10 with
a machine stamp and records that stamp as a caveat on every number it produces. Calling
it rather than reimplementing it is what makes these numbers the same numbers the UI
shows for the same slides.

**A marker gets two attempts, and the second restarts the alignment.** The one failure
this run is most likely to meet repeatedly is step 10 refusing on a reused registration -
VALIS hands back an empty summary and the step reports it as a path-limit problem, which
it is not. `restart_alignment=True` is the documented way through it, so the retry is
that rather than the same call again.

**History is written last, and only for markers that scored.** `archive_marker` *moves*
the work into `v<N>_data/data/history`; it does not copy it. So it is done once a case's numbers
are safely in `results/<case>.json`, never before - and a failed marker is left in the
demo tree where a retry can still pick it up.
"""

from __future__ import annotations

import pathlib
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pipeline

sys.path.insert(0, str(pipeline.BACKEND))
sys.path.insert(0, str(pipeline.BACKEND / "scripts"))


def _score_fields(he_upload_id: str, ihc_upload_id: str) -> dict:
    """Step 16's numbers for one pair, as the CSV's columns.

    Read back from `score_service` rather than parsed out of the runner's log line,
    because the log rounds for a person: it prints `60 % positive, intensity 1` where the
    record needs `59.77` and `0.3746` as well. The report is already on disk at this
    point, so this is a read and not a recomputation.
    """
    from app.services.score_service import score_service

    report = score_service.report(he_upload_id, ihc_upload_id)
    score = report.score
    return {
        "percent": score.percent,
        "intensity": score.intensity,
        "intensity_label": score.intensity_label,
        "percent_raw": round(score.percent_raw, 4),
        "intensity_raw": round(score.intensity_raw, 6),
        "percent_pooled": round(score.percent_pooled, 4),
        "intensity_pooled": round(score.intensity_pooled, 6),
        "compartment": score.compartment,
        "second_measure": score.second_measure,
        "cells": score.cells,
        "positive_cells": round(score.positive_cells, 2),
        "h_score": round(score.h_score, 2),
        "allred_proportion": score.allred_proportion,
        "allred_intensity": score.allred_intensity,
        "allred_total": score.allred_total,
        "percent_area_weighted": round(score.percent_area_weighted, 4),
        "percent_plain_mean": round(score.percent_plain_mean, 4),
        "cuts_provisional": score.cuts_provisional,
        "caveats": list(score.caveats),
        "status": score.status,
        "population": score.population,
        "percent_tumour_only": score.percent_tumour_only,
        "intensity_tumour_only": score.intensity_tumour_only,
        "cells_tumour_only": score.cells_tumour_only,
        "status_reasons": list(score.status_reasons),
        # What the score was made with (P-15). `run_all.py` compares this against the
        # current code, settings and cut file before it trusts a `done` case.
        "provenance": report.provenance,
    }


def _archive(case_id: str, scored: list[str], log: pathlib.Path) -> list[str]:
    """File each scored marker into `v<N>_data/data/history`. Returns the ones that moved.

    `refresh` first: it is what turns the runs this process just performed - which the
    history screen has never seen, because nothing went through the UI - into manifest
    entries that `archive_marker` can act on. Without it the first archive raises
    "nothing is recorded for case ...".

    Each marker is archived on its own and a failure to move one does not stop the rest.
    The shared H&E work travels with the last marker of the case, so the order here is
    the order the moves happen in and the H&E is the tail of the final one.
    """
    from app.services import history_service

    moved: list[str] = []
    try:
        history_service.refresh(case_id)
    except Exception as failure:  # noqa: BLE001 - archiving must not lose the scores
        pipeline.say(f"  history refresh failed: {failure!r}", log)
        return moved

    for letter in scored:
        try:
            history_service.archive_marker(case_id, letter)
            moved.append(letter)
            pipeline.say(f"  history: filed {letter}", log)
        except Exception as failure:  # noqa: BLE001
            pipeline.say(f"  history: {letter} could not be filed - {failure!r}", log)
    return moved


def run(case_id: str) -> int:
    """Score one case's five markers, record them, file them. Returns an exit code."""
    pipeline.ensure_dirs()
    log = pipeline.LOGS / f"{case_id}.log"
    started = time.monotonic()

    case_path = pipeline.SLIDES / case_id
    if not case_path.is_dir():
        pipeline.say(f"{case_id}: no such case folder", log)
        return 2

    from app import panel
    from app.services import case_service
    import run_case_scores
    from run_case_scores import run_marker
    from app.services import ihc_alignment_service

    # Both switches are set from the one place that knows the mode. `run_marker` reads
    # the module attribute rather than taking a parameter, because it is also reached
    # from the API path where the default must stay refined.
    run_case_scores.SKIP_REFINEMENT = pipeline.MODE == "tiles"
    ihc_alignment_service.roi_source = "tiles" if pipeline.MODE == "tiles" else "refined"
    if run_case_scores.SKIP_REFINEMENT:
        pipeline.say(
            f"{case_id}: SKIPPING step 11 - scoring on step 9's tile squares. These "
            f"numbers are not comparable with the refined run's.",
            log,
        )

    resolution = case_service.resolve_case(str(case_path))
    pipeline.say(f"===== {case_id} =====", log)
    pipeline.say(f"  {case_path}", log)
    pipeline.say(f"  found {sorted(resolution.found)} missing {sorted(resolution.missing)}", log)

    if "HE" not in resolution.found:
        pipeline.say(f"{case_id}: no H&E in this folder - nothing can be scored", log)
        return 2

    wanted = [one for one in pipeline.MARKERS if one in resolution.found]
    record = {
        "caseId": resolution.case_id,
        "casePath": str(resolution.case_path),
        "startedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
        "markers": [],
    }

    # The H&E is registered once and every later marker is handed its upload id. Steps 2
    # to 9 - step 8 alone is tens of minutes - then run once for the case rather than
    # once per marker, which is the difference between a case taking an hour and five.
    shared_he: str | None = None

    for letter in wanted:
        name = panel.spec(letter).full_name
        pipeline.say("", log)
        pipeline.say(f"--- {case_id} {letter} ({name}) ---", log)

        entry = {"marker": letter, "marker_name": name, "state": "pending", "error": ""}
        run_result = None

        # Every marker is attempted. A rule here used to skip the rest of a case after two
        # registration refusals, on the reasoning that the H&E all five share was the
        # thing failing. That stopped being true when registration became per IHC slide:
        # a refusal now belongs to one section, not to the case, so two pale sections
        # would have dropped three markers that register (P-19).
        for attempt in (1, 2):
            try:
                run_result = run_marker(
                    resolution.case_path,
                    letter,
                    shared_he=shared_he,
                    log=log,
                    # The second attempt throws away the stored registration. See this
                    # module's docstring: a reused VALIS result is the failure this run
                    # is most likely to meet more than once, and repeating the identical
                    # call would reproduce it exactly.
                    restart_alignment=attempt == 2,
                )
            except Exception as failure:  # noqa: BLE001 - a marker must not end the case
                pipeline.say(f"{letter}: attempt {attempt} raised {failure!r}", log)
                pipeline.say(traceback.format_exc(), log)
                continue

            if run_result.state == "scored":
                break
            pipeline.say(
                f"{letter}: attempt {attempt} ended {run_result.state} "
                f"at {run_result.stage} - {run_result.error}",
                log,
            )

            # The retry exists to throw away a reused VALIS registration, and a
            # registration refusal is the one failure it cannot help with: the diagnostics
            # show fresh runs refusing exactly as reused ones do - 0 and 5 matched
            # features with `reusedRegistration: false` - so the second attempt recomputes
            # the same refusal for another 16 to 50 minutes. Retrying anything else is
            # still worth it.
            if run_result.state == "alignment_refused":
                pipeline.say(
                    f"{letter}: not retrying - the refusal is the registration itself, "
                    "not a stale one",
                    log,
                )
                break

        if run_result is not None:
            entry["state"] = run_result.state
            entry["error"] = run_result.error
            entry["seconds"] = run_result.seconds
            entry["he_upload_id"] = run_result.he_upload_id
            entry["ihc_upload_id"] = run_result.ihc_upload_id
            if shared_he is None and run_result.he_upload_id:
                shared_he = run_result.he_upload_id

            if run_result.state == "scored":
                try:
                    entry.update(_score_fields(run_result.he_upload_id, run_result.ihc_upload_id))
                    entry["scored_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    pipeline.say(
                        f"{letter}: SCORED {entry['percent']} % positive, "
                        f"intensity {entry['intensity']:g} ({entry['intensity_label']})",
                        log,
                    )
                except Exception as failure:  # noqa: BLE001
                    entry["state"] = "score-unreadable"
                    entry["error"] = f"{type(failure).__name__}: {failure}"
                    pipeline.say(f"{letter}: scored but unreadable - {entry['error']}", log)
        else:
            entry["state"] = "failed"
            entry["error"] = "both attempts raised before producing a run record"

        record["markers"].append(entry)

        # Written after every marker, not at the end of the case. A case is hours; a
        # crash in marker four must not lose the three that scored before it, and the
        # CSV is rebuilt from exactly this file.
        record["finishedAt"] = time.strftime("%Y-%m-%d %H:%M:%S")
        pipeline.save_result(case_id, record)
        pipeline.write_csv()

    scored = [one["marker"] for one in record["markers"] if one["state"] == "scored"]
    pipeline.say("", log)
    pipeline.say(f"{case_id}: {len(scored)} of {len(wanted)} markers scored", log)

    if scored:
        record["archived"] = _archive(resolution.case_id, scored, log)
        pipeline.save_result(case_id, record)

    record["seconds"] = round(time.monotonic() - started, 1)
    pipeline.save_result(case_id, record)
    pipeline.write_csv()
    pipeline.say(f"{case_id}: done in {record['seconds'] / 60:.1f} min", log)

    return 0 if scored else 1


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("usage: python run_case.py <CASE_ID>", file=sys.stderr)
        raise SystemExit(2)
    # argv[2] is the region source, defaulting to the refined (BEETLE) path.
    pipeline.configure(sys.argv[2] if len(sys.argv) > 2 else "refined")
    raise SystemExit(run(sys.argv[1]))
