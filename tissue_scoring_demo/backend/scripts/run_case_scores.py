"""Drive one case from a folder path all the way to its ten numbers, unattended.

    python scripts/run_case_scores.py <case-folder>
    python scripts/run_case_scores.py <case-folder> --markers A,F --excel out.xlsx

The whole flow the demo performs, for every IHC slide in the folder, without a
browser: load each marker's H&E + IHC pair, run steps 2-9 on the H&E, step 10 to
carry the invasive regions across, steps 11-13 to find and shape the cells, and
steps 14-16 to measure and score them. Then write one Excel workbook holding the
result.

**The H&E is registered once and reused across markers.** `case_service` gives
every `load_case` call a fresh upload id, which would mean steps 2-9 - step 8
alone is tens of minutes - running again for each of the five markers on the
same file. The five markers of a case share one H&E, so this script passes the
first one's id to the rest. Everything it computes lands exactly where the API
would put it, so the UI picks the results up rather than recomputing them.

**It confirms step 10 itself, and says so on every row.** The alignment gate
exists because a person is meant to look at the two panels and say the regions
landed on the same tissue. A batch job cannot do that. Refusing would mean no
unattended run could ever produce a score; confirming silently would leave a
record indistinguishable from a human sign-off. So it confirms with a machine
stamp, and step 16 turns that stamp into a caveat printed beside every number.

**Every stage is restartable and logs as it goes.** A five-marker run is hours,
so it writes a progress log after each stage and skips whatever is already on
disk. Killing it and starting it again resumes rather than repeats.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app import panel  # noqa: E402
from app.services import case_service, upload_service  # noqa: E402
from app.services.binning_service import binning_service  # noqa: E402
from app.services.cell_typing_service import cell_typing_service  # noqa: E402
from app.services.compartment_service import compartment_service  # noqa: E402
from app.services.ihc_alignment_service import ihc_alignment_service  # noqa: E402
from app.services.nuclei_service import nuclei_service  # noqa: E402
from app.services.per_cell_service import per_cell_service  # noqa: E402
from app.services.roi_refinement_service import roi_refinement_service  # noqa: E402
from app.services.roi_selection_service import roi_selection_service  # noqa: E402
from app.services.roi_service import roi_service  # noqa: E402
from app.services.score_service import score_service  # noqa: E402
from app.services.tissue_type_service import tissue_type_service  # noqa: E402


@dataclass
class MarkerRun:
    """What happened to one marker, stage by stage, for the log and the workbook."""

    marker: str
    marker_name: str
    he_upload_id: str = ""
    ihc_upload_id: str = ""
    state: str = "pending"
    stage: str = ""
    error: str = ""
    seconds: float = 0.0
    stages: dict[str, str] = field(default_factory=dict)


def _say(message: str, log: pathlib.Path | None = None) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    if log is not None:
        with log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _he_for(
    case_path: str, marker: str, shared_he: str | None, log: pathlib.Path | None = None
) -> case_service.CaseSession:
    """Load a marker's pair, reusing the case's already-registered H&E if there is one.

    `load_case` gives every new marker a freshly registered H&E upload id, and a
    fresh id means steps 2 to 9 - step 8 alone is tens of minutes - run again on
    a file they have already been run on. The five markers of a case share one
    H&E section, so once the first marker has established an id, later markers
    are pointed at the same one.

    **A marker that already has a sidecar keeps it, whatever the shared id is.**
    That is the important half. Redirecting an existing pair onto a different
    H&E id would orphan its step 10 registration - which is keyed on the pair -
    and force VALIS to register the same two slides again to reach the same
    answer. So the reuse applies only where this run is creating the pairing.
    """
    existing = case_service._load_sidecar(  # noqa: SLF001
        pathlib.Path(case_path).name, marker.upper()
    )
    if (
        existing
        and upload_service.local_slide_still_valid(existing.he_upload_id)
        and upload_service.local_slide_still_valid(existing.ihc_upload_id)
    ):
        _say(f"{marker}: reusing the pair already registered for this case", log)
        return existing

    if shared_he is None or not upload_service.local_slide_still_valid(shared_he):
        return case_service.load_case(case_path, marker)

    # New pairing, and we have an H&E already carrying steps 2-9. Register this
    # marker's IHC slide only, and write the sidecar against the shared H&E -
    # rather than calling `load_case`, which would register a second H&E record
    # pointing at the identical file and leave it orphaned.
    resolution = case_service.resolve_case(case_path)
    letter = marker.upper()
    if letter not in resolution.found:
        raise case_service.CaseError(
            f"no {panel.spec(letter).name} ({letter}) slide in {resolution.case_path}"
        )

    session = case_service.CaseSession(
        case_id=resolution.case_id,
        marker=letter,
        case_path=resolution.case_path,
        he_upload_id=shared_he,
        ihc_upload_id=upload_service.register_local_slide(resolution.found[letter]),
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    path = case_service._sidecar_path(session.case_id, session.marker)  # noqa: SLF001
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(session), indent=2), encoding="utf-8")
    _say(
        f"{marker}: paired with the H&E already carrying steps 2-9 ({shared_he}), "
        "so step 8 does not run again",
        log,
    )
    return session


#: Skip step 11 (BEETLE per-pixel refinement) and carry step 10's coarse tile regions
#: instead. Off by default, and it must stay off by default: a score built on tile squares
#: and one built on BEETLE's traced boundary are measured inside different tissue, and the
#: reports look identical.
#:
#: It exists to answer a question that cannot be answered any other way: step 11 is the
#: most expensive stage in the pipeline, and how much the score actually moves without it
#: is unknown. Scoring a cohort both ways measures that in score units rather than in area.
#: Set by `--roi-source tiles`; the two runs write to different CSVs and every region
#: carries its `source` through to the report.
SKIP_REFINEMENT = False


def run_marker(
    case_path: str,
    marker: str,
    *,
    shared_he: str | None,
    log: pathlib.Path,
    restart_alignment: bool = False,
) -> MarkerRun:
    """Steps 1-16 for one antibody. Skips whatever is already on disk."""
    spec = panel.spec(marker)
    run = MarkerRun(marker=marker, marker_name=spec.full_name)
    started = time.monotonic()

    try:
        run.stage = "load"
        session = _he_for(case_path, marker, shared_he, log)
        run.he_upload_id = session.he_upload_id
        run.ihc_upload_id = session.ihc_upload_id
        _say(f"{marker}: H&E {session.he_upload_id}  IHC {session.ihc_upload_id}", log)

        # --- steps 2-9, on the H&E -----------------------------------------
        # Asking step 9 for its regions drives whatever of the chain is not
        # already cached. Step 8 is the long one, tens of minutes.
        run.stage = "steps 2-9 (H&E)"
        _say(f"{marker}: steps 2-9 on the H&E (step 8 is the slow one)", log)
        stage_started = time.monotonic()
        tissue_type_service.start(session.he_upload_id)
        tissue_type_service.execute(session.he_upload_id)
        roi = roi_service.build(session.he_upload_id)
        run.stages["steps2-9"] = f"{time.monotonic() - stage_started:.0f}s"

        invasive = roi.class_regions.get("invasive_epithelium", [])
        _say(f"{marker}: step 9 found {len(invasive)} invasive region(s)", log)
        if not invasive:
            run.state = "no_invasive_regions"
            run.error = (
                "step 9 found no invasive epithelium on this case's H&E, so there is "
                "nothing to carry onto the IHC slide and nothing to score inside."
            )
            return run

        # --- ROI selection and refinement, on the H&E ------------------------
        # These two run once per case, on the shared H&E, and everything after
        # them measures inside what they produce.
        #
        # **They are not optional and there is no fallback past them.** The
        # alignment step used to choose its own regions off the class map; it now
        # reads step 11's traced foci and refuses outright if they are absent,
        # because the coarse squares are a different denominator and nothing
        # downstream could tell the difference. Without this block every marker of
        # every case fails at the registration with "there is no pixel-level
        # invasive mask for this H&E slide yet".
        #
        # **Nobody ticks the regions here, and the record says so.** Selection is
        # a screen a person drives. `selected()` is written for exactly this
        # case - it falls back to the coverage rule's own default rather than
        # raising, and stores `chosenByPerson: false` beside it - so an unattended
        # run refines the regions the pipeline would have chosen for itself and
        # stays distinguishable from one a person approved. That is the same
        # bargain the alignment gate below makes with its machine stamp.
        run.stage = "steps 10-11 (region selection and refinement)"
        if SKIP_REFINEMENT:
            _say(
                f"{marker}: choosing invasive regions - SKIPPING step 11, carrying step "
                f"9's tile squares (scores from this run are NOT comparable with refined "
                f"ones)",
                log,
            )
        else:
            _say(f"{marker}: choosing invasive regions and refining them per pixel", log)
        stage_started = time.monotonic()
        selection = roi_selection_service.build(session.he_upload_id)
        chosen = list(selection.selected)
        _say(
            f"{marker}: {len(selection.candidates)} candidate region(s), "
            f"{len(chosen)} selected "
            f"({'a person chose' if selection.chosen_by_person else 'default coverage rule'})",
            log,
        )
        if not chosen:
            run.state = "no_regions_selected"
            run.error = (
                "step 10 offered no region to refine on this case's H&E, so step 11 "
                "has nothing to trace and nothing downstream has a denominator."
            )
            _say(f"{marker}: {run.error}", log)
            return run

        if SKIP_REFINEMENT:
            # Step 12 is told, by name, to carry the coarse regions. Setting this here
            # rather than inside the service keeps the decision at the level that made it.
            ihc_alignment_service.roi_source = "tiles"
            run.stages["step10"] = (
                f"{time.monotonic() - stage_started:.0f}s, "
                f"{len(chosen)} tile region(s), refinement skipped"
            )
            _say(
                f"{marker}: {len(chosen)} tile region(s) selected, step 11 skipped",
                log,
            )
        else:
            ihc_alignment_service.roi_source = "refined"

        if not SKIP_REFINEMENT:
            refinement = roi_refinement_service.start(session.he_upload_id)
            if refinement.state in {"queued", "running"}:
                # Blocking, on this thread. The API hands this to a background worker
                # because a request cannot be held open for it; a batch has nothing
                # else to do and must not return before it is finished.
                roi_refinement_service.execute(session.he_upload_id)
            refined = roi_refinement_service.report(session.he_upload_id)
            run.stages["steps10-11"] = (
                f"{time.monotonic() - stage_started:.0f}s, "
                f"{refined.completed}/{refined.selected} refined, "
                f"{refined.refined_mm2:.2f} mm2 of {refined.tile_mm2:.2f} mm2 kept"
            )
            _say(
                f"{marker}: step 11 refined {refined.completed} of {refined.selected} "
                f"region(s) - {refined.refined_mm2:.2f} mm2 invasive inside "
                f"{refined.tile_mm2:.2f} mm2 of squares "
                f"({refined.kept_share * 100:.0f}% kept)",
                log,
            )
            if refined.completed == 0:
                run.state = "refinement_failed"
                run.error = (
                    "step 11 refined none of the selected regions, so there is no "
                    "pixel-level invasive mask to carry onto the IHC slide."
                )
                _say(f"{marker}: {run.error}", log)
                return run

        # --- step 12, the registration --------------------------------------
        run.stage = "step 12 (alignment)"
        _say(f"{marker}: step 12, registering the two slides", log)
        stage_started = time.monotonic()
        ihc_alignment_service.start(
            session.he_upload_id, session.ihc_upload_id, restart=restart_alignment
        )
        ihc_alignment_service.execute(session.he_upload_id, session.ihc_upload_id)
        alignment = ihc_alignment_service.report(session.he_upload_id, session.ihc_upload_id)
        run.stages["step10"] = f"{time.monotonic() - stage_started:.0f}s ({alignment.state})"

        if alignment.state != "ready":
            run.state = "alignment_refused"
            run.error = "; ".join(alignment.refusal_reasons) or f"state {alignment.state}"
            _say(f"{marker}: step 12 refused - {run.error}", log)
            return run

        if not alignment.confirmed:
            # The gate. See the module docstring for why this is stamped rather
            # than simply set.
            ihc_alignment_service.confirm(
                session.he_upload_id,
                session.ihc_upload_id,
                confirmed=True,
                by="machine",
            )
            _say(
                f"{marker}: step 12 confirmed BY MACHINE - no person has looked at the "
                "panels, and every score from this pair carries that caveat",
                log,
            )

        # --- steps 11-13, finding and shaping the cells ----------------------
        run.stage = "step 13 (nuclei)"
        _say(f"{marker}: step 13, segmenting nuclei", log)
        stage_started = time.monotonic()
        nuclei_service.start(session.he_upload_id, session.ihc_upload_id)
        nuclei_service.execute(session.he_upload_id, session.ihc_upload_id)
        state = nuclei_service.state(session.he_upload_id, session.ihc_upload_id)
        if state.state != "ready":
            run.state = "nuclei_failed"
            run.error = state.error or state.message or "step 11 did not finish"
            _say(f"{marker}: step 11 failed - {run.error}", log)
            return run
        nuclei = nuclei_service.report(session.he_upload_id, session.ihc_upload_id)
        run.stages["step11"] = (
            f"{time.monotonic() - stage_started:.0f}s, {nuclei.counted:,} nuclei"
        )
        _say(
            f"{marker}: step 11 counted {nuclei.counted:,} nuclei over "
            f"{nuclei.sampled_mm2:.3f} mm2 "
            f"({nuclei.density_per_mm2:,.0f}/mm2)",
            log,
        )

        run.stage = "step 14 (cell typing)"
        typed = cell_typing_service.report(session.he_upload_id, session.ihc_upload_id)
        run.stages["step12"] = f"{typed.tumour_share:.1%} tumour of {typed.counted:,}"
        _say(
            f"{marker}: step 12 typed {typed.counted:,} cells, "
            f"{typed.tumour_share:.1%} tumour",
            log,
        )

        run.stage = "step 15 (compartments)"
        compartments = compartment_service.report(
            session.he_upload_id, session.ihc_upload_id
        )
        run.stages["step13"] = (
            f"{compartments.cells:,} {compartments.compartment} compartments"
        )
        _say(
            f"{marker}: step 13 built {compartments.cells:,} "
            f"{compartments.compartment} compartments",
            log,
        )

        # --- steps 14-16, the measurement and the score ----------------------
        run.stage = "step 16 (per-cell measurement)"
        stage_started = time.monotonic()
        measured = per_cell_service.report(session.he_upload_id, session.ihc_upload_id)
        run.stages["step14"] = (
            f"{time.monotonic() - stage_started:.0f}s, {measured.cells:,} cells"
        )
        _say(
            f"{marker}: step 16 measured {measured.cells:,} cells, "
            f"mean DAB {measured.mean_od:.4f} OD, "
            f"mean {measured.second_measure} {measured.mean_second:.3f}",
            log,
        )

        run.stage = "step 17 (intensity binning)"
        binned = binning_service.report(session.he_upload_id, session.ihc_upload_id)
        run.stages["step15"] = " / ".join(
            f"{entry.label}:{entry.count}" for entry in binned.bins
        )
        _say(f"{marker}: step 17 bins {run.stages['step15']}", log)

        run.stage = "step 18 (aggregate)"
        scored = score_service.report(session.he_upload_id, session.ihc_upload_id).score
        run.stages["step16"] = f"{scored.percent}% / {scored.intensity:g}"
        _say(
            f"{marker}: STEP 18 -> {scored.percent} % positive, intensity "
            f"{scored.intensity:g} ({scored.intensity_label})   "
            f"[raw {scored.percent_raw:.2f} % / {scored.intensity_raw:.4f} OD]",
            log,
        )
        for caveat in scored.caveats:
            _say(f"{marker}:   caveat: {caveat}", log)

        run.state = "scored"
        run.stage = "done"
    except Exception as failure:  # noqa: BLE001 - one marker must not end the run
        run.state = "failed"
        run.error = f"{type(failure).__name__}: {failure}"
        _say(f"{marker}: FAILED during {run.stage} - {run.error}", log)
        _say(traceback.format_exc(), log)
    finally:
        run.seconds = round(time.monotonic() - started, 1)

    return run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_path", help="folder holding the case's .svs files")
    parser.add_argument(
        "--markers",
        default="",
        help="comma-separated letters; default is every scored marker in the folder",
    )
    parser.add_argument(
        "--excel",
        default="",
        help="where to write the workbook; default <repo root>/<case>_scores.xlsx",
    )
    parser.add_argument(
        "--log",
        default="",
        help="progress log; default <repo root>/SCORING_RUN_LOG.md",
    )
    parser.add_argument("--restart-alignment", action="store_true")
    parser.add_argument(
        "--roi-source",
        choices=("refined", "tiles"),
        default="refined",
        help=(
            "refined: step 11's BEETLE pixel boundaries (default). "
            "tiles: step 9's coarse squares, skipping step 11 entirely. The two are "
            "measured inside different tissue and their scores are not comparable."
        ),
    )
    args = parser.parse_args()

    global SKIP_REFINEMENT
    SKIP_REFINEMENT = args.roi_source == "tiles"
    ihc_alignment_service.roi_source = args.roi_source

    root = pathlib.Path(__file__).resolve().parents[3]
    log = pathlib.Path(args.log) if args.log else root / "SCORING_RUN_LOG.md"
    log.parent.mkdir(parents=True, exist_ok=True)

    resolution = case_service.resolve_case(args.case_path)
    _say(f"case {resolution.case_id} at {resolution.case_path}", log)
    _say(f"  found {sorted(resolution.found)}, missing {sorted(resolution.missing)}", log)

    if args.markers:
        wanted = [letter.strip().upper() for letter in args.markers.split(",") if letter.strip()]
    else:
        wanted = [
            letter for letter in panel.SCORED_MARKERS if letter in resolution.found
        ]
    _say(f"  scoring {wanted}", log)

    if "HE" not in resolution.found:
        _say("  no H&E slide in this folder; every step from 2 to 10 runs on it", log)
        return 1

    runs: list[MarkerRun] = []
    shared_he: str | None = None
    for letter in wanted:
        _say("", log)
        _say(f"===== {letter} ({panel.spec(letter).full_name}) =====", log)
        run = run_marker(
            resolution.case_path,
            letter,
            shared_he=shared_he,
            log=log,
            restart_alignment=args.restart_alignment,
        )
        runs.append(run)
        if shared_he is None and run.he_upload_id:
            shared_he = run.he_upload_id
        _say(f"{letter}: {run.state} in {run.seconds:.0f}s", log)

    _say("", log)
    _say("===== summary =====", log)
    for run in runs:
        _say(
            f"  {run.marker} {run.marker_name:22s} {run.state:22s} "
            f"{run.stages.get('step16', '-')}",
            log,
        )

    from app.reporting.workbook import write_case_workbook

    destination = (
        pathlib.Path(args.excel)
        if args.excel
        else root / f"{resolution.case_id}_scores.xlsx"
    )
    written = write_case_workbook(
        resolution.case_id,
        destination,
        runs=[asdict(run) for run in runs],
        case_path=resolution.case_path,
    )
    _say(f"workbook: {written}", log)

    return 0 if any(run.state == "scored" for run in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
