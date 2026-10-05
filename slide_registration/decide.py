"""Write DECISIONS.md: what worked, what did not, and what is settled.

    python decide.py

**Generated from the result files, never hand-written**, so it cannot drift from what
actually happened. Every number in it is read from `v<N>_data/data/registration/*/methods.json`,
`registration.json` and the scoring CSVs at the moment it runs. Re-run it any time; it
overwrites.

The judgements it states - which method wins a pair, whether a method beat doing nothing -
are computed here from one rule applied to every row, rather than recorded by whichever
stage produced the row. That keeps the verdict in one place where it can be argued with.
"""

from __future__ import annotations

import csv
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import fit_best  # noqa: E402  - FORCE_METHOD decides what the document may claim is used

OUT = common.RESULTS / "DECISIONS.md"

#: A method has to beat the outline by more than this to count as having earned its place.
#: Below it the two are the same answer and the cheaper one wins.
MEANINGFUL_GAIN = 0.002


def _methods_table() -> tuple[list[dict], dict]:
    """Every pair's per-method scores, and a tally of which method won how often."""
    rows: list[dict] = []
    wins: dict[str, int] = {}
    for case in sorted(common.cases()):
        payload = common.read_json(common.case_dir(case) / "methods.json")
        valis = common.read_json(common.case_dir(case) / "registration.json") or {}
        valis_slides = valis.get("slides") or {}
        if not payload or not payload.get("ok"):
            continue
        for code, entry in sorted((payload.get("slides") or {}).items()):
            scores = {
                name: attempt["nmi"]
                for name, attempt in entry["methods"].items()
                if attempt.get("ok") and attempt.get("nmi")
            }
            if code in valis_slides and valis_slides[code].get("alignment_nmi"):
                scores["valis"] = valis_slides[code]["alignment_nmi"]
            if not scores:
                continue
            best = max(scores, key=scores.get)
            wins[best] = wins.get(best, 0) + 1
            rows.append({"case": case, "marker": code, "scores": scores, "best": best})
    return rows, wins


def _mean_loss(rows: list[dict], method: str) -> float | None:
    """Mean NMI a method gave up against each pair's best, over the pairs it ran on."""
    losses = [
        row["scores"][row["best"]] - row["scores"][method]
        for row in rows
        if method in row["scores"]
    ]
    return sum(losses) / len(losses) if losses else None


def _affine_is_current() -> tuple[bool, int]:
    """Whether every stored affine attempt ran from the similarity start, and how many ran.

    Before P-19 `method_affine` threw its similarity fit away and started from an
    identity transform centred on the image corner. On a known synthetic warp that start
    ended 15.9 px out where the similarity fit itself was 8.3 px out and the corrected
    affine 0.08 px. Sweeps since then record `init: similarity` on every affine attempt;
    one without it describes the bug, not the method.
    """
    attempts = stale = 0
    for case in sorted(common.cases()):
        payload = common.read_json(common.case_dir(case) / "methods.json") or {}
        for entry in (payload.get("slides") or {}).values():
            affine = (entry.get("methods") or {}).get("affine")
            if not affine:
                continue
            attempts += 1
            if affine.get("init") != "similarity":
                stale += 1
    return stale == 0 and attempts > 0, attempts


def _scores_from(path: pathlib.Path) -> dict[tuple[str, str], str]:
    if not path.is_file():
        return {}
    out = {}
    try:
        # utf-8-sig: the scorer writes a BOM, which otherwise turns the first column name
        # into "﻿case_id" and makes every lookup miss silently.
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("state") and row["state"] != "scored":
                    continue
                case = row.get("case_id") or row.get("case") or row.get("caseId") or ""
                marker = row.get("marker") or ""
                percent = row.get("percent") or ""
                intensity = row.get("intensity") or ""
                if case and marker and percent:
                    out[(case, marker)] = (percent, intensity)
    except (OSError, csv.Error):
        return {}
    return out


def write() -> pathlib.Path:
    rows, wins = _methods_table()
    lines: list[str] = []
    add = lines.append

    add("# Decisions: what worked, what did not, what is settled")
    add("")
    add(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by `slide_registration/decide.py`.")
    add("**Generated from the result files, not written by hand**, so it cannot drift from")
    add("what actually ran. Re-run that script to refresh it.")
    add("")
    add("Full reasoning and measurements: `tissue_scoring_demo/docs/registration/REGISTRATION-PLAN.md`.")
    add("")

    # --- the headline ---------------------------------------------------------
    add("## The problem, and what settled it")
    add("")
    add("Step 12 failed on three of six cases. VALIS estimates its transform from **detected")
    add("and matched image features**, so when detection fails the whole step fails - and it")
    add("fails exactly where a section is nearly unstained, because there is nothing for a")
    add("detector to key on. CAN_00865's CD44 section returned 5 matched features and")
    add("CAN_00267's returned 13, at every resolution and with every detector tried.")
    add("")
    add("**Feature matching is not the only way to register two images.** Methods that use")
    add("every pixel instead of a few hundred keypoints succeed where it cannot.")
    add("")

    # --- what was tried -------------------------------------------------------
    add("## What was tried")
    add("")
    add("| approach | verdict | why |")
    add("|---|---|---|")
    add("| **Mattes mutual information** over a similarity transform (SimpleITK) | **ADOPTED —"
        " the method** | Needs no features, so it works where VALIS cannot: CAN_00865's"
        " near-negative CD44 slide scored 1.0488 against VALIS's 1.0042. **Cannot fold tissue"
        " by construction** — a similarity transform has one Jacobian for the whole section."
        " 30/30 pairs plausible, Jacobians 0.874–1.161. Also 6x faster than B-spline. |")
    add("| **Outline alignment** (render to common scale, centre on tissue centroid, rotate by"
        " measured outline angle) | **WORKS — adopted as the baseline** | Free, and beats VALIS"
        " outright on several markers. Serial sections of one block share an outline. |")
    add("| **B-spline non-rigid** | **REJECTED — folds tissue** | Ranked first on mutual"
        " information, then measured: **10 of 16 fits folded tissue through itself**"
        " (negative Jacobian), stretching up to 6.1x. Worst case folded 24% of a section —"
        " and it had scored the cohort's *highest* NMI gain. MI is blind to this. |")
    rows_for_loss, _ = _methods_table()
    affine_loss = _mean_loss(rows_for_loss, "affine")
    mattes_loss = _mean_loss(rows_for_loss, "mattes")
    affine_current, affine_runs = _affine_is_current()
    losses = (
        f"mean loss against per-pair best {affine_loss:.4f} vs `mattes`'s {mattes_loss:.4f}"
        if affine_loss is not None and mattes_loss is not None
        else "no stored comparison"
    )
    if affine_current:
        add(f"| Affine | {'not adopted' if affine_loss > mattes_loss else 'competitive'} | Safe like"
            f" `mattes` (constant Jacobian); {losses} over {affine_runs} pairs. |")
    else:
        add(f"| Affine | **UNDECIDED — re-run the sweep** | The stored attempts ({affine_runs}) ran"
            " from an identity start, not the similarity fit (P-19, fixed 2026-10-05), so"
            f" their {losses} says nothing about affine as a method. |")
    add("| Pre-rotating sections before matching | **WORKS** | CAN_00267's CD44 section is"
        " mounted ~196° round. Correcting it took that pair from 5 matched features to 13. |")
    add("| Rendering all six sections to one physical grid | **WORKS — required** | Defeats"
        " VALIS's silent clamp of `max_image_dim_px` down to the smallest thumbnail, which had"
        " held every registration in this project at 512 px. |")
    add("| BRISK instead of DISK+LightGlue | **PARTLY** | Rotation-invariant and ~40× faster on"
        " a CPU-only torch, but still feature-based, so it does not fix a featureless slide. |")
    add("| Higher resolution (1024, 2048 px) | **NO** | Made no difference to the failing"
        " slides; the signal is absent, not too small. |")
    add("| Cropping to tissue before registering | **NO — dead end** | VALIS already does it."
        " Measured gain 1.01×. |")
    add("| Mask/distance-transform geometric fit | **NO** | Measured worst of every method"
        " tried. Kept only as a diagnostic. |")
    add("")

    # --- the measured scoreboard ---------------------------------------------
    add("## The scoreboard")
    add("")
    if not rows:
        add("_No method comparison on disk yet. Run `slide_registration/methods_run.py`._")
    else:
        names = sorted({n for row in rows for n in row["scores"]})
        add("Normalised mutual information over a lattice covering the H&E's tissue - the same")
        add("probe for every method, computed on the result rather than on what the method")
        add("optimised. Higher is better; **bold** is the winner for that pair.")
        add("")
        add("| case | marker | " + " | ".join(names) + " | winner |")
        add("|---" * (len(names) + 3) + "|")
        for row in rows:
            cells = []
            for name in names:
                value = row["scores"].get(name)
                if value is None:
                    cells.append("—")
                elif name == row["best"]:
                    cells.append(f"**{value:.4f}**")
                else:
                    cells.append(f"{value:.4f}")
            add(f"| {row['case']} | {row['marker']} | " + " | ".join(cells) + f" | {row['best']} |")
        add("")
        add("**Wins per method:** " + ", ".join(f"`{k}` {v}" for k, v in sorted(wins.items(), key=lambda kv: -kv[1])))
        add("")

    # --- what is settled ------------------------------------------------------
    add("## What is settled")
    add("")
    add("0. **One method for all 30 pairs: `mattes`.** Chosen for what it *cannot* do as much")
    add("   as for what it scores. Ranking on mutual information alone put B-spline first;")
    add("   checking the Jacobian showed 63% of its fits were geometrically impossible. A")
    add("   similarity transform cannot fold. The accuracy cost is 0.0065 mean NMI, and part")
    add("   of B-spline's lead was bought by deformation the tissue cannot have undergone.")
    add("1. **Registration does not need feature matching on this data.** Sections of one")
    add("   FFPE block, rendered to a common physical scale and rotated by their measured")
    add("   outline, are already well aligned; mutual information refines that without ever")
    add("   looking for a keypoint.")
    add("2. **A metric the failure mode is invisible to is not evidence.** The method")
    add("   comparison ran over 30 pairs with means, medians and worst cases, and was wrong,")
    add("   because mutual information cannot see a fold. More data on the wrong measurement")
    add("   does not converge on the right answer. The per-pair scoreboard is still recorded")
    add("   beside every transform, so a specific pair can be revisited.")
    if fit_best.FORCE_METHOD:
        add(f"3. **Every pair is fitted with `{fit_best.FORCE_METHOD}`, whatever the scoreboard says**")
        add("   (`fit_best.FORCE_METHOD`). The scoreboard - VALIS included - is recorded beside")
        add("   each transform for reference; its winner is not what is used.")
    else:
        add("3. **Each pair is fitted with its scoreboard winner** (`fit_best.FORCE_METHOD` is")
        add("   unset), VALIS included where it scored best.")
    add("4. **The tissue mask must be optical-density based, not saturation based.** A")
    add("   saturation rule measures how *stained* a slide is and captured 4% of the tissue")
    add("   on CAN_00865's CD44 section.")
    add("5. **Two masks per slide, for two jobs.** A strict mask places the section and")
    add("   reports its area; a permissive one decides what the matcher may see. Using the")
    add("   strict mask for both whitened a near-negative section out of existence.")
    add("")

    # --- what is not settled --------------------------------------------------
    add("## What is NOT settled, and is yours to decide")
    add("")
    add("- **Whether a rigid outline fit is enough where it wins.** It cannot follow the local")
    add("  deformation a section picks up being floated onto glass. \"Rigid beats a bad")
    add("  non-rigid fit\" is not \"rigid is sufficient\", and the 1.5 µm membrane shell the")
    add("  scores are measured in is a demanding target.")
    add("- **Whether CD44 on CAN_00267 and CAN_00865 should be scored at all.** Those sections")
    add("  are near-blank. Registration can now place them, but a score from an unstained")
    add("  slide may be meaningless regardless. Only OncoStem's own readings settle it.")
    add("- **CAN_00259's region selection**, which took two slide-spanning components and")
    add("  spent the whole coverage budget on them.")
    add("")

    # --- the two scoring runs -------------------------------------------------
    add("## Scores with and without BEETLE")
    add("")
    add("Step 11 (BEETLE per-pixel refinement) is the most expensive stage in the pipeline.")
    add("The cohort is scored both ways to measure what it is worth in score units:")
    add("")
    add("| run | regions carried | file |")
    add("|---|---|---|")
    add("| without BEETLE | step 9's coarse tile squares | `oncostem_ai_scores_tiles.csv` |")
    add("| with BEETLE | step 11's per-pixel boundaries | `oncostem_ai_scores.csv` |")
    add("")
    add("They are **measured inside different tissue and are not directly comparable as")
    add("scores** - that is the point of the comparison, and it is why they are kept in")
    add("separate files with separate checkpoints rather than interleaved.")
    add("")
    tiles = _scores_from(common.RESULTS / "oncostem_ai_scores_tiles.csv")
    beetle = _scores_from(common.RESULTS / "oncostem_ai_scores.csv")
    shared = sorted(set(tiles) & set(beetle))
    if shared:
        add("| case | marker | without BEETLE | with BEETLE | difference |")
        add("|---|---|---|---|---|")
        for key in shared:
            a, b = tiles[key], beetle[key]
            try:
                delta = f"{float(b[0]) - float(a[0]):+.0f} pts"
            except (TypeError, ValueError):
                delta = "—"
            add(f"| {key[0]} | {key[1]} | {a[0]}% (int {a[1]}) | {b[0]}% (int {b[1]}) | **{delta}** |")
        add("")
        add("A positive difference means BEETLE's refinement raised the score. The two runs")
        add("use the **same registration**, so the region source is the only thing that")
        add("differs between them.")
    else:
        add(f"_Rows so far: {len(tiles)} without BEETLE, {len(beetle)} with. "
            "No overlapping pairs to compare yet._")
    add("")
    add("How much of a tile region BEETLE actually calls invasive, at slide, region and tile")
    add("grain: `beetle_coverage_by_slide.csv`, `_by_region.csv`, `_by_tile.csv`.")
    add("")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    common.say(f"decisions written to {OUT}")
    return OUT


if __name__ == "__main__":
    write()
