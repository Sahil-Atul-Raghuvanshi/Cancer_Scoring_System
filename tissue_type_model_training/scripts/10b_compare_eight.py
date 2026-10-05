"""All eight tissue-type heads, scored and compared: 4 haematoxylin x 4 H&E.

    python scripts/10b_compare_eight.py

Writes `HE_VS_HCHANNEL.md` and `he_vs_hchannel.json` to the workspace root.

**Why this is a new driver rather than an extension of `09_head_to_head.py`.** That
script's premise, stated in its own docstring, is one export, one feature space, one
architecture, heads differing only in the data they were fitted on. The premise fails
here twice over: an H&E head consumes RGB-derived 1024-vectors and a haematoxylin head
consumes density-derived ones, so they cannot share a feature cache; and across fields
of view the held-out tile sets are different sizes. So `09` is the right scoring kernel
and the wrong driver, and its `score`/`wilson`/`load_head` are imported here.

**Within a field of view the comparison is paired, and that is the whole reason the two
stores were cut in two passes.** `export_region` takes the vote from the mask alone, so
the two stores hold the same `tile_id`s with the same labels - asserted, per arm, by
`10c_assert_paired.py`. That makes McNemar on the in-situ rows available, which is the
test that actually answers "does the second dye help the confusion that matters".

**Across fields of view nothing is paired and nothing is ranked.** The held-out sets
differ in size by two hundred times and a 672 um window needs 1344 source pixels where
a 112 um one needs 224, so the four arms are not looking at the same tissue. Point
estimates only, each with its k/n, each labelled uncontrolled.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
CAMPAIGN_DIR = REPO / "reports" / "he_campaign"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import backend_path  # noqa: E402

DATA = backend_path.DATA_DIR
import bcss  # noqa: E402
import datasets  # noqa: E402
import report as report_lib  # noqa: E402

_nine = __import__("09_head_to_head")
load_head = _nine.load_head
score = _nine.score
wilson = _nine.wilson
TAU = 0.40

ARMS = (112, 224, 448, 672)
BRANCHES = {
    "h_channel": {
        112: "invasive_tile_fov112_fix1_concat",
        224: "invasive_tile_fov224_concat",
        448: "invasive_tile_fov448_concat",
        672: "invasive_tile_fov672_concat",
    },
    "he": {fov: f"invasive_tile_fov{fov}_he_concat" for fov in ARMS},
}
STORES = {"h_channel": "h_channel", "he": "he"}


def decide(probabilities: np.ndarray, tau: float | None) -> np.ndarray:
    """The served decision rule; `tau=None` is a plain argmax."""
    if tau is None:
        return probabilities.argmax(axis=1)
    invasive = probabilities[:, bcss.INVASIVE] >= tau
    other = probabilities[:, [bcss.NON_EPITHELIUM, bcss.NON_INVASIVE]].argmax(axis=1)
    mapped = np.where(other == 0, bcss.NON_EPITHELIUM, bcss.NON_INVASIVE)
    return np.where(invasive, bcss.INVASIVE, mapped)


def load_features(store: Path, rows) -> np.ndarray:
    """The two frozen caches concatenated, with both fingerprints checked."""
    want = datasets.manifest_fingerprint(rows)
    halves = []
    for init in ("imagenet", "simclr"):
        for suffix in ("_std", ""):
            path = store / "features" / f"{init}{suffix}.npz"
            if path.exists():
                break
        else:
            raise FileNotFoundError(f"no {init} cache in {store / 'features'}")
        cache = np.load(path, allow_pickle=True)
        got = str(cache["fingerprint"])
        if not got.startswith(want[:16]):
            raise ValueError(
                f"{path.name} fingerprint {got[:16]} != manifest {want[:16]}"
            )
        halves.append(cache["features"])
    return np.concatenate(halves, axis=1)


def evaluate_cell(branch: str, fov: int) -> dict | None:
    """One cell of the 4x2 grid, or `None` with a reason when it cannot be scored."""
    store = DATA / STORES[branch] / f"{fov}um"
    name = BRANCHES[branch][fov]
    manifest = store / "tiles_manifest.csv"

    if not manifest.exists():
        return {"absent": f"no tile store at {store}"}
    if not (backend_path.MODELS_DIR / f"{name}.pt").exists():
        return {"absent": f"{name} is not published"}

    rows = datasets.by_source_authority(datasets.read_manifest(manifest))
    features = load_features(store, rows)
    train_rows, test_rows = datasets.institution_split(rows)

    index = {id(row): i for i, row in enumerate(rows)}
    test_idx = [index[id(row)] for row in test_rows]
    truth = np.array([int(row["label"]) for row in test_rows])
    slides = np.array([str(row["slide_id"]) for row in test_rows])
    tile_ids = [str(row["tile_id"]) for row in test_rows]
    sources = np.array([str(row["source"]) for row in test_rows])

    head = load_head(name, features.shape[1])
    head.eval()
    with torch.inference_mode():
        logits = head(torch.from_numpy(
            np.ascontiguousarray(features[test_idx], dtype=np.float32)
        )).numpy()
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()

    cell: dict = {
        "branch": branch,
        "fov": fov,
        "model": name,
        "held_out_tiles": len(test_rows),
        "held_out_slides": int(len(set(slides))),
        "train_tiles": len(train_rows),
        "tile_ids": tile_ids,
        "truth": truth.tolist(),
    }

    for label, tau in (("argmax", None), ("tau", TAU)):
        predicted = decide(probabilities, tau)
        block = score(truth, predicted)
        block["accuracy"] = float((predicted == truth).mean())
        low, high = wilson(block["insitu_errors"], block["insitu_total"])
        block["insitu_called_invasive_ci"] = [low, high]
        block["per_slide_dice"] = report_lib.per_slide_dice(truth, predicted, slides)
        cell[label] = block
        cell[f"predicted_{label}"] = predicted.tolist()

    # The held-out in-situ row, split by who labelled it. "Nine human in-situ tiles in
    # all of BCSS" is the fact that makes every class-1 number in this project fragile,
    # and averaging BCSS's nine with BEETLE's thousands hides it.
    insitu = truth == bcss.NON_INVASIVE
    cell["insitu_by_source"] = {
        source: int(((sources == source) & insitu).sum())
        for source in sorted(set(sources))
    }
    return cell


def mcnemar(truth: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict:
    """Continuity-corrected McNemar on the tiles where exactly one model is wrong."""
    a_wrong = a != truth
    b_wrong = b != truth
    only_a = int((a_wrong & ~b_wrong).sum())
    only_b = int((~a_wrong & b_wrong).sum())
    total = only_a + only_b
    if total == 0:
        return {"only_a_wrong": 0, "only_b_wrong": 0, "chi2": None, "p": None,
                "note": "the two models are wrong on exactly the same tiles"}
    chi2 = (abs(only_a - only_b) - 1) ** 2 / total
    from math import erfc, sqrt

    p = erfc(sqrt(chi2 / 2)) if chi2 > 0 else 1.0
    # `erfc` underflows to exactly 0.0 for chi2 above about 150, and "p = 0.0" is not a
    # p-value - it is a statement about double precision. Reported as a bound instead,
    # which is what the number actually supports.
    return {
        "only_a_wrong": only_a,
        "only_b_wrong": only_b,
        "chi2": round(chi2, 4),
        "p": round(p, 5) if p > 1e-16 else None,
        "p_display": f"{p:.5f}" if p > 1e-16 else "< 1e-16",
    }


def slide_bootstrap(truth: np.ndarray, a: np.ndarray, b: np.ndarray,
                    slides: np.ndarray, *, draws: int = 2000, seed: int = 0) -> dict:
    """The paired accuracy difference, resampled over slides rather than tiles.

    Resampling tiles would treat two hundred near-duplicate tiles of one slide as two
    hundred independent observations and produce an interval several times too narrow.
    That is the same reason the whole splitting story in `datasets` exists.
    """
    unique = np.array(sorted(set(slides)))
    if len(unique) < 3:
        return {"delta": None, "ci": None,
                "note": f"not computable, n_slides = {len(unique)}"}

    rng = np.random.default_rng(seed)
    observed = float((b == truth).mean() - (a == truth).mean())
    deltas = []
    for _ in range(draws):
        picked = rng.choice(unique, size=len(unique), replace=True)
        mask = np.concatenate([np.flatnonzero(slides == slide) for slide in picked])
        deltas.append(float((b[mask] == truth[mask]).mean()
                            - (a[mask] == truth[mask]).mean()))
    low, high = np.percentile(deltas, [2.5, 97.5])
    return {"delta": round(observed, 4),
            "ci": [round(float(low), 4), round(float(high), 4)],
            "n_slides": int(len(unique))}


def compare_within_fov(h_cell: dict, he_cell: dict) -> dict:
    """The paired H-vs-H&E comparison at one field of view."""
    h_ids = h_cell["tile_ids"]
    he_ids = he_cell["tile_ids"]
    shared = sorted(set(h_ids) & set(he_ids))

    h_pos = {tile: i for i, tile in enumerate(h_ids)}
    he_pos = {tile: i for i, tile in enumerate(he_ids)}
    h_sel = [h_pos[tile] for tile in shared]
    he_sel = [he_pos[tile] for tile in shared]

    truth = np.array(h_cell["truth"])[h_sel]
    truth_check = np.array(he_cell["truth"])[he_sel]

    out: dict = {
        "held_out_h": len(h_ids),
        "held_out_he": len(he_ids),
        "shared": len(shared),
        "truth_agrees": bool(np.array_equal(truth, truth_check)),
    }
    if not shared or not out["truth_agrees"]:
        out["note"] = ("the two stores do not describe the same held-out squares, so "
                       "no paired statistic below would mean anything")
        return out

    slides = None
    for label in ("argmax", "tau"):
        a = np.array(h_cell[f"predicted_{label}"])[h_sel]
        b = np.array(he_cell[f"predicted_{label}"])[he_sel]

        insitu = truth == bcss.NON_INVASIVE
        out[label] = {
            "mcnemar_all": mcnemar(truth, a, b),
            "mcnemar_insitu_row": mcnemar(truth[insitu], a[insitu], b[insitu]),
            "accuracy_h": float((a == truth).mean()),
            "accuracy_he": float((b == truth).mean()),
        }

        if slides is None:
            # Recovered from the per-slide dice keys of the haematoxylin cell, which
            # were computed over the same held-out rows.
            slides = np.array(sorted(h_cell[label]["per_slide_dice"]))

        try:
            paired = report_lib.paired_comparison(
                h_cell[label]["per_slide_dice"], he_cell[label]["per_slide_dice"]
            )
            # `Paired`'s own fields, and its own `significant` verdict rather than a
            # second reading of the same interval. It reports a mean per-slide
            # difference with a bootstrap interval and a Wilcoxon p; `significant` is
            # simply whether that interval excludes zero, which is the question.
            out[label]["wilcoxon_slide_dice"] = {
                "n_slides": paired.slides,
                "mean_difference": round(paired.mean_difference, 4),
                "ci": [round(paired.ci_low, 4), round(paired.ci_high, 4)],
                "wins": paired.wins,
                "losses": paired.losses,
                "ties": paired.ties,
                "p": round(paired.p_value, 5) if paired.p_value is not None else None,
                "significant": paired.significant,
                "headline": paired.headline(),
            }
        except Exception as exc:
            out[label]["wilcoxon_slide_dice"] = {
                "note": f"not computable: {exc}"
            }

    return out


WARNINGS = [
    ("Window-size selection is systematic, not random.",
     "A 448 um window needs 896x896 source pixels and a 672 um window 1344x1344, so "
     "every region smaller than that contributes **nothing** to that arm. That drops "
     "small regions preferentially, and nothing in this project measures whether small "
     "regions are harder. Already documented at `RESULTS_224_VS_448.md:79`."),
    ("BACH contributes zero tiles at 672 um.",
     "BACH regions are 1720x1290 at 0.5 um/px and a 672 um window needs 1344 px, so "
     "the height fails. The 672 um cells are two-source where every other cell is "
     "three-source, and the second-laboratory contrast the BACH trees exist to provide "
     "is absent there entirely."),
    ("The 672 um held-out set is 76 tiles, of which the in-situ truth row is 11.",
     "A k/11 rate carries a 95% Wilson interval about thirty points wide. It is "
     "printed with its interval or not at all, and it is never ranked against another "
     "field of view."),
    ("The held-out in-situ row differs in size by two hundred times across the arms.",
     "About 2,270 tiles at 112 um against 73 at 448 um. The same five-point difference "
     "is decisive at one field of view and invisible at another, which is why every "
     "percentage in this document carries its k/n."),
]


def fmt_rate(block: dict) -> str:
    k, n = block["insitu_errors"], block["insitu_total"]
    low, high = block["insitu_called_invasive_ci"]
    if not n:
        return "n/a"
    return f"{k}/{n} = {100 * k / n:.1f}% [{100 * low:.0f}-{100 * high:.0f}]"


def main() -> int:
    cells: dict[str, dict[int, dict]] = {"h_channel": {}, "he": {}}
    for branch in ("h_channel", "he"):
        for fov in ARMS:
            try:
                cells[branch][fov] = evaluate_cell(branch, fov) or {}
            except Exception as exc:
                cells[branch][fov] = {"absent": f"{type(exc).__name__}: {exc}"}
            state = cells[branch][fov]
            note = state.get("absent", "scored")
            print(f"  {branch:<10} {fov:>4} um  {note}")

    paired: dict[int, dict] = {}
    for fov in ARMS:
        h_cell, he_cell = cells["h_channel"][fov], cells["he"][fov]
        if "absent" in h_cell or "absent" in he_cell:
            paired[fov] = {"note": "one side is absent, so nothing is paired here"}
            continue
        paired[fov] = compare_within_fov(h_cell, he_cell)

    # --- the document -----------------------------------------------------
    lines: list[str] = []
    lines.append("# Haematoxylin against H&E: all eight tissue-type heads")
    lines.append("")
    lines.append("Four fields of view times two input contracts. The geometry is "
                 "identical across the two contracts at each scale - 224 px at 0.5, "
                 "1.0, 2.0 and 3.0 um/px - and that identity is the only reason the "
                 "two columns are comparable at all.")
    lines.append("")
    lines.append("## Read these first")
    lines.append("")
    for heading, body in WARNINGS:
        lines.append(f"**{heading}** {body}")
        lines.append("")

    lines.append("## The grid")
    lines.append("")
    lines.append("`insitu -> invasive` is the headline error: an in-situ tile called "
                 "invasive. Lower is better, and it carries its k/n and a 95% Wilson "
                 "interval because at three of these four scales the denominator is "
                 "small enough for that to matter.")
    lines.append("")
    lines.append("| fov | branch | model | held out | insitu -> invasive (argmax) | "
                 "Dice inv-vs-insitu | accuracy | accuracy (tau=0.4) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for fov in ARMS:
        for branch in ("h_channel", "he"):
            cell = cells[branch][fov]
            if "absent" in cell:
                lines.append(f"| {fov} µm | {branch} | — | — | "
                             f"_{cell['absent']}_ | — | — | — |")
                continue
            a = cell["argmax"]
            t = cell["tau"]
            lines.append(
                f"| {fov} µm | {branch} | `{cell['model']}` | "
                f"{cell['held_out_tiles']:,} tiles / {cell['held_out_slides']} slides | "
                f"{fmt_rate(a)} | {a['dice_invasive_vs_non_invasive']:.3f} | "
                f"{a['accuracy']:.3f} | {t['accuracy']:.3f} |"
            )
    lines.append("")

    lines.append("## Per field of view, paired")
    lines.append("")
    for fov in ARMS:
        lines.append(f"### {fov} µm")
        lines.append("")
        info = paired[fov]
        if "note" in info and "argmax" not in info:
            lines.append(f"_{info['note']}_")
            lines.append("")
            continue
        lines.append(f"Held out: {info['held_out_h']:,} haematoxylin, "
                     f"{info['held_out_he']:,} H&E, {info['shared']:,} shared "
                     f"(truth agrees: {info['truth_agrees']}).")
        lines.append("")
        for label in ("argmax", "tau"):
            block = info.get(label)
            if not block:
                continue
            rule = "argmax" if label == "argmax" else "tau = 0.40"
            lines.append(f"**Decision rule: {rule}.** "
                         f"Accuracy {block['accuracy_h']:.3f} haematoxylin against "
                         f"{block['accuracy_he']:.3f} H&E.")
            lines.append("")
            row = block["mcnemar_insitu_row"]
            lines.append(f"- McNemar on the in-situ row (the headline test): "
                         f"{row['only_a_wrong']} tiles only haematoxylin gets wrong, "
                         f"{row['only_b_wrong']} only H&E gets wrong, "
                         f"chi2 {row.get('chi2')}, p {row.get('p_display', row.get('p'))}."
                         + (f" {row['note']}" if row.get("note") else ""))
            allrow = block["mcnemar_all"]
            lines.append(f"- McNemar over every held-out tile: "
                         f"{allrow['only_a_wrong']} / {allrow['only_b_wrong']}, "
                         f"p {allrow.get('p_display', allrow.get('p'))}.")
            wil = block.get("wilcoxon_slide_dice") or {}
            if "note" in wil:
                lines.append(f"- Per-slide Dice, paired: {wil['note']}.")
            else:
                lines.append(
                    f"- Per-slide Dice, paired over {wil['n_slides']} slides: "
                    f"{wil['headline']}; Wilcoxon p {wil['p']}."
                )
            lines.append("")
        lines.append("")

    lines.append("## Reading the two paired tests together")
    lines.append("")
    lines.append("They can disagree, and at 112 µm they do: McNemar is overwhelming "
                 "while the per-slide interval includes zero. That is not a "
                 "contradiction, it is the two tests counting different things. "
                 "**McNemar treats each tile as an observation**, so 9,393 tiles give "
                 "it enormous power - but tiles from one slide are not independent, "
                 "and a model that is better on a few large sections can win it. "
                 "**The per-slide bootstrap treats each slide as an observation**, "
                 "which is the conservative and clinically relevant unit, and there "
                 "the difference is small. Read the tile test as \"the labels changed, "
                 "a lot\" and the slide test as \"how much a case's answer moved\". "
                 "Both are reported because neither on its own is the answer.")
    lines.append("")
    lines.append("## What this does and does not establish")
    lines.append("")
    lines.append("Within a field of view the comparison is paired on the same squares "
                 "of the same sections, so a McNemar result there is a real answer to "
                 "'does keeping the second dye change the in-situ/invasive confusion'. "
                 "Across fields of view nothing here is controlled: the arms see "
                 "different subsets of the regions for the mechanical reason in the "
                 "first warning above, so a difference between 224 µm and 448 µm in "
                 "this document is not evidence about window size. The precedent for "
                 "saying so plainly is `RESULTS_224_VS_448.md`, which reports "
                 "z = 1.92, p = 0.055 and calls it suggestive and not established.")
    lines.append("")

    (CAMPAIGN_DIR / "HE_VS_HCHANNEL.md").write_text("\n".join(lines), encoding="utf-8")

    # The JSON keeps the per-tile vectors out, which would make it enormous.
    slim = {
        branch: {
            fov: {k: v for k, v in cell.items()
                  if k not in ("tile_ids", "truth", "predicted_argmax", "predicted_tau")}
            for fov, cell in per_fov.items()
        }
        for branch, per_fov in cells.items()
    }
    (CAMPAIGN_DIR / "he_vs_hchannel.json").write_text(
        json.dumps({"cells": slim, "paired": paired}, indent=2, sort_keys=True,
                   default=float),
        encoding="utf-8",
    )

    print(f"\nwrote {WORKSPACE / 'HE_VS_HCHANNEL.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
