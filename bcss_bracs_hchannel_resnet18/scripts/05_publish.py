"""Gate G6: publish the checkpoint to models/tissue_type/, pin it, and prove it reloads.

    python scripts/05_publish.py --init imagenet
    python scripts/05_publish.py --init simclr
    python scripts/05_publish.py --both            # publish both arms of the A/B

Two artefacts per model: `invasive_tile_v1_<init>.pt` and its `manifest.json`. The
manifest is not documentation - it is the input contract, the metrics, the provenance
and a set of probe logits, and `models.load_pinned` refuses a checkpoint whose SHA-256
disagrees with it.

**The probe check is the most valuable test in the plan.** A fresh subprocess reloads
the checkpoint, runs twenty named tiles and must reproduce the logits recorded here. It
is the only test that catches a normalisation mismatch between training and serving,
because inside the training process both sides use the same wrong transform.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import numpy as np
import torch

import backend_path  # noqa: F401
import bcss
import datasets
import models
import train

LICENCES = {
    "imagenet": {"BCSS": "CC0 1.0", "torchvision resnet18": "BSD-3"},
    "simclr": {"BCSS": "CC0 1.0", "torchvision resnet18": "BSD-3",
               "ciga simclr resnet18": "MIT"},
}

#: What the borrowed in-situ class drags in with it. Merged into the manifest's licence
#: block whenever the export contains DCIS tiles, rather than being a separate note
#: someone has to remember to read - the licences dict is what a reader checks before
#: shipping, so the restriction has to be *in* it.
DCIS_LICENCES = {
    "BEETLE nnU-Net (labels for data/dcis)": "CC BY-NC-SA 4.0 - NON-COMMERCIAL",
    "BRACS (images behind data/dcis)": "non-commercial, research only",
}


def licences_for(init: str, export_summary: dict) -> dict:
    """The licence block, with the borrowed class's terms folded in when present."""
    block = dict(LICENCES[init])
    if export_summary.get("dcis"):
        block.update(DCIS_LICENCES)
        block["EFFECTIVE"] = (
            "RESEARCH ONLY - the most restrictive term governs. Do not ship this "
            "checkpoint; the permissive one is trained without --include-dcis."
        )
    return block

INIT_NAMES = {"imagenet": "imagenet1k_v1", "simclr": "simclr_ciga_tenpercent"}


def probe_rows(test_rows: list[dict], *, per_class: int = 7) -> list[dict]:
    """A stratified handful of held-out tiles, so the probe covers all three classes.

    Twenty tiles of stroma would reproduce perfectly and prove almost nothing.
    """
    rng = np.random.default_rng(0)
    picked: list[dict] = []
    for cls in range(len(bcss.CLASS_NAMES)):
        pool = [row for row in test_rows if int(row["label"]) == cls]
        if not pool:
            continue
        for index in rng.choice(len(pool), size=min(per_class, len(pool)), replace=False):
            picked.append(pool[int(index)])
    return picked


def publish_one(
    init: str,
    *,
    invert: bool,
    standardise: bool = False,
    all_sources: bool = False,
    version: str = "v1",
) -> dict:
    suffix = (f"{init}{'_inverted' if invert else ''}"
              f"{'_std' if standardise else ''}")
    report_path = backend_path.REPORTS_DIR / f"04_report_{suffix}.json"
    head_path = backend_path.REPORTS_DIR / f"04_head_{suffix}.pt"
    if not report_path.exists() or not head_path.exists():
        raise FileNotFoundError(
            f"{report_path.name} or {head_path.name} is missing - run "
            f"scripts/04_train.py --init {init} first."
        )

    metrics = json.loads(report_path.read_text(encoding="utf-8"))
    export_summary = json.loads(
        (backend_path.TILES_DIR / "export_summary.json").read_text(encoding="utf-8")
    )
    download = json.loads(
        (backend_path.BCSS_DIR / "download_manifest.json").read_text(encoding="utf-8")
    )

    # Assemble the served model: the frozen body plus the head that produced the report.
    # Assembled rather than re-fitted, so the published weights are exactly the ones
    # the numbers describe.
    net = models.resnet18_backbone(init, weights_dir=backend_path.PRETRAINED_DIR)
    models.freeze_body(net)
    net.fc.load_state_dict(torch.load(head_path, weights_only=True))
    net.eval()

    rows = datasets.read_manifest(backend_path.TILES_DIR / "tiles_manifest.csv")
    # The same filter 03 and 04 applied, or the probe tiles - the ones the manifest
    # records logits for, and the ones gate G6 replays - could be drawn from tiles this
    # model was never fitted on. G6 would still pass, because it only checks that the
    # transform reproduces the recorded numbers; it would just be checking the wrong
    # tiles, which is the sort of quietly-useless gate this project exists to avoid.
    if not all_sources:
        rows = datasets.by_source_authority(rows)
    _, test_rows = datasets.institution_split(rows)
    picked = probe_rows(test_rows)

    probe_set = datasets.TileDataset(picked, backend_path.TILES_DIR,
                                     augment=False, invert=invert,
                                     standardise=standardise)
    with torch.inference_mode():
        batch = torch.stack([probe_set[i][0] for i in range(len(probe_set))])
        logits = net(batch).numpy()

    correct = int(sum(1 for i, row in enumerate(picked)
                      if int(np.argmax(logits[i])) == int(row["label"])))
    print(f"  {len(picked)} probe tiles, {correct} predicted correctly")

    spec = export_summary["spec"]
    checkpoint, manifest_path = models.publish(
        net,
        name=f"invasive_tile_{version}_{suffix}",
        models_dir=backend_path.MODELS_DIR,
        init=INIT_NAMES[init],
        invert_polarity=invert,
        standardise=standardise,
        tile_px=int(spec["tile_px"]),
        mpp=float(spec["mpp"]),
        metrics=metrics,
        training_data={
            **training_data_block(export_summary, download),
            # Provenance, not decoration: which source was allowed to teach which
            # class is the single biggest decision behind this checkpoint, and a
            # reader comparing two of them needs it beside the metrics rather than
            # in a commit message.
            "source_authority": (
                {source: sorted(classes) for source, classes in datasets.SOURCE_CLASSES.items()}
                if not all_sources
                else "every source contributed every class"
            ),
            "source_authority_note": (
                "BRACS regions were selected for being DCIS-heavy, so its invasive "
                "and non-epithelium tiles are a side effect of that selection - 270 "
                "and 662 tiles guessed by BEETLE, against 5,282 and 5,435 drawn by "
                "hand in BCSS. They are excluded. The cost is that class 1 is then "
                "99.5% BRACS, so `source` and `class 1` are nearly the same "
                "question - see scripts/06_leakage_check.py."
                if not all_sources
                else "no source-authority rule was applied"
            ),
        },
        provenance={
            "exporter_spec": spec,
            "tiles_manifest_sha256": models.sha256_of(
                backend_path.TILES_DIR / "tiles_manifest.csv"),
            "pretrained_sha256": models.sha256_of(
                backend_path.PRETRAINED_DIR / models.INITS[init]),
            "licences": licences_for(init, export_summary),
            # Read off the module the fit actually used, so the manifest cannot
            # describe a recipe nobody ran.
            "training_recipe": {
                "epochs": train.EPOCHS, "lr": train.LR,
                "weight_decay": train.WEIGHT_DECAY, "seed": train.SEED,
                "frozen_body": True, "fitted_on": "cached frozen features",
            },
        },
        probe_tiles=[
            {"tile_id": row["tile_id"], "tile_path": row["tile_path"],
             "label": int(row["label"]),
             "logits": [round(float(value), 6) for value in logits[index]]}
            for index, row in enumerate(picked)
        ],
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"  {checkpoint.name}  {checkpoint.stat().st_size / 1e6:.1f} MB")
    print(f"  sha256 {manifest['sha256']}")
    return manifest


def training_data_block(export_summary: dict, download: dict) -> dict:
    """What this model was actually fitted on, including the borrowed class.

    Written from the export summary rather than stated as a constant. The constant used
    to read "BCSS (Amgad et al. 2019), CC0 1.0" unconditionally, which stopped being
    true the moment `02_export.py --include-dcis` existed - and it is baked into the
    published manifest, so a research-only model would have shipped describing itself as
    CC0. A licence claim that is a string literal is a licence claim that goes stale
    silently.
    """
    dcis = export_summary.get("dcis")
    block = {
        "source": "BCSS (Amgad et al. 2019), CC0 1.0",
        "regions_total": download["summary"]["regions"],
        "regions_trained_on": download["summary"]["train_regions"],
        "tiles": export_summary["tiles_kept"],
        "per_class": export_summary["per_class"],
        "pseudo_labels": False,
        "note": ("AICAN pseudo-labels (plan input B2) deferred: no H&E slide on "
                 "disk. BCSS-only is a complete model - see "
                 "docs/approach-1-training-plan.md Phase 5."),
    }
    if not dcis:
        return block

    block["pseudo_labels"] = True
    block["borrowed_in_situ_class"] = {
        "why": (
            "BCSS supplies 9 non_invasive_epithelium tiles in the whole release and "
            "none in the held-out institutions, so class 1 could be neither fitted nor "
            "measured from it alone."
        ),
        "source": "BRACS DCIS regions of interest, non-commercial",
        "labels": "BEETLE released nnU-Net ensemble, CC BY-NC-SA 4.0 - MODEL-GENERATED, "
                  "reviewed by no pathologist",
        "regions": dcis.get("regions"),
        "patients": dcis.get("cases"),
        "tiles": dcis.get("tiles"),
        "per_class": dcis.get("per_class"),
        "source_mpp": dcis.get("source_mpp"),
    }
    block["source"] = (
        "BCSS (CC0 1.0) + BRACS regions labelled by BEETLE (CC BY-NC-SA 4.0, "
        "non-commercial)"
    )
    block["licence_position"] = (
        "RESEARCH ONLY. This checkpoint inherits BEETLE's ShareAlike and BRACS's "
        "non-commercial terms and must not be shipped. The permissive model is the one "
        "trained without --include-dcis."
    )
    return block


def reload_check(name: str) -> None:
    """G6: a *fresh* process must reproduce the manifest's own logits.

    A subprocess, not this one. A check that runs in the process that just trained the
    model shares its imports, its module state and its transform - which is exactly
    what makes it unable to catch the fault it exists to catch.
    """
    code = textwrap.dedent(f"""
        import sys
        import numpy as np, torch
        sys.path.insert(0, r"{SRC}")
        import backend_path, datasets, models

        net, manifest = models.load_pinned("{name}", models_dir=backend_path.MODELS_DIR)
        probes = manifest["probe_tiles"]
        # EVERY input-affecting setting comes from the manifest, none from a default.
        # `standardise_tile_p99` was missing here and the check silently rebuilt the
        # input with a different transform from the one training used - which is the
        # exact fault this gate exists to detect, committed by the gate itself. It went
        # unnoticed because the caller was reload-checking a stale checkpoint whose
        # transform happened to match the defaults.
        dataset = datasets.TileDataset(
            [{{"tile_path": p["tile_path"], "label": p["label"], "tile_id": p["tile_id"]}}
             for p in probes],
            backend_path.TILES_DIR, augment=False,
            invert=manifest["input"]["invert_polarity"],
            standardise=manifest["input"]["standardise_tile_p99"],
        )
        with torch.inference_mode():
            logits = net(torch.stack([dataset[i][0] for i in range(len(dataset))])).numpy()
        worst = float(np.max(np.abs(logits - np.array([p["logits"] for p in probes]))))
        print(f"  worst logit difference: {{worst:.3e}}")
        assert worst < 1e-4, (
            "a freshly loaded checkpoint does not reproduce its own recorded logits - "
            "the transform at load time differs from the one at training time"
        )
        print("  G6 OK")
    """)
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    print((done.stdout or done.stderr).rstrip())
    if done.returncode != 0:
        raise SystemExit(f"G6 failed for {name} - do not deploy this checkpoint")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", choices=("imagenet", "simclr"), default="imagenet")
    parser.add_argument("--both", action="store_true", help="publish both arms")
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--standardise", action="store_true",
                        help="record standardise=True in the manifest and use it for "
                             "the reload probe. MUST match what 03 and 04 used.")
    parser.add_argument("--all-sources", action="store_true",
                        help="the old behaviour, in which BRACS also supplied invasive "
                             "and non-epithelium tiles. MUST match what 03 and 04 used.")
    parser.add_argument("--version", default="v1",
                        help="version tag in the published name. Use a new one rather "
                             "than overwriting a checkpoint someone may have scored a "
                             "slide with - the old numbers become uncheckable otherwise.")
    args = parser.parse_args()

    backend_path.ensure_dirs()
    inits = ("imagenet", "simclr") if args.both else (args.init,)

    published: dict[str, dict] = {}
    for init in inits:
        print(f"== publishing {init} ==")
        published[init] = publish_one(init, invert=args.invert,
                                      standardise=args.standardise,
                                      all_sources=args.all_sources,
                                      version=args.version)
        # Reload the checkpoint that was JUST written, by the name `publish_one` gave
        # it. This used to rebuild the suffix here and forget `_std`, which did not
        # fail loudly - it reload-checked whatever older `invasive_tile_v1_imagenet.pt`
        # happened to be sitting in the models directory and printed "G6 OK" for a file
        # this run had not produced. A gate that passes by verifying the wrong artefact
        # is worse than one that crashes, because the crash at least gets investigated.
        reload_check(published[init]["name"])
        print()

    ab_path = backend_path.REPORTS_DIR / "04_ab_imagenet_vs_simclr.json"
    ab = json.loads(ab_path.read_text(encoding="utf-8")) if ab_path.exists() else None

    print("=" * 76)
    print("Paste into models/README.md:\n")
    print("### tissue_type/ - the region model (pipeline step 8)\n")
    for init, manifest in published.items():
        held = manifest["metrics"].get("held_out", {})
        role = ""
        if ab:
            role = " - WINNER, pinned" if init == ab["winner"] else " - control, kept"
        print(f"- `{manifest['name']}.pt`{role}\n"
              f"  dice invasive vs non-invasive "
              f"{held.get('dice_invasive_vs_non_invasive', float('nan')):.3f} on held-out "
              f"institutions ({', '.join(manifest['held_out_institutions'])})\n"
              f"  sha256 `{manifest['sha256']}`\n"
              f"  licences: "
              f"{', '.join(f'{k} {v}' for k, v in manifest['provenance']['licences'].items())}")
    if ab:
        paired = ab["paired"]
        print(f"\nPaired per-slide difference (simclr - imagenet): "
              f"{paired['mean_difference']:+.3f} "
              f"[{paired['ci'][0]:+.3f}, {paired['ci'][1]:+.3f}], "
              f"{paired['wins']}W/{paired['losses']}L over {paired['slides']} slides - "
              f"{'a real difference' if paired['significant'] else 'inside the noise'}.")
        # The winner's real published name, not one rebuilt from parts - same reason
        # as the reload_check above.
        winner_name = published.get(ab["winner"], {}).get("name")
        if winner_name:
            print(f"\nSet settings.tissue_model_name = \"{winner_name}\".")
        else:
            print(f"\nA/B winner is {ab['winner']}, which this run did not publish.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
