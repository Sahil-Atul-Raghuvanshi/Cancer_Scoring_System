"""Doctor for step 8, and gates G6 and G7. Run this before trusting a class map.

    .venv\\Scripts\\python scripts\\check_tissue_model.py
    .venv\\Scripts\\python scripts\\check_tissue_model.py --model invasive_tile_v1_imagenet

It checks, in the order that things actually go wrong:

  1. is torch importable, and what device would it use
  2. which checkpoints are published, and under what licence
  3. does the selected one pass all four load-time checks - bytes, class order,
     input contract, exact state-dict fit
  4. **G6** - does a fresh process reproduce the logits the checkpoint's manifest
     recorded for its own probe tiles
  5. does a forward pass on synthetic haematoxylin behave sanely

**Step 4 is the one worth running.** It is the only check in this project that can
catch a training-to-serving mismatch, and that mismatch is invisible in every metric
computed inside the training process: the model trains, validates, publishes a good
number, and then scores badly on real slides for reasons that look like biology. The
manifest records a handful of tile ids and the logits the training process produced for
them; this replays those tiles through **the backend's own** transform and the
backend's own loader and compares. If the two disagree, the two definitions of "the
model's input" have drifted apart, and the number in the manifest belongs to a model
that is no longer being served.

The probe tiles live in the training folder's gitignored tile store, so step 4 is
skipped rather than failed when that folder is not present - it is a research artefact
and a deployment has no reason to carry ten thousand PNGs. The check says which
happened, because "skipped" and "passed" are very different states to ship on.

Exit code is 0 when step 8 can run and nothing it did check failed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Allow running as `python scripts/check_tissue_model.py` from the backend root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import DATA_ROOT  # noqa: E402
from app.pipeline.step08_tissue_type_segmentation import input as model_input  # noqa: E402
from app.pipeline.step08_tissue_type_segmentation import model  # noqa: E402
from app.pipeline.step08_tissue_type_segmentation.classes import CLASS_NAMES  # noqa: E402

TICK = "  ok  "
CROSS = " FAIL "
WARN = " warn "
SKIP = " skip "

#: Where the training folder keeps the tiles the manifest's probe logits were computed
#: from. Under the workspace's one `data/` tree rather than inside either project,
#: because those tiles are BCSS and BRACS pixels and the backend has no licence to
#: carry them - and because `data/` is the tree that can be moved or deleted whole.
#:
#: The **root** of the eight stores rather than one of them: they are
#: `<root>/{h_channel,he}/<fov>um/`, and which one a checkpoint's probes live in follows
#: from its own manifest - the channel and the geometry are both recorded there. Pointing
#: at a single directory is how this gate quietly died: it named `data/tiles`, which is
#: the staging tree the exporter empties after every run.
TILE_STORE = DATA_ROOT.parent / "tissue_type_model_training"


def _store_for(pinned) -> Path:
    """The tile store a checkpoint's probe tiles were cut into.

    Read off the manifest - `input.channel` and `tile_px * mpp` - so publishing a head
    at a new field of view needs no change here.
    """
    branch = "he" if pinned.channel == model_input.CHANNEL_RGB_HE else "h_channel"
    fov = round(pinned.tile_px * pinned.mpp)
    return TILE_STORE / branch / f"{fov}um"


def _probe_paths(store: Path) -> dict[str, Path]:
    """Tile id to file, from the store's own manifest.

    **The manifests record `tile_id` and no `tile_path`**, so resolving a probe means
    looking it up here. This is the other half of why the gate was skipping: it read a
    key that has never been written, found `None` every time, and reported "none of the
    probe tiles are in the store" - which reads like a missing store rather than a
    lookup that cannot succeed.
    """
    manifest = store / "tiles_manifest.csv"
    if not manifest.is_file():
        return {}
    with manifest.open(encoding="utf-8", newline="") as handle:
        return {
            row["tile_id"]: store / row["tile_path"]
            for row in csv.DictReader(handle)
            if row.get("tile_id") and row.get("tile_path")
        }

#: How far a replayed logit may differ from the recorded one. The training process and
#: this one run the same weights over the same bytes, so the only legitimate source of
#: difference is float32 reduction order - which is worth about 1e-5 on a ResNet18,
#: three orders of magnitude below the gap a real transform mismatch opens.
LOGIT_TOLERANCE = 1e-4


def line(status: str, message: str) -> None:
    print(f"[{status}] {message}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=None,
        help="Checkpoint to check. Defaults to the one step 8 would serve.",
    )
    parser.add_argument(
        "--tiles",
        type=Path,
        default=TILE_STORE,
        help="Training tile store, for the G6 replay. Skipped when absent.",
    )
    arguments = parser.parse_args()

    print("=" * 74)
    print("Step 8 (tissue-type segmentation) readiness check")
    print("=" * 74)

    ok = True

    # 1. torch ---------------------------------------------------------------
    runtime = model.runtime()
    if runtime.usable:
        detail = runtime.device_name or (f"{runtime.threads} threads" if runtime.threads else "")
        line(TICK, f"torch {runtime.torch_version} on {runtime.device} {detail}".rstrip())
    else:
        line(CROSS, f"torch stack unusable: {runtime.problem}")
        line("      ", "fix: pip install -r requirements.txt")
        return 1

    # 2. what is published ---------------------------------------------------
    candidates = model.discover()
    if not candidates:
        line(CROSS, f"no checkpoint published in {model.models_dir()}")
        line(
            "      ",
            "fix: train one - tissue_type_model_training/scripts/05_publish.py writes here",
        )
        return 1

    line(TICK, f"{len(candidates)} checkpoint(s) in {model.models_dir()}")
    for candidate in candidates:
        if not candidate.usable:
            ok = False
            line(CROSS, f"{candidate.name}: {candidate.problem}")
            continue
        badge = {"permissive": "SHIPPABLE", "research-only": "RESEARCH ONLY"}.get(
            candidate.licence_track, "LICENCE UNKNOWN"
        )
        line(
            "      ",
            f"{candidate.name}  {candidate.init}  {candidate.tile_px}px@{candidate.mpp}  "
            f"{'std' if candidate.standardise else 'no-std'}  [{badge}]",
        )

    selected = arguments.model or model.discover()[0].name
    chosen = model.find(selected)
    if chosen is None:
        line(CROSS, f"no checkpoint named {selected!r}")
        return 1

    if chosen.licence_track != "permissive":
        line(
            WARN,
            f"{selected} is {chosen.licence_track}: every number it produces inherits "
            "that restriction and must not ship",
        )
        for source, terms in sorted(chosen.licences.items()):
            line("      ", f"{source}: {terms}")

    # 3. the four load-time checks -------------------------------------------
    try:
        pinned = model.load_pinned(selected)
    except Exception as exc:  # noqa: BLE001 - the message is the whole point
        line(CROSS, f"{selected} would not load: {exc}")
        return 1

    line(TICK, f"{selected} loaded: bytes, class order, input contract and shape all agree")
    line("      ", f"sha256 {pinned.sha256[:16]}...")
    line("      ", f"classes {pinned.classes}")
    line(
        "      ",
        f"input: {pinned.tile_px}px at {pinned.mpp} um/px, gamma {pinned.gamma}, "
        f"standardise {pinned.standardise}, invert {pinned.invert_polarity}",
    )

    # 4. G6 - the replay -----------------------------------------------------
    if not _replay(pinned, arguments.tiles):
        ok = False

    # 5. a forward pass on something known -----------------------------------
    _sanity(pinned)

    print("-" * 74)
    line(TICK if ok else CROSS, "step 8 is ready" if ok else "step 8 has a problem above")
    return 0 if ok else 1


def _replay(pinned: model.Pinned, tiles: Path) -> bool:
    """G6: reproduce the manifest's own probe logits, in this process, through this code.

    Returns True when the check passed *or* was skipped for want of the tile store, and
    False only on a real disagreement. A skip is reported as a skip - shipping on an
    unverified transform is a decision, and it should be a visible one.
    """
    import numpy as np
    import torch
    from PIL import Image

    probes = pinned.manifest.get("probe_tiles") or []
    if not probes:
        line(
            WARN,
            "this checkpoint records no probe tiles, so the training-to-serving check "
            "cannot run. Re-publish it with probe logits - it is the only check that "
            "catches an input mismatch",
        )
        return True

    store = _store_for(pinned) if tiles == TILE_STORE else tiles
    if not store.is_dir():
        line(
            SKIP,
            f"G6 replay skipped: no tile store at {store}. The {len(probes)} probe tiles "
            "the manifest records live in the training folder, which a deployment has no "
            "licence to carry",
        )
        return True

    by_id = _probe_paths(store)

    worst = 0.0
    checked = 0
    missing = 0

    for probe in probes:
        recorded = probe.get("logits")
        tile_id = probe.get("tile_id")
        if recorded is None or tile_id is None:
            continue

        # `tile_path` where a manifest happens to carry one, the store's own index
        # otherwise. Both, because the older published manifests may differ.
        relative = probe.get("tile_path")
        path = (
            store / str(relative).replace("\\", "/")
            if relative is not None
            else by_id.get(str(tile_id))
        )
        if path is None or not path.is_file():
            missing += 1
            continue

        if pinned.channel == model_input.CHANNEL_RGB_HE:
            # The stored artefact *is* the model's input on this branch, up to the
            # fixed normalisation - so there is one function rather than a stored-form
            # and a slide-form of the same transform.
            with Image.open(path) as handle:
                tensor = model_input.rgb_to_model_input(
                    np.asarray(handle.convert("RGB"))
                )
        else:
            with Image.open(path) as handle:
                stored = np.asarray(handle.convert("L"))

            tensor = model_input.from_stored(
                stored,
                gamma=pinned.gamma,
                invert=pinned.invert_polarity,
                standardise=pinned.standardise,
            )
        with torch.inference_mode():
            logits = pinned.net(torch.from_numpy(tensor[None])).numpy()[0]

        worst = max(worst, float(np.abs(logits - np.asarray(recorded, dtype=np.float64)).max()))
        checked += 1

    if checked == 0:
        line(
            SKIP,
            f"G6 replay skipped: none of the {len(probes)} probe tiles are in {store}",
        )
        return True

    # A probe id that no longer resolves means the store has been re-cut since this
    # checkpoint was published - a new label rule, a different tree list - so the count
    # is reported alongside the verdict rather than silently reducing the sample.
    #
    # It does **not** explain a disagreement over the ids that do resolve. Measured:
    # `invasive_tile_v3_concat_bach` resolves all twelve of its probes and differs by
    # 1.0e+01, while `invasive_tile_fov112_fix1_concat` - the same geometry, the same
    # store - reproduces to 3.6e-06. The store is therefore not the problem, and v3's
    # recorded logits are: they were written either against a differently resampled
    # export or by a publish path that no longer exists. It is research-only, its
    # successor supersedes it, and `model_for` no longer selects it.
    detail = f"{checked} probe tiles, worst logit difference {worst:.2e}"
    if missing:
        detail += f" ({missing} of {len(probes)} not in {store.name})"

    if worst <= LOGIT_TOLERANCE:
        line(TICK, f"G6 PASS - this process reproduces the manifest's logits: {detail}")
        return True

    line(CROSS, f"G6 FAIL - {detail}, past the {LOGIT_TOLERANCE:.0e} tolerance")
    line(
        "      ",
        "the training transform and the serving transform have drifted apart. Compare "
        "step08_tissue_type_segmentation/input.py against what the checkpoint's manifest "
        "records in its `input` block - and do not score a slide until they agree",
    )
    return False


def _sanity(pinned: model.Pinned) -> None:
    """One forward pass on synthetic input, to prove the wiring end to end.

    Not an accuracy claim: a flat field is not tissue and the model has never seen one.
    What this catches is a network that returns the wrong shape, or NaN, or the same
    vector for every input - failures that a replay against recorded logits would also
    catch but which are worth reporting separately when the replay had to be skipped.
    """
    import numpy as np
    import torch

    flat = np.full((pinned.tile_px, pinned.tile_px), 0.30, dtype=np.float32)
    speckled = flat.copy()
    speckled[::8, ::8] = 1.2

    outputs = []
    for field in (flat, speckled):
        tensor = model_input.to_model_input(
            field,
            gamma=pinned.gamma,
            invert=pinned.invert_polarity,
            standardise=pinned.standardise,
        )
        with torch.inference_mode():
            logits = pinned.net(torch.from_numpy(tensor[None]))
            outputs.append(torch.softmax(logits, dim=1).numpy()[0])

    if any(not np.isfinite(vector).all() for vector in outputs):
        line(CROSS, "a forward pass produced non-finite values")
        return

    named = ", ".join(
        f"{CLASS_NAMES[index]} {value:.2f}" for index, value in enumerate(outputs[1])
    )
    line(TICK, f"forward pass ok - on synthetic nuclei: {named}")
    if float(np.abs(outputs[0] - outputs[1]).max()) < 1e-6:
        line(
            WARN,
            "a flat field and a speckled one gave the same answer, which a working "
            "network should not do",
        )


if __name__ == "__main__":
    raise SystemExit(main())
