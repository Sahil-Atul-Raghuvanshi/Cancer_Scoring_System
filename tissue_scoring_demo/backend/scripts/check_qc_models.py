"""Doctor for step 2. Run this before wiring anything to the UI.

    .venv\\Scripts\\python scripts\\check_qc_models.py

It checks, in the order that things actually go wrong:

  1. is scipy importable          (the classical metrics)
  2. is torch importable, and what device would it use
  3. is segmentation-models-pytorch importable
  4. are the four checkpoints on disk, and where did we look
  5. do they actually load - which is where a version mismatch surfaces
  6. does a forward pass produce the class ids GrandQC documents

Step 5 is the one worth running. The artefact checkpoints are pickled modules,
so a version skew in segmentation-models-pytorch shows up as an unpickling
error at load time rather than as anything visible at install time.

Exit code is 0 when step 2 can run - fully or in its degraded, no-tissue-model
form - and 1 when it cannot.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `python scripts/check_qc_models.py` from the backend root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline.step02_quality_control import models  # noqa: E402
from app.pipeline.step02_quality_control.features import features_available  # noqa: E402
from app.services.qc_service import qc_service  # noqa: E402

TICK = "  ok  "
CROSS = " FAIL "
WARN = " warn "


def line(status: str, message: str) -> None:
    print(f"[{status}] {message}")


def main() -> int:
    print("=" * 74)
    print("Step 2 (quality control) readiness check")
    print("=" * 74)

    ok = True

    # 1. classical half ------------------------------------------------------
    features_ok, features_problem = features_available()
    if features_ok:
        line(TICK, "scipy importable - classical feature metrics can run")
    else:
        ok = False
        line(CROSS, f"classical metrics unavailable: {features_problem}")
        line("      ", "fix: pip install -r requirements-qc.txt")

    # 2 and 3. torch stack ---------------------------------------------------
    runtime = models.runtime()
    if runtime.torch_version:
        line(TICK, f"torch {runtime.torch_version}")
    else:
        ok = False
        line(CROSS, f"{runtime.problem}")
        line("      ", "fix: pip install -r requirements-qc.txt")

    if runtime.smp_version:
        line(TICK, f"segmentation-models-pytorch {runtime.smp_version}")
    elif runtime.torch_version:
        ok = False
        line(CROSS, f"{runtime.problem}")

    if runtime.usable:
        where = runtime.device_name or runtime.device
        line(TICK if runtime.device != "cpu" else WARN, f"inference device: {where}")
        if runtime.device == "cpu":
            line(
                "      ",
                "no GPU - expect several minutes per slide. The 5x model "
                "(--model-mpp 2.0) is roughly twice as fast as 7x.",
            )

    # 4. checkpoints ---------------------------------------------------------
    print("-" * 74)
    found = models.discover_checkpoints()

    if found.root is None:
        ok = False
        line(CROSS, "no checkpoint directory found. Looked in:")
        for path in found.searched:
            print(f"         {path}")
    else:
        line(TICK, f"checkpoint directory: {found.root}")

    if found.tissue:
        size = found.tissue.stat().st_size / 1e6
        line(TICK, f"{models.TISSUE_CHECKPOINT}  ({size:.0f} MB)")
    else:
        line(WARN, f"{models.TISSUE_CHECKPOINT} missing - tissue falls back to Otsu")
        line("      ", "get it from https://zenodo.org/records/14507273")

    if not found.artefacts:
        ok = False
        line(CROSS, "no artefact checkpoint - step 2 cannot run at all")
        line("      ", "get one from https://zenodo.org/records/14041538")
    else:
        for mpp in sorted(found.artefacts):
            path = found.artefacts[mpp]
            size = path.stat().st_size / 1e6
            label = models.MPP_LABELS.get(mpp, "")
            line(TICK, f"{path.name}  ({label}, {mpp} um/px, {size:.0f} MB)")

    # 5 and 6. do they load and predict -------------------------------------
    if runtime.usable and (found.tissue or found.artefacts):
        print("-" * 74)
        ok = _try_loading(found, runtime.device) and ok

    # verdict ----------------------------------------------------------------
    print("=" * 74)
    capability = qc_service.capability()
    line(
        TICK if capability.mode != "unavailable" else CROSS,
        f"mode: {capability.mode} - {capability.reason}",
    )
    print("=" * 74)
    return 0 if capability.mode != "unavailable" else 1


def _try_loading(found: models.Checkpoints, device: str) -> bool:
    """Load each checkpoint and push one blank patch through it."""
    import numpy as np

    ok = True
    preprocess = models.preprocessing_fn()
    blank = np.full((512, 512, 3), 220, dtype=np.uint8)  # pale, like glass

    if found.tissue:
        try:
            loaded = models.load_tissue_model(found.tissue, device)
            line(TICK, f"loaded {found.tissue.name}: {loaded.classes} classes")
            _forward(loaded, blank, preprocess, expected="0 = tissue, 1 = background")
        except Exception as exc:  # noqa: BLE001
            ok = False
            line(CROSS, f"{found.tissue.name} would not load: {exc}")

    for mpp in sorted(found.artefacts):
        path = found.artefacts[mpp]
        try:
            loaded = models.load_artefact_model(path, device)
            line(TICK, f"loaded {path.name}: {loaded.classes} output channels")
            if loaded.classes < 7:
                ok = False
                line(
                    CROSS,
                    f"expected at least 7 channels (GrandQC codes classes 1-7), got "
                    f"{loaded.classes} - this is not a GrandQC artefact checkpoint",
                )
            _forward(
                loaded,
                blank,
                preprocess,
                expected=(
                    "1 tissue, 2 fold, 3 dark spot, 4 pen, 5 edge, "
                    "6 out of focus, 7 background"
                ),
            )
        except Exception as exc:  # noqa: BLE001
            ok = False
            line(CROSS, f"{path.name} would not load: {exc}")

    return ok


def _forward(loaded: models.LoadedModel, patch, preprocess, *, expected: str) -> None:
    """One forward pass, reporting which class ids came back."""
    import numpy as np
    import torch

    array = np.ascontiguousarray(preprocess(patch).transpose(2, 0, 1))[None].astype("float32")
    with torch.no_grad():
        logits = loaded.model(torch.from_numpy(array).to(loaded.device))
    ids = sorted(int(value) for value in np.unique(logits.argmax(dim=1).cpu().numpy()))
    line("      ", f"forward pass ok, predicted ids {ids} on a blank patch")
    line("      ", f"class coding: {expected}")


if __name__ == "__main__":
    raise SystemExit(main())
