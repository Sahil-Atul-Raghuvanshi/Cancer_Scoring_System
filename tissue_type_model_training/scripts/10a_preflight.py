"""Everything that must be true before the H&E campaign runs unattended.

    python scripts/10a_preflight.py            # report, exit non-zero on any FAIL
    python scripts/10a_preflight.py --json      # machine-readable, for the driver

Each item is `PASS`, `WARN`, `FAIL` or `MANUAL`. The driver refuses to start on a
single `FAIL`, and that is the whole point: every check here replaces a failure that
would otherwise surface hours later, in the dark, with nobody awake to read it. The
disk check replaces an `OSError` twenty-five minutes into the largest export. The
backbone check replaces a partial state-dict load forty-four minutes into a feature
pass. The region-count check replaces a pairing assertion that fires after half an
hour of cutting.

The one item worth reading before anything else is disk. This volume runs at 97% and
RGB tiles are three times the size of haematoxylin ones, so the campaign's headroom is
measured in single-digit gigabytes and *anything* else writing overnight - Windows
Update staging, a hibernation file, the demo's own cache - can take it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import backend_path  # noqa: E402
import bcss  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
DATA = backend_path.DATA_DIR

#: What the campaign is going to write, measured rather than guessed: the four
#: haematoxylin stores total 1.65 GB on disk and RGB re-encodes of real region crops
#: measure 3.0x the bytes of the H tiles, so the H&E stores come to about 4.9 GB. Plus
#: two feature caches per arm (~0.14 GB) and four 90 MB checkpoints (~0.36 GB).
NEED_GB = 5.4
FAIL_BELOW_GB = 12.0
WARN_BELOW_GB = 15.0

#: Exact image/mask pair counts per region tree, as the haematoxylin arms were cut
#: from. A tree that has grown means the two exports saw different data and the
#: pairing assertion will fail - better to know now.
EXPECTED_REGIONS = {
    "dcis": 239,
    "ic": 202,
    "normal": 213,
    "bach_insitu": 100,
    "bach_invasive": 100,
    "bach_normal": 100,
}

#: The four heads whose input contract must still verify after the descriptor change.
#: This is the gate that stops the branch work bricking the served models.
PUBLISHED_H_HEADS = (
    "invasive_tile_fov112_fix1_concat",
    "invasive_tile_fov224_concat",
    "invasive_tile_fov448_concat",
    "invasive_tile_fov672_concat",
)

results: list[dict[str, object]] = []


def record(name: str, status: str, detail: str) -> None:
    results.append({"item": name, "status": status, "detail": detail})


def check_disk() -> None:
    usage = shutil.disk_usage(str(WORKSPACE.anchor or "C:\\"))
    free_gb = usage.free / 1e9
    used_pct = 100.0 * usage.used / usage.total
    detail = (f"{free_gb:.1f} GB free of {usage.total / 1e9:.0f} GB ({used_pct:.0f}% "
              f"used); the campaign needs about {NEED_GB:.1f} GB")
    if free_gb < FAIL_BELOW_GB:
        record("disk", "FAIL", detail + f" and refuses to start below {FAIL_BELOW_GB:.0f} GB")
    elif free_gb < WARN_BELOW_GB:
        record("disk", "WARN", detail + " - tight; nothing else should write tonight")
    else:
        record("disk", "PASS", detail)


def check_torch() -> None:
    try:
        import torch
        import torchvision
    except Exception as exc:  # pragma: no cover - environment problem
        record("torch", "FAIL", f"torch does not import: {exc}")
        return
    version = torch.__version__
    detail = f"torch {version}, torchvision {torchvision.__version__}, python {sys.version.split()[0]}"
    if not version.startswith("2.8"):
        record("torch", "FAIL",
               detail + " - this project pins 2.8.x; 2.9 and above break on Windows "
                        "MAX_PATH and the export would die mid-run")
    else:
        record("torch", "PASS", detail)


def check_backbones() -> None:
    import models

    try:
        for init in ("imagenet", "simclr"):
            body = models.resnet18_backbone(init, weights_dir=backend_path.PRETRAINED_DIR)
            del body
    except Exception as exc:
        record("backbones", "FAIL",
               f"{init} backbone does not load: {exc}. This would otherwise surface "
               "forty minutes into a feature pass.")
        return
    record("backbones", "PASS",
           "both pretrained backbones load through resnet18_backbone with the "
           "fc-only assertion satisfied")


def check_regions() -> None:
    problems: list[str] = []
    counts: dict[str, int] = {}
    for tree, expected in EXPECTED_REGIONS.items():
        directory = backend_path.borrowed_dir(tree)
        sidecar = directory / f"{tree}_manifest.json"
        if not sidecar.exists():
            problems.append(f"{tree}: {sidecar.name} missing")
            continue
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        images = sorted((directory / "images").glob("*.png"))
        masks = sorted((directory / "masks").glob("*.png"))
        counts[tree] = len(images)

        if len(images) != len(masks):
            problems.append(f"{tree}: {len(images)} images but {len(masks)} masks")
        if len(images) != expected:
            problems.append(
                f"{tree}: {len(images)} regions on disk, {expected} when the "
                "haematoxylin arms were cut - the two exports would see different "
                "data and 10c would refuse the pairing"
            )
        if float(meta.get("source_mpp", 0)) != 0.5:
            problems.append(f"{tree}: source_mpp is {meta.get('source_mpp')}, not 0.5")
        teaches = tuple(sorted(meta.get("teaches_classes") or []))
        expected_teaches = tuple(sorted(bcss.BORROWED_TREES[tree].teaches))
        if teaches != expected_teaches:
            problems.append(
                f"{tree}: teaches {teaches} but bcss.BORROWED_TREES says "
                f"{expected_teaches} - the source-authority rule and the producing "
                "manifest have drifted apart"
            )

    detail = ", ".join(f"{tree} {count}" for tree, count in sorted(counts.items()))
    if problems:
        record("regions", "FAIL", "; ".join(problems))
    else:
        record("regions", "PASS", f"six trees at 0.5 um/px: {detail}")


def check_bcss() -> None:
    try:
        regions = bcss.find_regions(backend_path.BCSS_DIR)
    except Exception as exc:
        record("bcss", "FAIL", f"cannot enumerate BCSS regions: {exc}")
        return

    if not regions:
        record("bcss", "FAIL", f"no BCSS regions under {backend_path.BCSS_DIR}")
        return

    mpp_path = backend_path.BCSS_DIR / "slide_mpp.json"
    if not mpp_path.exists():
        record("bcss", "FAIL",
               f"{mpp_path} is missing and the export refuses to assume 0.25 um/px "
               "from the filenames")
        return

    measured = json.loads(mpp_path.read_text(encoding="utf-8"))
    if isinstance(measured, dict) and "resolutions" in measured:
        measured = measured["resolutions"]
    missing = sorted({r.slide_key for r in regions} - set(measured))
    if missing:
        record("bcss", "FAIL",
               f"{len(missing)} slides have no measured resolution, e.g. {missing[:5]}")
    else:
        record("bcss", "PASS",
               f"{len(regions)} regions, every slide's mpp measured "
               f"({len(measured)} slides recorded)")


def check_h_arms() -> None:
    import datasets
    import numpy as np

    problems: list[str] = []
    for fov in ("112um", "224um", "448um", "672um"):
        store = DATA / "h_channel" / fov
        manifest = store / "tiles_manifest.csv"
        if not manifest.exists():
            problems.append(f"{fov}: no manifest at {store}")
            continue
        rows = datasets.read_manifest(manifest)
        broken = sum(1 for row in rows if not (store / row["tile_path"]).is_file())
        if broken:
            problems.append(f"{fov}: {broken} of {len(rows)} tile paths do not resolve")
        fingerprint = datasets.manifest_fingerprint(datasets.by_source_authority(rows))
        for init in ("imagenet", "simclr"):
            cache = store / "features" / f"{init}_std.npz"
            if not cache.exists():
                problems.append(f"{fov}: {init}_std.npz missing")
                continue
            try:
                got = str(np.load(cache, allow_pickle=True)["fingerprint"])
            except Exception as exc:
                problems.append(f"{fov}/{init}: cache unreadable ({exc})")
                continue
            if not got.startswith(fingerprint[:16]):
                problems.append(f"{fov}/{init}: fingerprint {got[:16]} != {fingerprint[:16]}")

    if problems:
        record("h_arms", "FAIL", "; ".join(problems))
    else:
        record("h_arms", "PASS",
               "all four haematoxylin arms intact after the move: every tile path "
               "resolves and both feature caches match their manifest")


def check_regions_dir() -> None:
    name = backend_path.REGIONS_DIR.name
    if name != "regions":
        record("regions_dir", "FAIL",
               f"REGIONS_DIR resolved to {backend_path.REGIONS_DIR} - the export "
               "would read the wrong trees, silently")
    else:
        record("regions_dir", "PASS", str(backend_path.REGIONS_DIR))


def check_published_heads() -> None:
    sys.path.insert(0, str(WORKSPACE / "tissue_scoring_demo" / "backend"))
    try:
        from app.pipeline.step08_tissue_type_segmentation import model as region_model
        from app.pipeline.step08_tissue_type_segmentation.branches import ModelBranch
        from app.services import tiling_service
    except Exception as exc:
        record("published_heads", "FAIL", f"cannot import the demo backend: {exc}")
        return

    problems: list[str] = []
    for name in PUBLISHED_H_HEADS:
        try:
            pinned = region_model.load_pinned(name, verify=True)
        except Exception as exc:
            problems.append(f"{name}: {exc}")
            continue
        if pinned.branch is not ModelBranch.H_CHANNEL:
            problems.append(f"{name}: resolved to branch {pinned.branch.value}")

    # The branch filter has to be in place before an RGB head can be published, or
    # `discover()`'s ordering decides which of two heads at one geometry serves.
    for um in (112.0, 224.0, 448.0, 672.0):
        found = tiling_service.model_for(um, ModelBranch.H_CHANNEL)
        if found is None:
            problems.append(f"{um:g} um: no haematoxylin head resolves")
        elif found.branch is not ModelBranch.H_CHANNEL:
            problems.append(f"{um:g} um: resolved a {found.branch.value} head")

    if problems:
        record("published_heads", "FAIL", "; ".join(problems))
    else:
        record("published_heads", "PASS",
               "all four haematoxylin heads load and verify, and model_for filters "
               "on branch - an RGB head can be published without displacing them")


def check_other_python() -> None:
    try:
        out = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'", "get", "ProcessId,CreationDate"],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except Exception:
        record("other_python", "MANUAL",
               "could not enumerate processes - check by hand that no other torch job "
               "is running; a second feature pass halves throughput and blows every budget")
        return

    pids = [line.split()[-1] for line in out.splitlines()
            if line.strip() and line.split()[-1].isdigit()]
    others = [pid for pid in pids if pid != str(os.getpid())]
    if others:
        record("other_python", "WARN",
               f"{len(others)} other python.exe process(es) running (pids {others[:6]}). "
               "If any is a torch job the budgets below will not hold.")
    else:
        record("other_python", "PASS", "no other python.exe is running")


def check_sleep() -> None:
    try:
        out = subprocess.run(
            ["powercfg", "/query", "SCHEME_CURRENT", "SUB_SLEEP"],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except Exception as exc:
        record("sleep", "MANUAL", f"could not read the power scheme ({exc}) - confirm "
                                  "sleep and hibernate are disabled on AC by hand")
        return

    blocks = out.split("Power Setting GUID:")
    bad: list[str] = []
    for block in blocks:
        for label, key in (("standby", "STANDBYIDLE"), ("hibernate", "HIBERNATEIDLE")):
            if key in block:
                for line in block.splitlines():
                    if "Current AC Power Setting Index" in line:
                        value = line.split(":")[-1].strip()
                        if value not in ("0x00000000", "0"):
                            bad.append(f"{label}={value}")
    if bad:
        record("sleep", "FAIL",
               f"the machine will sleep on AC ({', '.join(bad)}). Fix with:\n"
               "      powercfg /change standby-timeout-ac 0\n"
               "      powercfg /change hibernate-timeout-ac 0")
    else:
        record("sleep", "PASS", "sleep and hibernate are disabled on AC")


def check_reboot_pending() -> None:
    try:
        out = subprocess.run(
            ["reg", "query",
             r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        record("reboot", "MANUAL", "could not check for a pending reboot")
        return
    if out.returncode == 0:
        record("reboot", "FAIL",
               "Windows Update has a reboot pending. It may restart the machine "
               "mid-campaign. Reboot now, then start the run.")
    else:
        record("reboot", "PASS", "no Windows Update reboot pending")


def check_writable() -> None:
    problems = []
    for target in (WORKSPACE, DATA / "he", REPO / "reports"):
        target.mkdir(parents=True, exist_ok=True)
        probe = target / ".preflight_write_probe"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            problems.append(f"{target}: {exc}")
    if problems:
        record("writable", "FAIL", "; ".join(problems))
    else:
        record("writable", "PASS", "the workspace root, data/he and reports are writable")


def check_backend_not_running() -> None:
    import socket

    for port in (8000, 8080):
        with socket.socket() as probe:
            probe.settimeout(0.4)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                record("backend", "WARN",
                       f"something is listening on 127.0.0.1:{port}. If it is the demo "
                       "API it holds checkpoints in its module cache and could serve a "
                       "half-published one. Stop it before the run.")
                return
    record("backend", "PASS", "the demo API is not listening on 8000 or 8080")


CHECKS = (
    check_disk,
    check_torch,
    check_regions_dir,
    check_backbones,
    check_regions,
    check_bcss,
    check_h_arms,
    check_published_heads,
    check_other_python,
    check_sleep,
    check_reboot_pending,
    check_writable,
    check_backend_not_running,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    for check in CHECKS:
        try:
            check()
        except Exception as exc:  # a broken check is itself a finding
            record(check.__name__.removeprefix("check_"), "FAIL",
                   f"the check itself raised: {exc!r}")

    failures = [r for r in results if r["status"] == "FAIL"]
    warnings = [r for r in results if r["status"] == "WARN"]

    if args.json:
        print(json.dumps({"ok": not failures, "results": results}, indent=2))
    else:
        print("\n== pre-flight ==\n")
        for entry in results:
            print(f"  {entry['status']:<7} {entry['item']:<18} {entry['detail']}")
        print()
        if failures:
            print(f"  {len(failures)} FAIL - the campaign must not start.")
        elif warnings:
            print(f"  clear to start, with {len(warnings)} warning(s) above.")
        else:
            print("  all clear.")
        print()

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
