#!/usr/bin/env python3
"""Set up the Breast Cancer IHC Tissue Scoring web app from a fresh clone.

Everything this repository does not carry in git, this script puts back:

  1. the backend virtualenv, from backend/requirements*.txt
  2. the frontend node_modules, from package-lock.json
  3. the model checkpoints, from models.lock.json
  4. backend/.env, from .env.example

Then it runs the project's own readiness checks - the same two scripts the
docs tell you to run - so a green finish means the app will actually serve
step 2 and step 8, not merely that pip exited 0.

Stdlib only, so it runs on a system Python before any venv exists.

    python setup.py                       # full setup
    python setup.py --check               # audit only, download nothing
    python setup.py --drive-folder <id>   # fetch weights from a Drive folder
    python setup.py --models-only         # skip venv and npm
    python setup.py --with-training       # also fetch training-only weights

The venv is disposable by design: delete backend/.venv and re-run this script.
That is why nothing in it is hand-installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / "tissue_scoring_demo"
BACKEND = APP / "backend"
FRONTEND = APP / "frontend"
VENV = BACKEND / ".venv"
LOCK = ROOT / "models.lock.json"

# Stdlib-only, so importing it here is safe before any venv exists. Checkpoints
# live in the active data version's models/ (v<N>_data/models/), or the newest
# earlier version's when this one has none; with none anywhere, models_root()
# names the active version's own folder, which is where a fresh setup installs.
sys.path.append(str(ROOT))  # data_versions.py lives at the repository root
import data_versions  # noqa: E402

# torch 2.9+ unpacks a licence tree deep enough to break MAX_PATH under this
# project's already-long path; requirements-qc.txt pins <2.9 for that reason.
# CPU wheels are also ~2 GB smaller, and no machine here has CUDA.
TORCH_INDEX = "https://download.pytorch.org/whl/cpu"

MIN_PYTHON = (3, 11)
MIN_NODE = 18

# GrandQC is a third-party clone, not our code, so it is not in this repo. The
# backend picks it up as a sibling checkout if it is present, but does not need
# it once v<N>_data/models/grandqc/ is populated.
GRANDQC_REPO = "https://github.com/cpath-ukk/grandqc.git"


# --------------------------------------------------------------------------
#  output
# --------------------------------------------------------------------------

_COLOUR = sys.stdout.isatty() and os.name != "nt" or bool(os.environ.get("WT_SESSION"))


def _paint(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOUR else text


def ok(msg: str) -> None:
    print(f"[  {_paint('32', 'ok')}  ] {msg}")


def warn(msg: str) -> None:
    print(f"[ {_paint('33', 'warn')} ] {msg}")


def fail(msg: str) -> None:
    print(f"[ {_paint('31', 'FAIL')} ] {msg}")


def skip(msg: str) -> None:
    print(f"[ skip ] {msg}")


def note(msg: str) -> None:
    print(f"[      ] {msg}")


def rule(title: str = "") -> None:
    print("-" * 74 if not title else f"\n{'=' * 74}\n {title}\n{'=' * 74}")


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


class Problems:
    """Collected failures, so setup reports every one instead of the first."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, msg: str) -> None:
        fail(msg)
        self.errors.append(msg)

    def warning(self, msg: str) -> None:
        warn(msg)
        self.warnings.append(msg)


# --------------------------------------------------------------------------
#  prerequisites
# --------------------------------------------------------------------------


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def check_prerequisites(p: Problems, *, skip_node: bool, why: str) -> None:
    rule("1/5  Prerequisites")

    if sys.version_info < MIN_PYTHON:
        p.error(
            f"Python {'.'.join(map(str, MIN_PYTHON))}+ required, "
            f"this is {platform.python_version()}"
        )
    else:
        ok(f"Python {platform.python_version()} ({sys.executable})")

    if skip_node:
        skip(f"Node check - {why}")
        return

    npm = shutil.which("npm")
    node = shutil.which("node")
    if not node or not npm:
        p.error(f"Node {MIN_NODE}+ and npm are required, and were not found on PATH")
        return

    try:
        raw = subprocess.run(
            [node, "-v"], capture_output=True, text=True, timeout=60
        ).stdout.strip()
        major = int(raw.lstrip("v").split(".")[0])
    except Exception:
        p.warning("could not read the Node version; continuing")
        return

    if major < MIN_NODE:
        p.error(f"Node {MIN_NODE}+ required, this is {raw}")
    else:
        ok(f"Node {raw}, npm present")


# --------------------------------------------------------------------------
#  backend virtualenv
# --------------------------------------------------------------------------


def run(cmd: list[str], *, cwd: Path | None = None, label: str) -> bool:
    """Run a subprocess, streaming nothing but reporting its tail on failure."""
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=3600
        )
    except subprocess.TimeoutExpired:
        fail(f"{label} timed out after an hour")
        return False
    except FileNotFoundError:
        fail(f"{label}: {cmd[0]} not found")
        return False

    if proc.returncode != 0:
        fail(f"{label} exited {proc.returncode}")
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-15:]
        for line in tail:
            note(f"  {line}")
        return False
    return True


def setup_venv(p: Problems, *, recreate: bool) -> None:
    rule("2/5  Backend virtualenv")

    if recreate and VENV.exists():
        note("removing the existing virtualenv (--recreate-venv)")
        shutil.rmtree(VENV, ignore_errors=True)

    py = venv_python()
    if py.exists():
        ok(f"virtualenv present: {VENV}")
    else:
        note("creating the virtualenv (a minute or two)")
        if not run([sys.executable, "-m", "venv", str(VENV)], label="python -m venv"):
            p.error("could not create the virtualenv")
            return
        ok(f"created {VENV}")

    if not py.exists():
        p.error(f"{py} is missing after venv creation")
        return

    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "--quiet"], label="pip upgrade")

    # **The lock first, when there is one (P-20).** The ranges in requirements*.txt let
    # numpy, scipy, torch, smp and timm float, so two machines set up a week apart ran
    # different code under the same commit. The lock is the venv as it was verified;
    # the range files below then only add what the lock does not already satisfy.
    lock = BACKEND / "requirements.lock.txt"
    if lock.exists():
        note(f"installing {lock.name} (every package at the version it was verified with)")
        if not run(
            [
                str(py), "-m", "pip", "install", "-r", str(lock),
                "--extra-index-url", TORCH_INDEX, "--quiet",
            ],
            label="pip install requirements.lock.txt",
        ):
            p.error("the locked dependencies failed to install")
            return
        ok("locked dependencies installed")

    core = BACKEND / "requirements.txt"
    note(f"installing {core.name} (FastAPI, tiffslide, scipy)")
    if not run(
        [str(py), "-m", "pip", "install", "-r", str(core), "--quiet"],
        label="pip install requirements.txt",
    ):
        p.error("core dependencies failed to install")
        return
    ok("core dependencies installed")

    tools = BACKEND / "requirements-tools.txt"
    if tools.exists() and run(
        [str(py), "-m", "pip", "install", "-r", str(tools), "--quiet"],
        label="pip install requirements-tools.txt",
    ):
        ok("offline tools installed (scikit-learn, openpyxl)")

    qc = BACKEND / "requirements-qc.txt"
    note(f"installing {qc.name} (torch CPU, smp, timm - about 500 MB, several minutes)")
    if not run(
        [
            str(py), "-m", "pip", "install", "-r", str(qc),
            "--extra-index-url", TORCH_INDEX, "--quiet",
        ],
        label="pip install requirements-qc.txt",
    ):
        p.error(
            "torch stack failed to install. If this was WinError 206, the path is "
            "too long: enable long paths (LongPathsEnabled=1, admin, reboot) or "
            "move the repository nearer the drive root."
        )
        return
    ok("torch stack installed (CPU wheels)")


def setup_env_file() -> None:
    example = BACKEND / ".env.example"
    target = BACKEND / ".env"
    if target.exists():
        ok(".env present")
    elif example.exists():
        shutil.copyfile(example, target)
        ok(f"created .env from {example.name}")
    else:
        warn(".env.example is missing; the app will fall back to its defaults")


# --------------------------------------------------------------------------
#  frontend
# --------------------------------------------------------------------------


def setup_frontend(p: Problems) -> None:
    rule("3/5  Frontend dependencies")

    if (FRONTEND / "node_modules").exists():
        ok("node_modules present")
        return

    npm = shutil.which("npm")
    if not npm:
        p.error("npm not found; cannot install frontend dependencies")
        return

    lockfile = FRONTEND / "package-lock.json"
    cmd = [npm, "ci"] if lockfile.exists() else [npm, "install"]
    note(f"running {' '.join(cmd[1:])} in {FRONTEND.name} (a minute or two)")
    if not run(cmd, cwd=FRONTEND, label=" ".join(cmd[1:])):
        p.error("frontend dependencies failed to install")
        return
    ok("frontend dependencies installed")


# --------------------------------------------------------------------------
#  models
# --------------------------------------------------------------------------


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, entry: dict) -> tuple[bool, str]:
    """Does the file on disk match the lock file? Returns (ok, reason)."""
    if not path.exists():
        return False, "missing"

    actual_bytes = path.stat().st_size
    if actual_bytes != entry["bytes"]:
        return False, f"{actual_bytes:,} bytes, expected {entry['bytes']:,}"

    actual = sha256_of(path)
    if actual == entry["sha256"]:
        return True, "bytes and sha256 match"

    # No exception for a locally re-saved checkpoint any more (P-20). It used to pass on
    # its byte count alone, which a different set of weights of the same architecture
    # matches exactly - a retrained step 8 head is the same size as the one it replaces.
    # The lock records this copy's own sha256 (`10e_update_models_lock.py` reads it from
    # the published manifest), so a mismatch means it is not the file that was locked.
    return False, f"sha256 mismatch: {actual[:16]}... expected {entry['sha256'][:16]}..."


def download(url: str, target: Path, expected_bytes: int) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    started = time.time()

    try:
        request = urllib.request.Request(url, headers={"User-Agent": "cancer-scoring-setup"})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
            done = 0
            while chunk := response.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if expected_bytes:
                    pct = 100.0 * done / expected_bytes
                    print(f"\r         {human(done)} / {human(expected_bytes)}  {pct:5.1f}%", end="")
        if expected_bytes:
            print(f"\r         {human(done)} in {time.time() - started:.0f}s" + " " * 20)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        print()
        note(f"  download failed: {exc}")
        partial.unlink(missing_ok=True)
        return False

    partial.replace(target)
    return True


def download_zip_member(url: str, member: str, target: Path, expected_bytes: int) -> bool:
    """Fetch an archive and lift one file out of it.

    InstanSeg publishes its model as a release **zip** - weights, metadata and
    the upstream's own test tensors together - rather than as three separate
    assets, so three lock entries share one download. The archive is fetched
    once into the cache below and every member after the first is extracted
    from the copy already on disk.

    `expected_bytes` is the *member's* size, not the archive's, and the caller
    still verifies the sha256 afterwards - so a mismatched or renamed member is
    caught by the same check that catches a truncated download.
    """
    cache = target.parent / ".archives"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / url.rsplit("/", 1)[-1]

    if not archive.exists():
        # Archive size is unknown here (the lock file records members), so the
        # progress line is suppressed rather than shown against a wrong total.
        if not download(url, archive, 0):
            return False

    try:
        with zipfile.ZipFile(archive) as bundle:
            with bundle.open(member) as source:
                payload = source.read()
    except (KeyError, zipfile.BadZipFile, OSError) as exc:
        note(f"  could not read {member} from {archive.name}: {exc}")
        return False

    if expected_bytes and len(payload) != expected_bytes:
        note(f"  {member} is {human(len(payload))}, expected {human(expected_bytes)}")
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return True


def drive_fetch(plan: list[dict], folder: str, py: Path) -> dict[str, dict]:
    """Fetch several files from the shared Drive folder in one pass.

    Batched on purpose: the folder is listed once rather than once per
    checkpoint, and only the requested paths are downloaded - the same tree
    holds BEETLE's 1.8 GB model.zip, which serving never needs.
    """
    if not plan:
        return {}

    if not py.exists():
        return {
            item["relpath"]: {
                "ok": False,
                "reason": "the Drive fetch needs the virtualenv, which does not exist yet",
            }
            for item in plan
        }

    if subprocess.run([str(py), "-c", "import gdown"], capture_output=True).returncode != 0:
        note("installing gdown into the virtualenv")
        if not run([str(py), "-m", "pip", "install", "gdown", "--quiet"],
                   label="pip install gdown"):
            return {item["relpath"]: {"ok": False, "reason": "gdown unavailable"} for item in plan}

    helper = ROOT / "tools" / "drive_fetch.py"
    if not helper.exists():
        return {
            item["relpath"]: {"ok": False, "reason": f"{helper} is missing"} for item in plan
        }

    plan_file = Path(tempfile.mkdtemp(prefix="setup_plan_")) / "plan.json"
    plan_file.write_text(json.dumps(plan), encoding="utf-8")

    note(f"listing the Drive folder and fetching {len(plan)} file(s)")
    try:
        proc = subprocess.run(
            [str(py), str(helper), folder, str(plan_file)],
            capture_output=True, text=True, timeout=7200,
        )
    except subprocess.TimeoutExpired:
        return {item["relpath"]: {"ok": False, "reason": "Drive fetch timed out"} for item in plan}
    finally:
        shutil.rmtree(plan_file.parent, ignore_errors=True)

    try:
        report = json.loads((proc.stdout or "").strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        tail = (proc.stderr or proc.stdout or "no output").strip().splitlines()[-6:]
        for line in tail:
            note(f"  {line}")
        return {
            item["relpath"]: {"ok": False, "reason": "the Drive helper returned no report"}
            for item in plan
        }

    if "_error" in report:
        note(f"  {report['_error']}")
        return {item["relpath"]: {"ok": False, "reason": report["_error"]} for item in plan}

    return report


def torchvision_resave(target: Path, py: Path) -> bool:
    """Regenerate the vendored ImageNet backbone the way training does."""
    if not py.exists():
        note("  needs the virtualenv, which does not exist yet")
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    script = (
        "import sys, torch, torchvision\n"
        "state = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True)\n"
        "torch.save(state, sys.argv[1])\n"
    )
    return run([str(py), "-c", script, str(target)], label="torchvision re-save")


def setup_models(
    p: Problems,
    *,
    check_only: bool,
    with_training: bool,
    drive_folder: str | None,
) -> None:
    rule("4/5  Model checkpoints")

    if not LOCK.exists():
        p.error(f"{LOCK.name} is missing; cannot resolve the model set")
        return

    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    # The lock's `path`s are relative to a models/ folder; which one is decided by
    # data_versions, not by the lock's legacy "models_root" field.
    models_root = data_versions.models_root()
    drive_folder = drive_folder or lock.get("drive_folder") or os.environ.get("MODELS_DRIVE_FOLDER")
    py = venv_python()

    note(f"target: {models_root}")
    if drive_folder:
        note(f"Drive folder: {drive_folder}")
    rule()

    wanted = [
        entry
        for entry in lock["files"]
        if entry["role"] != "training" or with_training
    ]
    deferred = [e for e in lock["files"] if e["role"] == "training" and not with_training]

    missing_runtime: list[dict] = []
    todo: list[dict] = []

    # First pass: report what is already correct, and collect the rest. The
    # Drive fetches are then batched, so the folder is listed once.
    for entry in wanted:
        path = models_root / entry["path"]
        good, reason = verify(path, entry)

        if good:
            tag = "" if entry["role"] == "runtime" else f"  ({entry['role']})"
            ok(f"{entry['path']}  {human(entry['bytes'])}{tag}")
            if "differs" in reason:
                note(f"  {reason}")
            continue

        if check_only:
            (p.error if entry["role"] == "runtime" else p.warning)(
                f"{entry['path']} - {reason}"
            )
            if entry["role"] == "runtime":
                missing_runtime.append(entry)
            continue

        warn(f"{entry['path']} - {reason}")
        todo.append(entry)

    # Second pass: direct sources, which need no shared listing.
    still_missing: list[dict] = []
    for entry in todo:
        path = models_root / entry["path"]
        fetched = False

        for source in entry["sources"]:
            if source["kind"] == "url":
                note(f"  {entry['path']} <- {source['url'].split('/')[2]}")
                fetched = download(source["url"], path, entry["bytes"])
            elif source["kind"] == "zip":
                note(f"  {entry['path']} <- {source['url'].split('/')[2]} ({source['member']})")
                fetched = download_zip_member(
                    source["url"], source["member"], path, entry["bytes"]
                )
            elif source["kind"] == "torchvision":
                note(f"  {entry['path']} <- torchvision {source['weights']}")
                fetched = torchvision_resave(path, py)
            else:
                continue

            if fetched:
                good, reason = verify(path, entry)
                if good:
                    ok(f"  {entry['path']} verified - {reason}")
                    break
                fail(f"  {entry['path']} downloaded but did not verify: {reason}")
                path.unlink(missing_ok=True)
                fetched = False

        if not fetched:
            still_missing.append(entry)

    # Third pass: everything left that Drive can supply, in one listing.
    drive_entries = [
        entry for entry in still_missing
        if any(source["kind"] == "drive" for source in entry["sources"])
    ]
    if drive_entries and not drive_folder:
        rule()
        note("Some checkpoints exist only on Google Drive. Re-run with:")
        note("  python setup.py --drive-folder <folder-id-or-url>")
        note("or set MODELS_DRIVE_FOLDER, or put the id in models.lock.json.")
    elif drive_entries:
        rule()
        report = drive_fetch(
            [
                {"relpath": entry["path"], "dest": str(models_root / entry["path"])}
                for entry in drive_entries
            ],
            drive_folder,
            py,
        )
        for entry in list(drive_entries):
            outcome = report.get(entry["path"], {"ok": False, "reason": "no result reported"})
            path = models_root / entry["path"]
            if not outcome.get("ok"):
                note(f"  {entry['path']}: {outcome.get('reason')}")
                continue
            good, reason = verify(path, entry)
            if good:
                ok(f"{entry['path']} verified - {reason}")
                still_missing.remove(entry)
            else:
                fail(f"{entry['path']} came down from Drive but did not verify: {reason}")
                path.unlink(missing_ok=True)

    for entry in still_missing:
        (p.error if entry["role"] == "runtime" else p.warning)(
            f"{entry['path']} could not be obtained"
        )
        if entry["role"] == "runtime":
            missing_runtime.append(entry)

    for entry in deferred:
        skip(f"{entry['path']}  {human(entry['bytes'])}  (training only, --with-training to fetch)")

    if missing_runtime:
        rule()
        note("The web app needs these to run steps 2 and 8:")
        for entry in missing_runtime:
            note(f"  {entry['path']}")
            for source in entry["sources"]:
                if source["kind"] == "url":
                    note(f"    {source['url']}")
                elif source["kind"] == "drive":
                    note(f"    Google Drive, at models/{entry['path']}")
            if "retrain_with" in entry:
                note(f"    or retrain: {entry['retrain_with']}")


def check_manifests(p: Problems) -> None:
    """A .pt without its .manifest.json will be refused by model.load_pinned."""
    tissue = data_versions.models_root() / "tissue_type"
    if not tissue.exists():
        return
    for checkpoint in sorted(tissue.glob("*.pt")):
        manifest = checkpoint.with_suffix(".manifest.json")
        if manifest.exists():
            continue
        p.error(
            f"{checkpoint.name} has no {manifest.name}. load_pinned refuses a "
            "checkpoint without its manifest; the manifests are committed, so "
            "this means the file was placed by hand rather than by setup."
        )


# --------------------------------------------------------------------------
#  readiness
# --------------------------------------------------------------------------


def run_readiness_checks(p: Problems) -> None:
    rule("5/5  Readiness checks")

    py = venv_python()
    if not py.exists():
        skip("no virtualenv, so the project checks cannot run")
        return

    for script, label in (
        ("scripts/check_qc_models.py", "step 2 (quality control)"),
        ("scripts/check_tissue_model.py", "step 8 (tissue type)"),
    ):
        path = BACKEND / script
        if not path.exists():
            skip(f"{script} not found")
            continue

        note(f"running {script}")
        proc = subprocess.run(
            [str(py), script], cwd=BACKEND, capture_output=True, text=True, timeout=900
        )
        body = (proc.stdout or "") + (proc.stderr or "")
        verdict = [
            line for line in body.splitlines()
            if "is ready" in line or "mode:" in line or "[ FAIL" in line or "Traceback" in line
        ]
        if proc.returncode == 0:
            ok(f"{label} ready")
            for line in verdict[:3]:
                note(f"  {line.strip()}")
        else:
            p.error(f"{label} check failed (exit {proc.returncode})")
            for line in body.strip().splitlines()[-12:]:
                note(f"  {line}")


def _locked(lock: Path) -> dict[str, str]:
    """`name==version` lines of a lock file, keyed by lower-case name."""
    pins: dict[str, str] = {}
    if not lock.is_file():
        return pins
    for line in lock.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "==" in line:
            name, version = line.split("==", 1)
            pins[name.strip().lower().replace("_", "-")] = version.strip()
    return pins


def check_registration_venv(p: Problems) -> None:
    """The Python 3.11 venv step 12 runs its transforms in, against its lock (P-20).

    Not built by this script - it needs a 3.11 interpreter beside the backend's 3.13,
    see `tissue_scoring_demo/valis_service/README.md` - but checked by it. SimpleITK is
    the registration method and numpy is held for VALIS's wheels, so a venv with either
    at another version fits different transforms under the same commit.
    """
    rule("Registration venv (valis_service)")
    service = APP / "valis_service"
    python = service / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        p.warning(
            f"{python} is missing, so step 12 cannot carry regions onto the IHC. Create it "
            "with Python 3.11 and install requirements.lock.txt - see valis_service/README.md."
        )
        return

    pins = _locked(service / "requirements.lock.txt")
    probe = subprocess.run(
        [str(python), "-c", "import SimpleITK, numpy; print(SimpleITK.Version_VersionString()); print(numpy.__version__)"],
        capture_output=True, text=True, timeout=300,
    )
    if probe.returncode != 0:
        p.warning(f"the registration venv cannot import SimpleITK and numpy: {probe.stderr.strip()[-200:]}")
        return
    sitk_version, numpy_version = probe.stdout.split()[:2]
    for name, found in (("simpleitk", sitk_version), ("numpy", numpy_version)):
        wanted = pins.get(name)
        if wanted and wanted != found:
            p.warning(f"registration venv has {name} {found}; the lock says {wanted}")
        else:
            ok(f"registration venv: {name} {found}")


def maybe_clone_grandqc(clone: bool) -> None:
    target = ROOT / "grandqc"
    if not clone:
        return
    if target.exists():
        ok("grandqc/ present")
        return
    git = shutil.which("git")
    if not git:
        warn("git not found; skipping the GrandQC clone")
        return
    note("cloning GrandQC (reference only; the app does not need it)")
    if run([git, "clone", "--depth", "1", GRANDQC_REPO, str(target)], label="git clone grandqc"):
        ok("grandqc/ cloned")


# --------------------------------------------------------------------------
#  entry point
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Set up the Breast Cancer IHC Tissue Scoring web app.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true",
                        help="audit what is present and download nothing")
    parser.add_argument("--models-only", action="store_true",
                        help="only fetch and verify checkpoints")
    parser.add_argument("--skip-models", action="store_true",
                        help="set up code dependencies but no checkpoints")
    parser.add_argument("--with-training", action="store_true",
                        help="also fetch training-only weights, including BEETLE (1.8 GB)")
    parser.add_argument("--drive-folder", metavar="ID_OR_URL",
                        help="shared Google Drive folder holding the checkpoints")
    parser.add_argument("--recreate-venv", action="store_true",
                        help="delete backend/.venv and rebuild it")
    parser.add_argument("--clone-grandqc", action="store_true",
                        help="also clone the upstream GrandQC repo for reference")
    args = parser.parse_args()

    rule("Breast Cancer IHC Tissue Scoring - setup")
    note(f"repository: {ROOT}")
    note(f"mode: {'check only' if args.check else 'install'}")

    if not APP.exists():
        fail(f"{APP.name} not found next to this script; is the clone complete?")
        return 1

    problems = Problems()
    check_prerequisites(
        problems,
        skip_node=args.models_only or args.check,
        why="--check audits checkpoints only" if args.check else "--models-only",
    )

    if not args.check and not args.models_only:
        setup_venv(problems, recreate=args.recreate_venv)
        setup_env_file()
        setup_frontend(problems)
        maybe_clone_grandqc(args.clone_grandqc)

    if not args.skip_models:
        setup_models(
            problems,
            check_only=args.check,
            with_training=args.with_training,
            drive_folder=args.drive_folder,
        )
        check_manifests(problems)

    if not args.models_only:
        check_registration_venv(problems)

    if not args.check and not args.models_only:
        run_readiness_checks(problems)

    rule("Summary")
    if problems.errors:
        fail(f"{len(problems.errors)} problem(s) block a working app:")
        for message in problems.errors:
            note(f"  - {message}")
    if problems.warnings:
        warn(f"{len(problems.warnings)} warning(s), none of them fatal:")
        for message in problems.warnings:
            note(f"  - {message}")

    if not problems.errors:
        ok("setup complete")
        if not args.check:
            print()
            note("Start the app, from this directory:")
            note("  start.bat        (Windows)   then stop.bat to shut it down")
            print()
            note("  app  http://localhost:5173")
            note("  api  http://127.0.0.1:8000/docs")
    print()
    return 1 if problems.errors else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)
