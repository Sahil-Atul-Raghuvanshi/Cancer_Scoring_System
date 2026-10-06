"""Build one isolated Python 3.11 environment for an approach that cannot share the backend's.

    python stage_env.py deepliif | hovernet | cellpose

Each gets its own venv under `envs/`, so nothing here can disturb the backend's or the
registration venv. Research packages rarely state their dependencies completely, so the
check is an actual import of what the stage will use, and a `ModuleNotFoundError` is
answered by installing the named module and trying again (bounded).
"""

from __future__ import annotations

import re
import subprocess
import sys

import common

TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
#: Import name -> distribution name, where they differ.
DISTRIBUTION = {
    "cv2": "opencv-python-headless", "skimage": "scikit-image", "PIL": "pillow",
    "sklearn": "scikit-learn", "yaml": "pyyaml", "imgaug": "imgaug",
}

SPECS = {
    "deepliif": {
        "packages": [
            ["torch==2.8.0", "torchvision", "--index-url", TORCH_INDEX],
            ["numpy<2", "opencv-python-headless", "scikit-image<0.23", "dask[array]", "numba",
             "dominate", "requests", "imagecodecs", "zarr<3", "tifffile", "pillow", "click"],
            ["deepliif==1.2.7", "--no-deps"],
        ],
        # deepliif.util imports bioformats/javabridge at module load for whole-slide I/O,
        # which needs a JVM; the in-memory inference used here never calls them.
        "stubs": {"bioformats": ["omexml"], "javabridge": []},
        "verify": "from deepliif.models import infer_modalities; print('ok')",
    },
    "hovernet": {
        "packages": [
            ["torch==2.8.0", "--index-url", TORCH_INDEX],
            ["numpy<2", "opencv-python-headless", "scipy", "scikit-image<0.23", "pillow"],
        ],
        "stubs": {},
        "paths": ["downloads/hovernet_code/hover_net-master"],
        "verify": "from models.hovernet.net_desc import HoVerNet; "
                  "from models.hovernet.post_proc import process; print('ok')",
    },
    "cellpose": {
        "packages": [
            ["torch==2.8.0", "--index-url", TORCH_INDEX],
            ["cellpose<4", "numpy<2"],
        ],
        "stubs": {},
        "verify": "from cellpose import models, train; print('ok')",
    },
}
LOG = common.LOGS / "env.log"


def python_of(name: str):
    return common.ENVS / name / "Scripts" / "python.exe"


def _pip(py, args: list[str]) -> int:
    return subprocess.run([str(py), "-m", "pip", "install", "--quiet", *args], check=False).returncode


def _stubs(py, stubs: dict[str, list[str]]) -> None:
    site = subprocess.run([str(py), "-c", "import site; print(site.getsitepackages()[-1])"],
                          capture_output=True, text=True, check=True).stdout.strip()
    from pathlib import Path

    for package, submodules in stubs.items():
        folder = Path(site) / package
        folder.mkdir(exist_ok=True)
        (folder / "__init__.py").write_text(
            "# P-03 stub: imported at module load, never called by in-memory inference.\n"
            "def __getattr__(name):\n    raise RuntimeError(f'{__name__}.{name} is a stub')\n",
            encoding="utf-8")
        for sub in submodules:
            (folder / f"{sub}.py").write_text("# P-03 stub\n", encoding="utf-8")


def build(name: str) -> int:
    spec = SPECS[name]
    ok_file = common.ENVS / f"{name}.ok"
    if ok_file.exists():
        return 0
    py = python_of(name)
    if not py.exists():
        common.say(f"{name}: creating venv with {common.PY311}", LOG)
        subprocess.run([str(common.PY311), "-m", "venv", str(common.ENVS / name)], check=True)
        _pip(py, ["--upgrade", "pip"])
    for group in spec["packages"]:
        common.heartbeat("env", f"{name} {group[0]}")
        rc = _pip(py, group)
        common.say(f"{name}: pip {group[:3]} -> {rc}", LOG)
    _stubs(py, spec.get("stubs", {}))

    paths = [str(common.OUT / p) for p in spec.get("paths", [])]
    env_code = "import sys; sys.path[:0] = %r; " % paths + spec["verify"]
    for attempt in range(12):
        result = subprocess.run([str(py), "-c", env_code], capture_output=True, text=True, check=False)
        if result.returncode == 0 and "ok" in result.stdout:
            ok_file.write_text("verified", encoding="utf-8")
            common.say(f"{name}: verified", LOG)
            return 0
        missing = re.search(r"No module named '([^'.]+)", result.stderr)
        common.say(f"{name}: verify attempt {attempt}: {result.stderr.strip()[-400:]}", LOG)
        if not missing:
            return 1
        module = missing.group(1)
        _pip(py, [DISTRIBUTION.get(module, module)])
    return 1


if __name__ == "__main__":
    common.ensure_dirs()
    sys.exit(build(sys.argv[1]))
