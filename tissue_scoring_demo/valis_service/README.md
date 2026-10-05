# `valis_service` — the isolated registration environment

A separate Python 3.11 environment that holds SimpleITK. Step 12 and
`slide_registration/` run the three scripts here as subprocesses:

| Script | Called by | What it does |
| --- | --- | --- |
| `register_methods.py` | `slide_registration/methods_run.py`, `fit_best.py` | fits each H&E→IHC pair by outline, Mattes mutual information, affine, B-spline and mask methods, and stores the winning transform |
| `apply_transform.py` | `slide_registration/transform_warp.py`, which step 12 calls | maps step 11's region rings through the stored transform |
| `check_plausible.py` | `slide_registration/check_transforms.py` | checks that a stored transform does not fold or tear tissue |

**VALIS is no longer used.** Its feature matching failed outright on nearly
unstained IHC sections, and Mattes mutual information over a similarity
transform replaced it (see
`tissue_scoring_demo/docs/registration/REGISTRATION-PLAN.md` section 14). The
VALIS driver, its stack registration, and its smoke, tuning and refusal tests
are in `decrecated_code/valis_registration/`. Their old run output is in
`data/deprecated/valis_service_runs/`.

The folder and the backend's `valis_service_dir` setting keep the old name, so
existing environments and `.env` files keep working.

## Why a second Python

`requirements.txt` still installs `valis-wsi`, because that is what brings in
SimpleITK at a version these scripts were measured against. `valis-wsi` pins
`numpy<2.0`, and there is no numpy 1.x build for the Python 3.13 the backend
runs on, so this cannot share the backend's interpreter.

## Setting it up

Python 3.11 (or 3.10/3.12) has to exist on the machine. On Windows:

```
winget install --id Python.Python.3.11 -e --scope user --silent
```

Then, from `tissue_scoring_demo/`:

```
"$LOCALAPPDATA/Programs/Python/Python311/python.exe" -m venv valis_service/.venv
valis_service/.venv/Scripts/python -m pip install --upgrade pip
valis_service/.venv/Scripts/python -m pip install -r valis_service/requirements.txt
```

**`pyvips-binary` is not optional** while `valis-wsi` is the dependency: the
`pyvips` wheel is only the Python binding, and without the binary package it
fails at import with `cannot load library 'libvips-42.dll'`.

## Checking it works

```
python slide_registration/transform_warp.py CAN_00303 --marker A --selftest
```

run from the repository root with the backend's Python. It warps points
through the stored transform and back, and reports the round-trip error.
The backend reports the environment as available when `.venv` and
`apply_transform.py` are both present.
