"""P-20: the transform chain step 12 relies on, round-tripped through real files.

Level-0 H&E -> aligned -> working -> stored SimpleITK transform -> working -> aligned ->
level-0 IHC. Every link is a matrix whose direction and sign could be read either way,
and none of them was tested: a transposed rotation or an inverted transform moves every
carried region and still produces plausible-looking rings.

So the transform is written to disk by SimpleITK itself, in the registration venv, read
back by the real worker (`apply_transform.py`) through `transform_warp`, and the answers
are checked against numbers worked out by hand - not against a second implementation of
the same formulas, which would agree with the first whatever convention both got wrong.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "slide_registration"))

import common  # noqa: E402
import transform_warp  # noqa: E402

pytestmark = pytest.mark.skipif(
    not common.VALIS_PYTHON.exists(),
    reason="the registration venv is not installed, so the real worker cannot run",
)

IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def _write_transform(path: Path, matrix: list[list[float]], translation: list[float]) -> None:
    """An AffineTransform written by SimpleITK in the registration venv - the real format."""
    script = (
        "import SimpleITK as sitk, sys, json;"
        "m, t, out = json.loads(sys.argv[1]), json.loads(sys.argv[2]), sys.argv[3];"
        "a = sitk.AffineTransform(2);"
        "a.SetMatrix([m[0][0], m[0][1], m[1][0], m[1][1]]);"
        "a.SetTranslation(t);"
        "sitk.WriteTransform(a, out)"
    )
    subprocess.run(
        [str(common.VALIS_PYTHON), "-c", script, json.dumps(matrix), json.dumps(translation), str(path)],
        check=True, capture_output=True, timeout=300,
    )
    assert path.is_file() and path.stat().st_size > 0


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A case folder `transform_warp` reads, with every matrix set by the test."""
    directory = tmp_path / "CASE"
    (directory / "transforms").mkdir(parents=True)
    monkeypatch.setattr(transform_warp, "_case_for", lambda he, ihc: ("CASE", "A"))
    monkeypatch.setattr(transform_warp.common, "case_dir", lambda name: directory)

    def build(*, he, ihc, work_scale, matrix, translation):
        (directory / "aligned.json").write_text(json.dumps({
            "targetMpp": 8.0,
            "slides": {"HE": {"level0ToAligned": he}, "A": {"level0ToAligned": ihc}},
        }), encoding="utf-8")
        (directory / "transforms.json").write_text(json.dumps({
            "ok": True, "work_scale": work_scale,
            "saved": {"A": {"ok": True, "file": "A.h5", "method": "mattes", "nmi": 1.0}},
        }), encoding="utf-8")
        _write_transform(directory / "transforms" / "A.h5", matrix, translation)

    return build


def _warp(points: list[list[float]]) -> np.ndarray:
    out = transform_warp.warp_for_pair("he", "ihc", [points])
    return np.asarray(out["rings_level0"][0])


def test_identity_everywhere_returns_the_points_unchanged(case):
    case(he=IDENTITY, ihc=IDENTITY, work_scale=1.0, matrix=[[1, 0], [0, 1]], translation=[0, 0])
    points = [[100.0, 200.0], [300.0, 50.0]]
    assert np.allclose(_warp(points), points, atol=1e-6)


def test_the_stored_transform_maps_he_to_ihc_not_the_reverse(case):
    """A +5 px x-shift in the file must move H&E points +5 px towards the IHC.

    SimpleITK's `TransformPoint` maps fixed to moving. If the chain had the direction
    backwards every point would land at -5 instead, and a region would be carried
    exactly the wrong way.
    """
    case(he=IDENTITY, ihc=IDENTITY, work_scale=1.0, matrix=[[1, 0], [0, 1]], translation=[5, 0])
    assert np.allclose(_warp([[100.0, 200.0]]), [[105.0, 200.0]], atol=1e-6)


def test_rotation_sign_in_the_placement_matrix(case):
    """`level0ToAligned` rows act on (x, y, 1): [[0,-1,0],[1,0,0]] sends (100, 0) to (0, 100).

    Checked as numbers rather than through the formula, because a transposed matrix -
    a rotation the other way - is exactly the mistake a re-derivation would share.
    """
    quarter_turn = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]
    case(he=quarter_turn, ihc=IDENTITY, work_scale=1.0, matrix=[[1, 0], [0, 1]], translation=[0, 0])
    assert np.allclose(_warp([[100.0, 0.0]]), [[0.0, 100.0]], atol=1e-6)


def test_the_ihc_placement_is_inverted_on_the_way_back(case):
    """The IHC's own placement is undone at the end: aligned -> its level-0."""
    doubled = [[2.0, 0.0, 10.0], [0.0, 2.0, 20.0]]  # level-0 (x, y) -> (2x + 10, 2y + 20)
    case(he=IDENTITY, ihc=doubled, work_scale=1.0, matrix=[[1, 0], [0, 1]], translation=[0, 0])
    # H&E (110, 220) is aligned (110, 220), which is IHC level-0 ((110-10)/2, (220-20)/2).
    assert np.allclose(_warp([[110.0, 220.0]]), [[50.0, 100.0]], atol=1e-6)


def test_the_working_scale_is_applied_and_removed(case):
    """A transform fitted at half resolution moves level-0 points by twice its shift."""
    case(he=IDENTITY, ihc=IDENTITY, work_scale=0.5, matrix=[[1, 0], [0, 1]], translation=[5, 0])
    assert np.allclose(_warp([[100.0, 200.0]]), [[110.0, 200.0]], atol=1e-6)


def test_the_whole_chain_with_every_link_non_trivial(case):
    """Rotation, scale and shift in all four links at once, against a hand-worked point."""
    he = [[0.0, -0.5, 1000.0], [0.5, 0.0, 0.0]]       # quarter turn, half scale, shift
    ihc = [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]          # half scale
    fit = [[0.0, 1.0], [-1.0, 0.0]]                   # a quarter turn the other way
    case(he=he, ihc=ihc, work_scale=0.25, matrix=fit, translation=[10.0, -4.0])
    # (400, 200) -> aligned (0.5*... ) = (-100 + 1000, 200) = (900, 200)
    #            -> working (225, 50)
    #            -> fit: (50 + 10, -225 - 4) = (60, -229)
    #            -> aligned (240, -916)
    #            -> IHC level-0 (480, -1832)
    assert np.allclose(_warp([[400.0, 200.0]]), [[480.0, -1832.0]], atol=1e-6)
