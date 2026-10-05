"""Carry rings from an H&E slide onto an IHC slide using the stored winning transform.

    python transform_warp.py CAN_00303 --marker A --selftest

This is what step 12 calls instead of `valis_align.warp_rings`. It owns the whole
coordinate chain, and the chain is the part where a mistake produces a mask on the wrong
tissue with no error message anywhere.

    H&E level-0   --M_he-------------->  H&E aligned render
    H&E aligned   --x work_scale------>  working resolution
    working       --stored transform->   IHC working resolution
    IHC working   --/ work_scale----->   IHC aligned render
    IHC aligned   --inverse(M_ihc)--->   IHC level-0

`M_he` and `M_ihc` are the 2x3 affines Phase 3 wrote into `aligned.json`: scale to the
render's resolution, translate the tissue centroid to the canvas centre, rotate into the
H&E's frame. `work_scale` is recorded in `transforms.json` rather than recomputed, because
the transform was fitted at a downscaled resolution and inferring that factor at the call
site is exactly how an off-by-a-scale-factor bug gets in.

Only the middle hop needs SimpleITK, which lives in the other virtual environment, so that
one step is a subprocess. Everything else is four lines of matrix arithmetic done here.

**A case is found by looking the upload ids up in the demo's own case records**, so this
needs no new registry and cannot disagree with what the rest of the pipeline thinks a case
is.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402


class NoStoredTransform(RuntimeError):
    """This pair has no stored transform, so nothing can be carried across."""


def _case_for(he_upload_id: str, ihc_upload_id: str) -> tuple[str, str]:
    """(case id, marker) for a pair, from the demo's own case records."""
    directory = common.DATA / "demo" / "cases"
    if directory.is_dir():
        for path in sorted(directory.glob("*.json")):
            record = common.read_json(path) or {}
            if (
                record.get("he_upload_id") == he_upload_id
                and record.get("ihc_upload_id") == ihc_upload_id
            ):
                return record.get("case_id", ""), record.get("marker", "")
    raise NoStoredTransform(
        f"no case record pairs H&E {he_upload_id} with IHC {ihc_upload_id}"
    )


def _matrix(rows) -> np.ndarray:
    out = np.eye(3)
    out[:2, :] = np.asarray(rows, dtype=float)
    return out


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (np.column_stack([points, np.ones(len(points))]) @ matrix.T)[:, :2]


class StoredWarper:
    """Warps rings for one case using the transform its method sweep chose."""

    def __init__(self, case: str) -> None:
        self.case = case
        directory = common.case_dir(case)
        self.aligned = common.read_json(directory / "aligned.json")
        self.manifest = common.read_json(directory / "transforms.json")
        if not self.aligned or not self.manifest or not self.manifest.get("ok"):
            raise NoStoredTransform(
                f"{case}: no stored transforms. Run methods_run.py then fit_best.py."
            )
        self.work_scale = float(self.manifest["work_scale"])
        self.directory = directory

    def method_for(self, marker: str) -> str:
        return (self.manifest.get("saved", {}).get(marker) or {}).get("method", "")

    def warp(self, rings, marker: str) -> dict:
        """Rings in H&E level-0 pixels -> the same rings in the marker's level-0 pixels."""
        saved = (self.manifest.get("saved") or {}).get(marker)
        if not saved or not saved.get("ok"):
            raise NoStoredTransform(
                f"{self.case}/{marker}: no usable stored transform "
                f"({saved.get('error') if saved else 'not fitted'})"
            )

        slides = self.aligned["slides"]
        if "HE" not in slides or marker not in slides:
            raise NoStoredTransform(f"{self.case}: {marker} or HE missing from aligned.json")

        he_affine = _matrix(slides["HE"]["level0ToAligned"])
        ihc_inverse = np.linalg.inv(_matrix(slides[marker]["level0ToAligned"]))

        # level-0 -> aligned -> working
        in_work = []
        for ring in rings:
            if not ring:
                in_work.append([])
                continue
            aligned = _apply(he_affine, np.asarray(ring, dtype=float))
            in_work.append((aligned * self.work_scale).tolist())

        request = {
            "transform": str((self.directory / "transforms" / saved["file"]).resolve()),
            "rings": in_work,
        }
        with tempfile.TemporaryDirectory() as scratch:
            req = pathlib.Path(scratch) / "request.json"
            res = pathlib.Path(scratch) / "response.json"
            req.write_text(json.dumps(request), encoding="utf-8")
            completed = subprocess.run(  # noqa: S603 - fixed interpreter and script
                [str(common.VALIS_PYTHON),
                 str(common.VALIS_DIR / "apply_transform.py"), str(req), str(res)],
                capture_output=True, text=True, timeout=900,
            )
            if not res.exists():
                tail = (completed.stderr or completed.stdout or "")[-1000:]
                raise NoStoredTransform(
                    f"{self.case}/{marker}: transform worker wrote nothing "
                    f"(exit {completed.returncode}). {tail}"
                )
            payload = json.loads(res.read_text(encoding="utf-8"))
        if not payload.get("ok"):
            raise NoStoredTransform(f"{self.case}/{marker}: {payload.get('error')}")

        # working -> aligned -> level-0
        out = []
        for ring in payload["warped_rings"]:
            if not ring:
                out.append([])
                continue
            aligned = np.asarray(ring, dtype=float) / self.work_scale
            out.append(_apply(ihc_inverse, aligned).tolist())

        payload["rings_level0"] = out
        payload["method"] = saved["method"]
        payload["nmi"] = saved.get("nmi")
        # The round trip comes back in working pixels; report it in microns, which is what
        # every threshold in this pipeline is stated in.
        mpp = float(self.aligned["targetMpp"]) / self.work_scale
        for key in ("round_trip_median_px", "round_trip_max_px"):
            if key in payload:
                payload[key.replace("_px", "_um")] = round(payload[key] * mpp, 3)
        return payload


def warp_for_pair(he_upload_id: str, ihc_upload_id: str, rings) -> dict:
    """The entry point step 12 uses: a pair of upload ids in, warped rings out."""
    case, marker = _case_for(he_upload_id, ihc_upload_id)
    return StoredWarper(case).warp(rings, marker)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case")
    parser.add_argument("--marker", default="A")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    warper = StoredWarper(args.case)
    print(f"{args.case}: work scale {warper.work_scale:.4f}")
    for code, saved in sorted((warper.manifest.get("saved") or {}).items()):
        print(f"  {code}: {saved.get('method'):9} nmi {saved.get('nmi')} ok={saved.get('ok')}")

    if args.selftest:
        # A lattice over the H&E's own tissue, carried across. The test is that the
        # points land inside the IHC slide and that the transform inverts.
        from PIL import Image

        manifest = common.read_json(common.case_dir(args.case) / "render.json")
        entry = manifest["slides"]["HE"]
        mask = np.asarray(
            Image.open(common.case_dir(args.case) / "render" / entry["maskFile"]).convert("L")
        ) > 127
        rows, cols = np.nonzero(mask)
        stride = max(1, rows.size // 300)
        rows, cols = rows[::stride], cols[::stride]
        scale, offset = entry["scale"], entry["offset"]
        level0 = np.column_stack([(cols - offset[0]) / scale, (rows - offset[1]) / scale])

        got = warper.warp([level0.tolist()], args.marker)
        carried = np.asarray(got["rings_level0"], dtype=float)[0]
        moved = np.linalg.norm(carried - level0, axis=1) * float(entry["baseMpp"])
        print()
        print(f"selftest {args.case}/{args.marker} via {got['method']}:")
        print(f"  points               {len(level0)}")
        print(f"  median displacement  {np.median(moved):.0f} um")
        print(f"  round trip median    {got.get('round_trip_median_um', 'n/a')} um")
        print(f"  round trip max       {got.get('round_trip_max_um', 'n/a')} um")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
