"""Gate G0 and G1 from the command line: fetch BCSS, and prove it is the right release.

The same code the notebook runs - everything lives in `src/`, and both callers are thin.
This exists because the download is hours long and a notebook kernel is a poor place to
put an hours-long job: no resume across a restart, no log to read afterwards, and a
progress bar that only exists while a browser tab is open.

    python scripts/01_download.py            # G0, then download, then G1
    python scripts/01_download.py --verify    # G1 only, on what is already on disk
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

import backend_path  # noqa: F401 - puts the demo backend on sys.path
import bcss
import download_bcss
import hchannel
from app.common.imaging import optical_density
from app.common.stains import REFERENCE_BY_NAME, RUIFROK_INVERSE, unmix


def gate_g0() -> None:
    """The H channel this notebook computes must be the H channel the API serves."""
    print("== G0: one definition of the haematoxylin channel ==")

    vector = np.asarray(REFERENCE_BY_NAME["haematoxylin"])
    rgb = np.tile((255.0 * 10.0 ** (-0.8 * vector)).astype(np.uint8), (64, 64, 1))
    white = (255.0, 255.0, 255.0)

    ours = hchannel.haematoxylin_od(rgb, white)
    od = optical_density(rgb, white)
    by_hand = unmix(od.reshape(-1, 3), RUIFROK_INVERSE)[:, 0].reshape(od.shape[:2])

    print(f"  ours {ours.mean():.4f} | by hand {by_hand.mean():.4f} | expected 0.8000")
    assert np.allclose(ours, by_hand, atol=1e-6), "the script and the pipeline disagree"

    dab = np.tile(
        (255.0 * 10.0 ** (-0.8 * np.asarray(REFERENCE_BY_NAME["dab"]))).astype(np.uint8),
        (64, 64, 1),
    )
    leak = float(hchannel.haematoxylin_od(dab, white).mean())
    print(f"  pure DAB leaks {leak:.4f} into H (the brown is what we drop)")
    assert abs(leak) < 0.05

    print("  G0 OK\n")


def gate_g1(root: Path, *, sample_masks: int) -> None:
    """Raw 22-class release, complete, with `dcis` separate from `tumor`."""
    print("== G1: is this the raw release, and is it whole? ==")
    summary = download_bcss.verify(root, sample_masks=sample_masks)

    print(f"  regions      : {summary['regions']} (expected {download_bcss.EXPECTED_REGIONS})")
    print(f"  slides       : {summary['slides']}")
    print(f"  institutions : {len(summary['institutions'])}")
    print(f"  train / test : {summary['train_regions']} / {summary['held_out_regions']}"
          f"  (held out: {' '.join(sorted(bcss.TEST_INSTITUTIONS))})")
    print(f"  on disk      : {summary['bytes_on_disk'] / 1e9:.2f} GB")
    print(f"  dcis  pixels sampled: {summary['dcis_pixels_sampled']:,}")
    print(f"  tumor pixels sampled: {summary['tumor_pixels_sampled']:,}")

    print("\n  code census over the sampled masks:")
    for name, count in sorted(summary["code_census"].items(), key=lambda kv: -kv[1]):
        print(f"    {name:<26} {count:>13,}")

    manifest = download_bcss.write_manifest(root, summary)
    print(f"\n  wrote {manifest}")
    print("  G1 OK - dcis is code 20 and separate from tumor\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true",
                        help="skip the download; check what is already on disk")
    parser.add_argument("--route", choices=("girder", "gdrive"), default="girder",
                        help="girder (default): crop each ROI server-side and pull its "
                             "mask from figshare - no quota. gdrive: the Drive folder, "
                             "which rate-limits after ~40 files")
    parser.add_argument("--sample-masks", type=int, default=20,
                        help="masks to read for the code census (default 20)")
    args = parser.parse_args()

    backend_path.ensure_dirs()
    gate_g0()

    root = backend_path.BCSS_DIR
    if not args.verify:
        free = shutil.disk_usage(backend_path.DATA_DIR).free / 1e9
        print(f"== download: {free:.1f} GB free ==")
        if free < 15:
            print("  ! under 15 GB free; BCSS is ~10 GB and the export needs headroom")

        codes = download_bcss.check_codes_file(
            download_bcss.fetch_gtruth_codes(root / "meta" / "gtruth_codes.tsv")
        )
        print(f"  gtruth_codes.tsv: {len(codes)} codes, matching bcss.GT_CODES")
        print(f"  tumor = {codes['tumor']}, dcis = {codes['dcis']} (these must differ)")

        if args.route == "girder":
            root = download_bcss.download_girder(root)
        else:
            root = download_bcss.download_gdrive(root)
        print(f"  images and masks landed in {root}\n")
    else:
        images, _ = bcss.locate_pairs(root)
        root = images.parent

    gate_g1(root, sample_masks=args.sample_masks)
    print(f"BCSS root: {root}")
    print("Next: python scripts/02_export.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
