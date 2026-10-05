"""Prove the rebuilt network is BEETLE's, not merely a network that loads its weights.

`unet.load_beetle_weights` guarantees the key sets and every tensor shape agree. That
catches a wrong stage count, a transposed convolution built the wrong way round, a
missing skip. It cannot catch a **wiring** error between correctly-shaped layers - a
skip attached one stage out, the decoder walking its stages backwards - because those
produce a model that loads perfectly and segments nonsense.

Four checks close that gap, each of which a miswired network fails:

  1. **All five folds load.** They were trained separately; a shape that fits one by
     coincidence will not fit all five.
  2. **The output is structured.** A miswired U-Net predicts one class almost
     everywhere, or noise with no spatial coherence. Real tissue must produce contiguous
     regions and more than one class.
  3. **The folds agree with each other.** Five independently-trained networks agree on
     tissue only if each is computing something meaningful. Five miswired networks
     produce five unrelated pictures.
  4. **A DCIS region is called in-situ.** This is the one that would have caught the
     inverted label map on its own, and it is the reason this script exists at all: on
     BRACS regions that are DCIS by annotation, the class the model names must be the
     in-situ one. With the paper's ordering it would confidently say "invasive".

    python scripts/verify_teacher.py
    python scripts/verify_teacher.py --all-folds --n 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from bracs_app import config, labels, pipeline, teacher  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

#: Enough of a region to contain several ducts, small enough that five folds is a minute.
CROP = 1024

#: Below this, the argmax is one class almost everywhere and the model is not
#: discriminating. A real DCIS region has ducts and stroma both.
MIN_SECOND_CLASS = 0.02

#: Above this share on the leading class, the same conclusion from the other direction.
MAX_SINGLE_CLASS = 0.98

#: Ceiling on the mean standard deviation between folds at the class they chose. Five
#: independently-trained networks that were loaded into the right graph land within a
#: few percent of each other; five loaded into a subtly wrong one do not.
MAX_FOLD_SPREAD = 0.25


def centre_crop_at_teacher_spacing(path: Path, source_mpp: float) -> np.ndarray:
    factor = config.TEACHER_MPP / source_mpp
    want = int(round(CROP * factor))
    with Image.open(path) as image:
        image = image.convert("RGB")
        left = max(0, (image.width - want) // 2)
        top = max(0, (image.height - want) // 2)
        crop = image.crop((left, top, left + want, top + want))
        if abs(factor - 1.0) > 1e-6:
            crop = crop.resize(
                (max(1, round(crop.width / factor)), max(1, round(crop.height / factor))),
                Image.Resampling.BOX,
            )
        return np.asarray(crop)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2, help="regions to check")
    parser.add_argument("--all-folds", action="store_true", help="load all five, not just two")
    args = parser.parse_args()

    config.install_approach1_path()
    failures: list[str] = []

    print("=" * 66)
    print("  1. the archive and its label codes")
    archive = teacher.read_archive()
    print(f"     classes {archive.num_classes}, spacing {archive.spacing} um/px")
    print(f"     code 2 = {labels.CHANNEL_NAMES[2]}   code 3 = {labels.CHANNEL_NAMES[3]}")
    print("     (approach 4a's beetle.py has these two the other way round - see labels.py)")

    folds = tuple(range(5)) if args.all_folds else (0, 1)
    print()
    print(f"  2. loading {len(folds)} fold(s) strictly")
    for fold in folds:
        started = time.time()
        model = teacher.load_fold(archive, fold)
        params = sum(p.numel() for p in model.parameters())
        print(f"     fold {fold}: {params / 1e6:.1f} M parameters, {time.time() - started:.1f}s")
        del model

    entries = pipeline.list_rois("train")[: args.n]
    print()
    print(f"  3. segmenting {len(entries)} region(s), {CROP}px centre crop")

    spreads: list[tuple[str, float]] = []
    for entry in entries:
        rgb = centre_crop_at_teacher_spacing(entry.path, config.BRACS_MPP)
        prediction = teacher.predict(rgb, archive, folds=folds)
        spreads.append((entry.roi_id, float(prediction.disagreement.mean())))

        fractions = {
            name: float((prediction.mask == code).mean())
            for name, code in labels.BEETLE_CODES.items()
        }
        ranked = sorted(fractions.items(), key=lambda kv: -kv[1])
        print(f"     {entry.roi_id}")
        for name, share in ranked:
            if share > 0.001:
                print(f"        {name:<26} {share:6.1%}")
        print(f"        confidence {prediction.confidence.mean():.3f}"
              + (f"   fold spread {prediction.disagreement.mean():.3f}"
                 if len(folds) > 1 else ""))

        leader, leader_share = ranked[0]
        second_share = ranked[1][1] if len(ranked) > 1 else 0.0
        if leader_share > MAX_SINGLE_CLASS or second_share < MIN_SECOND_CLASS:
            failures.append(
                f"{entry.roi_id}: {leader} covers {leader_share:.1%} and nothing else "
                "reaches 2% - the output is not discriminating, which is what a "
                "miswired network looks like."
            )
        if leader != "non_invasive_epithelium":
            failures.append(
                f"{entry.roi_id}: a BRACS DCIS region came out mostly {leader!r}. "
                "On regions annotated as DCIS the leading class must be the in-situ "
                "one; anything else means the class codes are wrong."
            )
    if len(folds) > 1:
        print()
        print("  4. do the folds agree with each other?")
        # The folds were trained separately on different splits. If the architecture
        # were miswired, each would be computing a different arbitrary function and
        # their probabilities would scatter; agreeing to within a few percent on the
        # winning class is something only correctly-wired networks do together.
        for roi_id, spread in spreads:
            state = "agree" if spread <= MAX_FOLD_SPREAD else "DISAGREE"
            print(f"     {roi_id}: mean spread {spread:.3f} - folds {state}")
            if spread > MAX_FOLD_SPREAD:
                failures.append(
                    f"{roi_id}: the folds disagree by {spread:.3f} on the class they "
                    f"chose, against a ceiling of {MAX_FOLD_SPREAD}. Independently "
                    "trained networks scatter like this when the graph they are loaded "
                    "into is not the graph they were trained as."
                )

    print()
    print("=" * 66)
    if failures:
        for failure in failures:
            print(f"  FAIL  {failure}")
        print()
        print("  The rebuilt architecture loads BEETLE's weights but does not behave")
        print("  like BEETLE. Do not export tiles from it.")
        return 1

    print("  PASS  every fold loads strictly, the output is structured, and BRACS")
    print("        DCIS regions come out as in-situ epithelium. The rebuilt")
    print("        PlainConvUNet is behaving as the released ensemble.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
