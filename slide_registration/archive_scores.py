"""Take a dated, read-only copy of everything the scoring pass has produced so far.

    python archive_scores.py [--label before-stack-registration]

**Copies, never moves.** The history service elsewhere in this project archives by
renaming, which is right for slide work that must not be duplicated on disk. This is the
opposite situation: what is being preserved is nine scored rows that took two overnight
passes to earn, and the reason to preserve them is that the next pass may overwrite them.
A move would leave the live pass with nothing to fall back to if the new approach turns
out worse; a copy costs a few hundred kilobytes.

Run before any re-scoring, so there is always a version of the numbers that predates the
change being evaluated. Two sets of scores produced by different registrations are only
comparable if both still exist.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402

ARCHIVE = common.DATA / "score_archive"

#: What a scoring pass produces, and therefore what has to be copied for the archive to
#: be a complete record rather than a souvenir. The CSV alone is not enough: it is
#: rebuilt from the per-case JSON, so without those the archive cannot be regenerated or
#: audited.
TARGETS = [
    common.RESULTS / "oncostem_ai_scores.csv",
    common.DATA / "score_all_slides" / "results",
    common.DATA / "score_all_slides" / "state",
    common.DATA / "score_all_slides" / "logs",
    # The tiles mode keeps entirely separate files, and the first version of this list
    # omitted all three - so an "archive everything" call quietly captured only half the
    # evidence. A backup that silently misses what you are about to delete is worse than
    # no backup, because it is trusted.
    common.RESULTS / "oncostem_ai_scores_tiles.csv",
    common.DATA / "score_all_slides" / "results_tiles",
    common.DATA / "score_all_slides" / "state_tiles",
]


def archive(label: str) -> pathlib.Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    destination = ARCHIVE / f"{stamp}_{label}"
    destination.mkdir(parents=True, exist_ok=True)

    copied = []
    for source in TARGETS:
        if not source.exists():
            common.say(f"  (absent, nothing to copy) {source.name}")
            continue
        target = destination / source.name
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)
        copied.append(source.name)
        common.say(f"  copied {source.name}")

    common.write_json(
        destination / "manifest.json",
        {
            "label": label,
            "createdAt": time.strftime("%Y-%m-%d %H:%M:%S"),
            "copied": copied,
            "note": (
                "Copied, not moved. The originals are untouched and the live pass still "
                "reads them."
            ),
        },
    )
    common.say(f"archived to {destination}")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="before-stack-registration")
    args = parser.parse_args()
    common.ensure_dirs()
    archive(args.label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
