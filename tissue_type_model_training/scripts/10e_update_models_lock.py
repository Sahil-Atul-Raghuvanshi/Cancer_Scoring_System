"""Bring `models.lock.json` up to date with every tissue-type head on disk.

    python scripts/10e_update_models_lock.py            # report what would change
    python scripts/10e_update_models_lock.py --write    # apply it

The lock file is what `setup.py --check` audits: bytes and sha256 for every checkpoint
the project needs, so a download that arrived truncated is caught rather than served.
It went stale — it lists the v1/v2 heads and has never carried the four `fov*_concat`
ones, let alone the H&E arms — and a lock file that does not list a runtime checkpoint
audits nothing about it.

**The hashes are not recomputed here.** Each published manifest already records the
sha256 that `models.publish` computed over its own `.pt` at publish time, and that is
the authoritative value: it is what `model.load_pinned` refuses a mismatch against. This
script reads that number, checks it still describes the file on disk, and writes it into
the lock. A second independent hashing would be a second answer to the same question.

`sources: []` on the trained heads, deliberately. They are not downloadable from
anywhere — they are produced by this repository's own publish step — and recording that
as a fact beats leaving a field that looks like an oversight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
sys.path.append(str(WORKSPACE))  # data_versions.py lives at the workspace root
import data_versions  # noqa: E402

LOCK = WORKSPACE / "models.lock.json"
#: The lock's paths are relative to `models/`; this is the active version's (or the
#: newest earlier version's) - see `data_versions.models_root()`.
MODELS = data_versions.models_root() / "tissue_type"

sys.path.insert(0, str(REPO / "src"))

#: The licence every head fitted on the BACH trees inherits. Stated in full rather than
#: as "research only", because the ND term is the unresolved one and a reader deciding
#: whether a number may leave the machine needs to see it named.
LICENCE_WITH_BACH = (
    "RESEARCH ONLY (BCSS CC0 + BRACS non-commercial + BEETLE CC BY-NC-SA 4.0 + "
    "BACH CC BY-NC-ND 4.0 - the ND term is UNRESOLVED: whether a model fitted on it "
    "is a derivative work has not been decided. Do not distribute externally.)"
)

RETRAIN = (
    "tissue_type_model_training: scripts/02_export.py --channel {channel} --mpp {mpp}, "
    "then 03_features.py for both inits, then 08_fit_concat_mlp.py --publish {name}"
)


def configured_default() -> str:
    """What `config.tissue_type_model` names, read from the backend rather than copied.

    Copying it here would be a second place to update, and the whole point of the
    `runtime` role is that it tracks the setting.
    """
    sys.path.insert(
        0, str(WORKSPACE / "tissue_scoring_demo" / "backend")
    )
    from app.core.config import settings

    return str(settings.tissue_type_model)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def entry_for(manifest_path: Path) -> dict | None:
    name = manifest_path.name.removesuffix(".manifest.json")
    checkpoint = MODELS / f"{name}.pt"
    if not checkpoint.is_file():
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec = manifest.get("input") or {}
    recorded = manifest.get("sha256")
    on_disk = sha256_of(checkpoint)
    if recorded != on_disk:
        raise SystemExit(
            f"{name}: the manifest records sha256 {recorded} and the file on disk "
            f"hashes to {on_disk}. Refusing to write a lock entry for a checkpoint "
            "whose own manifest does not describe it - `load_pinned` would refuse it "
            "too, and the lock must not disagree with the loader."
        )

    channel = str(spec.get("channel", "haematoxylin"))
    tile_px = spec.get("tile_px")
    mpp = spec.get("mpp")
    fov = round(tile_px * mpp) if tile_px and mpp else None
    branch = "H&E colour" if channel == "rgb_he" else "haematoxylin"

    # `runtime` means the web app cannot run step 8 without it, and exactly one head
    # meets that: the one `config.tissue_type_model` names, which is what a caller
    # naming nothing resolves to and what step 7's default field of view selects.
    # Every other head is a choice the reader can make on step 7, so the app runs
    # without any of them individually.
    default = configured_default()
    role = "runtime" if name == default else "optional"

    return {
        "path": f"tissue_type/{name}.pt",
        "bytes": checkpoint.stat().st_size,
        "sha256": on_disk,
        # Not `published`: nothing upstream serves these bytes. They are produced here.
        "hash": "local-resave",
        "role": role,
        "licence": LICENCE_WITH_BACH,
        "used_by": (
            f"step 8 - the {branch} branch at {fov} um"
            + (" (config.tissue_type_model, the default)" if role == "runtime" else "")
            if fov
            else "step 8"
        ),
        "retrain_with": RETRAIN.format(
            channel="he" if channel == "rgb_he" else "haematoxylin",
            mpp=mpp,
            name=name,
        ),
        "sources": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    existing = {entry["path"]: entry for entry in lock["files"]}

    added: list[str] = []
    updated: list[str] = []
    for manifest_path in sorted(MODELS.glob("*.manifest.json")):
        entry = entry_for(manifest_path)
        if entry is None:
            print(f"  skip  {manifest_path.name} - no .pt beside it")
            continue

        previous = existing.get(entry["path"])

        # An existing entry claiming to be the default when it is not. v2 carried
        # "the default checkpoint (config.tissue_type_model)" long after the default
        # moved, and a lock file that misdescribes what the app loads is worse than one
        # that omits it - somebody deleting an "optional" checkpoint would be reading
        # the wrong row.
        if (
            previous is not None
            and entry["role"] == "optional"
            and "config.tissue_type_model" in str(previous.get("used_by", ""))
        ):
            previous["role"] = "optional"
            previous["used_by"] = (
                str(previous["used_by"])
                .replace(
                    "the default checkpoint (config.tissue_type_model)",
                    "a superseded checkpoint, kept for comparison",
                )
                .replace("config.tissue_type_model", "no longer the default")
            )
            updated.append(
                (entry["path"], "a stale 'default' note corrected; bytes unchanged")
            )

        if previous is None:
            added.append(entry["path"])
            existing[entry["path"]] = entry
        elif (previous.get("sha256") != entry["sha256"]
              or previous.get("bytes") != entry["bytes"]):
            updated.append((entry["path"], "bytes and sha256 changed"))
            # The role and any hand-written note on an existing entry are kept: a
            # human decided those, and this script only knows the bytes.
            previous.update(
                bytes=entry["bytes"], sha256=entry["sha256"]
            )

    for path in added:
        print(f"  add     {path}")
    for path, why in updated:
        print(f"  update  {path}  - {why}")
    if not added and not updated:
        print("  lock file already matches every checkpoint on disk")

    if not args.write:
        print("\n  dry run - pass --write to apply")
        return 0

    # GrandQC and BEETLE first, then the tissue-type heads by name, so the file reads
    # in the order somebody would look for things in.
    def sort_key(entry: dict) -> tuple[int, str]:
        path = entry["path"]
        family = 0 if path.startswith("grandqc/") else 2 if path.startswith("beetle/") else 1
        return (family, path)

    lock["files"] = sorted(existing.values(), key=sort_key)
    LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(f"\n  wrote {LOCK} - {len(lock['files'])} entries")
    print("  audit it with: python setup.py --check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
