#!/usr/bin/env python3
"""Fetch named files out of a shared Google Drive folder, by relative path.

Called by setup.py, and runnable on its own. Runs inside the backend
virtualenv, because that is where gdown is installed.

The whole `models/` tree is mirrored on Drive, so a file is addressed by the
path it has under that folder - `tissue_type/invasive_tile_v2_imagenet_std.pt`
- and not by a per-file id that a human would have to paste in twelve times.

The folder is listed once and only the requested files are downloaded, which
is the difference between a 350 MB setup and a 2.2 GB one: BEETLE's model.zip
lives in the same tree and is never needed to serve the app.

    python tools/drive_fetch.py <folder-id-or-url> <plan.json>

`plan.json` is a list of {"relpath": ..., "dest": ...}. A JSON report is
written to stdout: {"relpath": {"ok": bool, "reason": str}}.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import sys
import tempfile


def folder_url(folder: str) -> str:
    if folder.startswith("http"):
        return folder
    return f"https://drive.google.com/drive/folders/{folder}"


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2

    folder, plan_path = sys.argv[1], pathlib.Path(sys.argv[2])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    try:
        import gdown
    except ImportError:
        print(json.dumps({"_error": "gdown is not installed in this interpreter"}))
        return 1

    # skip_download gives the listing without moving any bytes, so the folder
    # is walked once and the selection happens locally.
    try:
        listing = gdown.download_folder(
            url=folder_url(folder), skip_download=True, quiet=True, use_cookies=False
        )
    except Exception as exc:  # gdown raises a variety of things
        print(json.dumps({"_error": f"could not list the folder: {exc}"}))
        return 1

    if not listing:
        print(json.dumps({
            "_error": "the folder listing came back empty. Is it shared as "
                      "'Anyone with the link'? A folder over 50 files also "
                      "needs the Drive API rather than gdown."
        }))
        return 1

    # gdown reports each file's path as "<folder name>/<relative path>"; drop
    # that first segment so the keys match the plan's relpaths.
    available: dict[str, str] = {}
    for item in listing:
        parts = pathlib.PurePosixPath(item.path.replace("\\", "/")).parts
        relative = "/".join(parts[1:]) if len(parts) > 1 else parts[0]
        available[relative] = item.id
        available.setdefault(parts[-1], item.id)  # basename fallback

    report: dict[str, dict] = {}
    staging = pathlib.Path(tempfile.mkdtemp(prefix="drive_fetch_"))

    try:
        for wanted in plan:
            relpath, dest = wanted["relpath"], pathlib.Path(wanted["dest"])
            file_id = available.get(relpath) or available.get(pathlib.PurePosixPath(relpath).name)

            if not file_id:
                report[relpath] = {
                    "ok": False,
                    "reason": f"not in the Drive folder (it lists {len(listing)} files)",
                }
                continue

            temporary = staging / pathlib.PurePosixPath(relpath).name
            try:
                got = gdown.download(id=file_id, output=str(temporary), quiet=False)
            except Exception as exc:
                report[relpath] = {"ok": False, "reason": f"download failed: {exc}"}
                continue

            if not got or not temporary.exists():
                report[relpath] = {
                    "ok": False,
                    "reason": "download produced no file; a large file may have hit "
                              "the Drive quota or virus-scan interstitial",
                }
                continue

            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(temporary), str(dest))
            report[relpath] = {"ok": True, "reason": "downloaded"}
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
