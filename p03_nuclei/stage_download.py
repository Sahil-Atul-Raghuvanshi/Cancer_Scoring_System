"""Download and unpack what approaches 2 and 3 need, resumably, verified by checksum.

Runs beside the other stages rather than before them - the DeepLIIF model alone is 3 GB
and the line here is ~0.3 MB/s - and writes `<name>.ok` when an item is verified and
unpacked, which is what the stages that need it wait for.

  lynsec_model    LyNSeC's published IHC nucleus model (HoVer-Net weights), CC BY 4.0
  lynsec_data     LyNSeC's hand-labelled nuclei, IHC and H&E, CC BY 4.0
  deepliif_model  DeepLIIF's latest serialized model, CC BY 4.0
  hovernet_code   the HoVer-Net architecture the LyNSeC weights load into, MIT
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import urllib.request
import zipfile

import common

ITEMS = [
    # name, zenodo record, file key (None = direct url), url
    ("lynsec_model", "8065174", "ihc.tar", None),
    ("lynsec_data", "8065174", "lynsec.zip", None),
    ("hovernet_code", None, None, "https://github.com/vqdang/hover_net/archive/refs/heads/master.zip"),
    ("deepliif_model", "4751737", "DeepLIIF_Latest_Model.zip", None),
]
LOG = common.LOGS / "download.log"


def _zenodo(record: str, key: str) -> tuple[str, int, str | None]:
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{record}", timeout=120) as r:
        meta = json.load(r)
    for entry in meta["files"]:
        if entry["key"] == key:
            checksum = entry.get("checksum", "")
            md5 = checksum.split(":", 1)[1] if checksum.startswith("md5:") else None
            return entry["links"]["self"], int(entry["size"]), md5
    raise RuntimeError(f"{key} not in zenodo record {record}")


def _md5(path) -> str:
    digest = hashlib.md5()  # noqa: S324 - matching the publisher's checksum, not security
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(name: str, record: str | None, key: str | None, url: str | None) -> None:
    ok = common.DOWNLOADS / f"{name}.ok"
    if ok.exists():
        common.say(f"{name}: already verified", LOG)
        return
    size, md5 = None, None
    if record:
        url, size, md5 = _zenodo(record, key)
    target = common.DOWNLOADS / (key or f"{name}.zip")

    for attempt in range(1, 31):
        if size and target.exists() and target.stat().st_size >= size:
            break
        have = target.stat().st_size if target.exists() else 0
        common.say(f"{name}: attempt {attempt}, have {have / 1e6:.0f} MB"
                   + (f" of {size / 1e6:.0f} MB" if size else ""), LOG)
        common.heartbeat("download", f"{name} {have / 1e6:.0f} MB")
        # curl resumes with -C -; --speed-limit aborts a stalled connection so the loop
        # retries instead of hanging for hours on a dead socket.
        subprocess.run(
            ["curl", "-sSL", "-C", "-", "--retry", "5", "--retry-delay", "10",
             "--speed-limit", "1000", "--speed-time", "120", "-o", str(target), url],
            check=False,
        )
        if not size and target.exists() and target.stat().st_size > 0:
            break
    if size and (not target.exists() or target.stat().st_size != size):
        raise RuntimeError(f"{name}: incomplete after retries")
    if md5:
        actual = _md5(target)
        if actual != md5:
            target.unlink()
            raise RuntimeError(f"{name}: md5 {actual} != published {md5}; deleted for a clean retry")

    if zipfile.is_zipfile(target) and name != "lynsec_model":
        destination = common.DOWNLOADS / name
        destination.mkdir(exist_ok=True)
        with zipfile.ZipFile(target) as archive:
            archive.extractall(destination)
    ok.write_text("verified", encoding="utf-8")
    common.say(f"{name}: verified and unpacked", LOG)


def main() -> int:
    common.ensure_dirs()
    failed = []
    for item in ITEMS:
        try:
            fetch(*item)
        except Exception as exc:  # noqa: BLE001 - one item must not stop the others
            common.say(f"{item[0]}: FAILED {exc}", LOG)
            failed.append(item[0])
    common.write_json(common.STATE / "download.items.json", {"failed": failed})
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
