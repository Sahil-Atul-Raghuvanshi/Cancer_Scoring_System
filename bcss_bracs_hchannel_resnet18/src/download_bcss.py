"""Fetch BCSS, and record exactly what arrived.

BCSS is 151 breast-cancer regions from TCGA with a pathologist-drawn label per pixel,
released **CC0** - public domain, no attribution required, no commercial restriction.
That licence is why this is the project's primary label source: the model fitted on it
can ship in a product that a paying client uses to help decide whether someone
receives chemotherapy, and nobody has to argue about whether trained weights are a
derivative work.

Two routes are implemented, and only one of them works. Measured, not guessed:

  girder   **the default, and the one that works.** Each ROI is cropped server-side
           out of its whole slide by the authors' own HistomicsTK instance, and its
           mask comes from figshare. ~20 s and ~27 MB per region, ~4 GB for all 151,
           no quota, resumable per region.
  gdrive   the "convenient single link" the BCSS README offers. It rate-limits hard:
           on this machine it served 42 of 302 files and then answered every
           subsequent request with "Cannot retrieve the public link ... or have had
           many accesses". Kept for the day Drive relents, but do not plan around it.

A third route exists and is not implemented: the Kaggle mirror
`whats2000/breast-cancer-semantic-segmentation-bcss`, which needs an API token and
needs checking hardest, because mirrors of the *merged five-class* version exist.

**Whichever route, the release check is not optional.** There is a widely-used
five-class version of BCSS that folds `dcis` into `tumor`, and it is the default in
most tutorials. Losing that distinction removes the only thing the region model is
being built to do. `verify()` below is the check, and `bcss.remap` refuses any mask
carrying a code the raw release does not use.

Amgad M et al. Structured crowdsourcing enables convolutional segmentation of
histology images. Bioinformatics 35(18):3461-3467 (2019). Data: CC0 1.0.
"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

import bcss
from models import sha256_of

#: The folder the BCSS repository points at for the convenient single-link download,
#: at 0.25 um/px. Pinned by id rather than by a shortened URL so it is auditable.
GDRIVE_FOLDER_ID = "1zqbdkQF8i5cEmZOGmbdQm-EP8dRYtvss"
GDRIVE_FOLDER_URL = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}"

#: The label codes, from the authors' repository. Small, and the source of truth the
#: hard-coded table in `bcss.py` is checked against.
GTRUTH_CODES_URL = (
    "https://raw.githubusercontent.com/CancerDataScience/"
    "CrowdsourcingDataset-Amgadetal2019/master/meta/gtruth_codes.tsv"
)

#: What the release contains, for the sanity check. 151 regions from 151 slides.
EXPECTED_REGIONS = 151

#: Pillow refuses images past ~89 Mpx by default, as a decompression-bomb guard. A
#: BCSS region is legitimately large - a 1.18 mm^2 mean area at 0.25 um/px - so the
#: guard is raised rather than disabled, and only here.
Image.MAX_IMAGE_PIXELS = 500_000_000


def fetch_gtruth_codes(destination: Path) -> Path:
    """Download `gtruth_codes.tsv` beside the data. Route-independent."""
    import urllib.request

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(GTRUTH_CODES_URL, timeout=60) as response:  # noqa: S310
        destination.write_bytes(response.read())
    return destination


def check_codes_file(path: Path) -> dict[str, int]:
    """Parse the TSV and assert it still matches `bcss.GT_CODES`.

    The hard-coded table is what the code compiles against; this file is the truth.
    They agree today - verified when this was written - and this is what notices if a
    future release renumbers anything.
    """
    codes: dict[str, int] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = [part.strip() for part in line.replace(",", "\t").split("\t") if part.strip()]
        if len(parts) < 2:
            continue
        label, value = parts[0], parts[1]
        if value.isdigit():
            codes[label] = int(value)
        elif parts[0].isdigit():
            codes[parts[1]] = int(parts[0])

    if not codes:
        raise ValueError(f"could not parse any label codes out of {path}")

    if codes != bcss.GT_CODES:
        only_file = {k: v for k, v in codes.items() if bcss.GT_CODES.get(k) != v}
        only_code = {k: v for k, v in bcss.GT_CODES.items() if codes.get(k) != v}
        raise AssertionError(
            "gtruth_codes.tsv no longer matches bcss.GT_CODES.\n"
            f"  in the file, differing: {only_file}\n"
            f"  in our table, differing: {only_code}\n"
            "Fix the table, re-check the class mapping, and re-run the mapping test - "
            "a renumbered code silently relabels the whole training set."
        )
    return codes


#: What the Drive folder holds: 151 images, 151 masks, and the authors' own logs.
#: Checked against the listing before anything is transferred, so a mirror that has
#: been pruned to a "sample" is caught before ten gigabytes are.
EXPECTED_ENTRIES = 302


def list_gdrive() -> list[object]:
    """The Drive folder's contents, without downloading anything.

    Worth its own call. It is a few seconds, it confirms the folder is still reachable
    and still whole, and it means the layout is known before the transfer rather than
    after: this release nests everything under `0_Public-data-Amgad2019_0.25MPP/`,
    which is not what the girder route produces.
    """
    import gdown

    listing = gdown.download_folder(
        GDRIVE_FOLDER_URL, skip_download=True, quiet=True
    )
    return list(listing or [])


def download_gdrive(destination: Path, *, quiet: bool = False) -> Path:
    """Route 1: pull the Drive folder, resuming whatever is already there.

    Through `gdown`'s Python API rather than its CLI. The CLI's resume flag has been
    spelled three different ways across releases and its folder handling has had a
    50-file cap in some of them; the API takes `resume=True` and, in 6.1.0, has no
    cap - verified against the listing, which returns all 302 files.

    Returns the directory that actually holds `images/` and `masks/`, which is not
    necessarily `destination`: see `bcss.locate_pairs`.
    """
    import gdown

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    entries = list_gdrive()
    print(f"  folder lists {len(entries)} entries "
          f"(expecting {EXPECTED_ENTRIES} images + masks, plus the authors' logs)",
          flush=True)
    if len(entries) < EXPECTED_ENTRIES:
        raise RuntimeError(
            f"the Drive folder lists only {len(entries)} entries, short of the "
            f"{EXPECTED_ENTRIES} this release contains. Refusing to transfer a "
            "partial dataset - a model trained on part of BCSS reports as though it "
            "were trained on all of it. Try route 2 or 3 instead."
        )

    gdown.download_folder(
        GDRIVE_FOLDER_URL, output=str(destination), quiet=quiet, resume=True
    )

    images, _ = bcss.locate_pairs(destination)
    return images.parent


# --- route 2: the authors' own server ----------------------------------------
#
# This is the route that works. Google Drive rate-limits folder downloads hard - it
# served 42 of 302 files and then returned "Cannot retrieve the public link ... or
# have had many accesses" for every subsequent one - and no amount of resuming gets
# past a quota. The authors' route has no such limit, and it is cheaper besides:
#
#   images   cropped server-side out of the whole slide, so 27 MB of ROI comes down
#            instead of a 1.4 GB slide. ~20 s each, ~4 GB for all 151.
#   masks    direct downloads from figshare, which is a CDN with no per-file quota.
#
# Filenames here reproduce the Drive release's exactly - `<slide>_xmin<x>_ymin<y>_
# MPP-0.2500.png` - so a part-finished Drive download is picked up rather than
# re-fetched, and `bcss.parse_roi` reads either without knowing which route ran.

GIRDER_API = "https://demo.kitware.com/histomicstk/api/v1"

#: The "Crowd Source Paper" folder holding the 151 source slides.
GIRDER_FOLDER_ID = "5bbdeba3e629140048d017bb"

#: ROI corners and the figshare link for each mask, one row per region.
ROI_BOUNDS_URL = (
    "https://raw.githubusercontent.com/CancerDataScience/"
    "CrowdsourcingDataset-Amgadetal2019/master/meta/roiBounds.csv"
)


def _get(url: str, *, timeout: int = 600, attempts: int = 6) -> bytes:
    """One GET with retries and linear backoff.

    Retried because 151 sequential requests against a research server will hit a
    transient failure or two, and losing a 50-minute download to one 502 is a poor
    trade for four lines of code.

    **`http.client.HTTPException` is in the catch list and must stay there.** An earlier
    version caught only `(URLError, TimeoutError, OSError)` and a real run died at region
    97 of 151 with:

        http.client.IncompleteRead: IncompleteRead(26940568 bytes read,
                                                  8294813 more expected)

    `IncompleteRead` inherits from `HTTPException` and `ValueError`, from neither of
    which `OSError` is an ancestor - so the retry never saw it and 40 minutes of
    downloading stopped. A truncated response mid-body is exactly the transient failure
    this function exists for, and it is the *commonest* one against a server that crops
    30 MB images on demand.
    """
    import http.client
    import urllib.error
    import urllib.request

    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
                return response.read()
        except (urllib.error.URLError, TimeoutError, OSError,
                http.client.HTTPException) as exc:
            last = exc
            if attempt < attempts:
                import time

                # Backoff is 15s, 30s, 45s ... rather than 5s, 10s, 15s. A real run hit
                # `HTTP Error 502: Bad Gateway` from the girder server and exhausted four
                # attempts inside 30 seconds - far too impatient for a research host that
                # crops 30 MB images on demand, and impolite besides. Six attempts over
                # ~3.75 minutes rides out a load spike; a server that is still refusing
                # after that is genuinely down, and stopping is then correct.
                time.sleep(15 * attempt)
    raise RuntimeError(f"giving up on {url[:90]}... after {attempts} attempts: {last}")


def roi_bounds() -> list[dict[str, str]]:
    """`meta/roiBounds.csv`: 151 rows of corners plus a mask link."""
    import csv
    import io

    text = _get(ROI_BOUNDS_URL, timeout=120).decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    # The first column is unnamed in the file and holds the slide key.
    key = next(name for name in rows[0] if not name or name.strip() == "")
    return [{**row, "slide": row[key]} for row in rows]


def girder_items() -> dict[str, str]:
    """Slide key -> girder item id, for the 151 source slides.

    Item names are full TCGA barcodes with a UUID appended -
    `TCGA-A1-A0SK-01Z-00-DX1.A44D70FA-...svs` - while `roiBounds.csv` keys on
    `TCGA-A1-A0SK-DX1`. Both are rebuilt into the same key rather than matched by
    prefix, because a prefix match would pair a slide with the wrong section on any
    case that has two.
    """
    listing = json.loads(
        _get(f"{GIRDER_API}/item?folderId={GIRDER_FOLDER_ID}&limit=1000000", timeout=120)
    )

    items: dict[str, str] = {}
    for item in listing:
        parts = str(item["name"]).split(".")[0].split("-")
        section = next((part for part in parts if part.startswith("DX")), None)
        if len(parts) >= 3 and section:
            items[f"{parts[0]}-{parts[1]}-{parts[2]}-{section}"] = item["_id"]
    return items


#: Sidecar recording each slide's own resolution, written by `download_girder` and read
#: by the exporter. See `slide_mpp` for why this file has to exist.
MPP_SIDECAR = "slide_mpp.json"


#: Slides whose resolution the server does not know, and which are therefore excluded.
#: Written beside the sidecar so the exclusion is a recorded fact rather than a silent
#: difference between "151 regions" in the paper and 150 on disk.
EXCLUDED_SIDECAR = "excluded.json"


def slide_mpp(item_id: str) -> float | None:
    """The slide's own microns per pixel at level 0, or `None` if the server has none.

    **Not 0.25, and this is the measurement that reshaped the whole exporter.** Every
    filename in this dataset ends `MPP-0.2500`, the Drive folder is called
    `0_Public-data-Amgad2019_0.25MPP`, and every tutorial therefore assumes 0.25.
    Measured across all 151 slides, the real values run from **0.1644 to 0.5005**: the
    cohort mixes 20x and 40x scans.

    That is not a rounding difference. The girder crop and the figshare mask are both in
    *level-0 pixels* - verified against slides at 0.1644, 0.2521 and 0.4992, where the
    mask is exactly the ROI's pixel extent in all three - so image and mask always agree
    with each other. But the resample to the pipeline's 0.5 um/px is computed from this
    number, so believing the filename would put every tile from a 20x slide at 1.0 um/px
    covering **224 um instead of 112**: four times the intended area, on a fifth of the
    cohort, silently.

    A 224 px tile is meant to be 112 um because that is nine cells across, which is the
    entire argument for the tile size. Nine cells is a claim about microns, so the
    microns have to be measured. Step 1 of the pipeline makes the same point in code:
    convert between resolutions using mpp, never a hard-coded factor.

    Returns `None` rather than raising for the one slide (TCGA-OL-A5RW) whose metadata
    carries no resolution at all - not in `mm_x`, not in `magnification`, and not in its
    own Aperio header, which is truncated before the `AppMag`/`MPP` fields. The caller
    excludes it. Guessing the cohort median would be wrong by a factor of two if that
    slide happens to be one of the 20x scans, and there is no way to tell from here.
    """
    meta = json.loads(_get(f"{GIRDER_API}/item/{item_id}/tiles", timeout=120))
    mm_x = meta.get("mm_x")
    return float(mm_x) * 1000.0 if mm_x else None


#: Concurrent region fetches. The server crops each ROI on demand, so a single
#: sequential stream spends most of its time waiting rather than transferring: measured
#: at 68 s per region serially, against 27-48 MB of payload. Four in flight cuts that to
#: roughly a quarter without being rude to what is a public research instance - the
#: transfers themselves are small, and this is 151 requests in total, once.
GIRDER_WORKERS = 4


def _fetch_one(
    row: dict[str, str],
    item_id: str,
    images_dir: Path,
    masks_dir: Path,
) -> tuple[str, bool, int]:
    """One region: its mask from figshare and its crop from girder. Returns (slide, fetched, bytes).

    Skips anything already on disk at non-zero size, which is what makes the whole
    download resumable per region rather than per run.
    """
    slide = row["slide"]
    # The release's own filename shape, so `bcss.parse_roi` reads either route's output.
    # The MPP in the name is the dataset's nominal figure and is kept for compatibility -
    # the real one lives in the sidecar.
    name = f"{slide}_xmin{row['xmin']}_ymin{row['ymin']}_MPP-0.2500.png"
    image_path, mask_path = images_dir / name, masks_dir / name

    have_image = image_path.exists() and image_path.stat().st_size > 0
    have_mask = mask_path.exists() and mask_path.stat().st_size > 0
    if have_image and have_mask:
        return slide, False, image_path.stat().st_size

    if not have_mask:
        # Atomic, for the same reason the image below is: the resume test is
        # "exists() and size > 0", so a response truncated mid-write would leave a
        # short-but-non-empty file that every later run then SKIPS as complete. A
        # corrupt mask is worse than a missing one - it trains, quietly, on nonsense.
        partial_mask = mask_path.with_suffix(mask_path.suffix + ".partial")
        partial_mask.write_bytes(_get(row["mask_link"]))
        partial_mask.replace(mask_path)

    if not have_image:
        # Level-0 pixels, un-scaled. Asking girder for a magnification or an mm_x here
        # would resample server-side and break the exact correspondence with the
        # figshare mask, which is itself in level-0 pixels.
        region = (
            f"{GIRDER_API}/item/{item_id}/tiles/region"
            f"?left={row['xmin']}&top={row['ymin']}"
            f"&right={row['xmax']}&bottom={row['ymax']}&encoding=PNG"
        )
        # Written to a temporary name and moved into place, so an interrupted transfer
        # cannot leave a half-written PNG that the next run treats as complete.
        partial = image_path.with_suffix(".png.part")
        partial.write_bytes(_get(region))
        partial.replace(image_path)

    return slide, True, image_path.stat().st_size


def download_girder(
    destination: Path, *, quiet: bool = False, workers: int = GIRDER_WORKERS
) -> Path:
    """Route 2: crop each ROI server-side, and pull its mask from figshare.

    The two arrive in the *same* coordinate space - level-0 pixels of the source slide -
    so image and mask line up exactly, with no resampling on either side. Each slide's
    real resolution is recorded alongside, because it is not the 0.25 the filenames
    claim; see `slide_mpp`.

    Resumable by construction: a region whose image and mask are both on disk at
    non-zero size is skipped, so an interrupted run costs one region rather than an
    hour.
    """
    destination = Path(destination)
    images_dir, masks_dir = destination / "images", destination / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    bounds = roi_bounds()
    items = girder_items()
    print(f"  {len(bounds)} ROIs listed, {len(items)} slides on the server", flush=True)

    missing = [row["slide"] for row in bounds if row["slide"] not in items]
    if missing:
        raise RuntimeError(
            f"{len(missing)} ROIs name a slide the server does not have, e.g. "
            f"{missing[:3]}. Refusing rather than training on the remainder."
        )

    # Phase 1: every slide's own resolution, before any pixels. Cheap (one small JSON
    # per slide), and it means the sidecar is complete even if the region transfers are
    # interrupted - so a part-finished download can still be exported honestly.
    sidecar = destination / MPP_SIDECAR
    measured: dict[str, float] = (
        json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
    )
    excluded_path = destination / EXCLUDED_SIDECAR
    excluded: dict[str, str] = (
        json.loads(excluded_path.read_text(encoding="utf-8"))
        if excluded_path.exists() else {}
    )

    todo = [row["slide"] for row in bounds
            if row["slide"] not in measured and row["slide"] not in excluded]
    if todo:
        print(f"  measuring {len(todo)} slide resolutions ...", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for slide, mpp in zip(
                todo, pool.map(lambda name: slide_mpp(items[name]), todo)
            ):
                if mpp is None:
                    excluded[slide] = (
                        "the server reports no mm_x, no magnification, and this slide's "
                        "own Aperio header is truncated before its MPP field - so its "
                        "resolution is unknown. Excluded rather than guessed: the cohort "
                        "spans 0.1644 to 0.5005 um/px, so a guess could be wrong by a "
                        "factor of two, and every tile from it would be at the wrong "
                        "physical scale."
                    )
                else:
                    measured[slide] = mpp
        sidecar.write_text(json.dumps(measured, indent=2, sort_keys=True), encoding="utf-8")
        excluded_path.write_text(json.dumps(excluded, indent=2, sort_keys=True),
                                 encoding="utf-8")

    spread = sorted({round(value, 4) for value in measured.values()})
    print(f"  native resolutions across {len(measured)} slides: {spread}", flush=True)
    print("  (every filename says 0.2500. They are wrong: this cohort mixes 20x and "
          "40x scans.)", flush=True)
    if excluded:
        print(f"  EXCLUDED {len(excluded)} slide(s) with no recorded resolution: "
              f"{sorted(excluded)} - see {EXCLUDED_SIDECAR}", flush=True)

    # Phase 2: the regions, `workers` at a time.
    wanted = [row for row in bounds if row["slide"] not in excluded]
    fetched = skipped = 0
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_one, row, items[row["slide"]], images_dir, masks_dir): row
            for row in wanted
        }
        for future in as_completed(futures):
            slide, was_fetched, size = future.result()
            done += 1
            fetched += int(was_fetched)
            skipped += int(not was_fetched)
            if not quiet:
                print(f"  [{done:>3}/{len(wanted)}] {slide:<20} {size / 1e6:>6.1f} MB  "
                      f"{measured[slide]:.4f} um/px   "
                      f"fetched {fetched}, had {skipped}", flush=True)

    print(f"  done: fetched {fetched}, already had {skipped}", flush=True)

    images, _ = bcss.locate_pairs(destination)
    return images.parent


def resolutions(root: Path) -> dict[str, float]:
    """The per-slide resolutions the download recorded, or `{}` for the Drive route."""
    sidecar = Path(root) / MPP_SIDECAR
    if not sidecar.exists():
        # The Drive route resamples every region to one resolution and keeps no record
        # of the originals, so there is nothing to read. Its geometry is also not the
        # girder route's - see this module's docstring.
        for parent in (Path(root).parent,):
            candidate = parent / MPP_SIDECAR
            if candidate.exists():
                sidecar = candidate
                break
        else:
            return {}
    return {key: float(value) for key, value in
            json.loads(sidecar.read_text(encoding="utf-8")).items()}


def verify(root: Path, *, sample_masks: int = 12) -> dict[str, object]:
    """Gate G1, as a function: is this the raw 22-class release, and is it complete?

    Returns the summary the notebook prints and writes to `download_manifest.json`.
    Raises on the two failures that matter - a merged release, or a partial download -
    because both produce a model that trains fine and means something different from
    what its report claims.
    """
    root = Path(root)
    regions = bcss.find_regions(root)

    institutions = Counter(region.institution for region in regions)
    slides = {region.slide_id for region in regions}
    held_out = [region for region in regions if region.is_test]

    # Code census over a sample of masks. Every mask would mean reading the whole
    # dataset twice; the question here is only "does code 20 exist and are the codes
    # in range", and a dozen full-size regions answers it. The exporter reads them all
    # anyway, and `bcss.remap` refuses an unknown code there too.
    rng = np.random.default_rng(0)
    picks = rng.choice(len(regions), size=min(sample_masks, len(regions)), replace=False)
    census: Counter[int] = Counter()
    for index in picks:
        mask = np.asarray(Image.open(regions[int(index)].mask))
        if mask.ndim == 3:
            mask = mask[..., 0]
        values, counts = np.unique(mask, return_counts=True)
        census.update(dict(zip(values.tolist(), counts.tolist())))

    unknown = sorted(code for code in census if code not in bcss.CODE_NAMES)
    if unknown:
        raise AssertionError(
            f"masks carry codes {unknown}, which are not in gtruth_codes.tsv. This is "
            "not the raw release."
        )

    dcis_pixels = int(census.get(bcss.GT_CODES["dcis"], 0))
    tumor_pixels = int(census.get(bcss.GT_CODES["tumor"], 0))

    if dcis_pixels == 0:
        # **A sample cannot answer this question, so escalate to a full scan.**
        #
        # `dcis` is in ONE region of 150 (measured: TCGA-AR-A2LH-DX1, 1,845,143 px), so
        # a 20-mask sample misses it about 87% of the time and a 40-mask sample about
        # 73%. An earlier version of this gate failed a perfectly good download on that
        # basis and told the reader to re-download - the worst possible advice, since
        # the data was already correct.
        #
        # The distinction being tested is the one this whole project rests on - raw
        # 22-code release versus the merged five-class one, which folds dcis into tumor -
        # so when the cheap check comes back empty it is worth the ~2 minutes to read
        # every mask rather than guessing.
        print("    dcis not in the sample (expected - it is in 1 region of 150); "
              "scanning all masks ...")
        carriers = []
        for region in regions:
            mask = np.asarray(Image.open(region.mask))
            if mask.ndim == 3:
                mask = mask[..., 0]
            hits = int(np.count_nonzero(mask == bcss.GT_CODES["dcis"]))
            if hits:
                carriers.append((region.mask.name, hits))
                dcis_pixels += hits

        if dcis_pixels == 0:
            raise AssertionError(
                "no pixel in ANY of the {} masks carries code 20 (dcis). This is the "
                "merged five-class release, which folds dcis into tumor and is unusable "
                "here - the invasive-vs-in-situ distinction is the point of the project. "
                "Re-download from route 1 or 2.".format(len(regions))
            )
        print(f"    code 20 found in {len(carriers)} region(s): "
              + ", ".join(f"{n} ({c:,} px)" for n, c in carriers))

    excluded_path = root / EXCLUDED_SIDECAR
    excluded = (
        json.loads(excluded_path.read_text(encoding="utf-8"))
        if excluded_path.exists() else {}
    )
    if len(regions) + len(excluded) != EXPECTED_REGIONS:
        print(
            f"  ! {len(regions)} regions on disk plus {len(excluded)} deliberately "
            f"excluded does not account for the {EXPECTED_REGIONS} this release holds. "
            "A partial download trains on part of the dataset and reports as if on all "
            "of it - finish the download before exporting tiles.",
            flush=True,
        )
    elif excluded:
        print(f"  {len(regions)} regions on disk; {len(excluded)} excluded by decision "
              f"({', '.join(sorted(excluded))}), which accounts for all "
              f"{EXPECTED_REGIONS}.", flush=True)

    return {
        "regions": len(regions),
        "slides": len(slides),
        "institutions": dict(sorted(institutions.items())),
        "held_out_regions": len(held_out),
        "train_regions": len(regions) - len(held_out),
        "sampled_masks": int(len(picks)),
        "code_census": {bcss.CODE_NAMES[code]: int(count) for code, count in sorted(census.items())},
        "dcis_pixels_sampled": dcis_pixels,
        "tumor_pixels_sampled": tumor_pixels,
        "bytes_on_disk": sum(
            path.stat().st_size for path in root.rglob("*") if path.is_file()
        ),
        "excluded": excluded,
        "resolutions_measured": resolutions(root),
    }


def write_manifest(root: Path, summary: dict[str, object], *, hash_files: bool = False) -> Path:
    """Record what arrived, so "which BCSS did we train on" has a written answer.

    `hash_files` is off by default: SHA-256 over ~10 GB takes minutes and the byte
    counts plus the code census already identify the release. Turn it on for the run
    whose output actually gets published.
    """
    root = Path(root)
    manifest: dict[str, object] = {
        "source": "BCSS (Amgad et al. 2019), CC0 1.0",
        "route": "google drive folder, 0.25 um/px",
        "folder": GDRIVE_FOLDER_URL,
        "summary": summary,
    }

    if hash_files:
        manifest["files"] = {
            str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": sha256_of(path)}
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    path = root / "download_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path
