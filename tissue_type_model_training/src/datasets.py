"""The tile dataset, the augmentation, and the splits that do not leak.

Three things live here, and each is a place a tile classifier on histology is
routinely got wrong:

  the dataset      reads the stored uint8 haematoxylin tile and hands it to
                   `hchannel.from_stored` - the one input transform - rather than
                   doing any arithmetic of its own.
  the augmentation flips, 90-degree rotations, and the single-channel stain jitter
                   derived in `hchannel.to_model_input`. No stain normalisation: the
                   jitter is what replaces it.
  the splits       grouped by slide, and reported on held-out *institutions*. Never a
                   random tile split.

**Why a random tile split is not merely optimistic but useless.** Adjacent tiles from
one region share a scanner, a stain batch, a fixation, a patient and often literal
cells at the shared edge. A random split puts near-duplicates on both sides, so the
model can score well by recognising the slide rather than the tissue - and it will,
because recognising a slide is easier. The number that comes out is beautiful and
means nothing, and it will be believed because it is the first number anyone sees.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import bcss
import hchannel

# --- the augmentation ---------------------------------------------------------


@dataclass(frozen=True)
class Jitter:
    """Stain-strength jitter, as a range rather than a value.

    `alpha` is multiplicative - how strongly this section was counterstained - and
    `beta` additive in optical density units, for the small offset a different
    scanner's background leaves. Applied before the clip, inside
    `hchannel.to_model_input`, because a strongly counterstained slide genuinely does
    saturate more of its nuclei and the clip is where that happens.

    `gamma` is the **shape** term, and it is not a nicety. Gate G0b measured the H
    channel of one of our H&E slides against the IHC section of the same case and found
    the density ratio running from 0.06 at p75 to 2.06 at p99 - a 37x spread, where a
    pure strength difference would be a single constant. An IHC counterstain is sparse
    and punchy where H&E haematoxylin is broad and mid-toned, and **no `alpha`, and no
    per-slide normalisation, can change that**: every such correction is monotone, and a
    monotone map cannot alter the ratio between two quantiles of one image. Raising the
    normalised value to a power can, and fitting the measured quantiles wants gamma
    between about 2.0 and 3.6.

    So the range is log-uniform over (0.5, 3.5): it spans the measured H&E -> IHC
    direction *and* the other way, because nothing guarantees the next slide's
    counterstain is heavier rather than lighter than 00251's. Log-uniform because gamma
    is an exponent - 2.0 and 0.5 are the same size of change in opposite directions, and
    a uniform draw would spend most of its mass on gamma > 1.

    The ranges are wide on purpose. Together they are the only thing standing between a
    model fitted on TCGA H&E regions and the Morphle-scanned IHC sections it will be
    asked to run on, and the plan skips stain normalisation entirely on the strength
    of them.
    """

    alpha: tuple[float, float] = (0.80, 1.25)
    beta: tuple[float, float] = (-0.05, 0.05)
    gamma: tuple[float, float] = (0.5, 3.5)

    def sample(self, rng: np.random.Generator) -> tuple[float, float, float]:
        lo, hi = self.gamma
        g = float(lo) if lo == hi else float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        return (
            float(rng.uniform(*self.alpha)),
            float(rng.uniform(*self.beta)),
            g,
        )


NO_JITTER = Jitter(alpha=(1.0, 1.0), beta=(0.0, 0.0), gamma=(1.0, 1.0))


def geometric(tile: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Flips and 90-degree rotations. Free, and correct for this data.

    A slide has no canonical orientation - the section landed on the glass however it
    landed - so the eight symmetries of the square are all genuinely the same tissue.
    Rotations by other angles are not included: they need interpolation, which invents
    pixel values, and 90-degree turns need none.
    """
    if rng.random() < 0.5:
        tile = np.fliplr(tile)
    if rng.random() < 0.5:
        tile = np.flipud(tile)
    turns = int(rng.integers(0, 4))
    if turns:
        tile = np.rot90(tile, turns)
    return np.ascontiguousarray(tile)


# --- the manifest -------------------------------------------------------------


def read_manifest(path: Path) -> list[dict[str, str]]:
    """The exporter's CSV, as a list of rows. The dataset's only source of truth."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no rows - run the exporter (notebook 02) first")
    return rows


#: Which classes each source is allowed to contribute, by the rule that a dataset
#: should only teach the classes it was *selected* to represent well.
#:
#: **BCSS teaches all three.** It is 151 TCGA regions with pixel masks drawn by people,
#: abundant in invasive carcinoma and in stroma, fat and the rest.
#:
#: **BRACS teaches class 1 only.** The regions here were chosen *because* they are
#: DCIS-heavy, so its in-situ epithelium is the thing it represents well - and its
#: invasive and non-epithelium tiles are a side effect of that selection: 270 and 662
#: tiles, guessed by BEETLE's nnU-Net inside ROIs picked for their DCIS, against BCSS's
#: 5,282 and 5,435 drawn by hand. Letting a DCIS-selected, model-labelled minority vote
#: on what invasive carcinoma looks like is not a small amount of extra data, it is a
#: biased sample of the class it is worst at.
#:
#: This is the whole of the "conflict resolution" this project needs, because the two
#: sources are *different images* - there is no pixel a human and BEETLE both labelled,
#: so there is never a disagreement to arbitrate. What there is instead is a question
#: of which source is authoritative for which class, and that is what this table is.
#: **BCSS teaches all three, and each BRACS tree teaches what its consensus covers.**
#: Built from `bcss.BORROWED_TREES` rather than restated, so that the rule a tile is
#: filtered by and the rule its producing manifest recorded are one fact:
#:
#:     bcss          -> {0, 1, 2}   151 TCGA regions, pixel masks drawn by people
#:     bracs_dcis    -> {1}         in-situ, the class BCSS has nine tiles of
#:     bracs_ic      -> {2}         invasive, from a second laboratory
#:     bracs_normal  -> {0, 1}      no-carcinoma consensus: stroma/fat and normal duct
SOURCE_CLASSES: dict[str, frozenset[int]] = {
    "bcss": frozenset({0, 1, 2}),
    **{tree.source: tree.teaches for tree in bcss.BORROWED_TREES.values()},
}


def by_source_authority(
    rows: list[dict[str, str]],
    *,
    allowed: dict[str, frozenset[int]] | None = None,
) -> list[dict[str, str]]:
    """Keep only the tiles whose source is authoritative for their class.

    Order is preserved, which matters more than it looks: `manifest_fingerprint` hashes
    the tile order and notebooks 03 and 04 compare it, so filtering here and forgetting
    to filter there is caught rather than silently training a head on features whose
    rows mean something else.

    **The cost of this rule, stated rather than buried.** With only the DCIS tree
    borrowed, class 1 was 1,843 BRACS tiles against **9** from BCSS - so `source` became
    very nearly a synonym for "class 1", and a model could score well on it by
    recognising *which dataset it is looking at* rather than by seeing duct
    architecture. Keeping BRACS's class-0 and class-2 tiles was the only thing pushing
    against that, weak as it was, and this function drops them on purpose: a model that
    has learned "BRACS => in-situ" at least has a *correct* representation of in-situ to
    have learned it from, whereas one taught invasive carcinoma by 270 DCIS-adjacent
    guesses has a wrong one.

    **The `ic` and `normal` trees are what pays that cost down, and they do it without
    weakening the rule.** They do not let BRACS vote on more classes than its consensus
    covers; they add BRACS regions whose consensus covers *other* classes - `bracs_ic`
    supplies class 2 from regions three pathologists called invasive, `bracs_normal`
    supplies classes 0 and 1 from regions they called free of carcinoma. The confound is
    a property of the joint distribution of `source` and `label`, never of any single
    row, so the fix is BRACS rows outside class 1, which is exactly what these two are.

    **So it must be run with the leakage check, not instead of it.** If a throwaway
    classifier can predict `source` from the same frozen features, the class-1 number is
    inflated and no amount of this helps. `scripts/06_leakage_check.py` is that test,
    and re-running it once the two new trees are in the manifest is the measurement that
    says whether they worked.
    """
    table = SOURCE_CLASSES if allowed is None else allowed
    kept = [
        row
        for row in rows
        # A source nobody has ruled on is trusted for everything, so adding a third
        # dataset later fails open rather than silently dropping all of its tiles.
        if int(row["label"]) in table.get(row["source"], frozenset({0, 1, 2}))
    ]
    if not kept:
        raise ValueError(
            "the source-authority rule removed every tile. Check the manifest's "
            f"`source` column against {sorted(table)}"
        )
    return kept


def borrowed_summaries(summary: dict) -> dict[str, dict]:
    """The per-tree blocks of an `export_summary.json`, from either shape it has had.

    Exports written before there were three lesion trees carry a single `"dcis"` object;
    ones written since carry `"borrowed": {name: object}`. Read through here rather than
    by key, so an export already on disk still gates, still passes the determinism check
    and still publishes with the right licence - instead of silently losing all three
    because a key was renamed. A run that has no borrowed tiles returns `{}`, which is
    the honest answer and the one every caller branches on.
    """
    if summary.get("borrowed"):
        return {name: value for name, value in summary["borrowed"].items() if value}
    return {"dcis": summary["dcis"]} if summary.get("dcis") else {}


def channel_of(tiles_dir: Path) -> str:
    """Which input contract a tile store holds, read off its own export summary.

    One export, one channel, and the store is the authority on which. Every later
    stage takes a `--tiles` directory and nothing else, so asking the directory is
    what stops a 448 um RGB store being fitted as though it were density.

    Absent means `haematoxylin`, because every export cut before the H&E branch
    existed is one. The manifest's own `channel` column is cross-checked where it is
    present and a disagreement raises: the summary and the rows are written by the
    same function in the same pass, so if they differ the store has been assembled by
    hand and nothing about it can be trusted.
    """
    tiles_dir = Path(tiles_dir)
    summary_path = tiles_dir / "export_summary.json"
    if not summary_path.exists():
        raise SystemExit(
            f"{summary_path} is missing, so there is no record of what channel these "
            "tiles are. Re-run the export rather than guessing."
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    channel = str((summary.get("input") or {}).get("channel", hchannel.CHANNEL_HAEMATOXYLIN))

    manifest = tiles_dir / "tiles_manifest.csv"
    if manifest.exists():
        rows = read_manifest(manifest)
        recorded = {str(row["channel"]) for row in rows if row.get("channel")}
        if recorded and recorded != {channel}:
            raise SystemExit(
                f"{tiles_dir} says its channel is {channel!r} in export_summary.json "
                f"but its rows say {sorted(recorded)}. One export is one channel; this "
                "store has been assembled from more than one and cannot be trained on."
            )

    return channel


def manifest_fingerprint(rows: list[dict[str, str]]) -> str:
    """A hash of the tile order, so a feature cache cannot be paired with the wrong rows.

    Notebook 03 writes this beside the cached features and Notebook 04 checks it. The
    failure it prevents is the quietest one in the whole plan: a feature array whose
    rows are in a different order from the labels trains to about chance, and looks
    exactly like a model that will not learn.
    """
    import hashlib

    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["tile_id"].encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


# --- the dataset --------------------------------------------------------------


class TileDataset(Dataset):
    """Stored tiles and their labels, augmented or not, in either channel.

    `augment=False` is not merely "no jitter" - it is the exact transform inference
    uses, which is what makes the cached features in Notebook 03 the same features
    step 8 will compute. Do not add a "harmless" resize or a centre crop to this path.

    `channel` says which store `tiles_dir` is. On `haematoxylin` a stored tile is
    quantised optical density and goes through `from_stored`; on `rgb_he` it is a
    photograph and goes through `rgb_to_model_input`, which has no parameters at all.
    The value is read off the export summary by `channel_of` rather than passed by
    hand wherever possible, because the store knows what it is and a caller guessing
    is how a colour model ends up fitted on density.
    """

    def __init__(
        self,
        rows: list[dict[str, str]],
        tiles_dir: Path,
        *,
        augment: bool = False,
        jitter: Jitter = Jitter(),
        invert: bool = False,
        standardise: bool = False,
        seed: int = 0,
        channel: str = hchannel.CHANNEL_HAEMATOXYLIN,
    ) -> None:
        if channel == hchannel.CHANNEL_RGB_HE:
            # These three are optical-density concepts. On a photograph there is no
            # such thing as "the tile's own p99 density" to divide by, and a polarity
            # flip would invert a colour image into its negative. Silently ignoring
            # them would let a caller believe a transform ran, so they are refused.
            if invert or standardise:
                raise ValueError(
                    "the rgb_he store has no polarity and no per-tile "
                    f"standardisation; got invert={invert}, standardise={standardise}"
                )
            if augment and jitter != NO_JITTER:
                raise ValueError(
                    "Jitter's alpha, beta and gamma are defined in optical-density "
                    "units on a single channel. Applying them to RGB has no correct "
                    "interpretation - a Tellez-style HED jitter would be a different "
                    "augmentation, and nobody here has measured one. Pass "
                    "jitter=NO_JITTER for geometric-only augmentation on this store."
                )

        self.rows = rows
        self.tiles_dir = Path(tiles_dir)
        self.augment = augment
        self.jitter = jitter if augment else NO_JITTER
        self.invert = invert
        self.channel = channel
        # `standardise` is a property of the INPUT, not of the augmentation, so it
        # applies on both paths. Setting it for training and not for inference would
        # be the exact training/serving drift this module exists to prevent.
        self.standardise = standardise
        self._seed = seed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        path = self.tiles_dir / row["tile_path"]

        if self.channel == hchannel.CHANNEL_RGB_HE:
            stored = np.asarray(Image.open(path).convert("RGB"))
            if self.augment:
                # Geometric only. `geometric` flips and rotates over the first two
                # axes, which is HxW on both stores, so the same function serves both.
                stored = geometric(stored, np.random.default_rng((self._seed, index)))
            return (
                torch.from_numpy(hchannel.rgb_to_model_input(stored)),
                int(row["label"]),
            )

        stored = np.asarray(Image.open(path))

        if self.augment:
            # Seeded per (epoch-independent) index plus the dataset's seed, so a run
            # is reproducible without a global generator that a DataLoader worker
            # would fork a copy of.
            rng = np.random.default_rng((self._seed, index))
            stored = geometric(stored, rng)
            alpha, beta, gamma = self.jitter.sample(rng)
        else:
            alpha, beta, gamma = 1.0, 0.0, 1.0

        tensor = hchannel.from_stored(
            stored,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            invert=self.invert,
            standardise=self.standardise,
        )
        return torch.from_numpy(tensor), int(row["label"])


# --- the splits ---------------------------------------------------------------


def class_weights(rows: list[dict[str, str]]) -> torch.Tensor:
    """Cross-entropy weights inversely proportional to class frequency.

    Not optional, and not a refinement. BCSS is overwhelmingly non-epithelium, so an
    unweighted loss is minimised by a model that answers `0` to everything: it scores
    around 80% accuracy, it never finds a tumour, and the accuracy figure hides that
    completely. The plan asks for this to be *demonstrated* by ablation rather than
    asserted, because "we used class weights" is a claim and "here is what happens
    without them" is evidence.

    Normalised to mean 1 so the loss stays on a comparable scale between splits with
    different class balances.
    """
    counts = np.zeros(len(bcss.CLASS_NAMES), dtype=np.float64)
    for row in rows:
        counts[int(row["label"])] += 1

    if (counts == 0).any():
        absent = [bcss.CLASS_NAMES[i] for i, count in enumerate(counts) if count == 0]
        raise ValueError(
            f"these classes have no tiles: {absent}. A three-class head cannot be "
            "fitted on two classes, and a missing class here means the mapping or the "
            "vote thresholds are wrong rather than the dataset being small."
        )

    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def real_dcis_slides(rows: list[dict[str, str]]) -> set[str]:
    """BCSS slides carrying a human-drawn in-situ tile. Three of them, nine tiles.

    These are the only pixels in the entire project where a person looked at a duct and
    called it in-situ carcinoma. Everything else labelled class 1 is BEETLE's opinion.
    Nine tiles is a wretched test set and it is also the only real one there is.
    """
    return {
        row["slide_id"]
        for row in rows
        if int(row["label"]) == bcss.NON_INVASIVE
        and not bcss.is_borrowed(row.get("source", "bcss"))
    }


def institution_split(
    rows: list[dict[str, str]], *, holdout_real_dcis: bool | None = None
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """The held-out set, over two sources that need two different rules.

    **BCSS** keeps its published split: six tissue source sites held out entirely. Two
    regions from one hospital share a scanner and a staining protocol, so holding out
    hospitals rather than regions is the closest thing that dataset offers to "a slide
    from somewhere we have never seen".

    **BRACS/BEETLE** is held out by patient, because it has no institutions - every
    region comes from one Italian centre, so an institution split would be all-or-
    nothing. That covers all three borrowed trees, pooled: they are one laboratory and
    one set of patients, differing only in which lesion was annotated.
    `bcss.bracs_test_cases` hashes each patient id independently, so adding regions
    later cannot reshuffle an existing assignment - which is what makes it safe to add
    `ic` and `normal` to a manifest that already had `dcis` in it, and what puts a
    patient who contributes to two trees on one side of the split rather than both.

    **The third rule, and it is a deliberate departure from the published split.** BCSS
    has nine human-labelled in-situ tiles and all nine are in *training* institutions,
    so the published held-out set contains no real class 1 at all - which is why the
    original report shows class-1 recall of 0.000 with an empty truth row. Adding BRACS
    tiles does not fix that: it would only ever measure agreement with BEETLE. So the
    three slides carrying those nine tiles are moved to the held-out side.

    Moved as **whole slides**, not as the nine tiles. Taking only the class-1 tiles
    would leave the other 408 tiles of the same three slides in training, and a model
    scored on one tile of a slide it was fitted on is scored on its own memory. The cost
    is 417 training tiles, about 4 %; the gain is the only class-1 number in this
    project that a person can be held to.

    `holdout_real_dcis` defaults to **auto**: on when the manifest carries borrowed
    class-1 tiles, off when it does not. That default is not a convenience, it is the
    thing that keeps the BCSS-only pipeline working. Moving those three slides out of
    training leaves *no* class 1 to fit on at all, and `class_weights` refuses a class
    with zero tiles - correctly, because a three-class head cannot be fitted on two. So
    the nine real tiles can only be spent as a test set once there is a replacement to
    train on, and the condition is the replacement's existence rather than any
    particular tree's: `dcis` supplies it as in-situ carcinoma and `normal` as normal
    ducts, both landing in class 1. Pass `True` or `False` explicitly to override.
    """
    borrowed_rows = [row for row in rows if bcss.is_borrowed(row.get("source"))]
    # The auto default asks the question it actually depends on - "is there borrowed
    # class 1 to train on once the nine real tiles leave" - rather than "is the DCIS
    # tree present". Those were the same question while `dcis` was the only borrowed
    # tree; they are not any more, because `bracs_normal` supplies class 1 too, as
    # normal ducts under a no-carcinoma consensus. Keying on tree identity would refuse
    # to spend the nine real tiles on a manifest that has thousands of class-1 tiles to
    # fit on.
    if holdout_real_dcis is None:
        holdout_real_dcis = any(
            int(row["label"]) == bcss.NON_INVASIVE for row in borrowed_rows
        )

    moved = real_dcis_slides(rows) if holdout_real_dcis else set()
    # Hashed over every borrowed patient at once, across all three trees. A patient can
    # contribute a DCIS region and an IC region, and hashing per tree would be the same
    # draw anyway - `bracs_test_cases` depends on the id alone - but pooling says so,
    # and makes it impossible to introduce a per-tree seed later by accident.
    held_cases = bcss.bracs_test_cases(row["slide_id"] for row in borrowed_rows)

    def is_test(row: dict[str, str]) -> bool:
        if bcss.is_borrowed(row.get("source")):
            return row["slide_id"] in held_cases
        if row["slide_id"] in moved:
            return True
        return row["institution"].upper() in bcss.TEST_INSTITUTIONS

    train = [row for row in rows if not is_test(row)]
    test = [row for row in rows if is_test(row)]

    if not test:
        raise ValueError(
            "no tiles from the held-out institutions "
            f"({sorted(bcss.TEST_INSTITUTIONS)}) are in this manifest, so there is "
            "nothing to report a generalisation number on."
        )

    # A slide on both sides is the one failure this function exists to prevent, and it
    # is invisible in every metric it would inflate.
    overlap = {row["slide_id"] for row in train} & {row["slide_id"] for row in test}
    if overlap:
        raise AssertionError(
            f"these slides have tiles on both sides of the split: {sorted(overlap)[:5]}. "
            "Every number computed from this split would be measuring memorisation."
        )
    return train, test


def slide_folds(rows: list[dict[str, str]], *, folds: int = 5, seed: int = 0) -> list[np.ndarray]:
    """Fold assignments grouped by slide, for model selection on the training part.

    Grouped by `slide_id` and not by region: one slide contributes several regions in
    BCSS, and two regions of one slide are the same patient under the same scanner.

    Greedy balancing - the largest remaining slide goes to the smallest remaining fold
    - because a random assignment of 108 slides to 5 folds routinely produces one fold
    with twice the tiles of another, and a fold-to-fold comparison then measures fold
    size as much as anything else.
    """
    slides: dict[str, int] = {}
    for row in rows:
        slides[row["slide_id"]] = slides.get(row["slide_id"], 0) + 1

    rng = np.random.default_rng(seed)
    order = sorted(slides.items(), key=lambda item: (-item[1], item[0]))
    # Break ties in slide size deterministically but not alphabetically, so the fold
    # layout does not correlate with the barcode - which encodes the institution.
    rng.shuffle(order)
    order.sort(key=lambda item: -item[1])

    sizes = np.zeros(folds, dtype=np.int64)
    assignment: dict[str, int] = {}
    for slide, count in order:
        fold = int(np.argmin(sizes))
        assignment[slide] = fold
        sizes[fold] += count

    fold_of_row = np.asarray([assignment[row["slide_id"]] for row in rows])
    return [np.flatnonzero(fold_of_row == fold) for fold in range(folds)]


def assert_no_leak(*groups: list[dict[str, str]]) -> None:
    """No `slide_id` appears in more than one group. Asserted, never eyeballed.

    Gate G2 requires this and requires it as an assertion, because a leak is invisible
    in every downstream number except as an implausibly good one - and an implausibly
    good number is the last thing anyone questions.
    """
    seen: dict[str, int] = {}
    for index, group in enumerate(groups):
        for slide in {row["slide_id"] for row in group}:
            if slide in seen and seen[slide] != index:
                raise AssertionError(
                    f"slide {slide} appears in split {seen[slide]} and split {index}. "
                    "Tiles from one slide are near-duplicates, so this split would "
                    "report memory as generalisation."
                )
            seen[slide] = index
