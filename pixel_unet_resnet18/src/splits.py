"""Which tiles train and which are held out, and the assertion that they never mix.

Standalone, and simpler than the tile model's equivalent for one measured reason.

**The tile model had to depart from BCSS's published split; this one does not.** BCSS
carries nine human-drawn class-1 *tiles* and all nine are in training institutions, so its
held-out set had no class 1 at all and `datasets.institution_split` grew a
`holdout_real_dcis` flag that moves three whole slides across. At **pixel** level that
problem disappears: the same regions hold 4.75 M human class-1 pixels, and **0.38 M of
them across 7 regions are already in the published held-out institutions** (E2, EW, GM,
LL). The nine-tile figure was an artefact of the 50 % majority vote discarding every DCIS
focus too small to fill half a tile.

So this module implements BCSS's published split unmodified, and the held-out set is a
real, leak-free test set for DCIS boundaries drawn by a pathologist across four hospitals.

BRACS has no institutions - every region comes from one Italian centre - so it is held out
by **patient**, hashed rather than shuffled. A `random.sample` with a seed is reproducible
only while the set of patients never changes; add ten more and every previous assignment
moves, so two models both claiming seed 0 would be scored on different tiles. Hashing each
patient id independently makes a patient's side of the split a property of that patient,
and a later export that adds cases leaves existing assignments untouched.
"""

from __future__ import annotations

import hashlib

import classes

#: Share of BRACS patients held out.
DCIS_TEST_FRACTION = 0.25


def dcis_test_patients(patients) -> frozenset[str]:
    """Which BRACS patients are held out, decided by hash rather than by shuffle."""
    held = set()
    for patient in set(patients):
        digest = hashlib.sha256(patient.encode("utf-8")).digest()
        # First four bytes as a fraction of the range: a uniform draw in [0, 1) that
        # depends only on this patient id.
        if int.from_bytes(digest[:4], "big") / 2**32 < DCIS_TEST_FRACTION:
            held.add(patient)
    return frozenset(held)


def is_held_out(row: dict, dcis_held: frozenset[str]) -> bool:
    """One row's side of the split, by the rule its own source uses."""
    if row.get("source") == classes.SOURCE_DCIS:
        return row["patient"] in dcis_held
    return str(row["institution"]).upper() in classes.TEST_INSTITUTIONS


def split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """`(train, test)`. BCSS by institution, BRACS by hashed patient."""
    dcis_held = dcis_test_patients(
        row["patient"] for row in rows if row.get("source") == classes.SOURCE_DCIS
    )
    train = [row for row in rows if not is_held_out(row, dcis_held)]
    test = [row for row in rows if is_held_out(row, dcis_held)]

    if not test:
        raise ValueError(
            "nothing is held out, so there is no generalisation number to report. "
            f"Expected BCSS institutions {sorted(classes.TEST_INSTITUTIONS)} or some "
            "BRACS patients to fall on the test side."
        )
    if not train:
        raise ValueError("everything is held out; there is nothing to train on")

    assert_no_leak(train, test)
    return train, test


def assert_no_leak(*groups: list[dict]) -> None:
    """No patient appears in more than one group. Asserted, never eyeballed.

    A leak is invisible in every downstream number except as an implausibly good one, and
    an implausibly good number is the last thing anyone questions. `patient` is the unit
    rather than the region because two regions of one slide are the same tissue under the
    same scanner.
    """
    seen: dict[str, int] = {}
    for index, group in enumerate(groups):
        for patient in {row["patient"] for row in group}:
            if patient in seen and seen[patient] != index:
                raise AssertionError(
                    f"patient {patient} appears in split {seen[patient]} and split "
                    f"{index}. Tiles from one patient are near-duplicates, so this split "
                    "would report memory as generalisation."
                )
            seen[patient] = index


def folds(rows: list[dict], *, count: int = 5, seed: int = 0) -> list[list[int]]:
    """Fold assignments grouped by patient, as positional indices into `rows`.

    Greedy balancing - the largest remaining patient goes to the smallest remaining fold -
    because a random assignment routinely produces one fold with twice the tiles of
    another, and a fold-to-fold comparison then measures fold size as much as anything
    else.
    """
    import numpy as np

    sizes_by_patient: dict[str, int] = {}
    for row in rows:
        sizes_by_patient[row["patient"]] = sizes_by_patient.get(row["patient"], 0) + 1

    rng = np.random.default_rng(seed)
    order = sorted(sizes_by_patient.items(), key=lambda item: (-item[1], item[0]))
    # Break ties deterministically but not alphabetically, so the fold layout does not
    # correlate with the barcode - which encodes the institution.
    rng.shuffle(order)
    order.sort(key=lambda item: -item[1])

    totals = np.zeros(count, dtype=np.int64)
    assignment: dict[str, int] = {}
    for patient, size in order:
        fold = int(np.argmin(totals))
        assignment[patient] = fold
        totals[fold] += size

    out: list[list[int]] = [[] for _ in range(count)]
    for index, row in enumerate(rows):
        out[assignment[row["patient"]]].append(index)
    return out


def pixel_class_weights(
    rows: list[dict], *, source_authority: bool = True
) -> list[float]:
    """Cross-entropy weights inversely proportional to **pixel** frequency.

    Not tile frequency. The tile model's `class_weights` counts one label per tile, which
    is the wrong denominator here by three orders of magnitude: class 1 is 13.7 % of the
    tile model's tiles but 0.15 % of BCSS's pixels. Using tile counts would under-weight
    the rare class enormously and the model would learn to answer "stroma" everywhere.

    Read from the per-tile pixel counts the exporter recorded, so this costs a sum over
    the manifest rather than a second pass over every mask.
    """
    counts = [0.0, 0.0, 0.0]
    for row in rows:
        # Only the pixels that will actually supervise. Weighting against counts the
        # loss never sees would balance the model against supervision that is not
        # there - and the gap is not small: BRACS's blanked non-epithelium alone is
        # 58 M pixels, its blanked invasive another 16 M.
        authoritative = (
            classes.authoritative_pixels(row)
            if source_authority
            else {name: int(row[f"px_{name}"]) for name in classes.CLASS_NAMES}
        )
        for index, name in enumerate(classes.CLASS_NAMES):
            counts[index] += float(authoritative[name])

    total = sum(counts)
    if total <= 0:
        raise ValueError("no labelled pixels at all in these rows")
    absent = [classes.CLASS_NAMES[i] for i, c in enumerate(counts) if c == 0]
    if absent:
        raise ValueError(
            f"these classes have no labelled pixels: {absent}. A three-class model cannot "
            "be fitted on two, and a missing class here means the mapping or the export "
            "floor is wrong rather than the dataset being small."
        )

    weights = [total / (len(counts) * c) for c in counts]
    mean = sum(weights) / len(weights)
    return [w / mean for w in weights]
