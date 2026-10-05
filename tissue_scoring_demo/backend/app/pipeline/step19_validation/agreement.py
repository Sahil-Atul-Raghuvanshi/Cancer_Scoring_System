"""Comparing our numbers against pathologists', including where they disagree.

**The standard is inter-pathologist agreement, not a perfect ground truth.**
There is no true percent positive for a slide - four trained readers looking at
the same section produce four different numbers, and the spread between them is
the resolution of the measurement. A system landing inside that spread has done
what can be done; one reported against an imaginary correct answer is being
measured against a fiction.

From the 120 real readings, that spread is known: **no reading in the 120 percent
values is more than 10 points from its case mean**, and four sit at exactly 10.
So ±10 absolute percentage points is the target, and it is an absolute number
rather than a relative 10 %. The same rule appears not to apply to intensity -
reader disagreement there reaches 0.625 on a 0-2 scale - which is open question
Q4, so intensity agreement is reported without a pass/fail band on it.

Everything here degrades honestly when there is nothing to compare against.
`agreement()` on an empty reference returns a report that says so. It does not
compare our numbers against themselves, and it does not fill a missing reader
value with a case mean - both would produce an agreement figure that looks like
a result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Absolute percentage points. A reading further than this from the group mean
#: is referred back to that reader to re-examine, per OncoStem's own rule, and it
#: is therefore the width of the human spread we hold ourselves to.
PERCENT_TOLERANCE = 10.0


@dataclass(frozen=True)
class MarkerAgreement:
    """How one marker's numbers compare with the readers' on the same cases."""

    marker: str
    cases: int

    #: Ours minus theirs, in percentage points, over the matched cases.
    percent_bias: float
    percent_limits: tuple[float, float]
    percent_within_tolerance: int
    percent_max_error: float

    #: How many cases had an intensity to compare at all. Zero is a normal
    #: result, not a failure: the sheet may not carry that marker's intensity
    #: column, and a bias computed over nothing would print as 0.0 - which reads
    #: as perfect agreement.
    intensity_cases: int
    intensity_bias: float
    intensity_limits: tuple[float, float]
    #: Weighted kappa on the banded intensity, or None with fewer than two
    #: distinct bands - a kappa over a constant is undefined, not 1.0.
    intensity_kappa: float | None

    #: The readers' own spread on these cases, which is what our error is read
    #: against. Ours being larger than this is the finding; ours being smaller
    #: than this is not evidence of being better than a pathologist.
    reader_spread: float | None


@dataclass(frozen=True)
class AgreementReport:
    available: bool
    reason: str = ""
    markers: list[MarkerAgreement] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def bland_altman(ours: np.ndarray, theirs: np.ndarray) -> tuple[float, tuple[float, float]]:
    """Bias and 95 % limits of agreement, the standard method-comparison pair.

    Not a correlation. Two methods can correlate at 0.99 and differ by twenty
    points everywhere, and correlation would call that agreement - which is the
    specific way a validation section flatters a system that is consistently
    wrong.
    """
    difference = np.asarray(ours, dtype=np.float64) - np.asarray(theirs, dtype=np.float64)
    if difference.size == 0:
        return 0.0, (0.0, 0.0)
    bias = float(difference.mean())
    spread = float(difference.std(ddof=1)) if difference.size > 1 else 0.0
    return bias, (bias - 1.96 * spread, bias + 1.96 * spread)


def weighted_kappa(ours: np.ndarray, theirs: np.ndarray, levels: list[float]) -> float | None:
    """Linearly weighted Cohen's kappa over ordered bands.

    Weighted, because the bands are ordered: being one band out is a smaller
    error than being three out, and an unweighted kappa treats them the same.
    Returns None where it would be undefined - fewer than two distinct levels
    used, which on this panel is the normal state of N-cadherin.
    """
    ours = np.asarray(ours, dtype=np.float64)
    theirs = np.asarray(theirs, dtype=np.float64)
    if ours.size == 0 or len(levels) < 2:
        return None

    index = {value: position for position, value in enumerate(levels)}
    a = np.array([index.get(float(value), -1) for value in ours])
    b = np.array([index.get(float(value), -1) for value in theirs])
    keep = (a >= 0) & (b >= 0)
    a, b = a[keep], b[keep]
    if a.size == 0 or (len(set(a.tolist()) | set(b.tolist())) < 2):
        return None

    size = len(levels)
    observed = np.zeros((size, size))
    for first, second in zip(a, b, strict=True):
        observed[first, second] += 1
    observed /= observed.sum()

    row = observed.sum(axis=1)
    column = observed.sum(axis=0)
    expected = np.outer(row, column)

    weights = np.abs(np.subtract.outer(np.arange(size), np.arange(size))) / (size - 1)
    denominator = float((weights * expected).sum())
    if denominator == 0:
        return None
    return round(1.0 - float((weights * observed).sum()) / denominator, 4)


def agreement(
    ours: dict[str, dict[str, tuple[float, float]]],
    readers: dict[str, dict[str, list[tuple[float, float | None]]]],
    *,
    bands: list[float],
) -> AgreementReport:
    """Compare per marker, over the cases both sides have.

    `ours` is `{marker: {case_id: (percent, intensity)}}`.
    `readers` is `{marker: {case_id: [(percent, intensity | None), ...]}}` -
    every reader's own reading, not a pre-averaged consensus, so the readers'
    spread can be computed from the same data the comparison uses.

    A reader's intensity may be None where the sheet does not carry that
    marker's intensity column. Those cases are dropped from the intensity
    statistics and counted in `intensity_cases`; they are never filled in, since
    a substituted zero would read as the readers calling the slide negative.
    """
    if not readers:
        return AgreementReport(
            available=False,
            reason=(
                "No pathologist readings are on disk. 6Slide Reports2.xlsx holds the 120 "
                "(percent, intensity) pairs this step exists to compare against; without "
                "it there is nothing to validate against, and comparing our numbers "
                "against themselves would produce a figure that looks like a result."
            ),
            notes=[
                "Validation is what converts an output into a claim. Until this runs, the "
                "pipeline has produced numbers, not findings."
            ],
        )

    out: list[MarkerAgreement] = []
    for marker, cases in sorted(readers.items()):
        mine = ours.get(marker, {})
        matched = sorted(set(mine) & set(cases))
        if not matched:
            continue

        our_percent = np.array([mine[case][0] for case in matched], dtype=np.float64)
        # The consensus is a plain arithmetic mean of the readers - verified on
        # all 60 averages in the sheet: no weighting, no trimming, no dropping of
        # outliers. And it is what feeds OncoStem's own model, so it is what we
        # are measured against rather than any individual reader.
        their_percent = np.array(
            [float(np.mean([r[0] for r in cases[case]])) for case in matched]
        )
        # Only the cases where the readers actually recorded an intensity.
        with_intensity = [
            case
            for case in matched
            if any(reading[1] is not None for reading in cases[case])
        ]
        our_intensity_matched = np.array(
            [mine[case][1] for case in with_intensity], dtype=np.float64
        )
        their_intensity = np.array(
            [
                float(
                    np.mean(
                        [r[1] for r in cases[case] if r[1] is not None]
                    )
                )
                for case in with_intensity
            ]
        )

        spreads = [
            float(np.max([r[0] for r in cases[case]]) - np.min([r[0] for r in cases[case]]))
            for case in matched
            if len(cases[case]) > 1
        ]

        percent_bias, percent_limits = bland_altman(our_percent, their_percent)
        intensity_bias, intensity_limits = bland_altman(
            our_intensity_matched, their_intensity
        )
        errors = np.abs(our_percent - their_percent)

        out.append(
            MarkerAgreement(
                marker=marker,
                cases=len(matched),
                percent_bias=round(percent_bias, 2),
                percent_limits=(round(percent_limits[0], 2), round(percent_limits[1], 2)),
                percent_within_tolerance=int(np.count_nonzero(errors <= PERCENT_TOLERANCE)),
                percent_max_error=round(float(errors.max()), 2),
                intensity_cases=len(with_intensity),
                intensity_bias=round(intensity_bias, 3),
                intensity_limits=(
                    round(intensity_limits[0], 3),
                    round(intensity_limits[1], 3),
                ),
                intensity_kappa=weighted_kappa(
                    our_intensity_matched, their_intensity, bands
                ),
                reader_spread=round(float(np.mean(spreads)), 2) if spreads else None,
            )
        )

    return AgreementReport(
        available=bool(out),
        reason="" if out else "no case appears in both our results and the reader sheet",
        markers=out,
        notes=[
            f"The target is +/-{PERCENT_TOLERANCE:g} absolute percentage points against the "
            "reader consensus, because that is the spread four trained pathologists reach "
            "among themselves on these same slides.",
            "Intensity is reported without a pass band. Reader disagreement on intensity "
            "reaches 0.625 on a 0-2 scale, which no 10 % rule would tolerate, so whether "
            "the rule applies to intensity at all is an open question (Q4).",
            "A marker showing 0 intensity cases has no intensity column in the sheet "
            "being read. Its percent comparison still stands; its intensity is simply "
            "not something this data can speak to.",
            "A marker whose readers recorded the same value on every case - N-cadherin was "
            "exactly 80 % in all 24 reads - cannot be validated by agreement. Matching a "
            "constant demonstrates nothing, and its kappa is undefined rather than 1.0.",
        ],
    )


__all__ = [
    "PERCENT_TOLERANCE",
    "AgreementReport",
    "MarkerAgreement",
    "agreement",
    "bland_altman",
    "weighted_kappa",
]
