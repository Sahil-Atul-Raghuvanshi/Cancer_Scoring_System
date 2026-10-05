"""Read-only status of the registration pass. Safe to run while it is going.

    python watch.py            one snapshot
    python watch.py --follow   refresh every 30 seconds

Touches nothing. It reads the checkpoint and the per-case JSON and prints them, so it can
be run from another window at any point without any risk to the pass - which is the whole
reason it is a separate script rather than a flag on the driver.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402

#: Acceptance thresholds, so the snapshot says pass or fail rather than making a person
#: hold the numbers in their head at seven in the morning.
#:
#: `MIN_KEYPOINTS` is the pairwise gate's own floor, applied to the weakest link of the
#: chain rather than to an endpoint pair - a composed transform is only as good as the
#: worst join along it.
#:
#: **The NMI test is on the absolute value, not on the gain, and that is a correction.**
#: The plan's first draft asked for a gain of +0.010 over the identity transform, taken
#: from what the old pairwise registrations achieved. That number does not transfer:
#: Phase 1 already centres every section on its tissue centroid and Phase 3 already
#: rotates it into the H&E's frame, so "unregistered" here is a far better starting point
#: than the old raw-canvas overlay - it measures 1.0064 where the old baseline measured
#: 1.0028. The gain is smaller because the baseline is better, not because the
#: registration is worse. The absolute figure is what compares like with like: old
#: successes landed at 1.0096 to 1.0173, so 1.008 is the floor, and any gain at all above
#: noise distinguishes a real registration from the ~0.0000 the failures returned.
MIN_KEYPOINTS = 50
MIN_NMI = 1.008
MIN_NMI_GAIN = 0.002

#: Largest non-rigid residual a marker may carry, in microns. The pairwise gate's own
#: limit, and it has to be checked separately from the keypoint count because the two
#: fail independently: CAN_00865's chain breaks at its W-to-R join and R comes back
#: with a 2374 um residual, which no keypoint threshold would have caught.
MAX_RESIDUAL_UM = 250.0


def weakest_to(payload: dict, code: str) -> int | None:
    """Fewest matched keypoints on any link between the H&E and this marker.

    **Not the weakest link in the whole chain**, which is what the registration itself
    reports and what a first reading of this screen used. A chained registration composes
    the transform along the cut order, so the transform for a marker depends only on the
    joins *before* it. CAN_00865 registers as `HE -> U -> F -> W -> R -> A`, and its near-
    negative CD44 section sits last: judging U, F, W and R by A's 5 matched keypoints
    fails four markers for a slide none of their transforms pass through.

    Falls back to the whole-chain figure when the order is unknown, and for a rung that
    registered every slide straight to the reference the sub-chain is just that slide.
    """
    order = payload.get("order") or []
    slides = payload.get("slides") or {}
    if code not in order:
        return payload.get("chain_weakest_matched")
    if str(payload.get("chosen_rung", "")).find("direct") >= 0:
        return slides.get(code, {}).get("matched_keypoints")

    counts = []
    for other in order[1 : order.index(code) + 1]:
        got = slides.get(other, {}).get("matched_keypoints")
        if got is not None:
            counts.append(got)
    return min(counts) if counts else None


def verdict(payload: dict, entry: dict, code: str = "", mpp: float = 0.0) -> str:
    if not payload.get("ok"):
        return "FAIL"
    # **Mutual information decides, and the other two are context.** That ordering is a
    # correction, made after acting on the residual column nearly threw away a good
    # registration. CAN_00865's R slide reports a 2374 um non-rigid residual, which reads
    # as catastrophic - and R overlaps W at 0.908 IoU after registration, which is
    # excellent. Both of VALIS's error columns are distances between the keypoints the
    # transform was *fitted to*, so on a slide with five matches they are computed from
    # five points and describe that sample rather than the transform.
    #
    # NMI is measured over a lattice covering the whole tissue and owes nothing to the
    # fit, which is the entire reason it is worth computing.
    problems = []
    nmi = entry.get("alignment_nmi")
    gain = entry.get("alignment_nmi_gain")
    if nmi is None or nmi < MIN_NMI:
        problems.append(f"nmi {nmi}")
    if gain is None or gain < MIN_NMI_GAIN:
        problems.append(f"no gain over identity ({gain})")

    # A negative gain is not a small problem. It says the feature registration moved the
    # tissue further from agreement than the outline alignment Phase 1 and 2 had already
    # achieved - centroid, measured rotation, matched physical scale. When that happens the
    # honest transform to carry a mask with is the outline one, and the registered
    # transform should be discarded rather than used because it is the one that ran.
    if gain is not None and gain < 0:
        problems.append(f"WORSE than outline alone (outline {entry.get('alignment_nmi_unregistered')})")

    notes = []
    weakest = weakest_to(payload, code) if code else payload.get("chain_weakest_matched")
    thin = (weakest or 0) < MIN_KEYPOINTS
    if thin:
        notes.append(f"only {weakest} keypoints")

    residual = entry.get("non_rigid_error")
    if residual is not None and mpp and residual * mpp > MAX_RESIDUAL_UM:
        if thin:
            # Two unreliable signals do not make a reliable one. Say so rather than
            # compounding them into a confident-looking failure.
            notes.append(f"residual {residual * mpp:.0f} um, unreliable on so few points")
        else:
            problems.append(f"residual {residual * mpp:.0f} um on {weakest} points")

    if problems:
        return "CHECK: " + ", ".join(problems + notes)
    return "ok" if not notes else "ok (" + ", ".join(notes) + ")"


def snapshot() -> None:
    checkpoint = common.load_checkpoint()
    rows = checkpoint.get("cases", {})
    print("=" * 96)
    print(f"registration pass - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 96)
    if not rows:
        print("nothing recorded yet")
    for case in sorted(rows, key=lambda c: rows[c].get("updated", "")):
        row = rows[case]
        print(
            f"{case:12} {row.get('state', '?'):8} phase={row.get('phase', '-'):9} "
            f"rung={str(row.get('rung', '-')):20} weakest={str(row.get('weakestMatched', '-')):>6} "
            f"px={str(row.get('processedPx', '-')):>5} {row.get('seconds', '')}s"
        )
        if row.get("cutOrder"):
            print(
                f"{'':12} cut order {row['cutOrder']}  "
                f"(margin {row.get('orderMargin')}, "
                f"{'confident' if row.get('orderConfident') else 'NOT confident'})"
            )
        if row.get("error"):
            print(f"{'':12} error: {str(row['error'])[:140]}")

    print("-" * 96)
    print(
        f"{'case':12} {'mk':3} {'kp':>6} {'toHere':>7} {'outline':>8} {'nmi':>7} {'gain':>7} "
        f"{'nonrig um':>10}  verdict"
    )
    for case in sorted(rows):
        payload = common.read_json(common.case_dir(case) / "registration.json")
        if not payload:
            continue
        # VALIS reports its errors in the units of the images it registered, and those are
        # rendered PNGs with no physical scale recorded - so it says "pixel". Those are
        # render pixels, and one is `targetMpp` microns across. Converting here rather
        # than leaving a bare number labelled "pixel" on the screen at 7am, because the
        # thresholds this has to be judged against are all stated in microns.
        aligned = common.read_json(common.case_dir(case) / "aligned.json") or {}
        mpp = float(aligned.get("targetMpp") or 0.0)

        def microns(value):
            if value is None:
                return "-"
            return f"{value * mpp:.1f}" if mpp else f"{value:.1f}px"

        for code, entry in sorted((payload.get("slides") or {}).items()):
            print(
                f"{case:12} {code:3} {str(entry.get('matched_keypoints')):>6} "
                f"{str(weakest_to(payload, code)):>7} "
                f"{str(entry.get('alignment_nmi_unregistered')):>8} "
                f"{str(entry.get('alignment_nmi')):>7} {str(entry.get('alignment_nmi_gain')):>7} "
                f"{microns(entry.get('non_rigid_error')):>10}"
                f"  {verdict(payload, entry, code, mpp)}"
            )

    # Where an earlier attempt survives beside the current one, show both. A change that
    # was supposed to help and quietly did not is the thing a single column cannot show,
    # and the first-pass files exist precisely so that comparison is possible.
    compared = []
    for case in sorted(rows):
        previous = common.read_json(common.case_dir(case) / "registration.first-pass.json")
        current = common.read_json(common.case_dir(case) / "registration.json")
        if not previous or not current:
            continue
        compared.append(
            (
                case,
                previous.get("chain_weakest_matched"),
                current.get("chain_weakest_matched"),
                previous.get("chosen_rung"),
                current.get("chosen_rung"),
            )
        )
    if compared:
        print("-" * 96)
        print(f"{'case':12} {'first pass':>11} {'now':>8}   rung then -> now")
        for case, before, after, rung_before, rung_after in compared:
            arrow = "same" if before == after else ("better" if (after or 0) > (before or 0) else "WORSE")
            print(
                f"{case:12} {str(before):>11} {str(after):>8}   "
                f"{rung_before} -> {rung_after}  ({arrow})"
            )

    log = common.RUN_LOG
    if log.is_file():
        print("-" * 96)
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines()[-8:]:
            print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args()
    while True:
        snapshot()
        if not args.follow:
            return 0
        time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
