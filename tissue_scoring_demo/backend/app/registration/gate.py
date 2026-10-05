"""The confidence gate: the whole safety story of moving a mask across slides.

VALIS returns a transform for a pair it could not really register, and that
transform is wrong *silently* - a warped mask that looks like a mask. The
worked example in `docs/segmentation_research/segmentation-approaches-implementation.md` Part 7 is a
near-blank IHC section, 17.5 mm2 of tissue against the H&E's 84.8, for which a
plausible-looking transform still comes back.

So four independent things are checked, and a failure names all of them rather
than the first:

    matched keypoints   did the two sections have enough in common to fit
                        anything at all
    residual error      how far apart the matched features still are after
                        warping, in microns
    tissue-area ratio   are these even comparable amounts of tissue - the check
                        that catches the near-blank section, and the only one
                        that does not depend on the registration having
                        succeeded
    round-trip error    warp there and back: does the transform agree with
                        itself over the region actually being moved, rather
                        than over the keypoints it was fitted on

The last one earns its place because the first two are measured *on the
features the fit used*, which is exactly where a transform looks best. The
round trip is measured on the ROI's own vertices.

Refusing is a result, not an error condition. It is reported to the viewer with
its numbers, and no mask is emitted.
"""

from __future__ import annotations

from app.core.config import settings

from .exceptions import RegistrationRefused


#: Lowest alignment NMI an intensity-based registration may post. Measured: registrations
#: that work land at 1.02-1.15, and ones that do not sit at ~1.00 because mutual information
#: of 1.0 means the two images are independent - the alignment tells you nothing.
MIN_ALIGNMENT_NMI = 1.008


def evaluate(diagnostics: dict) -> list[str]:
    """Every reason this registration should not be trusted. Empty means fine."""
    reasons: list[str] = []

    # **Which checks apply depends on how the transform was found.** A feature-based fit is
    # judged on its matched keypoints and their residual; an intensity-based one has
    # neither, because it never detects a feature - it optimises mutual information over
    # every pixel. Asking it for a keypoint count and passing it when the answer is absent
    # would be a check that cannot fail, which is not a check.
    if diagnostics.get("registration_kind") == "intensity":
        nmi = diagnostics.get("alignment_nmi")
        if nmi is None:
            reasons.append(
                "no alignment score was recorded, so how well the two sections correspond "
                "is unknown and cannot be checked"
            )
        elif nmi < MIN_ALIGNMENT_NMI:
            reasons.append(
                f"alignment score {nmi:.4f} is below {MIN_ALIGNMENT_NMI} - mutual "
                f"information near 1.0 means the two sections are independent, so the "
                f"transform is not describing a correspondence"
            )

        # The transform's own geometry. A similarity transform cannot fold, so this is a
        # guard against a stored transform being something other than what was approved
        # rather than against the method itself.
        if diagnostics.get("folded_points"):
            reasons.append(
                f"the transform folds tissue through itself at "
                f"{diagnostics['folded_points']} point(s) - that is geometrically "
                f"impossible and the mask it carries would land on unrelated cells"
            )

        ratio = diagnostics.get("tissue_area_ratio")
        if ratio is not None and not (
            settings.registration_tissue_ratio_min <= ratio <= settings.registration_tissue_ratio_max
        ):
            reasons.append(
                f"tissue-area ratio {ratio:.2f} is outside "
                f"{settings.registration_tissue_ratio_min:g}-"
                f"{settings.registration_tissue_ratio_max:g} - one section holds far less "
                f"tissue than the other, so there is nothing to align it to"
            )

        round_trip = diagnostics.get("round_trip_median_um")
        if round_trip is not None and round_trip > settings.registration_max_round_trip_um:
            reasons.append(
                f"round-trip error {round_trip:.0f} um, over the "
                f"{settings.registration_max_round_trip_um:.0f} um limit - warping the "
                f"region across and back does not return it to where it started"
            )
        return reasons

    keypoints = diagnostics.get("matched_keypoints")
    if keypoints is None or keypoints < settings.registration_min_keypoints:
        reasons.append(
            f"only {keypoints if keypoints is not None else 'unknown'} matched features, "
            f"need at least {settings.registration_min_keypoints} - the two sections have "
            f"too little in common to fit a transform to"
        )

    residual = diagnostics.get("residual_error_um")
    if residual is None:
        # Not "no news is good news". A residual that was never measured is a
        # check that never ran, and this one has already been silently disabled
        # once - by a `dst_dir` deep enough that VALIS could not write the
        # summary it comes from (Windows' 260-character path limit). Refusing
        # is what makes that visible instead of leaving three gates doing the
        # work of four.
        detail = diagnostics.get("measure_error")
        reasons.append(
            "no residual error was recorded, so how far apart the matched features "
            "ended up is unknown and cannot be checked"
            + (f" ({detail})" if detail else "")
        )
    elif residual > settings.registration_max_residual_um:
        reasons.append(
            f"matched features still {residual:.0f} um apart after warping, over the "
            f"{settings.registration_max_residual_um:.0f} um limit - that is a different "
            f"part of the tissue, not a rounding error"
        )

    ratio = diagnostics.get("tissue_area_ratio")
    if ratio is not None and not (
        settings.registration_tissue_ratio_min <= ratio <= settings.registration_tissue_ratio_max
    ):
        he_mm2 = diagnostics.get("he_tissue_mm2")
        ihc_mm2 = diagnostics.get("ihc_tissue_mm2")
        detail = (
            f" ({he_mm2:.1f} mm2 of H&E against {ihc_mm2:.1f} mm2 of IHC)"
            if he_mm2 is not None and ihc_mm2 is not None
            else ""
        )
        reasons.append(
            f"tissue-area ratio {ratio:.2f}{detail} is outside "
            f"{settings.registration_tissue_ratio_min:g}-"
            f"{settings.registration_tissue_ratio_max:g} - one section holds far less "
            f"tissue than the other, so there is nothing to align it to"
        )

    round_trip = diagnostics.get("round_trip_median_um")
    if round_trip is not None and round_trip > settings.registration_max_round_trip_um:
        reasons.append(
            f"round-trip error {round_trip:.0f} um, over the "
            f"{settings.registration_max_round_trip_um:.0f} um limit - warping the region "
            f"across and back does not return it to where it started, so the transform "
            f"does not agree with itself over the region being moved"
        )

    return reasons


def check(diagnostics: dict) -> None:
    """Raise `RegistrationRefused` unless every measure is within its limit."""
    reasons = evaluate(diagnostics)
    if reasons:
        raise RegistrationRefused(reasons, diagnostics)


__all__ = ["check", "evaluate"]
