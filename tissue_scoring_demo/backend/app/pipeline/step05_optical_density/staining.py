"""Which dyes a section carries, read off step 5's point cloud.

**Nothing in this pipeline recorded this before, and step 7 now needs it.** The H&E
branch shows the model the colour photograph, which is only meaningful where there is a
second dye to see: on a DAB section "eosin" is a direction pointing at nothing, and a
model fitted on pink and blue would be reading a colour that is not there. So the branch
is offered on evidence rather than on a filename.

The evidence already exists. Step 5 fits a plane to a tile's optical densities, finds the
two edges of the wedge the data occupies, and reports for each edge which published
reference vector it lands nearest and how far away that is - and `REFERENCE_VECTORS`
includes eosin. It also reports `two_armed`, whose own docstring says false means "the
two reported arms are the two tails of a single lobe - what a tile carrying only a
counterstain looks like". That is already the "is there a second dye" test. This module
does not add a measurement; it names what the existing one implies.

**Step 6 treats an eosin-nearest arm as a fault, and that stays true.** Step 6
deconvolves on Ruifrok's haematoxylin-DAB basis, where an arm pointing at eosin means
the assay is not what was expected. Both readings are correct because they are answers
to different questions: step 6 asks "is this the H-DAB section I was told to unmix",
step 7 asks "does this section have two dyes at all". The prose on both screens says
which question it is answering rather than contradicting the other.

**The threshold is measured, and the measurement says the threshold is not really what
decides.** Over 18 tissue tiles from six of this project's sections, every H&E tile put
an arm 5.0 to 9.4 degrees from Ruifrok's eosin vector, and not one of the twelve
immunostained tiles produced an eosin-nearest arm at all - they came back haematoxylin
plus DAB, every time. So the discriminating fact is *which* published vector an arm
lands nearest, and `he_eosin_tolerance_deg` is the safety bound on top of that rather
than the discriminator itself. The full numbers are in that setting's docstring.

One thing the measurement contradicted, worth recording because the guess was wrong in
a direction that matters: an H&E section's wedge here is **narrower** than an
immunostained one's, not wider. H&E separations ran 18.8 to 38.1 degrees against 48.6
to 63.5 for haematoxylin-DAB. Had `separation` been made a criterion on the assumption
that two "real" dyes open a wider wedge, it would have rejected the H&E slides and
accepted the IHC ones. It is reported as evidence for exactly that reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.pipeline.step05_optical_density.density import Cloud

#: The reference vector's name in `app.common.stains.REFERENCE_VECTORS`. A constant
#: rather than a literal so a rename there is a NameError here and not a predicate
#: that silently stops matching.
EOSIN = "eosin"


class SlideStaining(str, Enum):
    """What step 5's cloud says this section carries."""

    #: Two dye directions, one of them eosin. Step 7's H&E branch is offered.
    HE = "he"
    #: Two directions, neither eosin - an immunostained section, which is what most
    #: of this project's slides are.
    HAEMATOXYLIN_DAB = "haematoxylin_dab"
    #: One lobe reported as two tails. A counterstain on its own.
    SINGLE_STAIN = "single_stain"
    #: Step 5 has not measured this slide, or measured something it cannot name.
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class StainingVerdict:
    """The verdict and every number behind it.

    The numbers travel because the verdict is going to grey out a button, and a
    greyed-out button with no explanation is the least useful thing a screen can do.
    `reason` is written for somebody who does not know what a stain vector is; the
    angles are there for somebody who does.
    """

    staining: SlideStaining
    is_he: bool
    reason: str
    source: str

    two_armed: bool | None = None
    separation: float | None = None
    reference_separation: float | None = None
    eosin_arm_degrees: float | None = None
    tolerance_deg: float | None = None
    arms: tuple[tuple[str, float], ...] = ()
    tile: tuple[int, int] | None = None


def _describe_arms(cloud: Cloud) -> str:
    return " and ".join(
        f"{arm.nearest} ({arm.degrees_from_nearest:.0f}° away)" for arm in cloud.arms
    )


def classify_staining(
    cloud: Cloud | None,
    *,
    tolerance_deg: float,
    tile: tuple[int, int] | None = None,
) -> StainingVerdict:
    """Name the dyes on a section from one tile's point cloud.

    The order of the tests is the order of the questions. Is there a measurement at
    all; is there more than one dye; is one of the dyes eosin. Only the last of those
    is a new judgement, and it is a single comparison against a published direction.

    A near miss - an arm that *is* nearest eosin but further away than the tolerance -
    is `UNKNOWN` rather than `HAEMATOXYLIN_DAB`. Calling it DAB would be asserting the
    opposite of what the measurement leans towards; `UNKNOWN` says the honest thing,
    which is that this tile does not settle it, and the reason suggests moving step 5's
    tile.
    """
    if cloud is None or len(cloud.arms) < 2:
        return StainingVerdict(
            staining=SlideStaining.UNKNOWN,
            is_he=False,
            source="step5-cloud" if cloud is not None else "step5-not-run",
            tolerance_deg=tolerance_deg,
            tile=tile,
            reason=(
                "Step 5 has not measured this slide's dye directions yet, so nothing "
                "has decided whether a second dye is present. Run step 5 and come back."
                if cloud is None else
                "Step 5 found fewer than two dye directions in this tile, so there is "
                "nothing to compare against eosin. Try step 5 on a tile with more "
                "tissue in it."
            ),
        )

    arms = tuple((arm.nearest, round(float(arm.degrees_from_nearest), 2))
                 for arm in cloud.arms)
    shared = {
        "two_armed": bool(cloud.two_armed),
        "separation": round(float(cloud.separation), 2),
        "reference_separation": round(float(cloud.reference_separation), 2),
        "tolerance_deg": tolerance_deg,
        "arms": arms,
        "tile": tile,
    }

    if not cloud.two_armed:
        return StainingVerdict(
            staining=SlideStaining.SINGLE_STAIN,
            is_he=False,
            source="step5-cloud",
            reason=(
                f"Step 5's two arms are only {cloud.separation:.0f}° apart, under the "
                "12° it needs to call a tile two-armed: that is one dye's cloud read "
                "from both sides, which is what an immunostained slide's counterstain "
                "looks like on its own. The H&E model needs both dyes to be there."
            ),
            **shared,
        )

    eosin_arms = [arm for arm in cloud.arms if arm.nearest == EOSIN]
    if eosin_arms:
        closest = min(arm.degrees_from_nearest for arm in eosin_arms)
        if closest <= tolerance_deg:
            note = ""
            if cloud.separation < 0.5 * cloud.reference_separation:
                # Common rather than alarming: measured H&E separations on this panel
                # run 19-38 degrees against 49-64 for haematoxylin-DAB, so an H&E
                # wedge is *expected* to be the narrower of the two. Worth saying,
                # because a narrow wedge does mean the deconvolution has less to
                # separate - not worth refusing over.
                note = (
                    f" The two directions are {cloud.separation:.0f}° apart, against "
                    f"{cloud.reference_separation:.0f}° for the reference pair - which "
                    "is normal for haematoxylin and eosin, whose colours are closer "
                    "together than haematoxylin and brown."
                )
            return StainingVerdict(
                staining=SlideStaining.HE,
                is_he=True,
                source="step5-cloud",
                eosin_arm_degrees=round(float(closest), 2),
                reason=(
                    f"Step 5 found two separate dye directions {cloud.separation:.0f}° "
                    f"apart, and one of them points within {closest:.0f}° of the "
                    "published eosin direction - so this section carries haematoxylin "
                    "and eosin, and the H&E model can read it." + note
                ),
                **shared,
            )

        return StainingVerdict(
            staining=SlideStaining.UNKNOWN,
            is_he=False,
            source="step5-cloud",
            eosin_arm_degrees=round(float(closest), 2),
            reason=(
                f"One arm points nearest eosin but {closest:.0f}° away, past the "
                f"{tolerance_deg:.0f}° needed to call it eosin. Too far to be sure, so "
                "the H&E option stays off. Try step 5 on a different tile - a field "
                "with more stroma in it makes the second dye easier to see."
            ),
            **shared,
        )

    return StainingVerdict(
        staining=SlideStaining.HAEMATOXYLIN_DAB,
        is_he=False,
        source="step5-cloud",
        reason=(
            f"Step 5's two arms are {_describe_arms(cloud)} - neither of them eosin, so "
            "this is an immunostained section rather than an H&E one. The blue-stain "
            "option on this screen is the one that reads it."
        ),
        **shared,
    )


def unknown(*, tolerance_deg: float) -> StainingVerdict:
    """The verdict for a slide step 5 has not been run on. No cloud, no guess."""
    return classify_staining(None, tolerance_deg=tolerance_deg)


__all__ = [
    "EOSIN",
    "SlideStaining",
    "StainingVerdict",
    "classify_staining",
    "unknown",
]
