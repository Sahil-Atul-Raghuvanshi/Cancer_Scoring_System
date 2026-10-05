"""The H&E gate: which dyes a section carries, and whether step 7 offers its branch.

Pure tests over synthetic clouds. No slide, no reader, no torch - the predicate is one
comparison against a published direction, and testing it through a whole step 5 run
would test step 5 rather than the rule.

The cases here are the four verdicts plus the two ways of not knowing, because the
interesting property of this gate is not that it says yes to an H&E slide: it is that it
says *no* for four distinguishable reasons, and puts each of them on screen in words.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.pipeline.step05_optical_density.staining import (
    SlideStaining,
    classify_staining,
    unknown,
)

TOLERANCE = 15.0


def _cloud(two_armed: bool, separation: float, arms, reference: float = 43.0):
    """Only the five attributes the predicate reads.

    A real `Cloud` needs a fitted plane and a point cloud to fit it to, and building
    one would be testing the eigendecomposition. The rule under test is which of the
    reported arms points at eosin.
    """
    return SimpleNamespace(
        two_armed=two_armed,
        separation=separation,
        reference_separation=reference,
        arms=tuple(
            SimpleNamespace(nearest=name, degrees_from_nearest=degrees)
            for name, degrees in arms
        ),
    )


def test_the_he_branch_needs_two_arms_and_one_of_them_eosin() -> None:
    """The predicate, as a table. Each row is a different reason to refuse."""
    cases = [
        # two dyes, one of them eosin, comfortably inside tolerance
        (_cloud(True, 58.0, [("haematoxylin", 4.0), ("eosin", 6.0)]),
         SlideStaining.HE, True),
        # the same arms, but step 5 says one lobe - a counterstain on its own
        (_cloud(False, 7.0, [("haematoxylin", 4.0), ("eosin", 6.0)]),
         SlideStaining.SINGLE_STAIN, False),
        # two dyes, neither eosin: an immunostained section
        (_cloud(True, 41.0, [("haematoxylin", 4.0), ("dab", 9.0)]),
         SlideStaining.HAEMATOXYLIN_DAB, False),
        # nearest eosin, but too far to call it eosin
        (_cloud(True, 50.0, [("haematoxylin", 4.0), ("eosin", 31.0)]),
         SlideStaining.UNKNOWN, False),
        # step 5 has not run
        (None, SlideStaining.UNKNOWN, False),
    ]

    for cloud, expected, is_he in cases:
        verdict = classify_staining(cloud, tolerance_deg=TOLERANCE)
        assert verdict.staining is expected, verdict.reason
        assert verdict.is_he is is_he


def test_a_near_miss_is_unknown_rather_than_immunostained() -> None:
    """Calling it DAB would assert the opposite of what the measurement leans towards.

    The arm *is* nearest eosin; it is simply not near enough to be sure. `UNKNOWN` says
    that, and the reason tells the viewer what to do about it.
    """
    verdict = classify_staining(
        _cloud(True, 50.0, [("haematoxylin", 4.0), ("eosin", 31.0)]),
        tolerance_deg=TOLERANCE,
    )
    assert verdict.staining is SlideStaining.UNKNOWN
    assert verdict.eosin_arm_degrees == 31.0
    assert "31" in verdict.reason and "15" in verdict.reason


def test_the_reason_names_the_number_that_decided() -> None:
    """A disabled option renders this verbatim, so it has to explain rather than
    announce - and an empty reason would be the worst possible greyed-out button."""
    refusals = [
        _cloud(False, 7.0, [("haematoxylin", 4.0), ("eosin", 6.0)]),
        _cloud(True, 41.0, [("haematoxylin", 4.0), ("dab", 9.0)]),
        _cloud(True, 50.0, [("haematoxylin", 4.0), ("eosin", 31.0)]),
        None,
    ]
    for cloud in refusals:
        verdict = classify_staining(cloud, tolerance_deg=TOLERANCE)
        assert not verdict.is_he
        assert verdict.reason.strip()
        assert any(character.isdigit() for character in verdict.reason) or cloud is None


def test_a_narrow_wedge_is_noted_and_not_refused() -> None:
    """`separation` is evidence, not a criterion - and the measurement is why.

    Measured on this project's slides, an H&E wedge is the *narrower* of the two: 18.8
    to 38.1 degrees against 48.6 to 63.5 for haematoxylin-DAB. So a threshold on
    separation, set on the intuition that two real dyes open a wider wedge, would have
    rejected every H&E section here and accepted every immunostained one. The 18.8
    degree case below is a real slide, and it must come back `he`.
    """
    verdict = classify_staining(
        _cloud(True, 18.8, [("haematoxylin", 6.1), ("eosin", 5.0)]),
        tolerance_deg=TOLERANCE,
    )
    assert verdict.is_he
    # The angle is reported so a reader can weigh it, and named as normal rather than
    # as a fault, because on this panel it is.
    assert "19°" in verdict.reason or "18.8" in verdict.reason
    assert "normal" in verdict.reason


def test_a_single_arm_cloud_is_not_enough_to_decide() -> None:
    """Fewer than two arms means there is nothing to compare against eosin."""
    verdict = classify_staining(
        _cloud(True, 0.0, [("haematoxylin", 4.0)]), tolerance_deg=TOLERANCE
    )
    assert verdict.staining is SlideStaining.UNKNOWN
    assert verdict.source == "step5-cloud"


def test_a_slide_step_5_has_not_seen_says_so_rather_than_guessing() -> None:
    verdict = unknown(tolerance_deg=TOLERANCE)
    assert verdict.staining is SlideStaining.UNKNOWN
    assert verdict.source == "step5-not-run"
    assert "Run step 5" in verdict.reason


def test_the_gate_does_not_run_step_5_as_a_side_effect(monkeypatch) -> None:
    """Step 7 asks this question on a screen that advertises itself as one pass over a
    mask. Forcing step 5 from there would read a tile, step 3's mask and step 4's white
    point behind that screen, and would evict the memo steps 5 and 6 share."""
    from app.services import density_service as module

    def explode(*args, **kwargs):  # pragma: no cover - the point is it is not called
        raise AssertionError("step 7 must not trigger a step 5 run")

    monkeypatch.setattr(module.density_service, "_run", explode)
    verdict = module.density_service.staining("a-slide-nobody-has-measured")
    assert verdict.source == "step5-not-run"


def test_the_tolerance_is_the_one_that_was_asked_for() -> None:
    """Not `density_angular_tolerance_deg`, which is a per-pixel stability bound at
    1.5 degrees. An arm is a weighted percentile of a whole tile against a literature
    vector, and 1.5 degrees there would refuse every real slide."""
    from app.core.config import settings

    assert settings.he_eosin_tolerance_deg > settings.density_angular_tolerance_deg

    close = _cloud(True, 50.0, [("haematoxylin", 4.0), ("eosin", 10.0)])
    assert classify_staining(close, tolerance_deg=15.0).is_he
    assert not classify_staining(close, tolerance_deg=1.5).is_he


@pytest.mark.parametrize("staining", list(SlideStaining))
def test_every_verdict_is_serialisable_as_its_own_wire_value(staining) -> None:
    """A `str` enum, so the value on the wire is the value in the code."""
    assert isinstance(staining.value, str)
    assert SlideStaining(staining.value) is staining


# --- what step 7 does with the verdict ---------------------------------------

# The gate has two consumers now, not one. Both of step 7's colour options are gated on
# it, so the tests below are about `_offered_branches` rather than about the predicate:
# the predicate says "there is a second dye", and this is where that becomes a button.


def _offer(cloud, *, tolerance=TOLERANCE):
    """Step 7's offered branches, for a slide whose step 5 cloud looked like this."""
    from app.services.tiling_service import _offered_branches

    verdict = classify_staining(cloud, tolerance_deg=tolerance)
    return {entry.id: entry for entry in _offered_branches(verdict)}, verdict


_HE_CLOUD = _cloud(True, 30.0, [("haematoxylin", 8.0), ("eosin", 6.0)])
_IHC_CLOUD = _cloud(True, 60.0, [("haematoxylin", 16.0), ("dab", 26.0)])


def test_both_colour_options_are_refused_on_an_immunostained_section() -> None:
    """`he` and `beetle` go together, and the blue-stain option is untouched.

    BEETLE used to be offered on an IHC slide whenever its archive was on disk, which
    made it the one option on this screen that ignored what step 5 measured. It reads the
    colour photograph with no deconvolution and no stain normalisation - `dataset.json`
    names all three channels `rgb_to_0_1` and `to_model_input` is a division by 255 - so
    the dyes on the section are its input distribution, and a DAB section is brown where
    that distribution is pink. It would not fail on such a slide; it would return a
    confident wrong segmentation, which is the failure mode this whole screen is arranged
    to prevent.
    """
    offered, _verdict = _offer(_IHC_CLOUD)

    assert offered["h_channel"].enabled, "the blue stain is on every slide in the panel"
    assert not offered["he"].enabled
    assert not offered["beetle"].enabled

    # Still listed, still implemented: an option that vanishes makes the pipeline look
    # smaller than it is, and the reason is the informative part.
    assert offered["beetle"].implemented
    assert offered["beetle"].reason


def test_the_beetle_refusal_names_the_stain_and_not_the_download() -> None:
    """The order of the two questions, which is the whole point of asking them in one.

    Availability is two facts: can this slide take it, and is it installed. Only the
    second is fixable, and mentioning it first would send a reader after a 1.9 GB
    download for a section that could never use the result. So the staining reason wins,
    and it carries step 5's own sentence rather than a second version of it - the same
    string step 5's screen now shows.
    """
    offered, verdict = _offer(_IHC_CLOUD)
    reason = offered["beetle"].reason

    assert verdict.reason in reason, "step 5's sentence is quoted, not restated"
    assert "brown where it expects pink" in reason
    assert "1.9 GB" not in reason and "Zenodo" not in reason


def test_beetle_is_offered_on_an_he_section_when_its_archive_is_there() -> None:
    """The positive case, and the reason this is not simply `is_he`.

    On an H&E slide the staining question passes and the *machine* question is asked, so
    the outcome here depends on whether the 1.9 GB archive was downloaded. Both outcomes
    are asserted rather than skipped, because both are correct behaviour and which one
    this machine sees is not a property of the code.
    """
    from app.pipeline.step08_tissue_type_segmentation import beetle

    offered, _verdict = _offer(_HE_CLOUD)
    installed, problem = beetle.available()

    assert offered["he"].enabled, "an H&E section is what the colour branches are for"
    if installed:
        assert offered["beetle"].enabled
        assert offered["beetle"].reason is None
    else:
        assert not offered["beetle"].enabled
        assert offered["beetle"].reason == problem


def test_a_slide_step_5_has_not_measured_offers_neither_colour_option() -> None:
    """Not-yet-measured is refused exactly as immunostained is, and says which it is.

    The fail-closed direction matters: an unknown verdict must not open a colour branch,
    because "we have not looked" and "there are two dyes" are not the same claim. What
    stops this being a dead end is that the verdict is a dependency of the branch payload
    on the client, so running step 5 re-asks the question.
    """
    offered, _verdict = _offer(None)

    assert offered["h_channel"].enabled
    assert not offered["he"].enabled
    assert not offered["beetle"].enabled
    assert "Run step 5" in offered["he"].reason
    assert "Run step 5" in offered["beetle"].reason


def test_the_fields_of_view_stay_the_models_question_not_the_slides() -> None:
    """A branch disabled by the staining still lists what scales it has.

    The two levels answer different questions, and collapsing them would lose
    information: branch `enabled` is "can this slide take it", field-of-view `available`
    is "does a model exist at this scale". BEETLE's four are available together or not at
    all, because one release covers every extent - so on an IHC slide the option is off
    while its scales still describe the release honestly.
    """
    offered, _verdict = _offer(_IHC_CLOUD)

    assert not offered["beetle"].enabled
    assert len(offered["beetle"].fields_of_view) == 4
    assert len({entry.available for entry in offered["beetle"].fields_of_view}) == 1
    assert all(entry.mpp == 0.5 for entry in offered["beetle"].fields_of_view)


#: What this project's own slides actually measured, three tissue tiles per section.
#: `(section, expected verdict, separation, [(nearest, degrees), ...])`.
#:
#: Recorded as a table rather than described in prose because it is the evidence the
#: whole gate rests on, and because it is the thing that will stop being true first: a
#: new scanner, a new dye lot or a re-stained block moves these angles, and this table
#: is where that shows up as a failure instead of as a quietly disabled option.
MEASURED = [
    ("CAN_00270_26_H&E", SlideStaining.HE, 31.1, [("haematoxylin", 8.7), ("eosin", 5.8)]),
    ("CAN_00270_26_H&E", SlideStaining.HE, 27.9, [("haematoxylin", 7.3), ("eosin", 9.1)]),
    ("CAN_00270_26_H&E", SlideStaining.HE, 30.5, [("haematoxylin", 8.1), ("eosin", 6.9)]),
    ("CAN_00267_26_H&E", SlideStaining.HE, 18.8, [("eosin", 17.8), ("eosin", 6.2)]),
    ("CAN_00267_26_H&E", SlideStaining.HE, 27.2, [("haematoxylin", 15.4), ("eosin", 9.4)]),
    ("CAN_00267_26_H&E", SlideStaining.HE, 38.1, [("haematoxylin", 6.1), ("eosin", 5.0)]),
    ("CAN_00270_26_A", SlideStaining.HAEMATOXYLIN_DAB, 63.5, [("haematoxylin", 17.6), ("dab", 29.4)]),
    ("CAN_00270_26_A", SlideStaining.HAEMATOXYLIN_DAB, 63.1, [("haematoxylin", 17.9), ("dab", 28.7)]),
    ("CAN_00270_26_A", SlideStaining.HAEMATOXYLIN_DAB, 63.1, [("haematoxylin", 16.4), ("dab", 28.0)]),
    ("CAN_00270_26_R", SlideStaining.HAEMATOXYLIN_DAB, 60.6, [("haematoxylin", 14.6), ("dab", 25.3)]),
    ("CAN_00270_26_R", SlideStaining.HAEMATOXYLIN_DAB, 60.0, [("haematoxylin", 19.1), ("dab", 23.9)]),
    ("CAN_00270_26_R", SlideStaining.HAEMATOXYLIN_DAB, 58.3, [("haematoxylin", 14.0), ("dab", 21.3)]),
    ("CAN_00267_26_W", SlideStaining.HAEMATOXYLIN_DAB, 59.4, [("haematoxylin", 15.2), ("dab", 27.2)]),
    ("CAN_00267_26_W", SlideStaining.HAEMATOXYLIN_DAB, 57.7, [("haematoxylin", 17.9), ("dab", 25.3)]),
    ("CAN_00267_26_W", SlideStaining.HAEMATOXYLIN_DAB, 49.2, [("haematoxylin", 16.6), ("dab", 20.8)]),
    ("CAN_00267_26_U", SlideStaining.HAEMATOXYLIN_DAB, 49.2, [("haematoxylin", 20.6), ("dab", 23.9)]),
    ("CAN_00267_26_U", SlideStaining.HAEMATOXYLIN_DAB, 50.8, [("haematoxylin", 17.6), ("dab", 26.5)]),
    ("CAN_00267_26_U", SlideStaining.HAEMATOXYLIN_DAB, 48.6, [("haematoxylin", 22.3), ("dab", 23.7)]),
]


@pytest.mark.parametrize(("section", "expected", "separation", "arms"), MEASURED)
def test_the_gate_agrees_with_what_the_real_slides_measured(
    section, expected, separation, arms
) -> None:
    """Eighteen for eighteen, at the shipped tolerance.

    The arms are replayed rather than re-measured - re-reading six whole-slide images
    would make this a ten-minute test of `tiffslide` - but the numbers are the ones the
    reader produced, so a change to the predicate or the tolerance that would misread a
    real section fails here.
    """
    from app.core.config import settings

    verdict = classify_staining(
        _cloud(True, separation, arms),
        tolerance_deg=settings.he_eosin_tolerance_deg,
    )
    assert verdict.staining is expected, f"{section}: {verdict.reason}"
    assert verdict.is_he is (expected is SlideStaining.HE)


def test_no_immunostained_tile_produced_an_eosin_arm_at_all() -> None:
    """The measurement's real finding, and the reason the tolerance is a safety bound
    rather than the discriminator: on all twelve IHC tiles the nearest vectors were
    haematoxylin and DAB, so the eosin comparison never even arises."""
    ihc = [entry for entry in MEASURED if entry[1] is SlideStaining.HAEMATOXYLIN_DAB]
    assert len(ihc) == 12
    assert not any(
        name == "eosin" for _s, _e, _sep, arms in ihc for name, _d in arms
    )

    from app.core.config import settings

    # Per tile, the *closest* eosin arm - which is what the predicate reads, and the
    # distinction is not academic. One real H&E tile (CAN_00267_26_H&E at 18.8 degrees
    # separation) has **both** arms nearest eosin, the second of them 17.8 degrees out,
    # past the shipped bound. Taking the maximum would call that tile a near miss and
    # refuse a section that is plainly H&E; taking the minimum reads it correctly off
    # the 6.2 degree arm. A tile whose *only* eosin arm sat at 17.8 would still be
    # refused, which is the intended behaviour - one poorly conditioned tile is not
    # enough to claim a second dye.
    he = [entry for entry in MEASURED if entry[1] is SlideStaining.HE]
    closest = [
        min(degrees for name, degrees in arms if name == "eosin")
        for _s, _e, _sep, arms in he
    ]
    assert max(closest) <= settings.he_eosin_tolerance_deg, closest
    # 9.4 degrees was the worst of the six. The bound is 15, so there is real headroom
    # rather than a threshold tuned to the data it was measured on.
    assert max(closest) < 10.0, closest


# --- what step 5's own screen says about an eosin arm -------------------------


def test_an_eosin_arm_is_a_fault_on_dab_and_expected_on_he() -> None:
    """The same measurement, two opposite readings, and the note has to pick one.

    Step 5's prose used to say - unconditionally - that an arm nearest eosin "points
    somewhere no dye on this section absorbs". On an immunostained section that is
    right. On an H&E section it is exactly backwards: that arm *is* the second dye, and
    it is the evidence step 7 reads to offer its full-colour option. The two cases are
    cleanly separable on this project's panel, so saying the wrong one was a choice
    rather than an unavoidable ambiguity.
    """
    from app.services.density_service import _arm_note

    he_arms = {"haematoxylin", "eosin"}
    dab_arms = {"haematoxylin", "dab"}

    on_he = _arm_note(he_arms, dab_arms, is_he=True)
    assert "expected" in on_he
    assert "fault" in on_he  # as in "rather than a fault"
    assert "no dye on this section absorbs" not in on_he

    on_dab = _arm_note(he_arms, dab_arms, is_he=False)
    assert "no dye on this section absorbs" in on_dab
    assert "expected finding" not in on_dab


def test_the_expected_pair_is_reported_as_the_claim_it_is() -> None:
    """Both arms landing on the two dyes the assay uses is the interesting case, and
    the note says why: nothing in step 5 was told what either looks like."""
    from app.services.density_service import _arm_note

    dab_arms = {"haematoxylin", "dab"}
    note = _arm_note(dab_arms, dab_arms, is_he=False)
    assert "the two stains this assay uses" in note
    assert "told what either looks like" in note


def test_the_he_note_does_not_claim_step_6_handles_h_and_e() -> None:
    """There is still no haematoxylin-eosin basis in this codebase.

    Step 6 unmixes an H&E section on the H-DAB basis, so its second channel is eosin
    read through a DAB vector. The note has to explain that rather than imply step 6 has
    a mode for this - and it is the reason the full-colour branch ignores that channel
    and hands the model the photograph.
    """
    from app.services.density_service import _arm_note

    note = _arm_note({"haematoxylin", "eosin"}, {"haematoxylin", "dab"}, is_he=True)
    assert "haematoxylin-DAB basis" in note
    assert "photograph" in note
