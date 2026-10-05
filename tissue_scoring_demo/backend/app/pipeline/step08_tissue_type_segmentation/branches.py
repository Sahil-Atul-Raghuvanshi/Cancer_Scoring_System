"""The three things step 7 can offer, and which input contract each one means.

Step 7's one consequential choice used to be the field of view. It now has two: the
field of view, and **what the model is shown**. The second is the interesting one,
because the two branches answer different questions.

`h_channel` throws the colour away and keeps Ruifrok's haematoxylin channel. That is
the one dye present on all six stains of a case - the H&E and all five IHC - so the
input contract is satisfiable on every slide of a case. It is the branch every
published checkpoint until now belongs to.

**It was designed to serve the whole panel, and measured on real slides it does not.**
Run against this project's five immunostained markers it returns **0% invasive** on
every one of them, while the same head returns 12.7% on the same case's H&E. The
confidence does not drop to admit it: it sits at 0.96-0.98 throughout. So the failure
is not "worse on IHC", it is a confident wrong answer that looks exactly like a right
one, which is the reason step 10 exists - the region is found on the H&E and
registered across, rather than found on each slide directly as
`docs/guides/demo-pipeline-guide.md` originally recommended. Satisfying a model's input
contract is not the same as being in its training distribution, and this is the
sentence that used to confuse the two.

`he` hands the network the colour photograph. On an H&E section the second dye is half
the evidence: eosin is what makes a duct wall and a collagen band look different, and
the H channel cannot see the difference. It costs generality - the branch is meaningless
on a DAB section, where "eosin" is a vector pointing at nothing - which is why it is
gated on step 5 actually finding two dyes rather than offered unconditionally.

`beetle` hands the slide to a published network from another research group instead of
one of ours, and it answers **per pixel** where the other two answer per window. It is
now built - `beetle.py` and `pixels.py` - and it is the one branch that resolves to no
checkpoint of ours at all: there is no head, no manifest and no input contract to
verify, because what it runs is a release with its own published preprocessing rule. It
is available at every field of view the moment `models/beetle/model.zip` is on disk,
rather than as heads are trained, and it reports **five** classes rather than three.

**A checkpoint declares its branch through its input contract, never through its
filename.** `input.descriptor`'s `channel` key is the fact; `branch_of_channel` is the
one-line map. A manifest with no `channel` at all is `h_channel`, which is what makes
every legacy checkpoint keep working without being re-published.

**BEETLE is deliberately absent from `BRANCH_CHANNEL`, and that is not an omission.**
That table maps a *manifest's* declared channel onto a branch, and it exists so
`model.Candidate.branch` can sort published ResNet checkpoints into the two branches
that have them. BEETLE publishes no manifest here, so no candidate can ever declare it -
which is exactly right: a scan of `models/tissue_type/` must never return BEETLE, and
`tiling_service` resolves it through `beetle.available()` instead of through
`model.discover()`. Adding a channel for it would invite the geometry scan to match it
and hand `model.load_pinned` a name it cannot load.

This module lives in step 8's package rather than step 7's because the existing
dependency runs step 7 -> step 8 (`tiling_service` already imports `model` from here);
putting it the other way would make a cycle. It imports `enum` and `input`, so it stays
torch-free and the capability listing can read it.
"""

from __future__ import annotations

from enum import Enum

from app.pipeline.step08_tissue_type_segmentation import input as model_input


class ModelBranch(str, Enum):
    """What step 7 offers and step 8 runs. A `str` enum so it is its own wire value."""

    H_CHANNEL = "h_channel"
    HE = "he"
    BEETLE = "beetle"


#: What a caller naming nothing gets, and what a manifest recording no channel is.
#: Every checkpoint published before the H&E branch existed belongs here, so this
#: default is what keeps them resolvable.
DEFAULT_BRANCH: ModelBranch = ModelBranch.H_CHANNEL

#: The branches step 7 can commit to and step 8 can run. All three, now that BEETLE is
#: built - but note that only the first two are *published checkpoints of ours*, which
#: is the distinction `SERVED_HEADS` keeps below.
SERVED_BRANCHES: tuple[ModelBranch, ...] = (
    ModelBranch.H_CHANNEL,
    ModelBranch.HE,
    ModelBranch.BEETLE,
)

#: The branches whose model is a checkpoint in `models/tissue_type/`, and therefore the
#: only ones `model.discover()` and `tiling_service.model_for` should ever be asked
#: about. BEETLE is absent because it is a downloaded release with no manifest of ours;
#: asking a geometry scan over our own checkpoints for it would either return nothing
#: (making a working branch look unavailable) or, worse, match one of ours by accident.
SERVED_HEADS: tuple[ModelBranch, ...] = (ModelBranch.H_CHANNEL, ModelBranch.HE)

#: The branch -> input contract map. The values are `input.py`'s own constants rather
#: than string literals, so a renamed channel is a NameError here and not a branch that
#: silently matches nothing.
BRANCH_CHANNEL: dict[ModelBranch, str] = {
    ModelBranch.H_CHANNEL: model_input.CHANNEL_HAEMATOXYLIN,
    ModelBranch.HE: model_input.CHANNEL_RGB_HE,
}

_CHANNEL_BRANCH: dict[str, ModelBranch] = {
    channel: branch for branch, channel in BRANCH_CHANNEL.items()
}


def branch_of_channel(channel: str | None) -> ModelBranch:
    """Which branch a manifest's `input.channel` means.

    `None` - a manifest predating the branch idea - is `h_channel`, because that is
    what every such checkpoint was fitted as. An unrecognised channel is *also*
    `h_channel` rather than an exception, because `_describe` promises never to raise:
    a checkpoint whose channel this code does not know still fails
    `_verify_input_contract` at load, which is the right place for that refusal and
    gives a far better message than a `KeyError` inside a directory listing.
    """
    if channel is None:
        return DEFAULT_BRANCH
    return _CHANNEL_BRANCH.get(str(channel), DEFAULT_BRANCH)


def parse_branch(value: str | None) -> ModelBranch:
    """A wire string to a branch, refusing anything else.

    Used at the API boundary, where an unknown branch must be a 422 rather than a
    quiet fall back to the default - the same reason `tiling_service.field_of_view`
    refuses 300 um instead of snapping it to 224.
    """
    if value is None:
        return DEFAULT_BRANCH
    try:
        return ModelBranch(str(value))
    except ValueError:
        offered = ", ".join(branch.value for branch in ModelBranch)
        raise ValueError(
            f"{value!r} is not a branch step 7 offers. The branches are: {offered}."
        ) from None


__all__ = [
    "BRANCH_CHANNEL",
    "DEFAULT_BRANCH",
    "SERVED_BRANCHES",
    "SERVED_HEADS",
    "ModelBranch",
    "branch_of_channel",
    "parse_branch",
]
