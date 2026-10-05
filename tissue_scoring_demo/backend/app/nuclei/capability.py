"""Can this machine segment nuclei, and if not, exactly why.

Reported rather than discovered on first use, the same way step 10 reports
whether VALIS is installed: a viewer should be told the model is missing before
they start a run, not after it. The reason is carried as text because the three
ways this fails need three different answers - install torch, run setup, or stop
and look at a parity failure.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.nuclei import model as nuclei_model


@dataclass(frozen=True)
class NucleiCapability:
    available: bool
    reason: str | None
    models_dir: str
    model_name: str | None = None
    model_version: str | None = None
    licence: str | None = None
    mpp: float | None = None


def capability() -> NucleiCapability:
    """Load the model if it is there, and turn any failure into a sentence."""
    directory = str(nuclei_model.models_dir())
    try:
        loaded = nuclei_model.load()
    except nuclei_model.ModelUnavailable as exc:
        return NucleiCapability(available=False, reason=str(exc), models_dir=directory)

    spec = loaded.manifest
    return NucleiCapability(
        available=True,
        reason=None,
        models_dir=directory,
        model_name=str(spec.get("name")),
        model_version=str(spec.get("version")),
        licence=str(spec.get("licence")),
        mpp=loaded.mpp,
    )


__all__ = ["NucleiCapability", "capability"]
