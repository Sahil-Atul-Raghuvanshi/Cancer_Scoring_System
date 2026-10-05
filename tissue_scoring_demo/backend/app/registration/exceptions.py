"""What can go wrong when moving a mask between two slides, named precisely.

The distinction that matters is between "this machine cannot register" and
"this pair should not be registered". The first is an installation problem and
says so; the second is a *result* - a measured judgement that the two sections
do not correspond well enough to carry a mask across - and it must never be
reported as a mask.
"""

from __future__ import annotations


class RegistrationError(RuntimeError):
    """Base class: registration did not produce a usable transform."""


class RegistrationUnavailable(RegistrationError):
    """The isolated registration environment is missing or will not start.

    An environment problem, not a judgement about the slides. Reported
    separately so a viewer is told to install something rather than being told
    their slides do not match.
    """


class RegistrationRefused(RegistrationError):
    """The registration ran, was measured, and is not trustworthy.

    Carries every reason it failed rather than the first, because "12 matched
    features AND a tissue-area ratio of 0.2" describes a near-blank section,
    while either alone is just a number.
    """

    def __init__(self, reasons: list[str], diagnostics: dict | None = None) -> None:
        self.reasons = reasons
        self.diagnostics = diagnostics or {}
        super().__init__("; ".join(reasons))


__all__ = ["RegistrationError", "RegistrationRefused", "RegistrationUnavailable"]
