"""Moving geometry between two slides of the same block.

Never in the per-slide path: registration only means anything when there are
two slides, so nothing in steps 1-9 imports this. Step 10 does.
"""

from .exceptions import RegistrationError, RegistrationRefused, RegistrationUnavailable

__all__ = ["RegistrationError", "RegistrationRefused", "RegistrationUnavailable"]
