"""InstanSeg, loaded and run as plain TorchScript.

Step 11 needs an instance segmenter. The guide's ranking picks InstanSeg and
the reasons hold up on inspection: Apache-2.0 **and** an Apache/CC-BY training
set, two of whose five datasets are IHC rather than H&E, which is the actual
evidence behind "it already handles IHC".

What is not obvious is that it can run here at all. `instanseg-torch` declares
Python 3.9-3.11 and publishes no 3.13 wheel; this backend is 3.13 on numpy 2.x.
That is the same collision that pushed VALIS into its own interpreter, and it
would have pushed this there too - except that the published model is a
TorchScript archive with its post-processing compiled in, so `torch.jit.load`
runs it against the torch already installed for step 2. No new dependency, no
second venv, and no `instanseg` import anywhere in this package.

`model.py` holds the loader and the one preprocessing step the model cannot be
run without; `capability.py` reports whether any of it is available, so the UI
can say "not installed" before somebody waits.
"""

from app.nuclei.capability import NucleiCapability, capability
from app.nuclei.model import (
    ModelUnavailable,
    ParityFailure,
    manifest,
    scale_range,
    segment_array,
)

__all__ = [
    "ModelUnavailable",
    "NucleiCapability",
    "ParityFailure",
    "capability",
    "manifest",
    "scale_range",
    "segment_array",
]
