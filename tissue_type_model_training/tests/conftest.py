"""Put `src/` on the path, so the tests import the same modules the notebooks do.

The modules under `src/` are flat and import each other flatly (`import bcss`), which
is what lets a notebook use them without a package install. The cost is this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
