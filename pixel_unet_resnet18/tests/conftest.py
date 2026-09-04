"""Put this folder's `src/` on `sys.path`, so tests import flatly like the scripts do.

The only thing this file does. No shared fixtures: the suite builds its synthetic tiles
with module-level helpers, so a reader can see what a test is working on without following
a fixture two files away.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
