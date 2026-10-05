"""Step 10 - Review and select the candidate regions.

Implemented, and it runs no model, opens no checkpoint and measures nothing new. What
it adds to the pipeline is a *decision point*: step 9 found 13 to 20 patches of invasive
tile, and step 11 is about to spend BEETLE on them, so this is where a person says which
ones are worth it.

`candidates.py` is the load-bearing module and the one to read first. Its one real idea
is that a candidate's id is its area rank - `ROI-001` is the largest invasive patch on
the slide - which makes the id a pure function of step 8's class map and therefore
reproducible, and which is also why a selection is keyed to the class map it was made
on. A selection carried onto a class map rebuilt at a different threshold would be a
set of ids pointing at different tissue.

Nothing here imports torch and nothing here writes a mask. The output is a list, a
picture per entry, and a set of ticked ids.
"""

from .candidates import (
    Bbox,
    Candidate,
    CandidateSet,
    build,
    default_selection,
    roi_id,
)
from .overlay import card_png

__all__ = [
    "Bbox",
    "Candidate",
    "CandidateSet",
    "build",
    "card_png",
    "default_selection",
    "roi_id",
]
