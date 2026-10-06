"""Write the two stain renders Cellpose is tried on, from the backend's own code (option 5).

    python stage_renders.py

The Cellpose venv cannot import the backend, so the renders are made here and saved:

  h       production's haematoxylin-only render (what step 13 feeds InstanSeg today)
  odsum   the optical-density-sum render (H + DAB), the same one `instanseg_odsum` reads

for every benchmark IHC field and every held-out labelled image, as
`renders/<render>/<unit>/<side>_<name>.png`. One white point per unit, as in approach 1.
"""

from __future__ import annotations

import sys

import numpy as np

import common
import stage_a1

LOG = common.LOGS / "renders.log"


def units():
    for case, marker in common.PAIRS:
        unit = common.pair_id(case, marker)
        meta = common.read_json(common.FIELDS / unit / "meta.json") or {"ihc": []}
        yield unit, "ihc", [(e["name"], common.FIELDS / unit / "ihc" / f"{e['name']}.png") for e in meta["ihc"]]
    for unit, (side, _) in common.LABELLED.items():
        yield unit, side, [(n, p) for n, p, _ in common.labelled_items(unit)]


def main() -> int:
    from PIL import Image

    common.ensure_dirs()
    checkpoint = common.Checkpoint("renders")
    for unit, side, items in units():
        if checkpoint.done(unit):
            continue
        if not items:
            common.say(f"{unit}: nothing to render yet", LOG)
            return 2  # its labelled set is not prepared yet: wait, do not fail
        images = {name: np.asarray(Image.open(path).convert("RGB")) for name, path in items}
        white = stage_a1._white(list(images.values()))
        for name, rgb in images.items():
            outputs = {"h": lambda: stage_a1._production_render(rgb, white),
                       "odsum": lambda: stage_a1._render(stage_a1._od_sum(rgb, white))}
            for render, make in outputs.items():
                path = common.RENDERS / render / unit / f"{side}_{name}.png"
                if path.exists():
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.asarray(make(), dtype=np.uint8)).save(path)
        checkpoint.mark(unit, images=len(images), white=[round(float(v), 1) for v in white])
        common.heartbeat("renders", unit)
        common.say(f"{unit}: {len(images)} images", LOG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
