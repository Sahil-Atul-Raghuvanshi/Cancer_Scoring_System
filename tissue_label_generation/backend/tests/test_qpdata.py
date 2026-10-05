"""`qpdata.py` reads QuPath's Java-serialised annotation files.

**Kept when the web app was removed, and this file is why.** It is the only reader in
the project for a pathologist-drawn annotation - BRACS's own `.qpdata` boxes, and the
OncoStem contour on `CAN_00270_26_H&E` - so it is the only path to a human ground truth
for the invasive-versus-in-situ question. A silent mis-read here would turn the one real
evaluation into a fiction, and an EMPTY ground truth would score every model as perfect.
Every test below guards that direction.

Recovered from `test_sixslides.py`, which went with the six-slide screen.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_a_non_qpdata_file_is_refused(tmp_path):
    from bracs_app import qpdata

    path = tmp_path / "not.qpdata"
    path.write_bytes(b"GeoJSON{}")
    with pytest.raises(ValueError, match="not a Java-serialised stream"):
        qpdata.parse(path)


def test_polygon_annotations_are_refused_rather_than_half_read(tmp_path):
    """A polygon file must fail loudly: a partial ground truth is worse than none."""
    from bracs_app import qpdata

    path = tmp_path / "poly.qpdata"
    path.write_bytes(b"\xac\xed\x00\x05" + b"junk qupath.lib.roi.PolygonROI more junk")
    with pytest.raises(ValueError, match="PolygonROI"):
        qpdata.parse(path)


def test_a_file_with_no_rectangles_is_refused(tmp_path):
    from bracs_app import qpdata

    path = tmp_path / "empty.qpdata"
    path.write_bytes(b"\xac\xed\x00\x05" + b"nothing useful here")
    with pytest.raises(ValueError, match="no RectangleROI"):
        qpdata.parse(path)


def test_the_real_bracs_annotations_parse_to_the_expected_lesions():
    """The actual file on disk, if it is there. Skipped rather than faked when absent."""
    from bracs_app import config, qpdata

    path = (config.BRACS_WSI_DIR / "annotations" / "test" / "test" / "Group_MT"
            / "Type_IC" / "BRACS_1284.qpdata")
    if not path.exists():
        pytest.skip("BRACS_1284.qpdata is not on this machine")

    found = qpdata.parse(path)
    assert len(found) == 16
    classes = {a.path_class for a in found}
    assert classes == {"DCIS-sure", "Malignant-sure", "UDH-sure"}
    # Every box inside a 22.6 x 20.5 mm slide, and none inverted.
    for a in found:
        assert 0 <= a.x < a.x2 < 90_000 and 0 <= a.y < a.y2 < 82_000
        assert a.width > 0 and a.height > 0
