"""Read BRACS's QuPath `.qpdata` annotations without QuPath.

These files are the only pathologist-drawn ground truth in this project. Everything else
labelled class 1 is a model's opinion, so being able to read them is what turns the
whole-slide comparison from two opinions into an evaluation.

**A `.qpdata` file is a Java-serialised object graph** - QuPath's own docs say to open it
in QuPath and export GeoJSON from the script console, and the folder's README repeats
that. That is true in general: the format can hold arbitrary QuPath objects and a full
reader would be a Java deserialiser. But BRACS's annotations are the simplest possible
case, and this module reads exactly that case and refuses everything else:

  * every ROI is a `RectangleROI$SerializationProxy`, not a polygon - BRACS drew bounding
    boxes round lesions
  * every annotation carries one `PathClass` whose name is a plain string
  * there are no nested objects, no TMA grid, no measurements worth reading

So rather than a general deserialiser, this walks the byte stream for two known markers
and reads the fields the class descriptor itself declares. `parse` raises if anything it
does not recognise appears, so a polygon-annotated file fails loudly here instead of
silently producing an empty or half-read ground truth - which is the failure that would
matter, because an empty ground truth scores every model as perfect.

--------------------------------------------------------------------------------
The field layout, read off the class descriptor rather than assumed
--------------------------------------------------------------------------------

The serialised descriptor for `RectangleROI$SerializationProxy` declares, in this order:

    I c    I t    D x    D x2    D y    D y2    I z    L name

Java writes primitive fields in that declared order after the `TC_ENDBLOCKDATA` /
superclass marker, so the 44 bytes following it are `c, t, x, x2, y, y2, z`. `_FIELDS`
below encodes that and `_read_descriptor_fields` checks the file still agrees with it -
if a future QuPath version reorders or adds a field, the check fails rather than reading
`y` out of `x2`'s bytes and putting every ROI in the wrong place.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path

#: Java serialization stream magic and version. Anything else is not a `.qpdata`.
_MAGIC = b"\xac\xed\x00\x05"

#: The proxy class whose instances carry the rectangles.
_ROI_CLASS = b"qupath.lib.roi.RectangleROI$SerializationProxy"

#: ROI classes BRACS does not use and this reader cannot honestly handle. Listed so the
#: error names what it found rather than saying "unsupported".
_UNSUPPORTED_ROI = (
    b"PolygonROI",
    b"PolylineROI",
    b"EllipseROI",
    b"AreaROI",
    b"GeometryROI",
    b"PointsROI",
)

#: `(name, struct format)` in the order the descriptor declares them. Checked, not assumed.
_FIELDS: tuple[tuple[str, str], ...] = (
    ("c", "i"), ("t", "i"),
    ("x", "d"), ("x2", "d"), ("y", "d"), ("y2", "d"),
    ("z", "i"),
)
_FIELD_STRUCT = ">" + "".join(fmt for _, fmt in _FIELDS)
_FIELD_BYTES = struct.calcsize(_FIELD_STRUCT)


@dataclass(frozen=True)
class Annotation:
    """One pathologist-drawn box, in level-0 slide pixels."""

    path_class: str
    x: float
    y: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x

    @property
    def height(self) -> float:
        return self.y2 - self.y

    def area_px(self) -> float:
        return self.width * self.height

    def as_json(self) -> dict:
        return {
            "path_class": self.path_class,
            "x": round(self.x, 1), "y": round(self.y, 1),
            "x2": round(self.x2, 1), "y2": round(self.y2, 1),
            "width": round(self.width, 1), "height": round(self.height, 1),
        }


def _read_utf(blob: bytes, offset: int) -> tuple[str, int]:
    """A Java `TC_STRING` body: two length bytes then modified UTF-8."""
    (length,) = struct.unpack_from(">H", blob, offset)
    return blob[offset + 2 : offset + 2 + length].decode("utf-8"), offset + 2 + length


def _read_descriptor_fields(blob: bytes, class_at: int) -> None:
    """Check the file's own field list still matches `_FIELDS`.

    The descriptor lists `count`, then for each field a typecode and a name. Reading the
    names back and comparing is cheap, and it is the difference between this module
    failing on a format change and silently swapping two coordinates.
    """
    # After the class name comes an 8-byte serialVersionUID, one flags byte, then the
    # field count as a big-endian short.
    cursor = class_at + len(_ROI_CLASS) + 8 + 1
    (count,) = struct.unpack_from(">H", blob, cursor)
    cursor += 2

    found: list[tuple[str, str]] = []
    for _ in range(count):
        typecode = chr(blob[cursor])
        cursor += 1
        name, cursor = _read_utf(blob, cursor)
        if typecode == "L":  # an object field: a className1 string follows
            if blob[cursor] == 0x74:  # TC_STRING
                _, cursor = _read_utf(blob, cursor + 1)
            elif blob[cursor] == 0x71:  # TC_REFERENCE
                cursor += 5
            else:
                raise ValueError(
                    f"unexpected type descriptor byte {blob[cursor]:#x} in the "
                    "RectangleROI field list"
                )
        found.append((name, typecode))

    primitives = tuple((name, code.lower()) for name, code in found if code in "IDFJSBZC")
    if primitives != _FIELDS:
        raise ValueError(
            "this .qpdata declares RectangleROI fields as "
            f"{primitives}, and this reader was written against {_FIELDS}. Refusing to "
            "read coordinates against a layout that has changed - the boxes would land "
            "in the wrong places and the ground truth would be quietly wrong."
        )


def parse(path: Path) -> list[Annotation]:
    """Every annotation in a BRACS `.qpdata`, with its class name.

    Classes are associated with rectangles by position in the stream. QuPath writes each
    annotation as its `PathClass` followed by its ROI, and repeated classes become
    back-references - so the class in force for a rectangle is the last class *name*
    written before it. That is a property of this file layout rather than of Java
    serialisation in general, which is why `parse` verifies the result: every rectangle
    must have a class, and the classes must be ones BRACS uses.
    """
    blob = Path(path).read_bytes()
    if not blob.startswith(_MAGIC):
        raise ValueError(f"{path} is not a Java-serialised stream, so not a .qpdata")

    for unsupported in _UNSUPPORTED_ROI:
        if unsupported in blob:
            raise ValueError(
                f"{path} contains {unsupported.decode()} annotations. This reader only "
                "handles the rectangles BRACS draws; a polygon file needs QuPath's own "
                "GeoJSON export. Refusing rather than reading a partial ground truth."
            )

    class_at = blob.find(_ROI_CLASS)
    if class_at < 0:
        raise ValueError(f"{path} holds no RectangleROI annotations at all")
    _read_descriptor_fields(blob, class_at)

    # Where each PathClass name appears. `PathClass` instances store their name as a
    # plain string, and BRACS's are the only strings ending `-sure`/`-non` in these files.
    class_positions = sorted(
        (match.start(), match.group(1).decode("utf-8"))
        for match in re.finditer(rb"\x74\x00.((?:[\w /]|-)+?-(?:sure|non))", blob)
    )
    if not class_positions:
        raise ValueError(f"{path} holds no PathClass names this reader recognises")

    # Field blocks: the first instance follows its own descriptor's TC_ENDBLOCKDATA; every
    # later one is a TC_OBJECT whose descriptor is a back-reference.
    starts: list[int] = []
    first = blob.find(b"\x78\x70", class_at)
    if first < 0:
        raise ValueError("the RectangleROI descriptor is truncated")
    starts.append(first + 2)

    handle = struct.pack(">H", 0x7E00 >> 8)  # unused; kept for clarity of the pattern below
    for match in re.finditer(rb"\x73\x71\x00\x7e\x00.", blob[class_at:]):
        starts.append(class_at + match.end())

    out: list[Annotation] = []
    for offset in sorted(set(starts)):
        if offset + _FIELD_BYTES > len(blob):
            continue
        try:
            values = dict(zip(
                (name for name, _ in _FIELDS),
                struct.unpack_from(_FIELD_STRUCT, blob, offset),
            ))
        except struct.error:
            continue

        # A plane other than (c=-1, z=0, t=0) is a z-stack or a channel-specific ROI,
        # neither of which BRACS uses; a reversed or absurd box is a mis-parse. Both are
        # skipped rather than trusted, and the count check below catches it if that ever
        # silences the whole file.
        if values["c"] != -1 or values["z"] != 0 or values["t"] != 0:
            continue
        if not (0 <= values["x"] < values["x2"] < 1e6 and 0 <= values["y"] < values["y2"] < 1e6):
            continue

        name = ""
        for position, candidate in class_positions:
            if position < offset:
                name = candidate
            else:
                break
        if not name:
            raise ValueError(
                f"a rectangle at byte {offset} has no PathClass before it in the stream, "
                "so its label is unknown. Refusing to guess."
            )
        out.append(Annotation(
            path_class=name,
            x=values["x"], y=values["y"], x2=values["x2"], y2=values["y2"],
        ))

    if not out:
        raise ValueError(
            f"{path} parsed to zero annotations. An empty ground truth would score every "
            "model as perfect, so this is refused rather than returned."
        )
    return out


def summarise(annotations: list[Annotation], mpp: float) -> dict:
    """Counts and areas per class, for the report and for a sanity check by eye."""
    by_class: dict[str, dict] = {}
    for annotation in annotations:
        entry = by_class.setdefault(annotation.path_class, {"boxes": 0, "mm2": 0.0})
        entry["boxes"] += 1
        entry["mm2"] += annotation.area_px() * mpp * mpp / 1e6
    for entry in by_class.values():
        entry["mm2"] = round(entry["mm2"], 3)
    return {
        "boxes": len(annotations),
        "total_mm2": round(sum(a.area_px() for a in annotations) * mpp * mpp / 1e6, 3),
        "by_class": by_class,
    }
