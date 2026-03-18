"""
scene_bridge.py
---------------
pymxs-based scene reader for the LayoutGPT bridge.

Reads the 3ds Max scene (via the pymxs Python API available inside 3ds Max
2024-2026) and extracts:
  - Room boundary splines (closed splines tagged ROOM_*)
  - Furniture prototype assets (tagged FURN_*)

All geometry queries are expressed in scene units (typically millimetres or
centimetres in a CRE workflow; the placement engine converts to LayoutGPT's
pixel space).

Public surface
--------------
    RoomInfo           dataclass: spline name, room_type, bounding box, area
    FurnitureAsset     dataclass: object name, layoutgpt_category, bbox dims
    read_selected_rooms()   → list[RoomInfo]
    read_furniture_library() → dict[str, list[FurnitureAsset]]
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .tag_mapper import normalize_tag, room_type_from_name

# Guard: pymxs is only available inside 3ds Max.  We import lazily so that
# the module can still be imported (and tested) outside Max.
try:
    import pymxs
    rt = pymxs.runtime          # alias: rt.Point3, rt.classOf, etc.
    _IN_MAX = True
except ImportError:
    rt = None
    _IN_MAX = False


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    """Axis-aligned bounding box in scene world units."""
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float

    @property
    def length_x(self) -> float:
        return self.max_x - self.min_x

    @property
    def length_y(self) -> float:
        return self.max_y - self.min_y

    @property
    def length_z(self) -> float:
        return self.max_z - self.min_z

    @property
    def center(self) -> tuple[float, float, float]:
        return (
            (self.min_x + self.max_x) / 2,
            (self.min_y + self.max_y) / 2,
            (self.min_z + self.max_z) / 2,
        )


@dataclass
class RoomInfo:
    """Information about one room boundary spline."""
    name: str                       # Max object name, e.g. "ROOM_bedroom_01"
    room_type: str                  # LayoutGPT room type, e.g. "bedroom"
    bbox: BoundingBox               # bounding box in scene units
    spline_obj: object = field(default=None, repr=False)  # pymxs node ref


@dataclass
class FurnitureAsset:
    """Information about one furniture prototype in the scene."""
    name: str                       # Max object name, e.g. "FURN_sofa_corner_01"
    layoutgpt_category: str         # 3D-FUTURE category, e.g. "l_shaped_sofa"
    bbox: BoundingBox               # asset bounding box (pivot at base centre)
    node: object = field(default=None, repr=False)  # pymxs node ref


# ---------------------------------------------------------------------------
# Low-level pymxs helpers
# ---------------------------------------------------------------------------

def _node_world_bbox(node) -> BoundingBox:
    """Return the world-space AABB of a node using its mesh/shape bounding box."""
    # Force an update and grab the bbox from the node's GetMesh or from the
    # bounding box of all vertices (works for both meshes and splines).
    with pymxs.attime(rt.currentTime):
        mn = rt.nodeGetBoundingBox(node, rt.Matrix3(1))  # identity matrix → world space
        # mn is a two-element MaxScript array: [min Point3, max Point3]
        min_pt = mn[0]
        max_pt = mn[1]
    return BoundingBox(
        min_x=float(min_pt.x), min_y=float(min_pt.y), min_z=float(min_pt.z),
        max_x=float(max_pt.x), max_y=float(max_pt.y), max_z=float(max_pt.z),
    )


def _is_closed_spline(node) -> bool:
    """Return True if the node is a closed spline shape."""
    if not rt.isKindOf(node, rt.SplineShape) and not rt.isKindOf(node, rt.shape):
        return False
    # Check the first sub-spline for closure
    try:
        return bool(rt.isClosed(node.shape, 1))
    except Exception:
        return False


def _match_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    n = name.lower()
    return any(n.startswith(p) for p in prefixes)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

ROOM_PREFIXES  = ("room_", "boundary_", "area_")
FURN_PREFIXES  = ("furn_", "furniture_")


def read_selected_rooms(selection=None) -> list[RoomInfo]:
    """
    Return a list of RoomInfo for every selected closed spline whose name
    matches ROOM_PREFIXES.  Pass *selection* to override (useful for testing).

    If nothing is selected, falls back to scanning the whole scene.
    """
    if not _IN_MAX:
        raise RuntimeError("scene_bridge.read_selected_rooms() requires pymxs (3ds Max).")

    nodes = selection if selection is not None else list(rt.selection)
    if not nodes:
        # Nothing selected – try scene-wide scan
        nodes = [obj for obj in rt.objects]

    rooms: list[RoomInfo] = []
    for node in nodes:
        name = str(node.name)
        if not _match_prefix(name, ROOM_PREFIXES):
            continue
        if not _is_closed_spline(node):
            continue
        bbox = _node_world_bbox(node)
        room_type = room_type_from_name(name)
        rooms.append(RoomInfo(name=name, room_type=room_type, bbox=bbox, spline_obj=node))

    return rooms


def read_furniture_library(prefixes: tuple[str, ...] = FURN_PREFIXES) -> dict[str, list[FurnitureAsset]]:
    """
    Scan the entire scene for furniture prototype nodes (FURN_* prefix).

    Returns a dict keyed by LayoutGPT category name, e.g.:
        {
            "l_shaped_sofa": [FurnitureAsset(...), ...],
            "double_bed":    [FurnitureAsset(...), ...],
        }

    Multiple assets with the same category are all collected; the placement
    engine can pick the most appropriate one (first match, or random).
    """
    if not _IN_MAX:
        raise RuntimeError("scene_bridge.read_furniture_library() requires pymxs (3ds Max).")

    library: dict[str, list[FurnitureAsset]] = {}
    for node in rt.objects:
        name = str(node.name)
        if not _match_prefix(name, prefixes):
            continue
        cat = normalize_tag(name)
        if cat is None:
            continue
        bbox = _node_world_bbox(node)
        asset = FurnitureAsset(name=name, layoutgpt_category=cat, bbox=bbox, node=node)
        library.setdefault(cat, []).append(asset)

    return library


def spline_floor_area(room: RoomInfo) -> float:
    """
    Approximate floor area (in scene units²) using the spline bounding box.
    For rectangular rooms this is exact; for L-shaped rooms it over-estimates.
    """
    return room.bbox.length_x * room.bbox.length_y


# ---------------------------------------------------------------------------
# Testing / offline mock
# ---------------------------------------------------------------------------

class MockRoomInfo:
    """Minimal stand-in used when running outside 3ds Max for unit tests."""

    @staticmethod
    def make(name="ROOM_bedroom_01", length_x=5000.0, length_y=4000.0) -> RoomInfo:
        bbox = BoundingBox(0, 0, 0, length_x, length_y, 2800.0)
        return RoomInfo(name=name, room_type=room_type_from_name(name), bbox=bbox)
