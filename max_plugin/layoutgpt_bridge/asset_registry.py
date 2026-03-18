"""
asset_registry.py
-----------------
Explicit, user-curated registry of rooms and furniture assets.

Instead of scanning the scene for naming-convention prefixes, the user
selects objects in the viewport and clicks "Add Selected Room" / "Add
Selected Furniture".  This registry stores those choices and drives
the rest of the pipeline.

Public surface
--------------
    RoomEntry       – dataclass: node name, room type, pymxs node ref
    FurnitureEntry  – dataclass: node name, layoutgpt category, room
                      assignment, max_instances (duplicate count), node ref
    AssetRegistry   – holds lists of both; serialisable to dict
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from .tag_mapper import ALL_LAYOUTGPT_CATEGORIES, normalize_tag, room_type_from_name


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RoomEntry:
    node_name: str              # Max object name, e.g. "Bedroom Boundary 01"
    room_type: str              # LayoutGPT key: "bedroom" | "livingroom"
    node: object = field(default=None, repr=False)   # pymxs node ref (runtime only)

    @property
    def display_label(self) -> str:
        return f"[{self.room_type}]  {self.node_name}"


@dataclass
class FurnitureEntry:
    node_name: str              # Max object name, e.g. "Sofa_Corner_01"
    category: str               # LayoutGPT 3D-FUTURE category
    room_assignment: str        # "all"  or  a RoomEntry.node_name
    max_instances: int          # how many copies LayoutGPT may place (1 = no duplicates)
    node: object = field(default=None, repr=False)   # pymxs node ref (runtime only)

    @property
    def display_label(self) -> str:
        dupe_tag = f"  ×{self.max_instances}" if self.max_instances > 1 else ""
        room_tag = f"  [{self.room_assignment}]" if self.room_assignment != "all" else ""
        return f"{self.category}{dupe_tag}  —  {self.node_name}{room_tag}"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class AssetRegistry:
    """
    User-curated collection of rooms and furniture assets.

    Instances of this class are held as globals inside the MAXScript UI and
    are serialised to/from a JSON string (stored in a Max custom attribute
    or written to a sidecar file) for persistence between Max sessions.
    """

    def __init__(self) -> None:
        self.rooms:     list[RoomEntry]     = []
        self.furniture: list[FurnitureEntry] = []

    # ------------------------------------------------------------------
    # Room management
    # ------------------------------------------------------------------

    def add_room(
        self,
        node_name: str,
        room_type: str | None = None,
        node=None,
    ) -> RoomEntry:
        """
        Add (or update) a room entry.
        room_type is auto-detected from node_name if not provided.
        """
        resolved_type = room_type or room_type_from_name(node_name)
        # Normalise to keys LayoutGPT understands
        if "dining" in resolved_type or "living" in resolved_type:
            resolved_type = "livingroom"
        elif resolved_type != "bedroom":
            resolved_type = "bedroom"   # safe fallback

        for existing in self.rooms:
            if existing.node_name == node_name:
                existing.room_type = resolved_type
                existing.node = node
                return existing

        entry = RoomEntry(node_name=node_name, room_type=resolved_type, node=node)
        self.rooms.append(entry)
        return entry

    def remove_room(self, node_name: str) -> None:
        self.rooms = [r for r in self.rooms if r.node_name != node_name]

    def room_names(self) -> list[str]:
        return [r.node_name for r in self.rooms]

    # ------------------------------------------------------------------
    # Furniture management
    # ------------------------------------------------------------------

    def add_furniture(
        self,
        node_name: str,
        category: str | None = None,
        room_assignment: str = "all",
        max_instances: int = 1,
        node=None,
    ) -> FurnitureEntry:
        """
        Add (or update) a furniture entry.
        category is auto-detected from node_name if not provided.
        Falls back to the first LayoutGPT category if detection fails.
        """
        resolved_cat = (
            category
            or normalize_tag(node_name)
            or ALL_LAYOUTGPT_CATEGORIES[0]
        )

        for existing in self.furniture:
            if existing.node_name == node_name:
                existing.category        = resolved_cat
                existing.room_assignment = room_assignment
                existing.max_instances   = max_instances
                existing.node            = node
                return existing

        entry = FurnitureEntry(
            node_name       = node_name,
            category        = resolved_cat,
            room_assignment = room_assignment,
            max_instances   = max_instances,
            node            = node,
        )
        self.furniture.append(entry)
        return entry

    def remove_furniture(self, node_name: str) -> None:
        self.furniture = [f for f in self.furniture if f.node_name != node_name]

    def update_furniture(
        self,
        node_name: str,
        category: str | None = None,
        room_assignment: str | None = None,
        max_instances: int | None = None,
    ) -> bool:
        """Update fields on an existing furniture entry. Returns True if found."""
        for entry in self.furniture:
            if entry.node_name == node_name:
                if category is not None:
                    entry.category = category
                if room_assignment is not None:
                    entry.room_assignment = room_assignment
                if max_instances is not None:
                    entry.max_instances = max_instances
                return True
        return False

    # ------------------------------------------------------------------
    # Library extraction  (used by placement pipeline)
    # ------------------------------------------------------------------

    def furniture_for_room(self, room: RoomEntry) -> list[FurnitureEntry]:
        """Furniture assigned to *room* or to 'all'."""
        return [
            f for f in self.furniture
            if f.room_assignment == "all" or f.room_assignment == room.node_name
        ]

    def build_library(self, room: RoomEntry) -> dict[str, list]:
        """
        Build a {category: [FurnitureEntry, ...]} dict for the given room.

        max_instances is honoured by repeating the entry — the placement engine
        sees multiple asset choices for that category, allowing duplicates.
        """
        library: dict[str, list] = {}
        for f in self.furniture_for_room(room):
            library.setdefault(f.category, [])
            for _ in range(f.max_instances):
                library[f.category].append(f)
        return library

    def available_categories(self, room: RoomEntry) -> list[str]:
        """Sorted unique category names available for a room."""
        return sorted({f.category for f in self.furniture_for_room(room)})

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps({
            "rooms": [
                {"node_name": r.node_name, "room_type": r.room_type}
                for r in self.rooms
            ],
            "furniture": [
                {
                    "node_name"       : f.node_name,
                    "category"        : f.category,
                    "room_assignment" : f.room_assignment,
                    "max_instances"   : f.max_instances,
                }
                for f in self.furniture
            ],
        }, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "AssetRegistry":
        data = json.loads(json_str)
        registry = cls()
        for r in data.get("rooms", []):
            registry.rooms.append(RoomEntry(
                node_name=r["node_name"],
                room_type=r["room_type"],
            ))
        for f in data.get("furniture", []):
            registry.furniture.append(FurnitureEntry(
                node_name       = f["node_name"],
                category        = f["category"],
                room_assignment = f.get("room_assignment", "all"),
                max_instances   = f.get("max_instances", 1),
            ))
        return registry
