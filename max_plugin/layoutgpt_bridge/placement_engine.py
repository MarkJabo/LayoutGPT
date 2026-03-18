"""
placement_engine.py
-------------------
Translates a list of LayoutGPT Placement objects into 3ds Max scene objects
using the pymxs Python API.

Design
------
For each Placement:
  1. Find the matching furniture prototype in the scene library
     (keyed by LayoutGPT category, falls back to first available asset).
  2. Create a pymxs *instance* of the prototype (non-destructive; shares mesh).
  3. Set position, Z-rotation, and optionally scale the instance to match
     the LLM-suggested bounding box dimensions.
  4. Name the instance descriptively and group it under a container named
     after the room spline.

All placed instances are logged and the engine returns a PlacementResult
with success/failure counts so the caller can report progress.

Public surface
--------------
    PlacementEngine(furniture_library)
        .place_all(placements, room_info, formatter) → PlacementResult
    PlacementResult   dataclass
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from .layoutgpt_runner import Placement
from .scene_bridge import FurnitureAsset, RoomInfo
from .room_formatter import RoomFormatter

# Guard: pymxs only available inside 3ds Max
try:
    import pymxs
    rt = pymxs.runtime
    _IN_MAX = True
except ImportError:
    rt = None
    _IN_MAX = False

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PlacementResult:
    placed:  list[str] = field(default_factory=list)   # names of placed instances
    skipped: list[str] = field(default_factory=list)   # categories with no asset found
    oob:     list[str] = field(default_factory=list)   # out-of-bound placements removed

    @property
    def n_placed(self) -> int:
        return len(self.placed)

    def summary(self) -> str:
        return (
            f"Placed: {self.n_placed}  |  "
            f"Skipped (no asset): {len(self.skipped)}  |  "
            f"Out-of-bound: {len(self.oob)}"
        )


# ---------------------------------------------------------------------------
# Placement helpers
# ---------------------------------------------------------------------------

def _deg_to_rad(deg: float) -> float:
    return deg * math.pi / 180.0


def _max_point3(x: float, y: float, z: float):
    """Create a pymxs Point3."""
    return rt.Point3(float(x), float(y), float(z))


def _instance_node(prototype, name: str):
    """Create a named instance of *prototype* in the Max scene."""
    instance = rt.instance(prototype)
    instance.name = name
    return instance


def _set_position(node, x: float, y: float, z: float) -> None:
    """Place node centre at world position (x, y, z)."""
    node.pos = _max_point3(x, y, z)


def _set_z_rotation(node, degrees: float) -> None:
    """Apply absolute Z-axis rotation to node (in degrees, CCW from above)."""
    rot = rt.eulerAngles(0.0, 0.0, float(degrees))
    node.rotation = rt.eulerToQuat(rot)


def _scale_to_fit(node, asset: FurnitureAsset, target: dict[str, float]) -> None:
    """
    Non-uniformly scale the instance so its bounding box matches the LLM target
    dimensions (dim_x, dim_y, dim_z in scene units).

    Falls back to no scaling if the asset bbox is degenerate.
    """
    src_x = asset.bbox.length_x
    src_y = asset.bbox.length_y
    src_z = asset.bbox.length_z
    if src_x < 1e-6 or src_y < 1e-6 or src_z < 1e-6:
        return  # degenerate asset – skip scaling

    tgt_x = target["dim_x"]
    tgt_y = target["dim_y"]
    tgt_z = target["dim_z"]
    if tgt_x < 1e-6 or tgt_y < 1e-6 or tgt_z < 1e-6:
        return  # degenerate target – skip scaling

    sx = tgt_x / src_x
    sy = tgt_y / src_y
    sz = tgt_z / src_z
    node.scale = _max_point3(sx, sy, sz)


def _place_on_floor(node, room: RoomInfo) -> None:
    """
    Snap the node's base (min Z) to the room floor level, preserving XY.
    This corrects any Z offset from LayoutGPT's 'depth' coordinate.
    """
    bbox_min_z = float(rt.nodeGetBoundingBox(node, rt.Matrix3(1))[0].z)
    current_z  = float(node.pos.z)
    node.pos.z = current_z + (room.bbox.min_z - bbox_min_z)


# ---------------------------------------------------------------------------
# Unique name generator
# ---------------------------------------------------------------------------

_placement_counters: dict[str, int] = {}

def _unique_instance_name(category: str, room_name: str) -> str:
    key = f"{room_name}_{category}"
    _placement_counters[key] = _placement_counters.get(key, 0) + 1
    return f"{room_name}_{category}_{_placement_counters[key]:02d}"


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class PlacementEngine:
    """
    Places LayoutGPT-generated furniture into the 3ds Max scene.

    Parameters
    ----------
    furniture_library : dict[str, list[FurnitureAsset]]  (from scene_bridge)
    scale_to_fit      : if True, non-uniformly scale instances to LLM dimensions
    snap_to_floor     : if True, snap all instances to the room floor plane
    seed              : random seed for asset selection when multiple variants exist
    """

    def __init__(
        self,
        furniture_library: dict[str, list[FurnitureAsset]],
        scale_to_fit: bool = False,
        snap_to_floor: bool = True,
        seed: int | None = 42,
    ):
        self.library     = furniture_library
        self.scale_to_fit = scale_to_fit
        self.snap_to_floor = snap_to_floor
        self._rng = random.Random(seed)

    def _pick_asset(self, category: str) -> FurnitureAsset | None:
        """Choose a furniture asset for *category* (random if multiple variants)."""
        assets = self.library.get(category)
        if not assets:
            return None
        return self._rng.choice(assets)

    # ------------------------------------------------------------------

    def place_all(
        self,
        placements: list[Placement],
        room: RoomInfo,
        formatter: RoomFormatter,
        group_name: str | None = None,
    ) -> PlacementResult:
        """
        Instantiate and position all furniture items in the Max scene.

        Parameters
        ----------
        placements  : list[Placement] from LayoutGPTRunner
        room        : RoomInfo for the target room
        formatter   : RoomFormatter for coordinate conversion
        group_name  : optional Max group name (defaults to room.name + "_layout")

        Returns
        -------
        PlacementResult with lists of placed / skipped / out-of-bound names.
        """
        if not _IN_MAX:
            raise RuntimeError("PlacementEngine.place_all() requires pymxs (3ds Max).")

        result = PlacementResult()
        placed_nodes = []

        for pl in placements:
            asset = self._pick_asset(pl.category)
            if asset is None:
                print(f"[PlacementEngine] No asset for '{pl.category}' – skipping")
                result.skipped.append(pl.category)
                continue

            # Bounds check (formatter already filtered OOB, but double-check)
            if not formatter.placement_in_bounds(pl.scene):
                result.oob.append(pl.category)
                continue

            inst_name = _unique_instance_name(pl.category, room.name)

            with pymxs.undo(True, f"Place {inst_name}"):
                inst = _instance_node(asset.node, inst_name)

                # Position
                _set_position(inst,
                               pl.scene["pos_x"],
                               pl.scene["pos_y"],
                               pl.scene["pos_z"])

                # Rotation
                _set_z_rotation(inst, pl.scene["rotation_deg"])

                # Optional scale-to-fit
                if self.scale_to_fit:
                    _scale_to_fit(inst, asset, pl.scene)

                # Snap base to floor
                if self.snap_to_floor:
                    _place_on_floor(inst, room)

            placed_nodes.append(inst)
            result.placed.append(inst_name)
            print(f"[PlacementEngine] Placed {inst_name} at "
                  f"({pl.scene['pos_x']:.1f}, {pl.scene['pos_y']:.1f}, {pl.scene['pos_z']:.1f}) "
                  f"rot={pl.scene['rotation_deg']:.1f}°")

        # Group all placed instances under one container
        if placed_nodes:
            grp_name = group_name or f"{room.name}_layout"
            rt.group(placed_nodes, name=grp_name)

        return result
