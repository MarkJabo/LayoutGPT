"""
room_formatter.py
-----------------
Convert a RoomInfo (from scene_bridge) into the LayoutGPT 3D prompt strings
and translate LayoutGPT's pixel coordinate space back to scene units.

LayoutGPT coordinate conventions (normalised px space, 256 px canvas)
----------------------------------------------------------------------
  left        → X position of furniture centre  (room origin = 0)
  top         → Y position of furniture centre  (depth / forward axis)
  depth       → Z position of furniture centre  (vertical, 0 = floor)
  length      → furniture extent along X
  width       → furniture extent along Y
  height      → furniture extent along Z (vertical)
  orientation → CCW rotation in degrees around vertical (Z) axis

3ds Max scene unit mapping
--------------------------
  scene_unit  : millimetres (default CRE workflow; configurable via SCENE_UNIT)
  px_per_unit : derived from room bounding box so 256 px = max(room_x, room_y)

Public surface
--------------
  RoomFormatter          – wraps a RoomInfo and builds prompts / back-converts
      .condition_prompt  – the "Condition:\\n…" string LayoutGPT expects
      .from_px()         – convert a LayoutGPT px placement → scene-unit dict
      .room_stats()      – dict describing room for system prompt
"""

from __future__ import annotations
import math
from typing import Any

from .scene_bridge import RoomInfo, BoundingBox
from .tag_mapper import layoutgpt_room_key

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# LayoutGPT normalises to a 256-px canvas where 256 px = the longer side of
# the room.  Set CANVAS_PX to match what you pass to the LLM.
CANVAS_PX: int = 256


class RoomFormatter:
    """
    Wraps one RoomInfo and provides helpers for the LayoutGPT prompt format.

    Parameters
    ----------
    room        : RoomInfo from scene_bridge
    canvas_px   : pixel canvas size used by LayoutGPT (default 256)
    """

    def __init__(self, room: RoomInfo, canvas_px: int = CANVAS_PX):
        self.room = room
        self.canvas_px = canvas_px

        # The room's longer plan dimension maps to `canvas_px` pixels.
        # We keep X and Y independently to handle non-square rooms.
        plan_x = room.bbox.length_x   # scene units along X
        plan_y = room.bbox.length_y   # scene units along Y (depth)
        longer = max(plan_x, plan_y)

        if longer == 0:
            raise ValueError(f"Room '{room.name}' has zero plan extent – check spline.")

        self._scale = canvas_px / longer          # scene_unit → px
        self._inv_scale = longer / canvas_px      # px → scene_unit

        # px room dimensions (rounded to int like LayoutGPT does)
        self.room_px_length = int(round(plan_x * self._scale))
        self.room_px_width  = int(round(plan_y * self._scale))

        # Room origin in scene world coordinates (lower-left corner of bbox)
        self._origin_x = room.bbox.min_x
        self._origin_y = room.bbox.min_y
        self._floor_z  = room.bbox.min_z

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    @property
    def condition_prompt(self) -> str:
        """
        The 'Condition:' block that LayoutGPT expects as the query.

        Example output
        --------------
        Condition:
        Room Type: bedroom
        Room Size: max length 270px, max width 252px
        """
        room_type = self.room.room_type
        # LayoutGPT uses "living room" (with space) in prompt text
        display_type = room_type.replace("livingroom", "living room")
        return (
            f"Condition:\n"
            f"Room Type: {display_type}\n"
            f"Room Size: max length {self.room_px_length}px, max width {self.room_px_width}px\n"
            f"Constraints: left must be 0–{self.room_px_length}px; "
            f"top must be 0–{self.room_px_width}px\n"
        )

    def room_stats(self) -> dict[str, Any]:
        """
        Summary dict used when building the system prompt for the LLM
        (available furniture list and frequencies are added by layoutgpt_runner).
        """
        return {
            "room_type"        : self.room.room_type,
            "layoutgpt_key"    : layoutgpt_room_key(self.room.room_type),
            "room_px_length"   : self.room_px_length,
            "room_px_width"    : self.room_px_width,
            "scene_length_x"   : self.room.bbox.length_x,
            "scene_length_y"   : self.room.bbox.length_y,
            "scene_length_z"   : self.room.bbox.length_z,
            "scale_px_per_unit": self._scale,
        }

    # ------------------------------------------------------------------
    # Coordinate conversion  (LayoutGPT px → Max scene units)
    # ------------------------------------------------------------------

    def from_px(self, placement: dict[str, float]) -> dict[str, float]:
        """
        Convert one LayoutGPT furniture placement (pixel space) into a dict of
        scene-unit values ready for the placement engine.

        Input keys (all in px, from LayoutGPT output)
        -----------------------------------------------
        left        – X centre relative to room origin
        top         – Y centre (depth) relative to room origin
        depth       – Z centre (floor offset; 0 = on floor plane in LayoutGPT)
        length      – furniture X extent
        width       – furniture Y extent
        height      – furniture Z (vertical) extent
        orientation – CCW degrees around vertical axis

        Output keys (all in scene units, world-space absolute)
        -------------------------------------------------------
        pos_x, pos_y, pos_z   – centre position in scene world coordinates
        dim_x, dim_y, dim_z   – object dimensions
        rotation_deg          – Z-axis rotation (CCW positive, Max convention)
        """
        s = self._inv_scale  # px → scene units

        # LayoutGPT image coordinate system: left=X (increases right), top=Y
        # but top=0 is the TOP of the image (far wall) and increases DOWNWARD.
        # 3ds Max world Y increases UPWARD, so top must be inverted:
        #   top=0  →  max_y (far wall)
        #   top=px →  max_y - px*scale  (toward near wall)
        pos_x = self._origin_x + float(placement["left"]) * s
        pos_y = self.room.bbox.max_y - float(placement["top"]) * s
        pos_z = self._floor_z  + float(placement.get("depth", 0.0)) * s

        # Dimensions must be computed before clamping (used for wall margin).
        dim_x = float(placement["length"]) * s
        dim_y = float(placement["width"])  * s
        dim_z = float(placement["height"]) * s

        # Clamp centre so the furniture *body* stays inside the room.
        # Shift the allowed range inward by half the object's footprint so
        # no piece clips through a wall (guard against degenerate oversized assets).
        wall_min_x = self.room.bbox.min_x + dim_x / 2
        wall_max_x = self.room.bbox.max_x - dim_x / 2
        wall_min_y = self.room.bbox.min_y + dim_y / 2
        wall_max_y = self.room.bbox.max_y - dim_y / 2
        if wall_min_x < wall_max_x:
            pos_x = max(wall_min_x, min(wall_max_x, pos_x))
        else:  # asset is wider than the room – fall back to centring
            pos_x = (self.room.bbox.min_x + self.room.bbox.max_x) / 2
        if wall_min_y < wall_max_y:
            pos_y = max(wall_min_y, min(wall_max_y, pos_y))
        else:
            pos_y = (self.room.bbox.min_y + self.room.bbox.max_y) / 2

        # Flipping Y changes chirality: CCW in image space → CW in world space.
        # Negate orientation so a sofa "facing down in image" faces -Y in Max.
        rotation_deg = -float(placement.get("orientation", 0.0))

        return {
            "pos_x"        : pos_x,
            "pos_y"        : pos_y,
            "pos_z"        : pos_z,
            "dim_x"        : dim_x,
            "dim_y"        : dim_y,
            "dim_z"        : dim_z,
            "rotation_deg" : rotation_deg,
        }

    def to_px(self, pos_x: float, pos_y: float, pos_z: float,
              dim_x: float, dim_y: float, dim_z: float,
              rotation_deg: float = 0.0) -> dict[str, float]:
        """
        Convert scene-unit furniture data back to LayoutGPT pixel space.
        Useful when building in-context examples from existing Max scenes.
        """
        s = self._scale  # scene_unit → px
        return {
            "left"       : (pos_x - self._origin_x) * s,
            "top"        : (self.room.bbox.max_y - pos_y) * s,   # Y inverted
            "depth"      : (pos_z - self._floor_z)  * s,
            "length"     : dim_x * s,
            "width"      : dim_y * s,
            "height"     : dim_z * s,
            "orientation": -rotation_deg,                         # negate back
        }

    # ------------------------------------------------------------------
    # Boundary check
    # ------------------------------------------------------------------

    def placement_in_bounds(self, scene_placement: dict[str, float]) -> bool:
        """Return True if the furniture centre lies inside the room bbox."""
        px, py = scene_placement["pos_x"], scene_placement["pos_y"]
        return (
            self.room.bbox.min_x <= px <= self.room.bbox.max_x and
            self.room.bbox.min_y <= py <= self.room.bbox.max_y
        )
