"""
tag_mapper.py
-------------
Bidirectional mapping between the user's 3ds Max furniture tag prefixes and
LayoutGPT's 3D-FUTURE category names.

Naming convention for Max scene objects
----------------------------------------
Furniture assets use a prefix tag that encodes category and commercial intent:
    FURN_<category>_<variant>_<index>
    e.g. FURN_sofa_corner_01, FURN_bed_double_A, FURN_chair_dining_02

Room boundary splines use:
    ROOM_<type>_<index>
    e.g. ROOM_bedroom_01, ROOM_livingroom_02, ROOM_dining_01

This module provides:
  - TAG_TO_LAYOUTGPT  : Max tag fragment → 3D-FUTURE category name
  - LAYOUTGPT_TO_TAGS : 3D-FUTURE category name → list of Max tag fragments
  - ROOM_TYPES        : canonical room type strings recognised by LayoutGPT
  - normalize_tag()   : resolve a Max object name → LayoutGPT category
  - denormalize_tag() : resolve a LayoutGPT category → preferred Max tag fragment
  - room_type_from_name(): extract LayoutGPT room-type string from spline name
"""

# ---------------------------------------------------------------------------
# Bedroom furniture vocabulary
# ---------------------------------------------------------------------------
_BEDROOM_MAP = {
    # Max tag fragment          : LayoutGPT 3D-FUTURE category
    "bed_single"                : "single_bed",
    "bed_twin"                  : "single_bed",
    "bed_double"                : "double_bed",
    "bed_queen"                 : "double_bed",
    "bed_king"                  : "double_bed",
    "bed_kids"                  : "kids_bed",
    "bed_bunk"                  : "kids_bed",
    "wardrobe"                  : "wardrobe",
    "closet"                    : "wardrobe",
    "armoire"                   : "wardrobe",
    "dresser"                   : "dresser",
    "chest_drawers"             : "dresser",
    "cabinet_bed"               : "cabinet",
    "cabinet_storage"           : "cabinet",
    "shelf_wall"                : "shelf",
    "shelf_floating"            : "shelf",
    "children_cabinet"          : "children_cabinet",
    "nightstand"                : "nightstand",
    "bedside"                   : "nightstand",
    "dressing_table"            : "dressing_table",
    "vanity"                    : "dressing_table",
    "tv_stand_bed"              : "tv_stand",
    "table_bed"                 : "table",
    "desk_bed"                  : "desk",
    "chair_desk"                : "chair",
    "chair_side"                : "chair",
    "dressing_chair"            : "dressing_chair",
    "stool_bed"                 : "stool",
    "armchair_bed"              : "armchair",
    "lamp_ceiling_bed"          : "ceiling_lamp",
    "lamp_floor_bed"            : "floor_lamp",
    "lamp_pendant_bed"          : "pendant_lamp",
    "bookshelf_bed"             : "bookshelf",
}

# ---------------------------------------------------------------------------
# Living-room furniture vocabulary
# ---------------------------------------------------------------------------
_LIVINGROOM_MAP = {
    "armchair_liv"              : "armchair",
    "lounge_chair"              : "lounge_chair",
    "chair_dining"              : "dining_chair",
    "stool_liv"                 : "stool",
    "sofa_corner"               : "l_shaped_sofa",
    "sofa_l_shape"              : "l_shaped_sofa",
    "sofa_2seater"              : "loveseat_sofa",
    "sofa_loveseat"             : "loveseat_sofa",
    "sofa_3seater"              : "multi_seat_sofa",
    "sofa_sectional"            : "multi_seat_sofa",
    "coffee_table"              : "coffee_table",
    "table_dining"              : "dining_table",
    "console_table"             : "console_table",
    "side_table_corner"         : "corner_side_table",
    "end_table_round"           : "round_end_table",
    "cabinet_liv"               : "cabinet",
    "wine_cabinet"              : "wine_cabinet",
    "bookshelf_liv"             : "bookshelf",
    "tv_stand_liv"              : "tv_stand",
    "lamp_ceiling_liv"          : "ceiling_lamp",
    "lamp_pendant_liv"          : "pendant_lamp",
    "desk_liv"                  : "desk",
}

# ---------------------------------------------------------------------------
# Merged forward map  (Max tag fragment → LayoutGPT category)
# ---------------------------------------------------------------------------
TAG_TO_LAYOUTGPT: dict[str, str] = {**_BEDROOM_MAP, **_LIVINGROOM_MAP}

# ---------------------------------------------------------------------------
# Reverse map  (LayoutGPT category → list[Max tag fragments])
# ---------------------------------------------------------------------------
LAYOUTGPT_TO_TAGS: dict[str, list[str]] = {}
for _tag, _cat in TAG_TO_LAYOUTGPT.items():
    LAYOUTGPT_TO_TAGS.setdefault(_cat, []).append(_tag)

# ---------------------------------------------------------------------------
# Room type strings
# ---------------------------------------------------------------------------
# Maps Max ROOM_* spline name fragments to the strings LayoutGPT expects
ROOM_TYPE_MAP: dict[str, str] = {
    "bedroom"       : "bedroom",
    "bed"           : "bedroom",
    "livingroom"    : "livingroom",
    "living"        : "livingroom",
    "living_dining" : "living room & dining room",
    "dining"        : "living room & dining room",
}

# Room types that LayoutGPT's dataset_stats cover
LAYOUTGPT_ROOM_TYPES = {"bedroom", "livingroom"}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def normalize_tag(max_object_name: str) -> str | None:
    """
    Resolve a 3ds Max object name to a LayoutGPT category string.

    Strategy:
    1. Strip the leading FURN_ prefix.
    2. Walk TAG_TO_LAYOUTGPT looking for the longest matching sub-string
       in the lowercase name (greedy longest-match to avoid false positives).
    3. Return None if no mapping is found.

    Example
    -------
    >>> normalize_tag("FURN_sofa_corner_01")
    'l_shaped_sofa'
    >>> normalize_tag("FURN_bed_double_A")
    'double_bed'
    """
    name_lower = max_object_name.lower()
    # strip furniture prefix if present
    for prefix in ("furn_", "furniture_", "obj_"):
        if name_lower.startswith(prefix):
            name_lower = name_lower[len(prefix):]
            break

    best_match: str | None = None
    best_len = 0
    for tag_fragment, layoutgpt_cat in TAG_TO_LAYOUTGPT.items():
        if tag_fragment in name_lower and len(tag_fragment) > best_len:
            best_match = layoutgpt_cat
            best_len = len(tag_fragment)

    return best_match


def denormalize_tag(layoutgpt_category: str) -> str:
    """
    Return the preferred Max tag fragment for a given LayoutGPT category.
    Falls back to the category name itself if no explicit mapping exists.

    Example
    -------
    >>> denormalize_tag("loveseat_sofa")
    'sofa_loveseat'
    """
    tags = LAYOUTGPT_TO_TAGS.get(layoutgpt_category)
    if tags:
        return tags[0]
    return layoutgpt_category


def room_type_from_name(spline_name: str) -> str:
    """
    Extract the LayoutGPT room-type string from a Max spline object name.

    The spline is expected to be named:   ROOM_<type>[_<index>]
    e.g. ROOM_bedroom_01  →  "bedroom"
         ROOM_livingroom  →  "livingroom"
         ROOM_dining_02   →  "living room & dining room"

    Returns the 3D-FUTURE room string, or "bedroom" as a safe fallback.
    """
    name_lower = spline_name.lower()
    for prefix in ("room_", "boundary_", "area_"):
        if name_lower.startswith(prefix):
            name_lower = name_lower[len(prefix):]
            break

    # strip trailing _01, _02 … indices
    import re
    name_lower = re.sub(r"_\d+$", "", name_lower)

    # longest-match against ROOM_TYPE_MAP
    best_match = "bedroom"
    best_len = 0
    for fragment, room_type in ROOM_TYPE_MAP.items():
        if name_lower.startswith(fragment) and len(fragment) > best_len:
            best_match = room_type
            best_len = len(fragment)

    return best_match


def layoutgpt_room_key(room_type: str) -> str:
    """
    Return the dataset key used by LayoutGPT ('bedroom' | 'livingroom').
    Dining rooms are treated as livingroom for dataset_stats lookup.
    """
    if room_type == "bedroom":
        return "bedroom"
    return "livingroom"


# ---------------------------------------------------------------------------
# Full sorted list of every LayoutGPT / 3D-FUTURE category
# Used to populate UI dropdowns; order is bedroom types first, then living-room
# ---------------------------------------------------------------------------
ALL_LAYOUTGPT_CATEGORIES: list[str] = sorted({
    # Bedroom
    "single_bed", "double_bed", "kids_bed",
    "wardrobe", "dresser", "cabinet", "shelf", "children_cabinet",
    "nightstand", "dressing_table", "tv_stand", "table", "desk",
    "chair", "dressing_chair", "stool", "armchair",
    "ceiling_lamp", "floor_lamp", "pendant_lamp", "bookshelf",
    # Living room (adds / overlaps)
    "lounge_chair", "dining_chair",
    "l_shaped_sofa", "loveseat_sofa", "multi_seat_sofa",
    "coffee_table", "dining_table", "console_table",
    "corner_side_table", "round_end_table",
    "wine_cabinet",
})
