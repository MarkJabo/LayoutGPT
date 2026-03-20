"""
layoutgpt_runner.py
-------------------
Calls the OpenAI Chat API with a LayoutGPT-style 3D scene synthesis prompt
and returns a parsed list of (category, placement_dict) pairs.

The prompt format mirrors run_layoutgpt_3d.py exactly so that the same
trained few-shot examples remain valid.

Public surface
--------------
    LayoutGPTRunner(api_key, model, ...)
        .run(formatter, furniture_library, dataset_stats) → list[Placement]

    Placement   namedtuple: category, px_placement, scene_placement
"""

from __future__ import annotations

import json
import math as _math
import os
import random as _random
import re
from dataclasses import dataclass
from typing import Any

import openai

from .tag_mapper import TAG_TO_LAYOUTGPT, LAYOUTGPT_TO_TAGS, layoutgpt_room_key
from .room_formatter import RoomFormatter

# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class Placement:
    """One furniture item as returned by LayoutGPT, in both coordinate spaces."""
    category: str               # LayoutGPT 3D-FUTURE category name
    px: dict[str, float]        # raw pixel-space placement from LLM output
    scene: dict[str, float]     # scene-unit placement (from RoomFormatter.from_px)


# ---------------------------------------------------------------------------
# Parser (matches parse_3D_layout in the original LayoutGPT utils.py)
# ---------------------------------------------------------------------------

def _parse_3d_line(line: str, unit: str = "px") -> tuple[str, dict[str, float]] | tuple[None, None]:
    """
    Parse one CSS-style furniture line from LayoutGPT output.

    Expected format:
        double_bed {length: 170px; width: 130px; height: 57px;
                    left: 129px; top: 111px; depth: 28px; orientation: 0 degrees;}

    Lenient: fields may be in any order; missing optional fields (depth,
    orientation) are defaulted rather than causing the whole line to be dropped.
    Required: category name, braces, and at minimum left/top/length/width.
    """
    # Must contain braces
    if "{" not in line:
        return None, None
    try:
        text, rest = line.split("{", 1)
    except ValueError:
        return None, None

    rest = rest.strip().rstrip("}").strip().rstrip(";")
    parts = [p.strip() for p in rest.split(";") if p.strip()]

    category = re.sub(r"\d", "", text.strip()).strip()
    if not category:
        return None, None

    parsed: dict[str, float] = {}
    for part in parts:
        try:
            key, val = part.split(":", 1)
            key = key.strip()
            # Strip unit suffix and any trailing whitespace
            val = val.strip()
            val = re.sub(r"(px|degrees?)\s*$", "", val, flags=re.IGNORECASE).strip()
            parsed[key] = float(val)
        except (ValueError, AttributeError):
            continue  # skip malformed fields rather than dropping the whole line

    # Require the four spatial fields that can't be defaulted
    required_min = {"length", "width", "left", "top"}
    if not required_min.issubset(parsed.keys()):
        return None, None

    # Default optional fields so downstream code always sees all seven keys
    parsed.setdefault("height", 60.0)
    parsed.setdefault("depth", 0.0)
    parsed.setdefault("orientation", 0.0)

    return category, parsed


# ---------------------------------------------------------------------------
# Semantic category ordering
# ---------------------------------------------------------------------------
# Anchor items (bed, sofa) must appear first so the LLM places them before
# deciding where accessories (chairs, lamps) go.  Unknown categories fall to
# the end of the list.

_BEDROOM_PRIORITY: list[str] = [
    "double_bed", "single_bed", "kids_bed", "bunk_bed",
    "nightstand", "bedside_table",
    "wardrobe", "cabinet", "dresser", "chest_of_drawers",
    "vanity", "desk",
    "desk_chair", "chair",
    "armchair", "lounge_chair",
    "shelf", "bookcase",
    "coffee_table", "side_table", "stool",
    "ceiling_lamp", "floor_lamp", "pendant_lamp",
]

_LIVINGROOM_PRIORITY: list[str] = [
    "multi_seat_sofa", "l_shaped_sofa", "sofa",
    "armchair", "lounge_chair",
    "coffee_table",
    "tv_stand", "media_console", "console_table",
    "side_table", "end_table",
    "shelf", "bookcase", "cabinet",
    "ceiling_lamp", "floor_lamp", "pendant_lamp",
]


def _sort_by_priority(categories: list[str], room_type: str) -> list[str]:
    """Return categories sorted by semantic placement priority for room_type."""
    norm = room_type.lower().replace(" ", "")
    priority = _LIVINGROOM_PRIORITY if "living" in norm else _BEDROOM_PRIORITY
    priority_map = {cat: i for i, cat in enumerate(priority)}
    return sorted(categories, key=lambda c: priority_map.get(c, len(priority)))


# ---------------------------------------------------------------------------
# Overlap detection and resolution
# ---------------------------------------------------------------------------

def _clashes_with_kept(
    candidate: "Placement",
    kept: list["Placement"],
    margin_px: float = 2.0,
) -> bool:
    """
    Return True if *candidate*'s bounding box overlaps any item in *kept*.

    Tests each kept item individually against the candidate so that
    pre-existing overlaps among kept items cannot cause false rejections.
    """
    # Build a tiny 2-element list for each kept item and test only that pair.
    for existing in kept:
        if _overlapping_pairs_px([existing, candidate], margin_px):
            return True
    return False


def _remove_overlapping_placements(
    placements: list["Placement"],
    margin_px: float = 2.0,
) -> list["Placement"]:
    """
    Deterministic post-processing: iterate through placements in list order
    (LLM's intended order — primary furniture is usually listed first) and
    keep each item only if it does not overlap any already-accepted item.

    This guarantees the returned list is overlap-free regardless of whether
    the LLM respected the no-overlap constraint.
    """
    kept: list["Placement"] = []
    for p in placements:
        if not _clashes_with_kept(p, kept, margin_px):
            kept.append(p)
        else:
            print(f"[LayoutGPTRunner] Post-process removed '{p.category}' "
                  f"(overlaps a kept item).")
    return kept


def _overlapping_pairs_px(
    placements: list["Placement"],
    margin_px: float = 2.0,
) -> list[tuple[str, str]]:
    """
    Return (cat_a, cat_b) pairs whose *pixel-space* bounding boxes overlap.

    We check pixel space (the LLM's intended positions) rather than world
    space so that wall-clamping artifacts don't produce false positives.
    A common false positive: an L-shaped sofa has a large AABB that covers
    its own open corner; a coffee table placed in that corner is fine
    physically but looks like a world-space collision.

    In pixel space: left/top are CENTER coords; length/width are axes before
    rotation; orientation (CCW degrees) swaps the axes for 90°/270° items.
    """
    boxes = []
    for p in placements:
        cx  = p.px["left"]
        cy  = p.px["top"]
        a   = _math.radians(p.px.get("orientation", 0.0))
        ca, sa = abs(_math.cos(a)), abs(_math.sin(a))
        hx  = ca * p.px["length"] / 2 + sa * p.px["width"] / 2
        hy  = sa * p.px["length"] / 2 + ca * p.px["width"] / 2
        boxes.append((p.category, cx - hx, cx + hx, cy - hy, cy + hy))

    overlaps = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            cat_a, ax1, ax2, ay1, ay2 = boxes[i]
            cat_b, bx1, bx2, by1, by2 = boxes[j]
            if (ax1 + margin_px < bx2 and ax2 - margin_px > bx1 and
                    ay1 + margin_px < by2 and ay2 - margin_px > by1):
                overlaps.append((cat_a, cat_b))
    return overlaps


# ---------------------------------------------------------------------------
# System / user prompt builders  (mirrors form_prompt_for_chatgpt)
# ---------------------------------------------------------------------------

_UNIT = "px"
_UNIT_NAME = "pixels"

# Keywords that put a category in each placement zone.
# Checked via substring match on the lowercased category name.
_WALL_KEYWORDS    = ("bed", "wardrobe", "dresser", "shelf", "bookshelf",
                     "cabinet", "tv_stand", "console", "wine", "desk",
                     "dressing_table", "children_cabinet")
_SOFA_KEYWORDS    = ("sofa", "armchair", "lounge_chair")
_FLOOR_KEYWORDS   = ("coffee_table", "dining_table", "round_end_table",
                     "corner_side_table")
_BESIDE_KEYWORDS  = ("nightstand", "bedside")


def _placement_zone(cat: str) -> str:
    """Return 'wall', 'sofa_wall', 'floor_center', 'beside_bed', or 'anywhere'."""
    c = cat.lower()
    if any(k in c for k in _BESIDE_KEYWORDS):
        return "beside_bed"
    if any(k in c for k in _WALL_KEYWORDS):
        return "wall"
    if any(k in c for k in _SOFA_KEYWORDS):
        return "sofa_wall"
    if any(k in c for k in _FLOOR_KEYWORDS):
        return "floor_center"
    return "anywhere"


def _required_items_block(
    available_furniture: list[str],
    category_counts: dict[str, int] | None,
) -> str:
    """
    Build the closing IMPORTANT instruction listing every category and how many
    CSS lines to generate.  Single-copy items are required; multi-copy items
    say 'up to N' so the LLM places as many as fit naturally — the placement
    engine will pick from available copies, and the overlap post-processor
    removes any that can't be spaced correctly.
    """
    counts = category_counts or {}
    required_lines = []
    optional_lines = []
    for cat in available_furniture:
        n = counts.get(cat, 1)
        if n == 1:
            required_lines.append(f"  1 × {cat}  (required — output exactly 1 line)")
        else:
            optional_lines.append(
                f"  up to {n} × {cat}  "
                f"(output 2–{n} lines if space permits; at least 1)"
            )
    all_lines = required_lines + optional_lines
    item_list = "\n".join(all_lines)
    total_required = len(required_lines) + len(optional_lines)  # at least 1 per category
    return (
        f"IMPORTANT: Output CSS lines for the following items:\n"
        f"{item_list}\n"
        "Required items (count = 1) MUST always appear.\n"
        "For items with 'up to N' copies, place as many as fit "
        "without crowding — prioritise good spacing over quantity.\n"
        "Do NOT add items not listed above.\n"
        "Multiple copies of the same category must be well-separated "
        "from each other and clearly spaced from other furniture.\n"
    )


def _build_system_prompt(
    available_furniture: list[str],
    class_freq: dict[str, float],
    asset_sizes: dict[str, dict[str, int]] | None = None,
    room_type: str = "bedroom",
    category_counts: dict[str, int] | None = None,
) -> str:
    freq_str = "; ".join(
        f"{obj}: {round(class_freq.get(obj, 0.0), 4)}" for obj in available_furniture
    )

    # Asset size block: tell the LLM the measured real dimensions so it doesn't
    # invent them.  The original paper derived sizes from the 3D-FUTURE dataset;
    # here we measure from the actual Max scene assets.
    if asset_sizes:
        size_lines = "\n".join(
            f"  {cat}: length={asset_sizes[cat]['length']}px, "
            f"width={asset_sizes[cat]['width']}px, "
            f"height={asset_sizes[cat]['height']}px"
            for cat in available_furniture
            if cat in asset_sizes
        )
        size_block = (
            f"Asset sizes (use these exact values for length/width/height):\n"
            f"{size_lines}\n\n"
        )
    else:
        size_block = ""

    return (
        # --- Identical to the paper's ChatGPT system prompt (form_prompt_for_chatgpt) ---
        "You are a 3D indoor scene designer.\n"
        "Instruction: synthesize the 3D layout of an indoor scene. "
        "The generated 3D layout should follow the CSS style, where each line starts "
        "with the furniture category and is followed by the 3D size, orientation and "
        "absolute position. "
        "Formally, each line should follow the template:\n"
        f"FURNITURE {{length: ?{_UNIT}; width: ?{_UNIT}; height: ?{_UNIT}; "
        f"left: ?{_UNIT}; top: ?{_UNIT}; depth: ?{_UNIT}; orientation: ? degrees;}}\n"
        f"All values are in {_UNIT_NAME} but the orientation angle is in degrees.\n\n"
        # --- Minimal coordinate note (the paper's ICL examples taught this implicitly;
        #     we spell it out once since we have fewer examples) ---
        "Note: left and top are the CENTRE of the item's floor footprint. "
        "depth = 0 for all floor-standing furniture.\n\n"
        # --- Asset sizes (replaces the paper's dataset-statistics-derived sizes) ---
        f"{size_block}"
        # --- Category list and frequencies (verbatim from the paper) ---
        f"Available furnitures: {', '.join(available_furniture)}\n"
        f"Overall furniture frequencies: ({freq_str})\n\n"
        # --- Required items (needed because we have no k-similar retrieval) ---
        + _required_items_block(available_furniture, category_counts)
    )


# ---------------------------------------------------------------------------
# Default few-shot examples
# ---------------------------------------------------------------------------
# These demonstrate correct semantic layout (wall placement, furniture
# relationships) and the exact pixel-centre coordinate convention.
# Room sizes are chosen to be realistic but different from any particular
# user scene, so the LLM must generalise rather than copy.

_BEDROOM_EXAMPLE: dict = {
    "condition": (
        "Condition:\n"
        "Room Type: bedroom\n"
        "Room Size: max length 270px, max width 252px\n"
        "Constraints: left must be 0–270px; top must be 0–252px\n"
    ),
    # Layout notes (not sent to LLM):
    #   double_bed  : against far wall (top=65=130/2), centred left–right
    #   nightstand  : flush beside bed on left side (left=25=50/2, same top)
    #   wardrobe    : against right wall (left=220=270−100/2), mid-height
    #   desk        : against left wall (left=55=110/2), near-wall side
    #   chair       : in front of desk (top=155 < desk top=210), facing desk (180°)
    "layout": (
        "double_bed {length: 170px; width: 130px; height: 57px; "
        "left: 135px; top: 65px; depth: 0px; orientation: 0 degrees;}\n"
        "nightstand {length: 50px; width: 40px; height: 45px; "
        "left: 25px; top: 65px; depth: 0px; orientation: 0 degrees;}\n"
        "wardrobe {length: 100px; width: 45px; height: 100px; "
        "left: 220px; top: 155px; depth: 0px; orientation: 0 degrees;}\n"
        "desk {length: 110px; width: 55px; height: 45px; "
        "left: 55px; top: 210px; depth: 0px; orientation: 0 degrees;}\n"
        "chair {length: 50px; width: 50px; height: 45px; "
        "left: 55px; top: 155px; depth: 0px; orientation: 180 degrees;}\n"
    ),
}

_LIVINGROOM_EXAMPLE: dict = {
    "condition": (
        "Condition:\n"
        "Room Type: living room\n"
        "Room Size: max length 256px, max width 200px\n"
        "Constraints: left must be 0–256px; top must be 0–200px\n"
    ),
    # Layout notes:
    #   multi_seat_sofa : against far wall (top=40=80/2), left of centre
    #   armchair        : against right wall, side-by-side with sofa
    #   coffee_table    : open floor in front of sofa
    #   tv_stand        : against near wall (top=178=200−40/2), facing sofa
    #   bookshelf       : against left wall, mid-depth
    "layout": (
        "multi_seat_sofa {length: 180px; width: 80px; height: 35px; "
        "left: 90px; top: 40px; depth: 0px; orientation: 0 degrees;}\n"
        "armchair {length: 70px; width: 70px; height: 40px; "
        "left: 221px; top: 40px; depth: 0px; orientation: 0 degrees;}\n"
        "coffee_table {length: 90px; width: 50px; height: 35px; "
        "left: 128px; top: 115px; depth: 0px; orientation: 0 degrees;}\n"
        "tv_stand {length: 130px; width: 40px; height: 40px; "
        "left: 128px; top: 178px; depth: 0px; orientation: 0 degrees;}\n"
        "bookshelf {length: 50px; width: 25px; height: 70px; "
        "left: 25px; top: 120px; depth: 0px; orientation: 0 degrees;}\n"
    ),
}


def _default_few_shot_examples(room_type: str) -> list[dict]:
    """Return built-in in-context examples for the given room type."""
    from .tag_mapper import layoutgpt_room_key
    key = layoutgpt_room_key(room_type)
    if key == "bedroom":
        return [_BEDROOM_EXAMPLE]
    return [_LIVINGROOM_EXAMPLE]


def _build_few_shot_messages(examples: list[dict]) -> list[dict]:
    """
    Convert a list of pre-formatted in-context examples into chat messages.

    Each example dict must have keys:
        condition  – the "Condition:\\n…" string
        layout     – the "Layout:\\n…" string
    """
    messages = []
    for ex in examples:
        messages.append({"role": "user",      "content": ex["condition"] + "Layout:\n"})
        messages.append({"role": "assistant", "content": ex["layout"].lstrip("Layout:\n")})
    return messages


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

class LayoutGPTRunner:
    """
    Wraps an OpenAI-compatible Chat API to produce LayoutGPT 3D placements.

    Works with OpenAI, Ollama (local), LM Studio, or any OpenAI-compatible
    endpoint by setting base_url.

    Parameters
    ----------
    api_key         : API key.  For Ollama pass "ollama" (any non-empty string).
                      Falls back to OPENAI_API_KEY env var.
    model           : chat model name, e.g. "gpt-4o" or "llama3.2:3b"
    base_url        : API base URL.  None → OpenAI default.
                      Ollama: "http://localhost:11434/v1"
                      LM Studio: "http://localhost:1234/v1"
    temperature     : sampling temperature (default 0.7)
    max_tokens      : max completion tokens (default 1024)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "No API key provided.  For Ollama use 'ollama' as the key; "
                "for OpenAI pass your sk- key or set OPENAI_API_KEY."
            )

        client_kwargs: dict[str, Any] = {"api_key": resolved_key, "timeout": 30.0}
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = openai.OpenAI(**client_kwargs)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        formatter: RoomFormatter,
        available_categories: list[str],
        class_frequencies: dict[str, float],
        few_shot_examples: list[dict] | None = None,
        n_results: int = 1,
        asset_sizes: dict[str, dict[str, int]] | None = None,
        category_counts: dict[str, int] | None = None,
    ) -> list[list[Placement]]:
        """
        Generate furniture placements for a room.

        Parameters
        ----------
        formatter           : RoomFormatter wrapping the target room
        available_categories: LayoutGPT category names present in the scene
        class_frequencies   : per-category frequency dict (from dataset_stats)
        few_shot_examples   : optional list of {condition, layout} dicts for ICL
        n_results           : number of independent layout variations to generate

        Returns
        -------
        list of length n_results; each element is a list[Placement] for one layout.
        """
        # Sort categories semantically so the LLM encounters anchor items
        # (bed, sofa) before accessories (chairs, lamps, side-tables).
        ordered_cats = _sort_by_priority(available_categories, formatter.room.room_type)
        print(f"[LayoutGPTRunner] Available categories (ordered): {ordered_cats}")
        print(f"[LayoutGPTRunner] Category counts: {category_counts}")
        system_msg = _build_system_prompt(
            ordered_cats, class_frequencies, asset_sizes,
            room_type=formatter.room.room_type,
            category_counts=category_counts,
        )
        if few_shot_examples is None:
            few_shot_examples = _default_few_shot_examples(formatter.room.room_type)

        _MAX_OVERLAP_RETRIES = 2
        retry_hint = ""

        for attempt in range(_MAX_OVERLAP_RETRIES + 1):
            # Unique variation tag busts OpenAI's prompt cache on every call.
            variation = _random.randint(10000, 99999)
            user_msg  = (
                formatter.condition_prompt
                + f"Variation: {variation}\n"
                + retry_hint
                + "Layout:\n"
            )
            if attempt == 0:
                print(f"[LayoutGPTRunner] User prompt:\n{user_msg}")

            messages: list[dict] = [{"role": "system", "content": system_msg}]
            if few_shot_examples:
                messages.extend(_build_few_shot_messages(few_shot_examples))
            messages.append({"role": "user", "content": user_msg})

            raw_content = self._call_api(messages, n=n_results)

            results: list[list[Placement]] = []
            for content in raw_content:
                placements = self._parse_response(content, formatter)
                results.append(placements)

            # Check the first result for overlaps in pixel space; retry if found.
            if results:
                bad_pairs = _overlapping_pairs_px(results[0])
                if bad_pairs:
                    pair_str = ", ".join(f"{a}&{b}" for a, b in bad_pairs)
                    print(f"[LayoutGPTRunner] Overlap detected ({pair_str}) "
                          f"– retry {attempt + 1}/{_MAX_OVERLAP_RETRIES}")
                    if attempt < _MAX_OVERLAP_RETRIES:
                        retry_hint = (
                            "IMPORTANT: the previous attempt had overlapping items "
                            f"({pair_str}). Re-space them so no two items share "
                            "floor area. Remember: rotating 90°/270° swaps "
                            "length↔width in the top axis.\n"
                        )
                        continue

            break  # no overlaps, or retries exhausted

        # Deterministic fallback: strip any remaining overlaps so nothing is
        # ever placed on top of another item, even if the LLM ignored the hints.
        results = [_remove_overlapping_placements(pl) for pl in results]

        return results

    # ------------------------------------------------------------------
    # API call (no retries – surface errors immediately so the UI stays
    # responsive and the user knows exactly what went wrong)
    # ------------------------------------------------------------------

    def _call_api(self, messages: list[dict], n: int = 1) -> list[str]:
        """Call OpenAI chat API, return list of assistant content strings."""
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=["Condition:"],
                n=n,
            )
            return [choice.message.content for choice in resp.choices]
        except openai.RateLimitError as exc:
            raise RuntimeError(f"Rate limited: {exc}") from exc
        except openai.APIStatusError as exc:
            raise RuntimeError(f"API error {exc.status_code}: {exc.message}") from exc
        except openai.APIConnectionError as exc:
            raise RuntimeError(f"Cannot reach API – check your endpoint/key: {exc}") from exc

    # ------------------------------------------------------------------
    # Response parser
    # ------------------------------------------------------------------

    def _parse_response(self, content: str, formatter: RoomFormatter) -> list[Placement]:
        print(f"[LayoutGPTRunner] Raw LLM response:\n{content}")
        placements: list[Placement] = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            category, px_placement = _parse_3d_line(line, unit=_UNIT)
            if category is None:
                continue
            try:
                scene_placement = formatter.from_px(px_placement)
            except (KeyError, TypeError, ValueError):
                continue

            # Bounds check – discard furniture outside the room
            if not formatter.placement_in_bounds(scene_placement):
                print(f"[LayoutGPTRunner] {category} out of bounds – skipping")
                continue

            placements.append(Placement(
                category=category,
                px=px_placement,
                scene=scene_placement,
            ))
        return placements


# ---------------------------------------------------------------------------
# Dataset stats loader  (mirrors LayoutGPT's dataset_stats.txt)
# ---------------------------------------------------------------------------

def load_dataset_stats(dataset_dir: str, room_type: str) -> dict[str, Any]:
    """
    Load dataset_stats.txt for a given room type.

    dataset_dir should point to the ATISS data_output folder, e.g.:
        ./ATISS/data_output/bedroom/dataset_stats.txt

    Returns dict with keys:
        object_types      : list[str]
        class_frequencies : dict[str, float]
    """
    stats_path = os.path.join(dataset_dir, room_type, "dataset_stats.txt")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"dataset_stats.txt not found at {stats_path}.  "
            "Run the ATISS pre-processing step or provide the correct dataset_dir."
        )
    with open(stats_path, "r") as fh:
        return json.load(fh)


def filter_stats_to_available(
    stats: dict[str, Any],
    available_categories: list[str],
) -> tuple[list[str], dict[str, float]]:
    """
    Filter full dataset stats down to only categories that exist in the scene.

    Returns (filtered_object_types, filtered_class_frequencies).
    """
    avail_set = set(available_categories)
    filtered_types = [t for t in stats["object_types"] if t in avail_set]
    filtered_freq  = {
        k: v for k, v in stats["class_frequencies"].items() if k in avail_set
    }
    return filtered_types, filtered_freq
