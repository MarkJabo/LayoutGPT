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
# Overlap detection
# ---------------------------------------------------------------------------

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


def _build_system_prompt(
    available_furniture: list[str],
    class_freq: dict[str, float],
    asset_sizes: dict[str, dict[str, int]] | None = None,
) -> str:
    freq_str = "; ".join(
        f"{obj}: {round(class_freq.get(obj, 0.0), 4)}" for obj in available_furniture
    )

    # If we have measured asset sizes, instruct the LLM to use them exactly so
    # its spatial reasoning reflects the actual mesh footprints.
    if asset_sizes:
        size_lines = "\n".join(
            f"  {cat}: length={asset_sizes[cat]['length']}px, "
            f"width={asset_sizes[cat]['width']}px, "
            f"height={asset_sizes[cat]['height']}px"
            for cat in available_furniture
            if cat in asset_sizes
        )
        size_block = (
            f"\nActual asset sizes (YOU MUST use these exact values for "
            f"length/width/height — do not invent your own):\n{size_lines}\n"
        )
    else:
        size_block = ""

    return (
        "You are a 3D indoor scene designer for commercial real estate visualisation.\n"
        "Instruction: synthesize the 3D layout of an indoor scene. "
        "The generated 3D layout should follow the CSS style, where each line starts "
        "with the furniture category and is followed by the 3D size, orientation and "
        "absolute position.\n"
        f"Formally, each line must follow the template:\n"
        f"FURNITURE {{length: ?{_UNIT}; width: ?{_UNIT}; height: ?{_UNIT}; "
        f"left: ?{_UNIT}; top: ?{_UNIT}; depth: ?{_UNIT}; orientation: ? degrees;}}\n"
        f"All values are in {_UNIT_NAME} but the orientation angle is in degrees.\n"
        f"{size_block}\n"
        f"Available furnitures: {', '.join(available_furniture)}\n"
        f"Overall furniture frequencies: ({freq_str})\n"
        "Placement rules:\n"
        "- Spread furniture across the ENTIRE room area — use the full range of left "
        "and top values, not just the centre.\n"
        "- Place large items (sofas, beds, wardrobes) against walls "
        "(left ≈ 0, left ≈ max, top ≈ 0, or top ≈ max).\n"
        "- Place small items (tables, lamps, chairs) away from walls, in the middle.\n"
        "- Do NOT place two items at the same (left, top) position.\n"
        "- depth should be 0 for all floor-standing furniture.\n"
        "Orientation: you may use 0, 90, 180, or 270 degrees to vary the layout.\n"
        "CRITICAL — when an item is rotated 90° or 270°, its footprint SWAPS: "
        "it will occupy WIDTH pixels in the horizontal (left) direction and "
        "LENGTH pixels in the vertical (top) direction.\n"
        "Example: a 93×28 item at orientation=90 is 28px wide and 93px tall — "
        "so if placed at top=118 in a 146px-tall room it would extend to top=211 "
        "which is OUT OF BOUNDS. Use orientation=0 or 180 for long items near "
        "the top/bottom walls, and 90/270 only when the item fits in both axes.\n"
        f"IMPORTANT: You MUST output exactly one line for EACH of the "
        f"{len(available_furniture)} available furniture categories listed above. "
        f"Do not skip any category.\n"
    )


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
        print(f"[LayoutGPTRunner] Available categories: {available_categories}")
        system_msg = _build_system_prompt(available_categories, class_frequencies, asset_sizes)

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
