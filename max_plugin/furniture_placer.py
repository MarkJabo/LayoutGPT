"""
furniture_placer.py
-------------------
Top-level entry point for the LayoutGPT furniture placement plugin.

Run this script from within 3ds Max 2024-2026 via:
    Python → Run Script → furniture_placer.py

Or call it from the MAXScript UI wrapper (furniture_placer.ms) which
exposes a simple dialog for configuration.

Workflow
--------
1. User selects one or more closed spline boundaries (ROOM_* naming).
2. Script reads room info and the scene's furniture library (FURN_*).
3. For each room, LayoutGPT generates a furniture layout via the OpenAI API.
4. Instances are placed in the scene, grouped per room.

Configuration
-------------
Edit the CONFIG block below or pass a config dict to main().
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Any

# Allow the bridge package to be imported regardless of sys.path state
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from layoutgpt_bridge.scene_bridge  import read_selected_rooms, read_furniture_library
from layoutgpt_bridge.room_formatter import RoomFormatter
from layoutgpt_bridge.tag_mapper    import normalize_tag, layoutgpt_room_key
from layoutgpt_bridge.layoutgpt_runner import (
    LayoutGPTRunner,
    load_dataset_stats,
    filter_stats_to_available,
)
from layoutgpt_bridge.placement_engine import PlacementEngine

# ---------------------------------------------------------------------------
# Default configuration  – edit here or pass overrides to main()
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: dict[str, Any] = {
    # OpenAI
    "openai_api_key"  : os.environ.get("OPENAI_API_KEY", ""),
    "openai_model"    : "gpt-4o",
    "temperature"     : 0.7,
    "max_tokens"      : 1024,

    # LayoutGPT dataset stats (for furniture frequency priors)
    # Point to the ATISS pre-processed data folder.
    # Set to None to skip frequency-based priors and use a flat distribution.
    "dataset_dir"     : None,   # e.g. r"C:\LayoutGPT\ATISS\data_output"

    # Layout generation
    "n_layouts"       : 1,      # number of layout variations per room
    "canvas_px"       : 256,    # must match what LayoutGPT was trained on

    # Placement options
    "scale_to_fit"    : False,  # non-uniform scale instances to LLM bbox
    "snap_to_floor"   : True,   # snap instance base to room floor Z

    # Furniture prefix tags scanned in the scene
    "furn_prefixes"   : ("furn_", "furniture_"),

    # Room boundary prefix tags
    "room_prefixes"   : ("room_", "boundary_", "area_"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_frequencies(categories: list[str]) -> dict[str, float]:
    """Return a uniform frequency dict when no dataset_stats are available."""
    freq = 1.0 / max(len(categories), 1)
    return {c: freq for c in categories}


def _build_few_shot_examples(
    dataset_stats: dict | None,
    dataset_dir: str | None,
    room_key: str,
) -> list[dict]:
    """
    Load a small number of hand-crafted in-context examples.

    In a full integration these would come from the LayoutGPT training set
    (selected via CLIP similarity).  Here we return a compact hard-coded
    example for each room type so the plugin works without the full dataset.
    """
    # Minimal hard-coded examples that illustrate the CSS format.
    # Replace / extend with real ATISS training examples for best results.
    BEDROOM_EXAMPLE = {
        "condition": (
            "Condition:\n"
            "Room Type: bedroom\n"
            "Room Size: max length 270px, max width 252px\n"
        ),
        "layout": (
            "Layout:\n"
            "double_bed {length: 170px; width: 130px; height: 57px; "
            "left: 129px; top: 111px; depth: 28px; orientation: 0 degrees;}\n"
            "nightstand {length: 40px; width: 40px; height: 45px; "
            "left: 60px; top: 111px; depth: 22px; orientation: 0 degrees;}\n"
            "nightstand {length: 40px; width: 40px; height: 45px; "
            "left: 216px; top: 111px; depth: 22px; orientation: 180 degrees;}\n"
            "wardrobe {length: 80px; width: 40px; height: 120px; "
            "left: 220px; top: 40px; depth: 60px; orientation: 90 degrees;}\n"
        ),
    }
    LIVINGROOM_EXAMPLE = {
        "condition": (
            "Condition:\n"
            "Room Type: living room\n"
            "Room Size: max length 320px, max width 280px\n"
        ),
        "layout": (
            "Layout:\n"
            "multi_seat_sofa {length: 180px; width: 80px; height: 60px; "
            "left: 120px; top: 180px; depth: 30px; orientation: 0 degrees;}\n"
            "coffee_table {length: 90px; width: 50px; height: 35px; "
            "left: 120px; top: 120px; depth: 17px; orientation: 0 degrees;}\n"
            "armchair {length: 70px; width: 70px; height: 60px; "
            "left: 40px; top: 160px; depth: 30px; orientation: 90 degrees;}\n"
            "tv_stand {length: 130px; width: 45px; height: 40px; "
            "left: 120px; top: 50px; depth: 20px; orientation: 0 degrees;}\n"
        ),
    }
    examples = [BEDROOM_EXAMPLE] if room_key == "bedroom" else [LIVINGROOM_EXAMPLE]
    return examples


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def process_room(
    room,
    furniture_library: dict,
    runner: LayoutGPTRunner,
    engine: PlacementEngine,
    config: dict,
    dataset_stats_cache: dict,
) -> None:
    """Run the full LayoutGPT pipeline for a single room."""
    print(f"\n{'='*60}")
    print(f"Processing room: {room.name}  (type: {room.room_type})")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Build RoomFormatter
    # ------------------------------------------------------------------
    try:
        formatter = RoomFormatter(room, canvas_px=config["canvas_px"])
    except ValueError as exc:
        print(f"[ERROR] Cannot format room '{room.name}': {exc}")
        return

    print(f"  Room dims: {formatter.room_px_length}px x {formatter.room_px_width}px  "
          f"(scale: {formatter._scale:.4f} px/unit)")

    # ------------------------------------------------------------------
    # Determine available categories for this room type
    # ------------------------------------------------------------------
    room_key = layoutgpt_room_key(room.room_type)
    available_cats = sorted(set(furniture_library.keys()))

    # Optionally load dataset_stats for frequency priors
    if config.get("dataset_dir"):
        if room_key not in dataset_stats_cache:
            try:
                dataset_stats_cache[room_key] = load_dataset_stats(
                    config["dataset_dir"], room_key
                )
            except FileNotFoundError as exc:
                print(f"[WARN] {exc}  – using flat frequencies")
                dataset_stats_cache[room_key] = None

        stats = dataset_stats_cache[room_key]
        if stats:
            available_cats, class_freq = filter_stats_to_available(stats, available_cats)
        else:
            class_freq = _flat_frequencies(available_cats)
    else:
        class_freq = _flat_frequencies(available_cats)

    if not available_cats:
        print(f"[WARN] No recognised furniture categories found for {room.room_type}.")
        return

    print(f"  Available categories ({len(available_cats)}): {', '.join(available_cats)}")

    # ------------------------------------------------------------------
    # Build few-shot examples
    # ------------------------------------------------------------------
    examples = _build_few_shot_examples(
        dataset_stats_cache.get(room_key),
        config.get("dataset_dir"),
        room_key,
    )

    # ------------------------------------------------------------------
    # Call LayoutGPT
    # ------------------------------------------------------------------
    print(f"  Calling LayoutGPT ({config['openai_model']}) …")
    print(f"  Prompt:\n{formatter.condition_prompt}")

    try:
        all_layouts = runner.run(
            formatter=formatter,
            available_categories=available_cats,
            class_frequencies=class_freq,
            few_shot_examples=examples,
            n_results=config.get("n_layouts", 1),
        )
    except Exception as exc:
        print(f"[ERROR] LayoutGPT inference failed: {exc}")
        traceback.print_exc()
        return

    # ------------------------------------------------------------------
    # Place furniture  (use first layout variant; extend for multi-variant UI)
    # ------------------------------------------------------------------
    for i, placements in enumerate(all_layouts):
        print(f"\n  Layout variant {i+1}: {len(placements)} furniture items")
        for pl in placements:
            print(f"    {pl.category:25s}  pos=({pl.scene['pos_x']:.1f}, "
                  f"{pl.scene['pos_y']:.1f})  rot={pl.scene['rotation_deg']:.1f}°")

        group_name = f"{room.name}_layout" + (f"_v{i+1}" if len(all_layouts) > 1 else "")
        result = engine.place_all(placements, room, formatter, group_name=group_name)
        print(f"  {result.summary()}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def main(config: dict | None = None) -> None:
    """
    Main entry point.  Call from MAXScript or directly from a Python console.

    Parameters
    ----------
    config : optional dict to override DEFAULT_CONFIG values.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    if not cfg["openai_api_key"]:
        raise ValueError(
            "OPENAI_API_KEY is not set.  "
            "Set the environment variable or pass openai_api_key in config."
        )

    # ------------------------------------------------------------------
    # Read scene
    # ------------------------------------------------------------------
    print("[FurniturePlacer] Reading room boundaries from selection …")
    rooms = read_selected_rooms()
    if not rooms:
        print("[FurniturePlacer] No ROOM_* closed splines found in selection or scene.")
        return
    print(f"[FurniturePlacer] Found {len(rooms)} room(s): {[r.name for r in rooms]}")

    print("[FurniturePlacer] Scanning furniture library …")
    furniture_library = read_furniture_library(
        prefixes=tuple(cfg["furn_prefixes"])
    )
    if not furniture_library:
        print("[FurniturePlacer] No FURN_* furniture assets found in scene.")
        return
    print(f"[FurniturePlacer] Furniture library: "
          f"{sum(len(v) for v in furniture_library.values())} assets, "
          f"{len(furniture_library)} categories")

    # ------------------------------------------------------------------
    # Initialise runner and engine (shared across rooms)
    # ------------------------------------------------------------------
    runner = LayoutGPTRunner(
        api_key    = cfg["openai_api_key"],
        model      = cfg["openai_model"],
        temperature= cfg["temperature"],
        max_tokens = cfg["max_tokens"],
    )
    engine = PlacementEngine(
        furniture_library = furniture_library,
        scale_to_fit      = cfg["scale_to_fit"],
        snap_to_floor     = cfg["snap_to_floor"],
    )

    dataset_stats_cache: dict = {}

    # ------------------------------------------------------------------
    # Process each room
    # ------------------------------------------------------------------
    for room in rooms:
        try:
            process_room(room, furniture_library, runner, engine, cfg, dataset_stats_cache)
        except Exception as exc:
            print(f"[ERROR] Failed to process room '{room.name}': {exc}")
            traceback.print_exc()

    print("\n[FurniturePlacer] Done.")


# ---------------------------------------------------------------------------
# Allow direct execution inside Max's Python listener
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
