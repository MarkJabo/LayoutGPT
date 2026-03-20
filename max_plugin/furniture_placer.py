"""
furniture_placer.py
-------------------
Python-side entry point for the LayoutGPT Furniture Placer plugin.

Called from furniture_placer.ms via python.execute().
Accepts a plain-dict config that the MAXScript UI serialises from its state.

Config keys
-----------
backend         : "ollama" | "openai" | "custom"
api_key         : string  (ignored for ollama backend)
model           : string  e.g. "llama3.2:3b", "gpt-4o"
base_url        : string or None  (overrides backend default)
temperature     : float
max_tokens      : int
snap_to_floor   : bool
scale_to_fit    : bool
registry_json   : JSON string produced by AssetRegistry.to_json()
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Auto-install dependencies silently before importing them
from installer import ensure_dependencies
_ok, _msg = ensure_dependencies()
if not _ok:
    raise RuntimeError(f"[FurniturePlacer] Dependency install failed: {_msg}")

from layoutgpt_bridge.asset_registry   import AssetRegistry, RoomEntry
from layoutgpt_bridge.scene_bridge     import read_furniture_library, BoundingBox, RoomInfo
from layoutgpt_bridge.room_formatter   import RoomFormatter
from layoutgpt_bridge.layoutgpt_runner import (
    LayoutGPTRunner,
    load_dataset_stats,
    filter_stats_to_available,
)
from layoutgpt_bridge.placement_engine import PlacementEngine, PlacementResult
from layoutgpt_bridge.tag_mapper       import layoutgpt_room_key

# ---------------------------------------------------------------------------
# Backend → base_url map
# ---------------------------------------------------------------------------
BACKEND_URLS: dict[str, str | None] = {
    "ollama"  : "http://localhost:11434/v1",
    "openai"  : None,
    "custom"  : None,   # overridden by config["base_url"]
}

BACKEND_DEFAULT_KEY: dict[str, str] = {
    "ollama" : "ollama",   # Ollama accepts any non-empty string
    "openai" : "",
    "custom" : "",
}

# ---------------------------------------------------------------------------
# Few-shot examples (compact hard-coded; extend with real ATISS examples)
# ---------------------------------------------------------------------------
_BEDROOM_EXAMPLE = {
    "condition": (
        "Condition:\nRoom Type: bedroom\n"
        "Room Size: max length 270px, max width 252px\n"
    ),
    "layout": (
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
_LIVINGROOM_EXAMPLE = {
    "condition": (
        "Condition:\nRoom Type: living room\n"
        "Room Size: max length 320px, max width 280px\n"
    ),
    "layout": (
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

_FEW_SHOT: dict[str, list[dict]] = {
    "bedroom"    : [_BEDROOM_EXAMPLE],
    "livingroom" : [_LIVINGROOM_EXAMPLE],
}


def _flat_frequencies(categories: list[str]) -> dict[str, float]:
    freq = 1.0 / max(len(categories), 1)
    return {c: freq for c in categories}


def _resolve_room_node(entry: RoomEntry) -> None:
    """Attach a live pymxs node to a RoomEntry by name lookup (if not already set)."""
    if entry.node is not None:
        return
    try:
        import pymxs
        rt = pymxs.runtime
        entry.node = rt.getNodeByName(entry.node_name)
    except Exception:
        pass


def _room_info_from_entry(entry: RoomEntry) -> RoomInfo | None:
    """Build a RoomInfo from a RoomEntry (resolves bbox via pymxs)."""
    _resolve_room_node(entry)
    if entry.node is None:
        print(f"[FurniturePlacer] WARNING: node '{entry.node_name}' not found in scene.")
        return None
    try:
        import pymxs
        rt = pymxs.runtime
        mn = rt.nodeGetBoundingBox(entry.node, rt.Matrix3(1))
        bbox = BoundingBox(
            min_x=float(mn[0].x), min_y=float(mn[0].y), min_z=float(mn[0].z),
            max_x=float(mn[1].x), max_y=float(mn[1].y), max_z=float(mn[1].z),
        )
        print(f"[FurniturePlacer] Room bbox '{entry.node_name}': "
              f"X[{bbox.min_x:.1f}, {bbox.max_x:.1f}]  "
              f"Y[{bbox.min_y:.1f}, {bbox.max_y:.1f}]  "
              f"Z[{bbox.min_z:.1f}, {bbox.max_z:.1f}]  "
              f"size: {bbox.length_x:.1f} x {bbox.length_y:.1f}")
        return RoomInfo(name=entry.node_name, room_type=entry.room_type,
                        bbox=bbox, spline_obj=entry.node)
    except Exception as exc:
        print(f"[FurniturePlacer] Cannot get bbox for '{entry.node_name}': {exc}")
        return None


def _furniture_asset_from_entry(fentry, FurnitureAsset_cls, BoundingBox_cls):
    """Build a FurnitureAsset from a FurnitureEntry."""
    try:
        import pymxs
        rt = pymxs.runtime
        node = rt.getNodeByName(fentry.node_name)
        if node is None:
            return None
        mn = rt.nodeGetBoundingBox(node, rt.Matrix3(1))
        bbox = BoundingBox_cls(
            min_x=float(mn[0].x), min_y=float(mn[0].y), min_z=float(mn[0].z),
            max_x=float(mn[1].x), max_y=float(mn[1].y), max_z=float(mn[1].z),
        )
        print(f"[FurniturePlacer] Furniture bbox '{fentry.node_name}' "
              f"({fentry.category}): "
              f"size {bbox.length_x:.1f} x {bbox.length_y:.1f} x {bbox.length_z:.1f}  "
              f"world pos [{bbox.min_x:.1f},{bbox.max_x:.1f}] x [{bbox.min_y:.1f},{bbox.max_y:.1f}]")
        from layoutgpt_bridge.scene_bridge import FurnitureAsset
        return FurnitureAsset(
            name=fentry.node_name,
            layoutgpt_category=fentry.category,
            bbox=bbox,
            rotation_offset=float(fentry.rotation_offset),
            node=node,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(config: dict[str, Any]) -> str:
    """
    Run the full pipeline.  Called from MAXScript via python.execute().

    Returns a human-readable status string for the UI status bar.
    """
    # ------------------------------------------------------------------
    # Resolve LLM backend
    # ------------------------------------------------------------------
    backend  = config.get("backend", "openai")
    api_key  = config.get("api_key") or BACKEND_DEFAULT_KEY.get(backend, "")
    model    = config.get("model", "gpt-4o")
    base_url = config.get("base_url") or BACKEND_URLS.get(backend)

    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return "ERROR: No API key. Add your key in the LLM Setup panel."

    # ------------------------------------------------------------------
    # Restore registry
    # ------------------------------------------------------------------
    registry_json = config.get("registry_json", "{}")
    try:
        registry = AssetRegistry.from_json(registry_json)
    except Exception as exc:
        return f"ERROR: Could not load registry: {exc}"

    if not registry.rooms:
        return "No rooms added. Use 'Add Selected Room' to add room boundaries."
    if not registry.furniture:
        return "No furniture added. Use 'Add Selected Furniture'."

    # ------------------------------------------------------------------
    # Build live furniture library from registry entries
    # ------------------------------------------------------------------
    from layoutgpt_bridge.scene_bridge import FurnitureAsset

    # ------------------------------------------------------------------
    # Shared runner + engine
    # ------------------------------------------------------------------
    try:
        runner = LayoutGPTRunner(
            api_key     = api_key,
            model       = model,
            base_url    = base_url,
            temperature = float(config.get("temperature", 0.7)),
            max_tokens  = int(config.get("max_tokens", 1024)),
        )
    except Exception as exc:
        return f"ERROR: LLM setup failed: {exc}"

    all_results: list[str] = []
    stats_cache: dict = {}

    for room_entry in registry.rooms:
        room_info = _room_info_from_entry(room_entry)
        if room_info is None:
            all_results.append(f"SKIP {room_entry.node_name}: node not found.")
            continue

        # Build furniture library dict for this room
        library: dict[str, list[FurnitureAsset]] = {}
        for f_entry in registry.furniture_for_room(room_entry):
            asset = _furniture_asset_from_entry(f_entry, FurnitureAsset, BoundingBox)
            if asset is None:
                continue
            # Respect max_instances by repeating the asset
            for _ in range(f_entry.max_instances):
                library.setdefault(f_entry.category, []).append(asset)

        if not library:
            all_results.append(f"SKIP {room_entry.node_name}: no furniture assets resolved.")
            continue

        engine = PlacementEngine(
            furniture_library = library,
            scale_to_fit      = bool(config.get("scale_to_fit", False)),
            snap_to_floor     = bool(config.get("snap_to_floor", True)),
        )

        try:
            formatter = RoomFormatter(room_info)
        except ValueError as exc:
            all_results.append(f"SKIP {room_entry.node_name}: {exc}")
            continue

        room_key = layoutgpt_room_key(room_entry.room_type)
        available_cats = sorted(library.keys())

        # Optionally load frequency priors
        if config.get("dataset_dir"):
            if room_key not in stats_cache:
                try:
                    stats_cache[room_key] = load_dataset_stats(config["dataset_dir"], room_key)
                except FileNotFoundError:
                    stats_cache[room_key] = None
            stats = stats_cache[room_key]
            if stats:
                available_cats, class_freq = filter_stats_to_available(stats, available_cats)
            else:
                class_freq = _flat_frequencies(available_cats)
        else:
            class_freq = _flat_frequencies(available_cats)

        examples = _FEW_SHOT.get(room_key, [_BEDROOM_EXAMPLE])

        # Measure actual asset footprints and pass to LLM so it can reason
        # about real sizes rather than guessing from category statistics.
        asset_px_sizes: dict[str, dict[str, int]] = {}
        for cat, assets in library.items():
            a = assets[0]  # use first asset as representative
            asset_px_sizes[cat] = {
                "length": max(1, int(round(a.bbox.length_x * formatter._scale))),
                "width":  max(1, int(round(a.bbox.length_y * formatter._scale))),
                "height": max(1, int(round(a.bbox.length_z * formatter._scale))),
            }
        print(f"[FurniturePlacer] Asset px sizes: {asset_px_sizes}")

        print(f"\n[FurniturePlacer] Generating layout for '{room_entry.node_name}' …")
        print(f"  {formatter.condition_prompt.strip()}")

        try:
            layouts = runner.run(
                formatter            = formatter,
                available_categories = available_cats,
                class_frequencies    = class_freq,
                few_shot_examples    = examples,
                n_results            = 1,
                asset_sizes          = asset_px_sizes,
            )
        except Exception as exc:
            all_results.append(f"ERROR {room_entry.node_name}: LLM call failed – {exc}")
            traceback.print_exc()
            continue

        placements = layouts[0] if layouts else []
        result: PlacementResult = engine.place_all(
            placements = placements,
            room       = room_info,
            formatter  = formatter,
        )
        all_results.append(f"{room_entry.node_name}: {result.summary()}")
        print(f"  {result.summary()}")

    if not all_results:
        return "Done — nothing was processed."
    return "\n".join(all_results)


if __name__ == "__main__":
    print("Run furniture_placer.ms inside 3ds Max to use this plugin.")
