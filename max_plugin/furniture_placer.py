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
    load_train_examples,
    load_atiss_training_data,
    _make_target_bitmap,
)
from layoutgpt_bridge.placement_engine import PlacementEngine, PlacementResult
from layoutgpt_bridge.tag_mapper       import layoutgpt_room_key

# ---------------------------------------------------------------------------
# Backend → base_url map
# ---------------------------------------------------------------------------
# Default data directory: bundled alongside this file in max_plugin/data/
_DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Splits JSON files (bedroom_splits.json / livingroom_splits.json) bundled
# inside max_plugin/data/ — used to restrict ATISS loading to rect_train rooms.
_SPLITS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

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
        available_cats = list(library.keys())  # ordering applied by runner

        # Resolve dataset directory: prefer explicit config, fall back to bundled data
        data_dir = config.get("dataset_dir") or _DEFAULT_DATA_DIR

        # Load frequency priors from dataset_stats.txt
        if room_key not in stats_cache:
            try:
                stats_cache[room_key] = load_dataset_stats(data_dir, room_key)
            except FileNotFoundError:
                stats_cache[room_key] = None
        stats = stats_cache[room_key]
        if stats:
            available_cats, class_freq = filter_stats_to_available(stats, available_cats)
        else:
            class_freq = _flat_frequencies(available_cats)

        # Warn the user if any selected categories are rare for this room type.
        # Low-frequency items (<10% of training rooms) degrade LLM quality because
        # the few-shot examples almost never include them — the model improvises.
        if stats:
            _RARE_THRESHOLD = 0.10
            raw_freq = stats.get("class_frequencies", {})
            rare = [
                (cat, raw_freq.get(cat, 0.0))
                for cat in library.keys()
                if raw_freq.get(cat, 0.0) < _RARE_THRESHOLD
            ]
            if rare:
                rare_lines = "\n".join(
                    f"  {c}  ({f:.0%} of {room_key} training rooms)" for c, f in rare
                )
                warn_msg = (
                    f"The following categories are rare in {room_key} scenes "
                    f"and may produce poor layout results:\n\n"
                    + rare_lines
                    + "\n\nThe model has little training data for these placements. "
                    "Proceed anyway?"
                )
                proceed = True
                try:
                    import pymxs
                    proceed = bool(pymxs.runtime.queryBox(warn_msg, title="Category Warning"))
                except Exception:
                    # pymxs unavailable (e.g. running outside Max) – log and continue
                    print(f"[FurniturePlacer] Category warning (auto-proceeding): {rare}")
                if not proceed:
                    all_results.append(
                        f"[{room_entry.node_name}] Cancelled: rare categories for {room_key}: "
                        + ", ".join(c for c, _ in rare)
                    )
                    continue

        # Detect whether dataset_dir contains real ATISS preprocessed data
        # (any subdirectory with a boxes.npz file) or only bundled JSON examples.
        _room_data_dir = os.path.join(data_dir, room_key)
        _has_atiss = False
        if os.path.isdir(_room_data_dir):
            for _d in os.listdir(_room_data_dir):
                if (os.path.isdir(os.path.join(_room_data_dir, _d)) and
                        os.path.exists(os.path.join(_room_data_dir, _d, "boxes.npz"))):
                    _has_atiss = True
                    break

        if _has_atiss:
            # Real ATISS data: load all training rooms + 64×64 floor plan features
            _splits_path = os.path.join(_SPLITS_DIR, f"{room_key}_splits.json")
            train_examples, train_features = load_atiss_training_data(
                data_dir, room_key, _splits_path)
            try:
                target_feature = _make_target_bitmap(
                    formatter.room_px_length, formatter.room_px_width)
            except Exception as exc:
                print(f"[FurniturePlacer] WARNING: could not compute target bitmap: {exc}")
                target_feature = None
            if train_examples:
                print(f"[FurniturePlacer] Using real ATISS data: "
                      f"{len(train_examples)} rooms loaded ({room_key})")
            else:
                # ATISS data directory existed but loading produced no valid
                # examples (e.g. all rooms skipped due to errors or empty
                # layouts).  Fall back to bundled JSON so the user still gets
                # at least a few training examples.
                print(f"[FurniturePlacer] WARNING: ATISS data found but 0 rooms "
                      f"loaded — falling back to bundled examples ({room_key})")
                train_examples = load_train_examples(data_dir, room_key)
                train_features = None
                target_feature = None
        else:
            # Bundled JSON fallback
            train_examples = load_train_examples(data_dir, room_key)
            train_features = None
            target_feature = None
            if train_examples:
                print(f"[FurniturePlacer] Loaded {len(train_examples)} bundled train examples "
                      f"for k-similar retrieval ({room_key})")

        # Measure actual asset footprints and pass to LLM so it can reason
        # about real sizes rather than guessing from category statistics.
        asset_px_sizes: dict[str, dict[str, int]] = {}
        # category_counts: how many copies the engine may place per category
        # (library stores repeated entries for max_instances > 1)
        category_counts: dict[str, int] = {}
        for cat, assets in library.items():
            a = assets[0]  # use first asset as representative
            asset_px_sizes[cat] = {
                "length": max(1, int(round(a.bbox.length_x * formatter._scale))),
                "width":  max(1, int(round(a.bbox.length_y * formatter._scale))),
                "height": max(1, int(round(a.bbox.length_z * formatter._scale))),
            }
            category_counts[cat] = len(assets)
        print(f"[FurniturePlacer] Asset px sizes: {asset_px_sizes}")
        print(f"[FurniturePlacer] Category counts: {category_counts}")

        print(f"\n[FurniturePlacer] Generating layout for '{room_entry.node_name}' …")
        print(f"  {formatter.condition_prompt.strip()}")

        try:
            layouts = runner.run(
                formatter            = formatter,
                available_categories = available_cats,
                class_frequencies    = class_freq,
                n_results            = 1,
                asset_sizes          = asset_px_sizes,
                category_counts      = category_counts,
                train_examples       = train_examples,
                train_features       = train_features,
                target_feature       = target_feature,
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
    tip = (
        "\n\nTIP: If any asset appears backwards or rotated incorrectly, "
        "select it in the Furniture panel and set its Rotation Offset "
        "(e.g. 180 for a desk imported facing away from the user)."
    )
    return "\n".join(all_results) + tip


if __name__ == "__main__":
    print("Run furniture_placer.ms inside 3ds Max to use this plugin.")
