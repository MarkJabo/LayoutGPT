# LayoutGPT – 3ds Max Furniture Placement Plugin

## Project Overview

This repo adapts the [LayoutGPT research paper](https://arxiv.org/abs/2305.15393) into a practical 3ds Max plugin. Given a room boundary node and a library of furniture prototypes registered by the user, it:

1. Reads room geometry via `pymxs`
2. Retrieves k-similar training examples from ATISS-derived data
3. Sends a few-shot CSS-style prompt to an LLM (Ollama, OpenAI, or custom endpoint)
4. Parses the response and places instanced furniture objects into the Max scene

The plugin is entirely self-contained inside `max_plugin/`. The repo also contains the original LayoutGPT research scripts and ATISS training infrastructure at the top level, but those are reference material only.

---

## Repository Layout

```
LayoutGPT/
├── CLAUDE.md                          ← this file
├── max_plugin/
│   ├── furniture_placer.ms            ← MAXScript UI + entry point
│   ├── furniture_placer.py            ← Python entry point (called by .ms)
│   ├── installer.py                   ← silent pip + Ollama manager
│   ├── layoutgpt_bridge/
│   │   ├── asset_registry.py          ← user-curated room + furniture registry
│   │   ├── layoutgpt_runner.py        ← LLM orchestration, k-similar retrieval
│   │   ├── placement_engine.py        ← pymxs scene placement
│   │   ├── room_formatter.py          ← coordinate conversion (px ↔ scene units)
│   │   ├── scene_bridge.py            ← pymxs scene reading, BoundingBox, etc.
│   │   └── tag_mapper.py              ← 3ds Max tag ↔ LayoutGPT category mapping
│   ├── data/
│   │   ├── bedroom/
│   │   │   ├── dataset_stats.txt      ← object_types + class_frequencies
│   │   │   └── train_examples.json    ← few-shot examples (condition + layout)
│   │   ├── livingroom/
│   │   │   ├── dataset_stats.txt
│   │   │   └── train_examples.json
│   │   ├── bedroom_splits.json        ← optional ATISS room_id splits
│   │   └── livingroom_splits.json
│   └── tests/
│       └── test_bridge_offline.py     ← pytest suite (runs outside Max)
├── run_layoutgpt_3d.py                ← original research reference
├── parse_llm_output.py                ← original CSS layout parser reference
└── utils.py                           ← original dataset utilities reference
```

---

## Core Data Flow

```
User picks room node + furniture prototypes in Max UI
    ↓
AssetRegistry serialised to JSON → furniture_placer.py
    ↓
scene_bridge.py reads world-space BoundingBoxes via pymxs.nodeGetBoundingBox
    ↓
room_formatter.py computes px scale (256px = shorter room dimension)
    ↓
layoutgpt_runner._make_target_bitmap() → 64×64 white rectangle feature
    ↓
select_similar_examples_by_feature() → k=8 training rooms by bitmap MSE
    (tiebreak: L2 on room dimensions, critical when all rooms are white rectangles)
    ↓
LayoutGPTRunner.run() → OpenAI-compatible chat API call with few-shot prompt
    ↓
_parse_3d_line() → list[Placement] in pixel space
    ↓
room_formatter.from_px() → scene-unit coordinates (includes Y-axis flip)
    ↓
placement_engine.py → pymxs instances placed, rotated, scaled, snapped to floor
```

---

## Key Concepts

### Coordinate Systems
- **Pixel space**: 256px = shorter room dimension. Top-left origin. LLM outputs here.
- **Scene space**: 3ds Max world units (typically mm). Y-axis is flipped vs image space.
- Conversion lives in `room_formatter.py:RoomFormatter.from_px()`.
- Orientations are negated during conversion (image CCW → world CW due to Y flip).

### LLM Prompt Format
CSS-style, matching the original LayoutGPT paper:
```
double_bed { length: 212px; width: 200px; height: 110px;
             left: 30px; top: 10px; depth: 20px; orientation: 0deg; }
```
`length`/`width` are the footprint in px. `left`/`top` are the bbox top-left corner. `depth` is vertical offset from floor (usually 0). `orientation` is CCW rotation in image space.

### k-Similar Retrieval
Training features are 64×64 grayscale bitmaps of ATISS room floor plans (255 = floor, 0 = wall). The synthetic target bitmap for a rectangular Max room is a solid white rectangle. Since all rectangular rooms produce identical bitmaps, the **dimension L2 tiebreaker** (`[length_px, width_px]`) drives the actual selection.

### Pymxs Gotchas
- **Position after ±180° rotation**: `node.pos` returns negated X/Y — always set rotation before position.
- **Point3 mutation**: `node.pos.z = v` is silently ignored; assign a full `rt.Point3(x, y, z)`.
- **Pivot vs bbox center**: LLM gives bbox centers; `node.pos` sets the pivot. Offset must be rotated by the final angle before applying.

### Asset Registry
Persisted as a JSON string in a 3ds Max custom attribute on a dummy node. Contains:
- `RoomEntry`: node name, room type (bedroom/livingroom), optional ATISS data dir
- `FurnitureEntry`: node name, user tag (e.g. `FURN_sofa_corner_01`)

Tags are normalised through `tag_mapper.py` to LayoutGPT category names (e.g. `FURN_sofa_corner_01` → `l_shaped_sofa`).

---

## LLM Backends

| Backend | Default base_url | Notes |
|---------|-----------------|-------|
| `ollama` | `http://localhost:11434/v1` | Free, local; `installer.py` manages process + model pulls |
| `openai` | OpenAI API | Requires `api_key` |
| `custom` | User-supplied `base_url` | Any OpenAI-compatible endpoint |

Model and temperature are user-configurable in the MAXScript UI.

---

## Running Tests (Outside 3ds Max)

```bash
cd max_plugin
python -m pytest tests/test_bridge_offline.py -v
```

Tests cover: `tag_mapper`, `room_formatter`, `asset_registry`, and the LLM response parser. They deliberately avoid `pymxs` imports so they run in any standard Python environment.

---

## Dependencies

Installed automatically by `installer.py` into Max's Python interpreter at plugin startup:
- `openai` – OpenAI-compatible chat client
- `numpy` – bitmap operations and k-similar computation
- `scipy` – convex hull for floor-plan bitmap generation (if present)

---

## Adding New Room Types

1. Add a `dataset_stats.txt` with `object_types` and `class_frequencies`.
2. Add a `train_examples.json` with `condition` + `layout` string pairs.
3. Add the room type key to `tag_mapper.py:layoutgpt_room_key()`.
4. Optionally add an ATISS-format `boxes.npz` data directory and point the `RoomEntry.atiss_data_dir` at it for bitmap-based k-similar.

---

## File-by-File Quick Reference

| File | Responsibility |
|------|---------------|
| `furniture_placer.ms` | MAXScript UI rollout, calls `python.execute()` |
| `furniture_placer.py` | Orchestrator: reads registry, builds library, calls runner, calls engine |
| `installer.py` | Silent pip install + Ollama lifecycle management |
| `asset_registry.py` | `RoomEntry` / `FurnitureEntry` dataclasses + JSON serialisation |
| `scene_bridge.py` | `BoundingBox`, `RoomInfo`, `read_furniture_library()` via pymxs |
| `room_formatter.py` | `RoomFormatter`: px↔scene unit conversion, prompt condition string |
| `layoutgpt_runner.py` | `LayoutGPTRunner.run()`, k-similar retrieval, ATISS data loading, LLM call |
| `placement_engine.py` | `PlacementEngine`: pymxs instance creation, rotation, scaling, snapping |
| `tag_mapper.py` | Bidirectional tag↔category mapping, `normalize_tag()`, `denormalize_tag()` |
