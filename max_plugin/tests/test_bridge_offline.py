"""
test_bridge_offline.py
----------------------
Unit tests for the LayoutGPT bridge — runs outside 3ds Max (no pymxs).
Covers tag_mapper, room_formatter, asset_registry, and the LLM response parser.

Run with:
    cd max_plugin
    python -m pytest tests/test_bridge_offline.py -v
"""
import sys
import os
import math
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from layoutgpt_bridge.tag_mapper import (
    normalize_tag,
    denormalize_tag,
    room_type_from_name,
    layoutgpt_room_key,
    TAG_TO_LAYOUTGPT,
    LAYOUTGPT_TO_TAGS,
    ALL_LAYOUTGPT_CATEGORIES,
)
from layoutgpt_bridge.scene_bridge import BoundingBox, RoomInfo, MockRoomInfo
from layoutgpt_bridge.room_formatter import RoomFormatter
from layoutgpt_bridge.layoutgpt_runner import _parse_3d_line, Placement
from layoutgpt_bridge.asset_registry import AssetRegistry, RoomEntry, FurnitureEntry


# ===========================================================================
# tag_mapper tests
# ===========================================================================

class TestNormalizeTag:
    def test_sofa_corner(self):
        assert normalize_tag("FURN_sofa_corner_01") == "l_shaped_sofa"

    def test_bed_double(self):
        assert normalize_tag("FURN_bed_double_A") == "double_bed"

    def test_bed_king(self):
        assert normalize_tag("FURN_bed_king_04") == "double_bed"

    def test_nightstand(self):
        assert normalize_tag("FURN_nightstand_02") == "nightstand"

    def test_wardrobe(self):
        assert normalize_tag("FURN_wardrobe_sliding_01") == "wardrobe"

    def test_unknown_tag(self):
        assert normalize_tag("FURN_unknown_object_99") is None

    def test_strips_furn_prefix(self):
        assert normalize_tag("furn_armchair_liv_01") == "armchair"

    def test_case_insensitive(self):
        assert normalize_tag("FURN_COFFEE_TABLE_01") == "coffee_table"

    def test_furniture_prefix(self):
        assert normalize_tag("furniture_tv_stand_liv_01") == "tv_stand"


class TestDenormalizeTag:
    def test_loveseat(self):
        tag = denormalize_tag("loveseat_sofa")
        assert tag in ("sofa_2seater", "sofa_loveseat")

    def test_fallback_to_category(self):
        assert denormalize_tag("mystery_furniture") == "mystery_furniture"


class TestRoomTypeFromName:
    def test_bedroom(self):
        assert room_type_from_name("ROOM_bedroom_01") == "bedroom"

    def test_living(self):
        assert room_type_from_name("ROOM_livingroom_02") == "livingroom"

    def test_dining(self):
        assert room_type_from_name("ROOM_dining_01") == "living room & dining room"

    def test_fallback(self):
        assert room_type_from_name("ROOM_studio_01") == "bedroom"

    def test_boundary_prefix(self):
        assert room_type_from_name("boundary_bedroom") == "bedroom"

    def test_strips_index(self):
        assert room_type_from_name("ROOM_living_03") == "livingroom"


class TestLayoutgptRoomKey:
    def test_bedroom(self):
        assert layoutgpt_room_key("bedroom") == "bedroom"

    def test_livingroom(self):
        assert layoutgpt_room_key("livingroom") == "livingroom"

    def test_dining_maps_to_livingroom(self):
        assert layoutgpt_room_key("living room & dining room") == "livingroom"


class TestAllCategories:
    def test_non_empty(self):
        assert len(ALL_LAYOUTGPT_CATEGORIES) > 20

    def test_sorted(self):
        assert ALL_LAYOUTGPT_CATEGORIES == sorted(ALL_LAYOUTGPT_CATEGORIES)

    def test_known_categories_present(self):
        for cat in ("double_bed", "l_shaped_sofa", "coffee_table", "wardrobe"):
            assert cat in ALL_LAYOUTGPT_CATEGORIES

    def test_no_duplicates(self):
        assert len(ALL_LAYOUTGPT_CATEGORIES) == len(set(ALL_LAYOUTGPT_CATEGORIES))


class TestRoundtrip:
    def test_all_tags_roundtrip(self):
        for tag_fragment, expected_cat in TAG_TO_LAYOUTGPT.items():
            fake_name = f"FURN_{tag_fragment}_01"
            got = normalize_tag(fake_name)
            assert got == expected_cat, \
                f"normalize_tag('{fake_name}') returned {got!r}, expected {expected_cat!r}"


# ===========================================================================
# room_formatter tests
# ===========================================================================

class TestRoomFormatter:
    @pytest.fixture
    def bedroom_room(self):
        return MockRoomInfo.make("ROOM_bedroom_01", length_x=5000.0, length_y=4000.0)

    @pytest.fixture
    def formatter(self, bedroom_room):
        return RoomFormatter(bedroom_room, canvas_px=256)

    def test_condition_prompt_format(self, formatter):
        prompt = formatter.condition_prompt
        assert prompt.startswith("Condition:\n")
        assert "Room Type: bedroom" in prompt
        assert "Room Size: max length" in prompt
        assert "px" in prompt

    def test_room_px_length_longer_axis(self, formatter):
        assert formatter.room_px_length == 256
        assert formatter.room_px_width == int(round(4000 / 5000 * 256))

    def test_from_px_position(self, formatter):
        px = {
            "left": 128.0, "top": 100.0, "depth": 0.0,
            "length": 50.0, "width": 40.0, "height": 30.0,
            "orientation": 45.0,
        }
        scene = formatter.from_px(px)
        scale = 5000 / 256
        assert abs(scene["pos_x"] - 128 * scale) < 0.01
        assert abs(scene["pos_y"] - 100 * scale) < 0.01
        assert scene["rotation_deg"] == 45.0

    def test_to_px_roundtrip(self, formatter):
        pos_x, pos_y, pos_z = 2500.0, 2000.0, 0.0
        dim_x, dim_y, dim_z = 1000.0, 800.0, 500.0
        rot = 30.0
        px = formatter.to_px(pos_x, pos_y, pos_z, dim_x, dim_y, dim_z, rot)
        scene = formatter.from_px(px)
        assert abs(scene["pos_x"] - pos_x) < 0.01
        assert abs(scene["pos_y"] - pos_y) < 0.01
        assert abs(scene["rotation_deg"] - rot) < 0.01

    def test_placement_in_bounds(self, formatter):
        assert formatter.placement_in_bounds({"pos_x": 2500.0, "pos_y": 2000.0})
        assert not formatter.placement_in_bounds({"pos_x": 6000.0, "pos_y": 2000.0})

    def test_zero_extent_raises(self):
        zero_bbox = BoundingBox(0, 0, 0, 0, 0, 2800)
        room = RoomInfo(name="ROOM_bad", room_type="bedroom", bbox=zero_bbox)
        with pytest.raises(ValueError):
            RoomFormatter(room, canvas_px=256)


# ===========================================================================
# asset_registry tests
# ===========================================================================

class TestAssetRegistry:
    @pytest.fixture
    def registry(self):
        r = AssetRegistry()
        r.add_room("Bedroom Spline 01", "bedroom")
        r.add_room("Living Room Spline", "livingroom")
        r.add_furniture("Sofa_Corner_01", "l_shaped_sofa", "all", 1)
        r.add_furniture("Bed_Double_A",   "double_bed",    "Bedroom Spline 01", 1)
        r.add_furniture("Armchair_01",    "armchair",      "Living Room Spline", 2)
        return r

    def test_room_count(self, registry):
        assert len(registry.rooms) == 2

    def test_furniture_count(self, registry):
        assert len(registry.furniture) == 3

    def test_add_room_deduplicates(self, registry):
        registry.add_room("Bedroom Spline 01", "livingroom")
        assert len(registry.rooms) == 2
        assert registry.rooms[0].room_type == "livingroom"

    def test_add_furniture_deduplicates(self, registry):
        registry.add_furniture("Sofa_Corner_01", "multi_seat_sofa", "all", 3)
        assert len(registry.furniture) == 3
        assert registry.furniture[0].category == "multi_seat_sofa"
        assert registry.furniture[0].max_instances == 3

    def test_remove_room(self, registry):
        registry.remove_room("Bedroom Spline 01")
        assert len(registry.rooms) == 1

    def test_remove_furniture(self, registry):
        registry.remove_furniture("Sofa_Corner_01")
        assert len(registry.furniture) == 2

    def test_furniture_for_room_all(self, registry):
        bedroom = registry.rooms[0]
        furn = registry.furniture_for_room(bedroom)
        # Sofa_Corner_01 (all) + Bed_Double_A (bedroom)
        names = {f.node_name for f in furn}
        assert "Sofa_Corner_01" in names
        assert "Bed_Double_A"   in names
        assert "Armchair_01"    not in names

    def test_furniture_for_room_living(self, registry):
        living = registry.rooms[1]
        furn = registry.furniture_for_room(living)
        names = {f.node_name for f in furn}
        assert "Sofa_Corner_01" in names
        assert "Armchair_01"    in names
        assert "Bed_Double_A"   not in names

    def test_build_library_max_instances(self, registry):
        living = registry.rooms[1]
        lib = registry.build_library(living)
        # Armchair has max_instances=2 → 2 entries in library
        assert len(lib.get("armchair", [])) == 2

    def test_available_categories(self, registry):
        bedroom = registry.rooms[0]
        cats = registry.available_categories(bedroom)
        assert "double_bed" in cats
        assert "l_shaped_sofa" in cats

    def test_update_furniture(self, registry):
        ok = registry.update_furniture("Sofa_Corner_01", category="multi_seat_sofa", max_instances=3)
        assert ok
        assert registry.furniture[0].category == "multi_seat_sofa"
        assert registry.furniture[0].max_instances == 3

    def test_update_furniture_not_found(self, registry):
        assert not registry.update_furniture("Ghost Object", category="armchair")

    def test_json_roundtrip(self, registry):
        json_str = registry.to_json()
        restored = AssetRegistry.from_json(json_str)
        assert len(restored.rooms)     == len(registry.rooms)
        assert len(restored.furniture) == len(registry.furniture)
        assert restored.rooms[0].node_name      == "Bedroom Spline 01"
        assert restored.furniture[1].category   == "double_bed"
        assert restored.furniture[2].max_instances == 2

    def test_display_labels(self, registry):
        assert "bedroom" in registry.rooms[0].display_label
        assert "Bedroom Spline 01" in registry.rooms[0].display_label
        # Armchair has max_instances=2 → ×2 in label
        assert "×2" in registry.furniture[2].display_label


# ===========================================================================
# layoutgpt_runner parser tests
# ===========================================================================

class TestParse3DLine:
    def test_valid_line(self):
        line = ("double_bed {length: 170px; width: 130px; height: 57px; "
                "left: 129px; top: 111px; depth: 28px; orientation: 0 degrees;}")
        cat, px = _parse_3d_line(line, unit="px")
        assert cat == "double_bed"
        assert px["length"] == 170.0
        assert px["width"]  == 130.0
        assert px["left"]   == 129.0
        assert px["orientation"] == 0.0

    def test_non_zero_orientation(self):
        line = ("armchair {length: 70px; width: 70px; height: 60px; "
                "left: 40px; top: 160px; depth: 30px; orientation: 90 degrees;}")
        cat, px = _parse_3d_line(line, unit="px")
        assert cat == "armchair"
        assert px["orientation"] == 90.0

    def test_negative_orientation(self):
        line = ("nightstand {length: 40px; width: 40px; height: 45px; "
                "left: 60px; top: 111px; depth: 22px; orientation: -45 degrees;}")
        cat, px = _parse_3d_line(line, unit="px")
        assert px["orientation"] == -45.0

    def test_invalid_line_returns_none(self):
        cat, px = _parse_3d_line("this is not a valid line")
        assert cat is None
        assert px is None

    def test_wrong_field_count(self):
        line = ("sofa {length: 100px; width: 80px; height: 60px; "
                "left: 50px; top: 50px; orientation: 0 degrees;}")
        cat, px = _parse_3d_line(line, unit="px")
        assert cat is None

    def test_meters_unit(self):
        line = ("double_bed {length: 1.70m; width: 1.30m; height: 0.57m; "
                "left: 1.29m; top: 1.11m; depth: 0.28m; orientation: 0 degrees;}")
        cat, px = _parse_3d_line(line, unit="m")
        assert cat == "double_bed"
        assert abs(px["length"] - 1.70) < 1e-6

    def test_category_strips_digits(self):
        line = ("nightstand1 {length: 40px; width: 40px; height: 45px; "
                "left: 216px; top: 111px; depth: 22px; orientation: 180 degrees;}")
        cat, px = _parse_3d_line(line, unit="px")
        assert cat == "nightstand"
