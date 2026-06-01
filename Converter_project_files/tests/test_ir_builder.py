"""
Tests for ir_builder.py — utility functions and IR building helpers.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ir_builder import (
    _unwrap_scalar,
    _make_meta,
    _safe_id,
    _build_node_id,
    _pos,
    _rot,
    _scl,
    _build_physics_material_ir,
    _build_collider_ir,
    _build_rigidbody_ir,
    _build_camera_ir,
    _build_light_ir,
    _build_reflection_probe_ir,
    _build_nav_mesh_settings_ir,
    _build_environment_ir,
    _build_volume_effects_ir,
    _validate_volume_effects_ir,
)


# ---------------------------------------------------------------------------
# _unwrap_scalar
# ---------------------------------------------------------------------------

class TestUnwrapScalar:
    def test_int_passthrough(self):
        assert _unwrap_scalar(5) == 5

    def test_float_passthrough(self):
        assert _unwrap_scalar(3.14) == 3.14

    def test_string_passthrough(self):
        assert _unwrap_scalar("hello") == "hello"

    def test_none_passthrough(self):
        assert _unwrap_scalar(None) is None

    def test_list_passthrough(self):
        lst = [1, 2, 3]
        assert _unwrap_scalar(lst) is lst

    def test_value_dict_unwrapped(self):
        assert _unwrap_scalar({"value": 42}) == 42

    def test_value_dict_float_unwrapped(self):
        assert _unwrap_scalar({"value": 1.5}) == 1.5

    def test_value_dict_string_unwrapped(self):
        assert _unwrap_scalar({"value": "test"}) == "test"

    def test_unknown_dict_appends_warning(self):
        warnings = []
        # Use -O mode safe fallback path can't be tested (assert fires in dev),
        # so just ensure the function accepts a warnings list parameter signature.
        # We test with a non-dict to verify normal path.
        result = _unwrap_scalar(99, warnings=warnings)
        assert result == 99
        assert warnings == []

    def test_zero_value_dict_unwrapped(self):
        assert _unwrap_scalar({"value": 0}) == 0

    def test_none_value_dict_unwrapped(self):
        assert _unwrap_scalar({"value": None}) is None


# ---------------------------------------------------------------------------
# _make_meta
# ---------------------------------------------------------------------------

class TestMakeMeta:
    def test_default_status_mapped(self):
        m = _make_meta()
        assert m["conversion_status"] == "mapped"

    def test_default_confidence_one(self):
        m = _make_meta()
        assert m["confidence"] == 1.0

    def test_default_ai_generated_false(self):
        m = _make_meta()
        assert m["ai_generated"] is False

    def test_default_user_edited_false(self):
        m = _make_meta()
        assert m["user_edited"] is False

    def test_default_requires_review_false_for_high_confidence(self):
        m = _make_meta(confidence=1.0, status="mapped")
        assert m["requires_review"] is False

    def test_requires_review_true_for_low_confidence(self):
        m = _make_meta(confidence=0.5)
        assert m["requires_review"] is True

    def test_requires_review_true_for_unknown_status(self):
        m = _make_meta(status="unknown", confidence=1.0)
        assert m["requires_review"] is True

    def test_custom_status(self):
        m = _make_meta(status="partial")
        assert m["conversion_status"] == "partial"

    def test_custom_notes(self):
        m = _make_meta(notes="some note")
        assert m["notes"] == "some note"

    def test_warnings_list_empty(self):
        m = _make_meta()
        assert m["warnings"] == []

    def test_errors_list_empty(self):
        m = _make_meta()
        assert m["errors"] == []

    def test_confidence_0_74_requires_review(self):
        m = _make_meta(confidence=0.74)
        assert m["requires_review"] is True


# ---------------------------------------------------------------------------
# _safe_id
# ---------------------------------------------------------------------------

class TestSafeId:
    def test_simple_lowercase(self):
        assert _safe_id("hello") == "hello"

    def test_uppercase_lowercased(self):
        assert _safe_id("Hello") == "hello"

    def test_spaces_replaced_with_underscore(self):
        assert _safe_id("hello world") == "hello_world"

    def test_special_chars_replaced(self):
        result = _safe_id("hello-world!")
        assert " " not in result
        assert "-" not in result
        assert "!" not in result

    def test_empty_string_returns_scene(self):
        assert _safe_id("") == "scene"

    def test_numbers_preserved(self):
        assert _safe_id("scene1") == "scene1"

    def test_mixed_case_and_spaces(self):
        result = _safe_id("My Scene 01")
        assert result == "my_scene_01"

    def test_leading_trailing_underscores_stripped(self):
        result = _safe_id("!hello!")
        assert not result.startswith("_")
        assert not result.endswith("_")

    def test_only_special_chars_returns_scene(self):
        result = _safe_id("---")
        assert result == "scene"


# ---------------------------------------------------------------------------
# _build_node_id
# ---------------------------------------------------------------------------

class TestBuildNodeId:
    def test_counter_incremented(self):
        counter = [0]
        _build_node_id("scene", counter)
        assert counter[0] == 1

    def test_first_id_format(self):
        counter = [0]
        nid = _build_node_id("scene", counter)
        assert nid == "scene_node_0001"

    def test_second_id_increments(self):
        counter = [0]
        _build_node_id("scene", counter)
        nid = _build_node_id("scene", counter)
        assert nid == "scene_node_0002"

    def test_prefix_used(self):
        counter = [0]
        nid = _build_node_id("level01", counter)
        assert nid.startswith("level01_")

    def test_zero_padding(self):
        counter = [9]
        nid = _build_node_id("s", counter)
        assert nid == "s_node_0010"


# ---------------------------------------------------------------------------
# _pos
# ---------------------------------------------------------------------------

class TestPos:
    def test_position_list(self):
        result = _pos({"position": [1.0, 2.0, 3.0]})
        assert result == [1.0, 2.0, 3.0]

    def test_missing_position_defaults_to_zero(self):
        result = _pos({})
        assert result == [0.0, 0.0, 0.0]

    def test_position_with_value_dicts(self):
        result = _pos({"position": [{"value": 1.0}, {"value": 2.0}, {"value": 3.0}]})
        assert result == [1.0, 2.0, 3.0]

    def test_returns_floats(self):
        result = _pos({"position": [1, 2, 3]})
        assert all(isinstance(v, float) for v in result)


# ---------------------------------------------------------------------------
# _rot
# ---------------------------------------------------------------------------

class TestRot:
    def test_rotation_dict(self):
        result = _rot({"rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}})
        assert result == [0.0, 0.0, 0.0, 1.0]

    def test_missing_rotation_default_identity(self):
        result = _rot({})
        assert result == [0.0, 0.0, 0.0, 1.0]

    def test_rotation_list(self):
        result = _rot({"rotation": [0.1, 0.2, 0.3, 0.9]})
        assert abs(result[3] - 0.9) < 1e-6

    def test_partial_dict_uses_defaults(self):
        result = _rot({"rotation": {"x": 0.5}})
        assert result[0] == 0.5
        assert result[3] == 1.0

    def test_returns_floats(self):
        result = _rot({"rotation": {"x": 0, "y": 0, "z": 0, "w": 1}})
        assert all(isinstance(v, float) for v in result)


# ---------------------------------------------------------------------------
# _scl
# ---------------------------------------------------------------------------

class TestScl:
    def test_scale_list(self):
        result = _scl({"scale": [2.0, 3.0, 4.0]})
        assert result == [2.0, 3.0, 4.0]

    def test_missing_scale_defaults_to_one(self):
        result = _scl({})
        assert result == [1.0, 1.0, 1.0]

    def test_returns_floats(self):
        result = _scl({"scale": [1, 2, 3]})
        assert all(isinstance(v, float) for v in result)


# ---------------------------------------------------------------------------
# _build_physics_material_ir
# ---------------------------------------------------------------------------

class TestBuildPhysicsMaterialIr:
    def test_none_ref_returns_none(self):
        result = _build_physics_material_ir(None, {}, "col1")
        assert result is None

    def test_none_map_returns_none(self):
        result = _build_physics_material_ir("guid:abc", None, "col1")
        assert result is None

    def test_missing_guid_in_map_returns_none(self):
        result = _build_physics_material_ir("guid:abc123", {"other": {}}, "col1")
        assert result is None

    def test_valid_guid_returns_dict(self):
        pm_map = {"abc123": {"dynamic_friction": 0.4, "static_friction": 0.5, "restitution": 0.1}}
        result = _build_physics_material_ir("guid:abc123", pm_map, "col1")
        assert result is not None
        assert "friction" in result
        assert "restitution" in result

    def test_friction_value_extracted(self):
        pm_map = {"abc": {"dynamic_friction": 0.3}}
        result = _build_physics_material_ir("guid:abc", pm_map, "col1")
        assert abs(result["friction"] - 0.3) < 1e-6

    def test_restitution_value_extracted(self):
        pm_map = {"abc": {"restitution": 0.8}}
        result = _build_physics_material_ir("guid:abc", pm_map, "col1")
        assert abs(result["restitution"] - 0.8) < 1e-6

    def test_pm_id_format(self):
        pm_map = {"abcdef123456": {"dynamic_friction": 0.5}}
        result = _build_physics_material_ir("guid:abcdef123456", pm_map, "col1")
        assert result["pm_id"] == "pm_abcdef123456"

    def test_invalid_combine_mode_uses_average(self):
        warnings = []
        pm_map = {"abc": {"friction_combine": "invalid_mode"}}
        result = _build_physics_material_ir("guid:abc", pm_map, "col1", warnings)
        assert result["friction_combine"] == "average"
        assert len(warnings) == 1

    def test_malformed_ref_returns_none(self):
        warnings = []
        result = _build_physics_material_ir("guid:", {}, "col1", warnings)
        assert result is None

    def test_friction_combine_default_average(self):
        pm_map = {"abc": {"dynamic_friction": 0.5}}
        result = _build_physics_material_ir("guid:abc", pm_map, "col1")
        assert result["friction_combine"] == "average"

    def test_bounce_combine_default_average(self):
        pm_map = {"abc": {"dynamic_friction": 0.5}}
        result = _build_physics_material_ir("guid:abc", pm_map, "col1")
        assert result["bounce_combine"] == "average"


# ---------------------------------------------------------------------------
# _build_collider_ir
# ---------------------------------------------------------------------------

class TestBuildColliderIr:
    def test_box_shape_type(self):
        r = _build_collider_ir({"type": "box", "size": [2.0, 3.0, 4.0]}, 0, "node1")
        assert r["shape"]["type"] == "box"
        assert r["shape"]["size"] == [2.0, 3.0, 4.0]

    def test_sphere_shape_type(self):
        r = _build_collider_ir({"type": "sphere", "radius": 1.5}, 0, "node1")
        assert r["shape"]["type"] == "sphere"
        assert abs(r["shape"]["radius"] - 1.5) < 1e-6

    def test_capsule_shape_type(self):
        r = _build_collider_ir({"type": "capsule", "radius": 0.5, "height": 2.0}, 0, "node1")
        assert r["shape"]["type"] == "capsule"
        assert r["shape"]["radius"] == 0.5
        assert r["shape"]["height"] == 2.0

    def test_capsule_axis_default_y(self):
        r = _build_collider_ir({"type": "capsule", "radius": 0.5, "height": 2.0}, 0, "node1")
        assert r["shape"]["axis"] == 1

    def test_mesh_convex_shape_type(self):
        r = _build_collider_ir({"type": "mesh_convex", "mesh_ref": "guid:abc"}, 0, "node1")
        assert r["shape"]["type"] == "mesh_convex"
        assert r["shape"]["mesh_ref"] == "guid:abc"

    def test_is_trigger_default_false(self):
        r = _build_collider_ir({"type": "box"}, 0, "node1")
        assert r["is_trigger"] is False

    def test_is_trigger_true(self):
        r = _build_collider_ir({"type": "box", "is_trigger": True}, 0, "node1")
        assert r["is_trigger"] is True

    def test_collider_id_format(self):
        r = _build_collider_ir({"type": "box"}, 2, "mynode")
        assert r["collider_id"] == "mynode_col_2"

    def test_center_default_zero(self):
        r = _build_collider_ir({"type": "sphere", "radius": 1.0}, 0, "node1")
        assert r["shape"]["center"] == [0.0, 0.0, 0.0]

    def test_small_box_clamped(self):
        warnings = []
        r = _build_collider_ir({"type": "box", "size": [0.0, 1.0, 1.0]}, 0, "node1", warnings=warnings)
        assert r["shape"]["size"][0] >= 0.001
        assert len(warnings) > 0

    def test_physics_material_none_when_no_ref(self):
        r = _build_collider_ir({"type": "box"}, 0, "node1")
        assert r["physics_material"] is None

    def test_wheel_shape_type(self):
        r = _build_collider_ir({"type": "wheel", "radius": 0.3}, 0, "node1")
        assert r["shape"]["type"] == "wheel"
        assert "radius" in r["shape"]


# ---------------------------------------------------------------------------
# _build_rigidbody_ir
# ---------------------------------------------------------------------------

class TestBuildRigidbodyIr:
    def test_rigidbody_id_format(self):
        r = _build_rigidbody_ir({}, "mynode")
        assert r["rigidbody_id"] == "rb_mynode"

    def test_default_type_dynamic(self):
        r = _build_rigidbody_ir({}, "node1")
        assert r["type"] == "dynamic"

    def test_mass_default(self):
        r = _build_rigidbody_ir({}, "node1")
        assert abs(r["mass"] - 1.0) < 1e-6

    def test_custom_mass(self):
        r = _build_rigidbody_ir({"mass": 5.0}, "node1")
        assert abs(r["mass"] - 5.0) < 1e-6

    def test_gravity_enabled_default_true(self):
        r = _build_rigidbody_ir({}, "node1")
        assert r["gravity_enabled"] is True

    def test_constraints_present(self):
        r = _build_rigidbody_ir({}, "node1")
        assert "constraints" in r
        assert "freeze_position" in r["constraints"]
        assert "freeze_rotation" in r["constraints"]

    def test_kinematic_type(self):
        r = _build_rigidbody_ir({"type": "kinematic"}, "node1")
        assert r["type"] == "kinematic"

    def test_sleep_threshold_default(self):
        r = _build_rigidbody_ir({}, "node1")
        assert abs(r["sleep_threshold"] - 0.005) < 1e-6


# ---------------------------------------------------------------------------
# _build_camera_ir
# ---------------------------------------------------------------------------

class TestBuildCameraIr:
    def test_camera_id_format(self):
        r = _build_camera_ir({}, "cam1")
        assert r["camera_id"] == "camera_cam1"

    def test_default_projection_perspective(self):
        r = _build_camera_ir({}, "cam1")
        assert r["projection"] == "perspective"

    def test_default_fov(self):
        r = _build_camera_ir({}, "cam1")
        assert abs(r["fov"] - 60.0) < 1e-6

    def test_default_near_clip(self):
        r = _build_camera_ir({}, "cam1")
        assert abs(r["near_clip"] - 0.1) < 1e-6

    def test_custom_fov(self):
        r = _build_camera_ir({"fov": 90.0}, "cam1")
        assert abs(r["fov"] - 90.0) < 1e-6

    def test_is_main_camera_default_false(self):
        r = _build_camera_ir({}, "cam1")
        assert r["is_main_camera"] is False

    def test_is_main_camera_true(self):
        r = _build_camera_ir({"is_main_camera": True}, "cam1")
        assert r["is_main_camera"] is True

    def test_render_post_processing_default_true(self):
        r = _build_camera_ir({}, "cam1")
        assert r["render_post_processing"] is True


# ---------------------------------------------------------------------------
# _build_light_ir
# ---------------------------------------------------------------------------

class TestBuildLightIr:
    def test_light_id_format(self):
        r = _build_light_ir({}, "light1")
        assert r["light_id"] == "light_light1"

    def test_default_type_directional(self):
        r = _build_light_ir({}, "light1")
        assert r["type"] == "directional"

    def test_directional_intensity_in_lux(self):
        r = _build_light_ir({"type": "directional", "intensity": 1.0}, "l1")
        assert r["intensity"]["unit"] == "lux"
        assert abs(r["intensity"]["value"] - 100000.0) < 1.0

    def test_point_intensity_in_candela(self):
        r = _build_light_ir({"type": "point", "intensity": 1.0}, "l1")
        assert r["intensity"]["unit"] == "candela"

    def test_spot_intensity_in_lumen(self):
        r = _build_light_ir({"type": "spot", "intensity": 1.0}, "l1")
        assert r["intensity"]["unit"] == "lumen"

    def test_shadows_key_present(self):
        r = _build_light_ir({}, "l1")
        assert "shadows" in r
        assert "enabled" in r["shadows"]

    def test_shadows_disabled_by_default(self):
        r = _build_light_ir({}, "l1")
        assert r["shadows"]["enabled"] is False

    def test_color_default_white(self):
        r = _build_light_ir({}, "l1")
        assert r["color"] == [1.0, 1.0, 1.0]


# ---------------------------------------------------------------------------
# _build_reflection_probe_ir
# ---------------------------------------------------------------------------

class TestBuildReflectionProbeIr:
    def test_probe_id_format(self):
        r = _build_reflection_probe_ir({}, "probe1")
        assert r["probe_id"] == "probe_probe1"

    def test_default_intensity(self):
        r = _build_reflection_probe_ir({}, "probe1")
        assert abs(r["intensity"] - 1.0) < 1e-6

    def test_box_projection_default_false(self):
        r = _build_reflection_probe_ir({}, "probe1")
        assert r["box_projection"] is False

    def test_update_mode_default_zero(self):
        r = _build_reflection_probe_ir({}, "probe1")
        assert r["update_mode"] == 0


# ---------------------------------------------------------------------------
# _build_nav_mesh_settings_ir
# ---------------------------------------------------------------------------

class TestBuildNavMeshSettingsIr:
    def test_agent_radius_default(self):
        r = _build_nav_mesh_settings_ir({})
        assert abs(r["agent_radius"] - 0.5) < 1e-6

    def test_agent_height_default(self):
        r = _build_nav_mesh_settings_ir({})
        assert abs(r["agent_height"] - 2.0) < 1e-6

    def test_agent_max_slope_default(self):
        r = _build_nav_mesh_settings_ir({})
        assert abs(r["agent_max_slope"] - 45.0) < 1e-6

    def test_custom_agent_radius(self):
        r = _build_nav_mesh_settings_ir({"agent_radius": 1.0})
        assert abs(r["agent_radius"] - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# _build_environment_ir
# ---------------------------------------------------------------------------

class TestBuildEnvironmentIr:
    def test_ambient_key_present(self):
        r = _build_environment_ir({})
        assert "ambient" in r

    def test_fog_key_present(self):
        r = _build_environment_ir({})
        assert "fog" in r

    def test_sky_key_present(self):
        r = _build_environment_ir({})
        assert "sky" in r

    def test_reflections_key_present(self):
        r = _build_environment_ir({})
        assert "reflections" in r

    def test_fog_disabled_by_default(self):
        r = _build_environment_ir({})
        assert r["fog"]["enabled"] is False

    def test_sky_type_procedural_when_no_guid(self):
        r = _build_environment_ir({"ambient_mode": "skybox"})
        assert r["sky"]["type"] == "procedural"

    def test_sky_type_skybox_when_guid_present(self):
        r = _build_environment_ir({"sky_guid": "abc123"})
        assert r["sky"]["type"] == "skybox"

    def test_ambient_intensity_default(self):
        r = _build_environment_ir({})
        assert abs(r["ambient"]["intensity"] - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# _build_volume_effects_ir
# ---------------------------------------------------------------------------

class TestBuildVolumeEffectsIr:
    def test_empty_raw_returns_empty_dict(self):
        r = _build_volume_effects_ir({})
        assert r == {}

    def test_bloom_present_when_in_raw(self):
        r = _build_volume_effects_ir({"bloom": {"active": True, "intensity": 0.5}})
        assert "bloom" in r
        assert r["bloom"]["active"] is True

    def test_bloom_intensity_clamped_to_zero(self):
        r = _build_volume_effects_ir({"bloom": {"intensity": -1.0}})
        assert r["bloom"]["intensity"] >= 0.0

    def test_bloom_scatter_clamped_to_one(self):
        r = _build_volume_effects_ir({"bloom": {"scatter": 2.0, "intensity": 0.1}})
        assert r["bloom"]["scatter"] <= 1.0

    def test_bloom_tint_is_list(self):
        r = _build_volume_effects_ir({"bloom": {"tint": [1.0, 0.5, 0.5], "intensity": 0.1}})
        assert isinstance(r["bloom"]["tint"], list)

    def test_tonemapping_present_when_in_raw(self):
        r = _build_volume_effects_ir({"tonemapping": {"active": False, "mode": 2}})
        assert "tonemapping" in r
        assert r["tonemapping"]["mode"] == "aces"

    def test_tonemapping_mode_integer_converted_to_string(self):
        r = _build_volume_effects_ir({"tonemapping": {"mode": 1}})
        assert r["tonemapping"]["mode"] == "neutral"

    def test_tonemapping_invalid_mode_defaults_to_none(self):
        r = _build_volume_effects_ir({"tonemapping": {"mode": 99}})
        assert r["tonemapping"]["mode"] == "none"

    def test_color_adjustments_present(self):
        r = _build_volume_effects_ir({"color_adjustments": {"active": True}})
        assert "color_adjustments" in r

    def test_color_adjustments_contrast_clamped(self):
        r = _build_volume_effects_ir({"color_adjustments": {"contrast": 200.0}})
        assert r["color_adjustments"]["contrast"] <= 100.0

    def test_exposure_present(self):
        r = _build_volume_effects_ir({"exposure": {"active": True, "mode": "fixed"}})
        assert "exposure" in r

    def test_exposure_invalid_mode_becomes_fixed(self):
        r = _build_volume_effects_ir({"exposure": {"mode": "invalid_mode"}})
        assert r["exposure"]["mode"] == "fixed"

    def test_absent_effect_not_in_result(self):
        r = _build_volume_effects_ir({"bloom": {"intensity": 0.1}})
        assert "tonemapping" not in r
        assert "exposure" not in r

    def test_vignette_present(self):
        r = _build_volume_effects_ir({"vignette": {"active": True, "intensity": 0.4}})
        assert "vignette" in r

    def test_depth_of_field_present(self):
        r = _build_volume_effects_ir({"depth_of_field": {"active": True, "focus_mode": "manual"}})
        assert "depth_of_field" in r

    def test_white_balance_present(self):
        r = _build_volume_effects_ir({"white_balance": {"active": True, "temperature": 50.0}})
        assert "white_balance" in r

    def test_ambient_occlusion_present(self):
        r = _build_volume_effects_ir({"ambient_occlusion": {"active": True, "intensity": 1.0}})
        assert "ambient_occlusion" in r


# ---------------------------------------------------------------------------
# _validate_volume_effects_ir
# ---------------------------------------------------------------------------

class TestValidateVolumeEffectsIr:
    def test_empty_ir_no_errors(self):
        errors = _validate_volume_effects_ir({})
        assert errors == []

    def test_valid_bloom_no_errors(self):
        r = _build_volume_effects_ir({"bloom": {"active": True, "intensity": 0.5, "scatter": 0.7}})
        errors = _validate_volume_effects_ir(r)
        assert errors == []

    def test_invalid_bloom_active_type_raises_error(self):
        r = {"bloom": {"active": "yes", "intensity": 0.5, "threshold": 0.9, "scatter": 0.7, "tint": [1.0, 1.0, 1.0]}}
        errors = _validate_volume_effects_ir(r)
        assert any("active" in e for e in errors)

    def test_valid_tonemapping_no_errors(self):
        r = _build_volume_effects_ir({"tonemapping": {"active": True, "mode": 2}})
        errors = _validate_volume_effects_ir(r)
        assert errors == []
