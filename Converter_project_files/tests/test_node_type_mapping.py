"""
Tier 3 — _GODOT_TO_IR_TYPE mapping and _DEFAULT_IR_TYPE.

Tests verify:
  • Known Godot node types map to correct IR types
  • Unknown types fall back to _DEFAULT_IR_TYPE ("group")
  • CollisionShape3D maps to "group" (handled inline on parent, not its own entity)
  • CollisionShape3D parsed via parse_text produces a node of type "group"
  • All listed types in the mapping produce non-empty string values
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from godot_to_unity.godot_scene_parser import _GODOT_TO_IR_TYPE, _DEFAULT_IR_TYPE, GodotSceneParser


# ---------------------------------------------------------------------------
# _DEFAULT_IR_TYPE
# ---------------------------------------------------------------------------

class TestDefaultIrType:
    def test_default_is_group(self):
        assert _DEFAULT_IR_TYPE == "group"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ir_type_for(godot_type: str) -> str:
    """Parse a minimal TSCN and return the top-level node's node_type."""
    tscn = f"[gd_scene format=3]\n\n[node name=\"N\" type=\"{godot_type}\"]\n"
    ir = GodotSceneParser().parse_text(tscn, source_file="Test.tscn")
    return ir["nodes"][0].get("node_type", "")


# ---------------------------------------------------------------------------
# Direct mapping dict lookups
# ---------------------------------------------------------------------------

class TestMappingDictContent:
    def test_node3d_in_map(self):
        assert _GODOT_TO_IR_TYPE["Node3D"] == "group"

    def test_meshinstance3d_in_map(self):
        assert _GODOT_TO_IR_TYPE["MeshInstance3D"] == "entity"

    def test_staticbody3d_in_map(self):
        assert _GODOT_TO_IR_TYPE["StaticBody3D"] == "entity"

    def test_rigidbody3d_in_map(self):
        assert _GODOT_TO_IR_TYPE["RigidBody3D"] == "entity"

    def test_characterbody3d_in_map(self):
        assert _GODOT_TO_IR_TYPE["CharacterBody3D"] == "entity"

    def test_collisionshape3d_is_group_in_map(self):
        assert _GODOT_TO_IR_TYPE["CollisionShape3D"] == "group"

    def test_camera3d_in_map(self):
        assert _GODOT_TO_IR_TYPE["Camera3D"] == "camera"

    def test_directionlight3d_in_map(self):
        assert _GODOT_TO_IR_TYPE["DirectionalLight3D"] == "light"

    def test_omnilight3d_in_map(self):
        assert _GODOT_TO_IR_TYPE["OmniLight3D"] == "light"

    def test_spotlight3d_in_map(self):
        assert _GODOT_TO_IR_TYPE["SpotLight3D"] == "light"

    def test_audiostreamplayer3d_in_map(self):
        assert _GODOT_TO_IR_TYPE["AudioStreamPlayer3D"] == "audio_source"

    def test_gpuparticles3d_in_map(self):
        assert _GODOT_TO_IR_TYPE["GPUParticles3D"] == "particle_system"

    def test_animationplayer_in_map(self):
        assert _GODOT_TO_IR_TYPE["AnimationPlayer"] == "group"

    def test_area3d_in_map(self):
        assert _GODOT_TO_IR_TYPE["Area3D"] == "group"

    def test_all_values_are_strings(self):
        for k, v in _GODOT_TO_IR_TYPE.items():
            assert isinstance(v, str) and v, f"empty value for {k}"

    def test_all_values_are_known_ir_types(self):
        allowed = {"group", "entity", "camera", "light", "audio_source",
                   "particle_system", "text_3d", "ui_canvas", "ui_element",
                   "sprite"}
        for k, v in _GODOT_TO_IR_TYPE.items():
            assert v in allowed, f"unknown IR type '{v}' for {k}"


# ---------------------------------------------------------------------------
# Via parse_text round-trip
# ---------------------------------------------------------------------------

class TestMappingViaParser:
    def test_node3d_maps_to_group(self):
        assert _ir_type_for("Node3D") == "group"

    def test_meshinstance3d_maps_to_entity(self):
        assert _ir_type_for("MeshInstance3D") == "entity"

    def test_staticbody3d_maps_to_entity(self):
        assert _ir_type_for("StaticBody3D") == "entity"

    def test_rigidbody3d_maps_to_entity(self):
        assert _ir_type_for("RigidBody3D") == "entity"

    def test_camera3d_maps_to_camera(self):
        assert _ir_type_for("Camera3D") == "camera"

    def test_directionallight3d_maps_to_light(self):
        assert _ir_type_for("DirectionalLight3D") == "light"

    def test_omnilight3d_maps_to_light(self):
        assert _ir_type_for("OmniLight3D") == "light"

    def test_animationplayer_maps_to_group(self):
        assert _ir_type_for("AnimationPlayer") == "group"

    def test_characterbody3d_maps_to_entity(self):
        assert _ir_type_for("CharacterBody3D") == "entity"

    def test_area3d_maps_to_group(self):
        assert _ir_type_for("Area3D") == "group"

    def test_unknown_type_defaults_to_group(self):
        assert _ir_type_for("SomeFancyCustomNode") == "group"

    def test_audiostream_maps_to_audio_source(self):
        assert _ir_type_for("AudioStreamPlayer3D") == "audio_source"


# ---------------------------------------------------------------------------
# CollisionShape3D special handling
# ---------------------------------------------------------------------------

class TestCollisionShape3DMapping:
    def test_collisionshape3d_maps_to_group_not_entity(self):
        assert _GODOT_TO_IR_TYPE["CollisionShape3D"] == "group"
        assert _GODOT_TO_IR_TYPE["CollisionShape3D"] != "entity"

    def test_collisionshape3d_parser_gives_group(self):
        assert _ir_type_for("CollisionShape3D") == "group"

    def test_collisionshape3d_child_has_colliders_in_its_own_components(self):
        tscn = (
            "[gd_scene load_steps=3 format=3]\n\n"
            "[sub_resource type=\"BoxShape3D\" id=\"1\"]\n"
            "size = Vector3(2, 2, 2)\n\n"
            "[node name=\"Body\" type=\"StaticBody3D\"]\n\n"
            "[node name=\"Shape\" type=\"CollisionShape3D\" parent=\".\"]\n"
            "shape = SubResource(\"1\")\n"
        )
        ir = GodotSceneParser().parse_text(tscn, source_file="Test.tscn")
        body = ir["nodes"][0]
        assert body["node_name"] == "Body"
        shape_children = body.get("children", [])
        assert len(shape_children) == 1
        shape_node = shape_children[0]
        assert shape_node["node_name"] == "Shape"
        assert "colliders" in shape_node.get("components", {}), (
            "CollisionShape3D's colliders component should be on the CollisionShape3D node itself"
        )
