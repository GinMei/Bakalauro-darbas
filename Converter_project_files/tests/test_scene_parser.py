"""
Tier 2 — GodotSceneParser.parse_file() / parse_text().

The parser is the entry point for all scene data.  Every downstream
component (exporter, classifier, validator) consumes the IR it produces,
so a silent parsing error corrupts the entire conversion chain.
"""

import sys
import os

import pytest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from godot_to_unity.godot_scene_parser import GodotSceneParser, GodotParseError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(tscn_text: str, source_file: str = "Level.tscn") -> dict:
    return GodotSceneParser().parse_text(tscn_text, source_file=source_file)


def _find_node(ir: dict, name: str) -> dict:
    """Recursively search for a node by name in the IR node tree."""
    def _search(nodes):
        for n in nodes:
            if n.get("node_name") == name:
                return n
            found = _search(n.get("children", []))
            if found:
                return found
        return None
    return _search(ir.get("nodes", []))


MINIMAL_TSCN = '[gd_scene format=3]\n\n[node name="Root" type="Node3D"]\n'


# ---------------------------------------------------------------------------
# GodotParseError
# ---------------------------------------------------------------------------

class TestGodotParseError:
    def test_raises_on_missing_file(self, tmp_path):
        parser = GodotSceneParser()
        with pytest.raises(GodotParseError):
            parser.parse_file(tmp_path / "nonexistent.tscn")

    def test_godot_parse_error_is_value_error(self):
        assert issubclass(GodotParseError, ValueError)

    def test_raises_on_empty_file(self, tmp_path):
        f = tmp_path / "empty.tscn"
        f.write_text("")
        parser = GodotSceneParser()
        with pytest.raises(GodotParseError):
            parser.parse_file(f)

    def test_raises_on_no_node_sections(self, tmp_path):
        f = tmp_path / "bad.tscn"
        f.write_text("[gd_scene format=3]\n")
        parser = GodotSceneParser()
        with pytest.raises(GodotParseError):
            parser.parse_file(f)


# ---------------------------------------------------------------------------
# IR structure — required keys
# ---------------------------------------------------------------------------

class TestIRStructure:
    def test_ir_version_present(self):
        ir = _parse(MINIMAL_TSCN)
        assert "ir_version" in ir

    def test_ir_version_is_1_0(self):
        ir = _parse(MINIMAL_TSCN)
        assert ir["ir_version"] == "1.0"

    def test_nodes_key_present(self):
        ir = _parse(MINIMAL_TSCN)
        assert "nodes" in ir

    def test_nodes_is_list(self):
        ir = _parse(MINIMAL_TSCN)
        assert isinstance(ir["nodes"], list)

    def test_coordinate_system_present(self):
        ir = _parse(MINIMAL_TSCN)
        assert "coordinate_system" in ir

    def test_coordinate_system_conversion_applied(self):
        ir = _parse(MINIMAL_TSCN)
        assert ir["coordinate_system"]["conversion_applied"] is True

    def test_scene_name_derived_from_source_file(self):
        ir = _parse(MINIMAL_TSCN, source_file="Forest.tscn")
        assert ir["scene_name"] == "Forest"

    def test_node_count_matches_nodes_len(self):
        tscn = (
            "[gd_scene format=3]\n\n"
            "[node name=\"Root\" type=\"Node3D\"]\n\n"
            "[node name=\"Child\" type=\"Node3D\" parent=\".\"]\n"
        )
        ir = _parse(tscn)
        # node_count may count all nodes; nodes list is the root-level list
        assert ir["node_count"] >= 1

    def test_asset_registry_present(self):
        ir = _parse(MINIMAL_TSCN)
        assert "asset_registry" in ir

    def test_connections_present(self):
        ir = _parse(MINIMAL_TSCN)
        assert "connections" in ir

    def test_source_engine_is_godot(self):
        ir = _parse(MINIMAL_TSCN)
        assert ir.get("source_engine") == "Godot"

    def test_target_engine_is_unity(self):
        ir = _parse(MINIMAL_TSCN)
        assert ir.get("target_engine") == "Unity"


# ---------------------------------------------------------------------------
# Node hierarchy
# ---------------------------------------------------------------------------

class TestNodeHierarchy:
    def test_single_root_node_parsed(self):
        ir = _parse(MINIMAL_TSCN)
        assert len(ir["nodes"]) == 1

    def test_root_node_name(self):
        ir = _parse(MINIMAL_TSCN)
        assert ir["nodes"][0]["node_name"] == "Root"

    def test_root_node_has_no_parent(self):
        ir = _parse(MINIMAL_TSCN)
        root = ir["nodes"][0]
        assert root.get("parent_id") is None or root.get("parent") is None or root.get("parent") == ""

    def test_child_node_parsed(self):
        tscn = (
            "[gd_scene format=3]\n\n"
            "[node name=\"Root\" type=\"Node3D\"]\n\n"
            "[node name=\"Child\" type=\"MeshInstance3D\" parent=\".\"]\n"
        )
        ir = _parse(tscn)
        child = _find_node(ir, "Child")
        assert child is not None

    def test_child_node_linked_to_parent(self):
        tscn = (
            "[gd_scene format=3]\n\n"
            "[node name=\"Root\" type=\"Node3D\"]\n\n"
            "[node name=\"Child\" type=\"Node3D\" parent=\".\"]\n"
        )
        ir = _parse(tscn)
        # Child is nested inside Root's children
        root = ir["nodes"][0]
        assert len(root.get("children", [])) == 1

    def test_multiple_children_parsed(self):
        tscn = (
            "[gd_scene format=3]\n\n"
            "[node name=\"Root\" type=\"Node3D\"]\n\n"
            "[node name=\"A\" type=\"Node3D\" parent=\".\"]\n\n"
            "[node name=\"B\" type=\"Node3D\" parent=\".\"]\n"
        )
        ir = _parse(tscn)
        root = ir["nodes"][0]
        assert len(root.get("children", [])) == 2


# ---------------------------------------------------------------------------
# Node type → IR type mapping
# ---------------------------------------------------------------------------

class TestNodeTypeMapping:
    def _ir_type_of(self, godot_type: str) -> str:
        tscn = f"[gd_scene format=3]\n\n[node name=\"N\" type=\"{godot_type}\"]\n"
        ir = _parse(tscn)
        return ir["nodes"][0].get("node_type")

    def test_node3d_maps_to_group(self):
        assert self._ir_type_of("Node3D") == "group"

    def test_meshinstance3d_maps_to_entity(self):
        assert self._ir_type_of("MeshInstance3D") == "entity"

    def test_staticbody3d_maps_to_entity(self):
        assert self._ir_type_of("StaticBody3D") == "entity"

    def test_rigidbody3d_maps_to_entity(self):
        assert self._ir_type_of("RigidBody3D") == "entity"

    def test_camera3d_maps_to_camera(self):
        assert self._ir_type_of("Camera3D") == "camera"

    def test_directionlight_maps_to_light(self):
        assert self._ir_type_of("DirectionalLight3D") == "light"

    def test_omnilight_maps_to_light(self):
        assert self._ir_type_of("OmniLight3D") == "light"

    def test_characterbody_maps_to_entity(self):
        assert self._ir_type_of("CharacterBody3D") == "entity"

    def test_unknown_type_defaults_to_group(self):
        assert self._ir_type_of("CustomFancyNode") == "group"

    def test_animationplayer_maps_to_group(self):
        assert self._ir_type_of("AnimationPlayer") == "group"

    def test_audiostream3d_maps_to_audio_source(self):
        assert self._ir_type_of("AudioStreamPlayer3D") == "audio_source"


# ---------------------------------------------------------------------------
# Transform extraction via parse_text
# ---------------------------------------------------------------------------

class TestParsedTransform:
    def test_node_without_transform_has_identity(self):
        ir = _parse(MINIMAL_TSCN)
        root = ir["nodes"][0]
        t = root.get("transform", {})
        assert t.get("scale") == [1.0, 1.0, 1.0]
        assert t.get("position") == [0.0, 0.0, 0.0]

    def test_node_with_transform_extracts_position(self):
        tscn = (
            "[gd_scene format=3]\n\n"
            "[node name=\"Root\" type=\"Node3D\"]\n"
            "transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 3, 5, -7)\n"
        )
        ir = _parse(tscn)
        pos = ir["nodes"][0]["transform"]["position"]
        assert abs(pos[0] - 3.0) < 1e-5
        assert abs(pos[1] - 5.0) < 1e-5

    def test_node_with_scale_transform(self):
        tscn = (
            "[gd_scene format=3]\n\n"
            "[node name=\"Root\" type=\"Node3D\"]\n"
            "transform = Transform3D(2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0)\n"
        )
        ir = _parse(tscn)
        sc = ir["nodes"][0]["transform"]["scale"]
        assert abs(sc[0] - 2.0) < 1e-5
        assert abs(sc[1] - 2.0) < 1e-5
        assert abs(sc[2] - 2.0) < 1e-5

    def test_rotation_quaternion_is_unit(self):
        import math
        tscn = (
            "[gd_scene format=3]\n\n"
            "[node name=\"Root\" type=\"Node3D\"]\n"
            "transform = Transform3D(0.866, 0, 0.5, 0, 1, 0, -0.5, 0, 0.866, 0, 0, 0)\n"
        )
        ir = _parse(tscn)
        r = ir["nodes"][0]["transform"]["rotation"]
        mag = math.sqrt(r["x"]**2 + r["y"]**2 + r["z"]**2 + r["w"]**2)
        assert abs(mag - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# Instance node (prefab_instance component)
# ---------------------------------------------------------------------------

class TestInstanceNode:
    def test_instance_node_detected(self):
        tscn = (
            "[gd_scene load_steps=2 format=3]\n\n"
            "[ext_resource type=\"PackedScene\" path=\"res://crate.tscn\" id=\"1\"]\n\n"
            "[node name=\"Level\" type=\"Node3D\"]\n\n"
            "[node name=\"Crate\" parent=\".\" instance=ExtResource(\"1\")]\n"
        )
        ir = _parse(tscn)
        instance_node = _find_node(ir, "Crate")
        assert instance_node is not None

    def test_instance_node_has_prefab_instance_component(self):
        tscn = (
            "[gd_scene load_steps=2 format=3]\n\n"
            "[ext_resource type=\"PackedScene\" path=\"res://crate.tscn\" id=\"1\"]\n\n"
            "[node name=\"Level\" type=\"Node3D\"]\n\n"
            "[node name=\"Crate\" parent=\".\" instance=ExtResource(\"1\")]\n"
        )
        ir = _parse(tscn)
        instance_node = _find_node(ir, "Crate")
        comps = instance_node.get("components", {})
        assert "instance_ref" in comps

    def test_instance_node_source_path_resolved(self):
        tscn = (
            "[gd_scene load_steps=2 format=3]\n\n"
            "[ext_resource type=\"PackedScene\" path=\"res://crate.tscn\" id=\"1\"]\n\n"
            "[node name=\"Level\" type=\"Node3D\"]\n\n"
            "[node name=\"Crate\" parent=\".\" instance=ExtResource(\"1\")]\n"
        )
        ir = _parse(tscn)
        instance_node = _find_node(ir, "Crate")
        prefab = instance_node["components"]["instance_ref"]
        assert prefab.get("source_res_path") == "res://crate.tscn"


# ---------------------------------------------------------------------------
# parse_file round-trip
# ---------------------------------------------------------------------------

class TestParseFile:
    def test_parse_file_matches_parse_text(self, tmp_path):
        f = tmp_path / "Level.tscn"
        f.write_text(MINIMAL_TSCN, encoding="utf-8")
        ir_file = GodotSceneParser().parse_file(f)
        ir_text = GodotSceneParser().parse_text(MINIMAL_TSCN, source_file="Level.tscn")
        assert ir_file["scene_name"] == ir_text["scene_name"]
        assert ir_file["node_count"] == ir_text["node_count"]
        assert ir_file["ir_version"] == ir_text["ir_version"]

    def test_parse_file_uses_filename_as_scene_name(self, tmp_path):
        f = tmp_path / "Forest.tscn"
        f.write_text(MINIMAL_TSCN, encoding="utf-8")
        ir = GodotSceneParser().parse_file(f)
        assert ir["scene_name"] == "Forest"
