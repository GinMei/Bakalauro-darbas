"""
Tests for godot_exporter.py — _ir_to_tscn_text() and save_tscn() with
trimesh mocked so no heavy dependency is needed.
"""

import sys
import os
import types
from unittest.mock import MagicMock, patch
from pathlib import Path

# ---------------------------------------------------------------------------
# Stub trimesh before importing godot_exporter
# ---------------------------------------------------------------------------

def _stub_trimesh():
    if "trimesh" not in sys.modules:
        trimesh = types.ModuleType("trimesh")
        trimesh.Scene = MagicMock
        trimesh.Trimesh = MagicMock
        trimesh.load = MagicMock(return_value=MagicMock())
        trimesh.transformations = MagicMock()
        trimesh.transformations.translation_matrix = MagicMock(return_value=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

        creation = types.ModuleType("trimesh.creation")
        creation.box = MagicMock(return_value=MagicMock())
        trimesh.creation = creation
        sys.modules["trimesh"] = trimesh
        sys.modules["trimesh.creation"] = creation

_stub_trimesh()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unity_to_godot.godot_exporter import _ir_to_tscn_text, save_tscn, export_gltf


# ---------------------------------------------------------------------------
# _ir_to_tscn_text — pure Python, no trimesh needed
# ---------------------------------------------------------------------------

class TestIrToTscnText:
    def _simple_ir(self, nodes=None):
        return {"nodes": nodes or []}

    def test_returns_string(self):
        result = _ir_to_tscn_text(self._simple_ir())
        assert isinstance(result, str)

    def test_has_gd_scene_header(self):
        result = _ir_to_tscn_text(self._simple_ir())
        assert "[gd_scene" in result

    def test_has_root_node(self):
        result = _ir_to_tscn_text(self._simple_ir())
        assert 'name="Root"' in result

    def test_node_name_in_output(self):
        nodes = [{"name": "MyObject", "type": "Node3D"}]
        result = _ir_to_tscn_text({"nodes": nodes})
        assert "MyObject" in result

    def test_node_type_in_output(self):
        nodes = [{"name": "Camera", "type": "Camera3D"}]
        result = _ir_to_tscn_text({"nodes": nodes})
        assert "Camera3D" in result

    def test_transform3d_for_spatial_node(self):
        nodes = [{"name": "Cube", "type": "Node3D",
                  "transform": {"position": [1.0, 2.0, 3.0]}}]
        result = _ir_to_tscn_text({"nodes": nodes})
        assert "Transform3D" in result

    def test_position_values_in_output(self):
        nodes = [{"name": "Cube", "type": "Node3D",
                  "transform": {"position": [1.0, 2.0, 3.0]}}]
        result = _ir_to_tscn_text({"nodes": nodes})
        assert "1.0000" in result or "1.000" in result

    def test_mesh_null_for_mesh_instance(self):
        nodes = [{"name": "Mesh", "type": "MeshInstance3D"}]
        result = _ir_to_tscn_text({"nodes": nodes})
        assert "mesh = null" in result

    def test_camera_current_true(self):
        nodes = [{"name": "Cam", "type": "Camera3D"}]
        result = _ir_to_tscn_text({"nodes": nodes})
        assert "current = true" in result

    def test_light_energy_for_directional(self):
        nodes = [{"name": "Sun", "type": "DirectionalLight3D"}]
        result = _ir_to_tscn_text({"nodes": nodes})
        assert "light_energy" in result

    def test_light_energy_for_omni(self):
        nodes = [{"name": "Lamp", "type": "OmniLight3D"}]
        result = _ir_to_tscn_text({"nodes": nodes})
        assert "light_energy" in result

    def test_audio_autoplay_false(self):
        nodes = [{"name": "Sound", "type": "AudioStreamPlayer3D"}]
        result = _ir_to_tscn_text({"nodes": nodes})
        assert "autoplay = false" in result

    def test_reflection_probe_intensity(self):
        nodes = [{"name": "Probe", "type": "ReflectionProbe"}]
        result = _ir_to_tscn_text({"nodes": nodes})
        assert "intensity = 1.0" in result

    def test_ui_control_uses_vector2(self):
        nodes = [{"name": "Btn", "type": "Button",
                  "transform": {"position": [10.0, 20.0, 0.0]}}]
        result = _ir_to_tscn_text({"nodes": nodes})
        assert "Vector2" in result

    def test_sprite2d_uses_vector2(self):
        nodes = [{"name": "Spr", "type": "Sprite2D",
                  "transform": {"position": [5.0, 10.0, 0.0]}}]
        result = _ir_to_tscn_text({"nodes": nodes})
        assert "Vector2" in result

    def test_nested_children_included(self):
        nodes = [{"name": "Parent", "type": "Node3D",
                  "children": [{"name": "Child", "type": "Node3D"}]}]
        result = _ir_to_tscn_text({"nodes": nodes})
        assert "Child" in result

    def test_empty_nodes_list(self):
        result = _ir_to_tscn_text({"nodes": []})
        assert "[gd_scene" in result

    def test_output_ends_with_newline(self):
        result = _ir_to_tscn_text(self._simple_ir())
        assert result.endswith("\n")


# ---------------------------------------------------------------------------
# save_tscn — with trimesh mocked
# ---------------------------------------------------------------------------

class TestSaveTscn:
    def test_writes_tscn_file(self, tmp_path):
        ir = {"nodes": [{"name": "Root", "type": "Node3D"}]}
        tscn = tmp_path / "scene.tscn"
        glb = tmp_path / "scene.glb"
        save_tscn(ir, tscn, glb_path=glb)
        assert tscn.exists()

    def test_tscn_content_has_yaml_like_header(self, tmp_path):
        ir = {"nodes": []}
        tscn = tmp_path / "scene.tscn"
        glb = tmp_path / "scene.glb"
        save_tscn(ir, tscn, glb_path=glb)
        content = tscn.read_text(encoding="utf-8")
        assert "[gd_scene" in content

    def test_default_glb_path(self, tmp_path):
        ir = {"nodes": []}
        tscn = tmp_path / "scene.tscn"
        save_tscn(ir, tscn)
