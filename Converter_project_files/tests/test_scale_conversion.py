"""
Tier 3 — scale conversion helpers.

Tests verify:
  • _scale_godot_to_unity() — passthrough: output equals input
  • Scale extracted from parse_text() IR matches original transform values
  • Uniform and non-uniform scales preserved
"""

import sys
import os
import math
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from godot_to_unity.godot_scene_parser import _scale_godot_to_unity, GodotSceneParser


# ---------------------------------------------------------------------------
# _scale_godot_to_unity — passthrough
# ---------------------------------------------------------------------------

class TestScaleGodotToUnityPassthrough:
    def test_identity_scale_unchanged(self):
        assert _scale_godot_to_unity([1.0, 1.0, 1.0]) == [1.0, 1.0, 1.0]

    def test_uniform_scale_unchanged(self):
        result = _scale_godot_to_unity([3.0, 3.0, 3.0])
        assert result == [3.0, 3.0, 3.0]

    def test_non_uniform_scale_unchanged(self):
        result = _scale_godot_to_unity([2.0, 4.0, 0.5])
        assert result == [2.0, 4.0, 0.5]

    def test_zero_scale_unchanged(self):
        result = _scale_godot_to_unity([0.0, 0.0, 0.0])
        assert result == [0.0, 0.0, 0.0]

    def test_fractional_scale_unchanged(self):
        result = _scale_godot_to_unity([0.1, 0.2, 0.3])
        assert abs(result[0] - 0.1) < 1e-9
        assert abs(result[1] - 0.2) < 1e-9
        assert abs(result[2] - 0.3) < 1e-9

    def test_large_scale_unchanged(self):
        result = _scale_godot_to_unity([100.0, 200.0, 300.0])
        assert result == [100.0, 200.0, 300.0]

    def test_returns_list_of_three(self):
        result = _scale_godot_to_unity([1.0, 2.0, 3.0])
        assert isinstance(result, list)
        assert len(result) == 3

    def test_x_component_preserved(self):
        result = _scale_godot_to_unity([7.0, 1.0, 1.0])
        assert result[0] == 7.0

    def test_y_component_preserved(self):
        result = _scale_godot_to_unity([1.0, 5.0, 1.0])
        assert result[1] == 5.0

    def test_z_component_preserved(self):
        result = _scale_godot_to_unity([1.0, 1.0, 9.0])
        assert result[2] == 9.0

    def test_x_y_z_order_unchanged(self):
        inp = [2.0, 3.0, 4.0]
        out = _scale_godot_to_unity(inp)
        assert out[0] == 2.0
        assert out[1] == 3.0
        assert out[2] == 4.0


# ---------------------------------------------------------------------------
# Scale round-trip via parse_text
# ---------------------------------------------------------------------------

class TestScaleFromParser:
    def _parse_scale(self, sx: float, sy: float, sz: float):
        tscn = (
            "[gd_scene format=3]\n\n"
            "[node name=\"Root\" type=\"Node3D\"]\n"
            f"transform = Transform3D({sx}, 0, 0, 0, {sy}, 0, 0, 0, {sz}, 0, 0, 0)\n"
        )
        ir = GodotSceneParser().parse_text(tscn, source_file="Test.tscn")
        return ir["nodes"][0]["transform"]["scale"]

    def test_identity_scale_parsed_correctly(self):
        sc = self._parse_scale(1.0, 1.0, 1.0)
        assert abs(sc[0] - 1.0) < 1e-5
        assert abs(sc[1] - 1.0) < 1e-5
        assert abs(sc[2] - 1.0) < 1e-5

    def test_uniform_scale_3_parsed(self):
        sc = self._parse_scale(3.0, 3.0, 3.0)
        assert abs(sc[0] - 3.0) < 1e-5
        assert abs(sc[1] - 3.0) < 1e-5
        assert abs(sc[2] - 3.0) < 1e-5

    def test_non_uniform_scale_parsed(self):
        sc = self._parse_scale(2.0, 4.0, 0.5)
        assert abs(sc[0] - 2.0) < 1e-5
        assert abs(sc[1] - 4.0) < 1e-5
        assert abs(sc[2] - 0.5) < 1e-4

    def test_scale_is_list_of_three(self):
        sc = self._parse_scale(1.0, 1.0, 1.0)
        assert isinstance(sc, list)
        assert len(sc) == 3

    def test_no_transform_gives_identity_scale(self):
        tscn = "[gd_scene format=3]\n\n[node name=\"Root\" type=\"Node3D\"]\n"
        ir = GodotSceneParser().parse_text(tscn, source_file="Test.tscn")
        sc = ir["nodes"][0]["transform"]["scale"]
        assert abs(sc[0] - 1.0) < 1e-5
        assert abs(sc[1] - 1.0) < 1e-5
        assert abs(sc[2] - 1.0) < 1e-5
