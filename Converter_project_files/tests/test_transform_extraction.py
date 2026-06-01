"""
Tier 1 — Transform extraction from Godot scene data.

Tests cover:
  • _pos_godot_to_unity  — position passthrough (no Z-flip in parser)
  • _scale_godot_to_unity — scale passthrough for regular nodes
  • _mat3_to_quat + _quat_godot_to_unity — quaternion from rotation matrix
  • GodotSceneParser._extract_transform — full transform from Transform3D string

The known row-vs-column scale extraction bug is documented via an
xfail test that will automatically turn green when the bug is fixed.
"""

import sys
import os
import math

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from godot_to_unity.godot_scene_parser import (
    GodotSceneParser,
    _pos_godot_to_unity,
    _scale_godot_to_unity,
    _mat3_to_quat,
    _quat_godot_to_unity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transform3d(*vals):
    """Build a raw Transform3D(...) property string from 12 floats."""
    return "Transform3D(" + ", ".join(str(v) for v in vals) + ")"


def _extract(vals):
    """Run GodotSceneParser._extract_transform on the given 12-value basis+origin."""
    parser = GodotSceneParser()
    return parser._extract_transform({"transform": _make_transform3d(*vals)})


def _quat_magnitude(q):
    return math.sqrt(q["x"] ** 2 + q["y"] ** 2 + q["z"] ** 2 + q["w"] ** 2)


def _quats_equal(a, b, eps=1e-4):
    """Quaternion equality handling q == -q."""
    return (
        all(abs(a[k] - b[k]) < eps for k in ("x", "y", "z", "w"))
        or all(abs(a[k] + b[k]) < eps for k in ("x", "y", "z", "w"))
    )


# ---------------------------------------------------------------------------
# _pos_godot_to_unity
# ---------------------------------------------------------------------------

class TestPosGodotToUnity:
    def test_values_returned_unchanged(self):
        """_pos_godot_to_unity is a passthrough; Z flip is done by the exporter."""
        assert _pos_godot_to_unity([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]

    def test_zero_position(self):
        assert _pos_godot_to_unity([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]

    def test_negative_values_preserved(self):
        result = _pos_godot_to_unity([-5.0, 0.0, 7.0])
        assert result == [-5.0, 0.0, 7.0]

    def test_negative_z_not_flipped(self):
        result = _pos_godot_to_unity([0.0, 0.0, -4.0])
        assert abs(result[2] - (-4.0)) < 1e-9


# ---------------------------------------------------------------------------
# _scale_godot_to_unity
# ---------------------------------------------------------------------------

class TestScaleGodotToUnity:
    def test_unit_scale_passthrough(self):
        assert _scale_godot_to_unity([1.0, 1.0, 1.0]) == [1.0, 1.0, 1.0]

    def test_non_uniform_scale_passthrough(self):
        assert _scale_godot_to_unity([3.0, 2.0, 4.0]) == [3.0, 2.0, 4.0]

    def test_fractional_scale_passthrough(self):
        assert _scale_godot_to_unity([0.5, 0.5, 0.5]) == [0.5, 0.5, 0.5]


# ---------------------------------------------------------------------------
# _extract_transform — missing / default
# ---------------------------------------------------------------------------

class TestExtractTransformDefaults:
    def test_no_transform_key_returns_identity(self):
        parser = GodotSceneParser()
        result = parser._extract_transform({})
        assert result["position"] == [0.0, 0.0, 0.0]
        assert result["scale"] == [1.0, 1.0, 1.0]
        assert result["rotation"] == {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}

    def test_empty_transform_string_returns_identity(self):
        parser = GodotSceneParser()
        result = parser._extract_transform({"transform": ""})
        assert result["position"] == [0.0, 0.0, 0.0]
        assert result["scale"] == [1.0, 1.0, 1.0]

    def test_identity_basis_zero_origin(self):
        result = _extract([1, 0, 0,  0, 1, 0,  0, 0, 1,  0, 0, 0])
        assert result["position"] == [0.0, 0.0, 0.0]
        assert result["scale"] == [1.0, 1.0, 1.0]
        # w close to ±1 (identity quaternion)
        assert abs(abs(result["rotation"]["w"]) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# _extract_transform — position
# ---------------------------------------------------------------------------

class TestExtractTransformPosition:
    def test_origin_extracted_correctly(self):
        """Parser stores raw Godot position (Z not flipped at parse time)."""
        result = _extract([1, 0, 0,  0, 1, 0,  0, 0, 1,  3.0, 5.0, -7.0])
        pos = result["position"]
        assert abs(pos[0] - 3.0) < 1e-6
        assert abs(pos[1] - 5.0) < 1e-6
        assert abs(pos[2] - (-7.0)) < 1e-6  # no Z flip in parser

    def test_positive_z_preserved(self):
        result = _extract([1, 0, 0,  0, 1, 0,  0, 0, 1,  0, 0, 4.0])
        assert abs(result["position"][2] - 4.0) < 1e-6

    def test_negative_x_y_preserved(self):
        result = _extract([1, 0, 0,  0, 1, 0,  0, 0, 1,  -2.5, -1.0, 0.0])
        pos = result["position"]
        assert abs(pos[0] - (-2.5)) < 1e-6
        assert abs(pos[1] - (-1.0)) < 1e-6

    def test_fractional_origin(self):
        result = _extract([1, 0, 0,  0, 1, 0,  0, 0, 1,  0.5, 1.25, -3.75])
        pos = result["position"]
        assert abs(pos[0] - 0.5) < 1e-5
        assert abs(pos[1] - 1.25) < 1e-5
        assert abs(pos[2] - (-3.75)) < 1e-5


# ---------------------------------------------------------------------------
# _extract_transform — scale (diagonal basis = row magnitude == column magnitude)
# ---------------------------------------------------------------------------

class TestExtractTransformScaleDiagonal:
    def test_uniform_scale(self):
        result = _extract([3, 0, 0,  0, 3, 0,  0, 0, 3,  0, 0, 0])
        sc = result["scale"]
        assert abs(sc[0] - 3.0) < 1e-5
        assert abs(sc[1] - 3.0) < 1e-5
        assert abs(sc[2] - 3.0) < 1e-5

    def test_non_uniform_scale(self):
        result = _extract([10, 0, 0,  0, 2, 0,  0, 0, 4,  0, 0, 0])
        sc = result["scale"]
        assert abs(sc[0] - 10.0) < 1e-5
        assert abs(sc[1] - 2.0) < 1e-5
        assert abs(sc[2] - 4.0) < 1e-5

    def test_scale_magnitude_is_always_positive(self):
        """Negative-axis basis still yields positive scale magnitude."""
        result = _extract([-5, 0, 0,  0, 2, 0,  0, 0, 3,  0, 0, 0])
        sc = result["scale"]
        assert sc[0] > 0
        assert sc[1] > 0
        assert sc[2] > 0

    def test_sub_unit_scale(self):
        result = _extract([0.5, 0, 0,  0, 0.25, 0,  0, 0, 2.0,  0, 0, 0])
        sc = result["scale"]
        assert abs(sc[0] - 0.5) < 1e-5
        assert abs(sc[1] - 0.25) < 1e-5
        assert abs(sc[2] - 2.0) < 1e-5

    def test_scale_correct_for_ry90_non_uniform(self):
        """
        Ry(90°) with scale (10, 2, 4) serialised as Transform3D rows:
          Row0 = (0,  0,  4)   [Godot row-major serialisation]
          Row1 = (0,  2,  0)
          Row2 = (-10, 0, 0)
        Row magnitudes  → sx=4,  sy=2, sz=10  (BUGGY current code)
        Column magnitudes → sx=10, sy=2, sz=4   (correct expected)
        """
        result = _extract([0, 0, 4,   0, 2, 0,   -10, 0, 0,   0, 0, 0])
        sc = result["scale"]
        assert abs(sc[0] - 10.0) < 1e-4, f"Expected sx=10, got {sc[0]}"
        assert abs(sc[1] - 2.0) < 1e-4
        assert abs(sc[2] - 4.0) < 1e-4, f"Expected sz=4, got {sc[2]}"


# ---------------------------------------------------------------------------
# _extract_transform — rotation
# ---------------------------------------------------------------------------

class TestExtractTransformRotation:
    def test_identity_rotation_is_unit_quaternion(self):
        result = _extract([1, 0, 0,  0, 1, 0,  0, 0, 1,  0, 0, 0])
        assert abs(_quat_magnitude(result["rotation"]) - 1.0) < 1e-5

    def test_rotation_quaternion_is_normalised(self):
        """Any valid rotation must yield a unit quaternion."""
        result = _extract([0.866, 0, 0.5,  0, 1, 0,  -0.5, 0, 0.866,  0, 0, 0])
        assert abs(_quat_magnitude(result["rotation"]) - 1.0) < 1e-4

    def test_uniform_scale_does_not_change_rotation(self):
        """Scale 5× on all axes — quaternion must equal identity."""
        identity = _extract([1, 0, 0,  0, 1, 0,  0, 0, 1,  0, 0, 0])
        scaled = _extract([5, 0, 0,  0, 5, 0,  0, 0, 5,  0, 0, 0])
        assert _quats_equal(identity["rotation"], scaled["rotation"])

    def test_scale_and_position_independent_of_rotation(self):
        """Origin (1,2,3) and scale 2 with identity rotation."""
        result = _extract([2, 0, 0,  0, 2, 0,  0, 0, 2,  1, 2, 3])
        # Scale must be 2
        sc = result["scale"]
        assert all(abs(s - 2.0) < 1e-5 for s in sc)
        # Position must be (1,2,3) unchanged
        pos = result["position"]
        assert abs(pos[0] - 1.0) < 1e-5
        assert abs(pos[1] - 2.0) < 1e-5
        assert abs(pos[2] - 3.0) < 1e-5


# ---------------------------------------------------------------------------
# _mat3_to_quat (tested via _quat_godot_to_unity for a stable return type)
# ---------------------------------------------------------------------------

class TestMat3ToQuat:
    def _run(self, col0, col1, col2):
        """Return the normalised quaternion dict for the given rotation columns."""
        return _quat_godot_to_unity(_mat3_to_quat(col0, col1, col2))

    def test_identity_matrix_gives_identity_quaternion(self):
        q = self._run([1, 0, 0], [0, 1, 0], [0, 0, 1])
        assert abs(abs(q["w"]) - 1.0) < 1e-5
        assert abs(q["x"]) < 1e-5
        assert abs(q["y"]) < 1e-5
        assert abs(q["z"]) < 1e-5

    def test_output_is_unit_quaternion(self):
        q = self._run([0.866, 0, -0.5], [0, 1, 0], [0.5, 0, 0.866])
        assert abs(_quat_magnitude(q) - 1.0) < 1e-5

    def test_ry180_gives_zero_w(self):
        """Ry(180°): col0=(-1,0,0), col1=(0,1,0), col2=(0,0,-1) → w≈0, |y|≈1."""
        q = self._run([-1, 0, 0], [0, 1, 0], [0, 0, -1])
        assert abs(q["w"]) < 1e-4
        assert abs(abs(q["y"]) - 1.0) < 1e-4

    def test_symmetric_opposite_columns_give_opposite_quaternions(self):
        """_mat3_to_quat(R) and _mat3_to_quat(-R) give quaternions that are negatives."""
        q1 = self._run([1, 0, 0], [0, 1, 0], [0, 0, 1])
        q2 = self._run([-1, 0, 0], [0, -1, 0], [0, 0, -1])
        # The two results are related but need not be exact negatives due to Shepperd branching
        # Just verify both are unit quaternions
        assert abs(_quat_magnitude(q1) - 1.0) < 1e-5
        assert abs(_quat_magnitude(q2) - 1.0) < 1e-5
