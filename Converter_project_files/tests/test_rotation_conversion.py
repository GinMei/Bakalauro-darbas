"""
Godot Transform3D → Unity quaternion conversion.

Tests verify _godot_rot_to_unity() by:
  1. Extracting the rotation from a Godot Transform3D matrix (column-magnitude scale
     removal + Shepperd quaternion).
  2. Running through _quat_godot_to_unity (normalise) and _godot_rot_to_unity
     (handedness + optional FBX correction).
  3. Comparing the resulting Unity quaternion against the known-correct expected
     quaternion.

All Godot input matrices in these test cases are pure Y-axis rotations with
uniform scale, so the expected Unity quaternions are straightforward:
  Non-FBX: Godot Ry(θ) → Unity Ry(-θ)  (handedness flip negates Y component)
  FBX:     adds a fixed 180° correction around the (Y+Z)/√2 axis on top of non-FBX

Expected values are expressed directly as (x, y, z, w) quaternion tuples and were
verified against the _godot_rot_to_unity implementation.
"""

import math
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from godot_to_unity.godot_scene_parser import _mat3_to_quat, _quat_godot_to_unity
from godot_to_unity.unity_scene_exporter import _godot_rot_to_unity


class Transform3D:
    """Minimal wrapper for the 12-value Godot Transform3D basis+origin."""

    def __init__(self, *vals):
        assert len(vals) == 12, "Transform3D requires exactly 12 values"
        self.vals = list(vals)


def convert_rotation(t: Transform3D, is_fbx: bool = False) -> dict:
    """Extract the Unity rotation quaternion from a Godot Transform3D.

    Returns a dict {x, y, z, w}.
    Uses column magnitudes for scale removal so the result is correct even
    when scale != 1 or rotation is non-identity.
    """
    v = t.vals
    ax, ay, az = v[0], v[1], v[2]
    bx, by, bz = v[3], v[4], v[5]
    cx, cy, cz = v[6], v[7], v[8]

    sx = math.sqrt(ax * ax + bx * bx + cx * cx)
    sy = math.sqrt(ay * ay + by * by + cy * cy)
    sz = math.sqrt(az * az + bz * bz + cz * cz)

    def _nc(c0, c1, c2, s):
        return [c0 / s, c1 / s, c2 / s] if s > 1e-9 else [c0, c1, c2]

    col0 = _nc(ax, bx, cx, sx)
    col1 = _nc(ay, by, cy, sy)
    col2 = _nc(az, bz, cz, sz)

    godot_quat = _mat3_to_quat(col0, col1, col2)
    unity_quat = _quat_godot_to_unity(godot_quat)
    return _godot_rot_to_unity(unity_quat, is_fbx=is_fbx)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quat_close(q1, q2, eps=1e-3):
    """Quaternion equality handling q == -q."""
    return (
        all(abs(a - b) < eps for a, b in zip(q1, q2)) or
        all(abs(a + b) < eps for a, b in zip(q1, q2))
    )


def _as_tuple(rot: dict):
    """Return (x, y, z, w) from a rotation dict."""
    return (rot["x"], rot["y"], rot["z"], rot["w"])


# ---------------------------------------------------------------------------
# Test cases — (godot_matrix_12_vals, expected_unity_quat_xyzw)
#
# Godot matrices are scale-10 pure Y-rotations:
#   identity  Ry(0°)     Ry(90°)    Ry(180°)   Ry(30°)    Ry(-60°)   Ry(-30°)
#
# Non-FBX: Godot Ry(θ)  →  Unity Ry(-θ)
# FBX:     non-FBX result  ×  180° correction around (Y+Z)/√2
# ---------------------------------------------------------------------------

NON_FBX_CASES = [
    #0
    (
        (0, 0, -1, 0, 1, 0, 1, 0, 0, 1, 0.5, -3),
        (0.0, -0.7071068, 0.0, 0.7071068),
    ),
    #1
    (
        (1, 0, 0, 0, 1, 0, 0, 0, 1, -3, 0.5, -7),
        (0.0, 1, 0.0, 0),
    ),
    #2
    (
        (-1, 0, 0, 0, 1, 0, 0, 0, -1, -1, 0.5, 7),
        (0.0, 0.0, 0.0, 1.0),
    ), 
    #3 (0,90,0)
    (
        (0, 0, 1, 0, 1, 0, -1, 0, 0, -7, 0.5, -3),
        (0.0, 0.7071068, 0.0, 0.7071068),
    ),
    #4
    (
        (-0.8384117, -0.005720854, -0.5450074, 0.5025975, 0.37873447, -0.77714604, 0.210859, -0.92548776, -0.31466013, 1, 1, -1),
        (0.5350274, -0.23751967, -0.15613608 , 0.79558253),
    ),
    #5
    (
        (-0.8270823, 0.46113002, -0.32139382, -0.513353, -0.3868393, 0.76604444, 0.22891837, 0.7985703, 0.5566704, 1, 1, -1),
        (0.83225477, 0.29272383, -0.027778573, -0.46999273),
    ),
    #6
    (
        (0.87499994, -0.21650635, 0.43301266, 0.43301266, 0.75, -0.49999997, -0.21650633, 0.625, 0.74999994, 1, 3, 1),
        (-0.17677663, 0.91855866, -0.30618626, 0.17677674),
    ),
    #7
    (
        (0.17364822, 0, -0.9848077, 0, 1, 0, 0.9848077, 0, 0.17364822, 1, 0.5, -3),
        (0, 0.7660445, -0, -0.6427876),
    ),
    #8
    (
        (-0.5675957, -0.793412, 0.21984631, 0.6040228, -0.21984631, 0.76604444, -0.5594565, 0.5675957, 0.6040228, 1, 3, 1),
        (-0.77321815, 0.45182422, 0.109804034, 0.431198),
    ),
]

FBX_CASES = [
    (
        (10, 0, 0, 0, 10, 0, 0, 0, 10, -37, 1.2, -26),
        (0.0, 0.707107, 0.707107, 0.0),
    ),
    (
        (0, 0, 10, 0, 10, 0, -10, 0, 0, -28, 1.2, -36),
        (0.5, 0.5, 0.5, -0.5),
    ),
    (
        (-10, 0, 0, 0, 10, 0, 0, 0, -10, 18, -0.4, -2),
        (0.707107, 0.0, 0.0, -0.707107),
    ),
    (
        (8.660254, 0, 5, 0, 10, 0, -5, 0, 8.660254, -37, 1.2, -26),
        (0.183013, 0.683013, 0.683013, -0.183013),
    ),
    (
        (4.9999995, 0, -8.6602545, 0, 10, 0, 8.6602545, 0, 4.9999995, -37, 1.2, -26),
        (-0.353553, 0.612372, 0.612372, 0.353553),
    ),
    (
        (8.660254, 0, -5, 0, 10, 0, 5, 0, 8.660254, -37, 1.2, -26),
        (-0.183013, 0.683013, 0.683013, 0.183013),
    ),
]

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("matrix, expected_xyzw", NON_FBX_CASES)
def test_non_fbx_rotation_conversion(matrix, expected_xyzw):
    t = Transform3D(*matrix)
    result   = tuple(round(v, 5) for v in _as_tuple(convert_rotation(t, is_fbx=False)))
    expected = tuple(round(v, 5) for v in expected_xyzw)
    assert quat_close(result, expected), (
        f"Non-FBX: got {result}, expected {expected}"
    )


@pytest.mark.parametrize("matrix, expected_xyzw", FBX_CASES)
def test_fbx_rotation_conversion(matrix, expected_xyzw):
    t = Transform3D(*matrix)
    result = _as_tuple(convert_rotation(t, is_fbx=True))
    assert quat_close(result, expected_xyzw), (
        f"FBX: got {result}, expected {expected_xyzw}"
    )
