import sys
import os
import types
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Stub yaml so tests can import unity_to_godot.unity_parser without PyYAML
# ---------------------------------------------------------------------------

def _stub_yaml():
    if "yaml" not in sys.modules:
        yaml_mod = types.ModuleType("yaml")

        class SafeLoader:
            @classmethod
            def add_multi_constructor(cls, _tag, _fn):
                pass

        yaml_mod.SafeLoader = SafeLoader
        yaml_mod.load = MagicMock(return_value={})
        yaml_mod.YAMLError = Exception

        class MappingNode:
            pass
        class SequenceNode:
            pass
        class Node:
            pass

        yaml_mod.MappingNode  = MappingNode
        yaml_mod.SequenceNode = SequenceNode
        yaml_mod.Node         = Node
        sys.modules["yaml"]   = yaml_mod

_stub_yaml()


# ---------------------------------------------------------------------------
# Minimal .tscn content helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_tscn():
    """Minimal valid .tscn with a single Node3D root."""
    return (
        '[gd_scene load_steps=1 format=3 uid="uid://abc"]\n\n'
        '[node name="Root" type="Node3D"]\n'
    )


@pytest.fixture
def tscn_with_child():
    """A .tscn with a root and one MeshInstance3D child."""
    return (
        '[gd_scene load_steps=1 format=3]\n\n'
        '[node name="Root" type="Node3D"]\n\n'
        '[node name="Mesh" type="MeshInstance3D" parent="."]\n'
    )


@pytest.fixture
def tscn_with_transform():
    """A .tscn root node carrying a non-trivial Transform3D."""
    return (
        '[gd_scene load_steps=1 format=3]\n\n'
        '[node name="Root" type="Node3D"]\n'
        'transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 3, 5, -7)\n'
    )


@pytest.fixture
def tscn_with_instance():
    """A .tscn that instances another scene via ExtResource."""
    return (
        '[gd_scene load_steps=2 format=3]\n\n'
        '[ext_resource type="PackedScene" path="res://objects/crate.tscn" id="1"]\n\n'
        '[node name="Level" type="Node3D"]\n\n'
        '[node name="Crate" parent="." instance=ExtResource("1")]\n'
    )


# ---------------------------------------------------------------------------
# Reference graph helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_ref_graph():
    """An empty reference graph with all expected keys present."""
    return {
        "referenced_res_paths": set(),
        "main_scene_res_path": "",
        "file_to_res": {},
        "script_loaded_res_paths": set(),
        "script_instantiate_paths": set(),
        "script_scene_change_paths": set(),
        "dynamic_scene_prefixes": set(),
        "dynamic_prefab_prefixes": set(),
        "usage_map": {},
    }


# ---------------------------------------------------------------------------
# Minimal IR fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_ir():
    """Engine-agnostic IR for a single root Node3D with identity transform."""
    return {
        "version": "1.0",
        "coordinate_system": {"conversion_applied": True},
        "scene_name": "TestScene",
        "nodes": [
            {
                "id": "node_0",
                "name": "Root",
                "godot_type": "Node3D",
                "ir_type": "group",
                "parent_id": None,
                "transform": {
                    "position": [0.0, 0.0, 0.0],
                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                    "scale": [1.0, 1.0, 1.0],
                },
                "components": {},
                "children": [],
            }
        ],
    }


@pytest.fixture
def ir_with_mesh():
    """IR for a root node with one MeshInstance3D child carrying a mesh reference."""
    return {
        "version": "1.0",
        "coordinate_system": {"conversion_applied": True},
        "scene_name": "MeshScene",
        "nodes": [
            {
                "id": "node_0",
                "name": "Root",
                "godot_type": "Node3D",
                "ir_type": "group",
                "parent_id": None,
                "transform": {
                    "position": [0.0, 0.0, 0.0],
                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                    "scale": [1.0, 1.0, 1.0],
                },
                "components": {},
                "children": ["node_1"],
            },
            {
                "id": "node_1",
                "name": "MeshNode",
                "godot_type": "MeshInstance3D",
                "ir_type": "entity",
                "parent_id": "node_0",
                "transform": {
                    "position": [1.0, 0.0, 0.0],
                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                    "scale": [1.0, 1.0, 1.0],
                },
                "components": {
                    "mesh": {"mesh_source_path": "res://meshes/cube.glb"}
                },
                "children": [],
            },
        ],
    }
