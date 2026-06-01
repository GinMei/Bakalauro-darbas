"""
Tests for unity_scene_exporter.py — component-specific node emission.

Each test exports a minimal scene IR containing a specific component type
and verifies that the expected Unity class ID appears in the .unity output.
Also covers pure helpers: _stable_local_id, _read_meta_guid, write_folder_meta.
"""

import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from godot_to_unity.unity_scene_exporter import (
    UnitySceneExporter,
    _stable_local_id,
    _read_meta_guid,
    write_folder_meta,
    _stable_guid,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_node(name: str = "Node", godot_type: str = "Node3D", **extra) -> dict:
    node = {
        "id": "node_0",
        "node_name": name,
        "godot_type": godot_type,
        "ir_type": "group",
        "transform": {
            "position": [0.0, 0.0, 0.0],
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            "scale": [1.0, 1.0, 1.0],
        },
        "components": {},
        "children": [],
    }
    node.update(extra)
    return node


def _ir(nodes: list, scene_name: str = "test") -> dict:
    return {
        "ir_version": "1.0",
        "coordinate_system": {"conversion_applied": True},
        "scene_name": scene_name,
        "nodes": nodes,
        "animations": [],
    }


def _export_content(ir: dict, tmp_path: Path) -> str:
    path = UnitySceneExporter().export(ir, tmp_path)
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# _stable_local_id
# ---------------------------------------------------------------------------

class TestStableLocalId:
    def test_returns_negative_int(self):
        result = _stable_local_id("test_key")
        assert result < 0

    def test_deterministic(self):
        assert _stable_local_id("x") == _stable_local_id("x")

    def test_different_keys_differ(self):
        assert _stable_local_id("a") != _stable_local_id("b")

    def test_stays_in_int64_range(self):
        result = _stable_local_id("anything")
        assert result >= -(2 ** 62)


# ---------------------------------------------------------------------------
# _read_meta_guid
# ---------------------------------------------------------------------------

class TestReadMetaGuid:
    def test_returns_guid(self, tmp_path):
        meta = tmp_path / "tex.png.meta"
        meta.write_text("fileFormatVersion: 2\nguid: abc123\n", encoding="utf-8")
        assert _read_meta_guid(meta) == "abc123"

    def test_returns_empty_when_no_guid(self, tmp_path):
        meta = tmp_path / "f.meta"
        meta.write_text("fileFormatVersion: 2\n", encoding="utf-8")
        assert _read_meta_guid(meta) == ""

    def test_returns_empty_for_missing_file(self, tmp_path):
        assert _read_meta_guid(tmp_path / "nonexistent.meta") == ""

    def test_strips_whitespace(self, tmp_path):
        meta = tmp_path / "a.meta"
        meta.write_text("  guid:   dead1234  \n", encoding="utf-8")
        assert _read_meta_guid(meta) == "dead1234"


# ---------------------------------------------------------------------------
# write_folder_meta
# ---------------------------------------------------------------------------

class TestWriteFolderMeta:
    def test_creates_meta_file(self, tmp_path):
        folder = tmp_path / "Assets" / "Scenes"
        folder.mkdir(parents=True)
        write_folder_meta(folder, tmp_path)
        meta = tmp_path / "Assets" / "Scenes.meta"
        assert meta.exists()

    def test_meta_has_folder_asset_flag(self, tmp_path):
        folder = tmp_path / "Folder"
        folder.mkdir()
        write_folder_meta(folder, tmp_path)
        content = (tmp_path / "Folder.meta").read_text(encoding="utf-8")
        assert "folderAsset: yes" in content

    def test_meta_has_guid(self, tmp_path):
        folder = tmp_path / "Dir"
        folder.mkdir()
        write_folder_meta(folder, tmp_path)
        content = (tmp_path / "Dir.meta").read_text(encoding="utf-8")
        assert "guid:" in content

    def test_does_not_overwrite_existing(self, tmp_path):
        folder = tmp_path / "Dir"
        folder.mkdir()
        meta = tmp_path / "Dir.meta"
        meta.write_text("existing content", encoding="utf-8")
        write_folder_meta(folder, tmp_path)
        assert meta.read_text(encoding="utf-8") == "existing content"

    def test_folder_outside_output_dir_uses_name(self, tmp_path, tmp_path_factory):
        other_root = tmp_path_factory.mktemp("root")
        folder = tmp_path / "SubDir"
        folder.mkdir()
        write_folder_meta(folder, other_root)
        meta = tmp_path / "SubDir.meta"
        assert meta.exists()


# ---------------------------------------------------------------------------
# Node type — MeshInstance3D  (exercises MeshFilter + MeshRenderer)
# ---------------------------------------------------------------------------

class TestExportMeshNode:
    def test_mesh_filter_class_id_present(self, tmp_path):
        node = _base_node("Cube", "MeshInstance3D")
        node["components"] = {
            "mesh": {"mesh_type": "box", "mesh_source_path": "res://cube.fbx"}
        }
        content = _export_content(_ir([node]), tmp_path)
        assert "!u!33 &" in content  # MeshFilter

    def test_mesh_renderer_class_id_present(self, tmp_path):
        node = _base_node("Cube", "MeshInstance3D")
        node["components"] = {
            "mesh": {"mesh_type": "box", "mesh_source_path": "res://cube.fbx"}
        }
        content = _export_content(_ir([node]), tmp_path / "run2")
        assert "!u!23 &" in content  # MeshRenderer

    def test_mesh_with_guid_map(self, tmp_path):
        node = _base_node("Hero", "MeshInstance3D")
        node["components"] = {
            "mesh": {"mesh_type": "static", "mesh_source_path": "res://hero.fbx"}
        }
        exporter = UnitySceneExporter()
        path = exporter.export(
            _ir([node]),
            tmp_path,
            mesh_guid_map={"res://hero.fbx": "abc123def456abc1"},
        )
        content = path.read_text(encoding="utf-8")
        assert "abc123def456abc1" in content


# ---------------------------------------------------------------------------
# Node type — Camera3D  (exercises Camera component)
# ---------------------------------------------------------------------------

class TestExportCameraNode:
    def test_camera_class_id_present(self, tmp_path):
        node = _base_node("Cam", "Camera3D")
        node["components"] = {
            "camera": {"fov": 75.0, "near_clip": 0.1, "far_clip": 1000.0}
        }
        content = _export_content(_ir([node]), tmp_path)
        assert "!u!20 &" in content  # Camera

    def test_camera_fov_in_output(self, tmp_path):
        node = _base_node("Cam", "Camera3D")
        node["components"] = {"camera": {"fov": 60.0, "near_clip": 0.05, "far_clip": 500.0}}
        content = _export_content(_ir([node]), tmp_path)
        assert "Camera:" in content


# ---------------------------------------------------------------------------
# Node type — DirectionalLight3D / OmniLight3D  (exercises Light component)
# ---------------------------------------------------------------------------

class TestExportLightNode:
    def test_directional_light_class_id_present(self, tmp_path):
        node = _base_node("Sun", "DirectionalLight3D")
        node["components"] = {
            "light": {
                "light_type": "directional",
                "color": {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0},
                "intensity": 1.0,
                "cast_shadows": True,
            }
        }
        content = _export_content(_ir([node]), tmp_path)
        assert "!u!108 &" in content  # Light

    def test_point_light_range_used(self, tmp_path):
        node = _base_node("Lamp", "OmniLight3D")
        node["components"] = {
            "light": {
                "light_type": "point",
                "color": {"r": 1.0, "g": 0.8, "b": 0.5, "a": 1.0},
                "intensity": 2.0,
                "range": 10.0,
                "cast_shadows": False,
            }
        }
        content = _export_content(_ir([node]), tmp_path)
        assert "Light:" in content


# ---------------------------------------------------------------------------
# Node type — RigidBody3D  (exercises Rigidbody component)
# ---------------------------------------------------------------------------

class TestExportRigidbodyNode:
    def test_rigidbody_class_id_present(self, tmp_path):
        node = _base_node("Box", "RigidBody3D")
        node["components"] = {
            "rigidbody": {
                "mass": 1.0,
                "drag": 0.0,
                "angular_drag": 0.05,
                "is_kinematic": False,
                "use_gravity": True,
            }
        }
        content = _export_content(_ir([node]), tmp_path)
        assert "!u!54 &" in content  # Rigidbody

    def test_rigidbody_mass_in_output(self, tmp_path):
        node = _base_node("Heavy", "RigidBody3D")
        node["components"] = {
            "rigidbody": {"mass": 5.0, "drag": 0.1, "angular_drag": 0.1,
                          "is_kinematic": False, "use_gravity": True}
        }
        content = _export_content(_ir([node]), tmp_path)
        assert "Rigidbody:" in content


# ---------------------------------------------------------------------------
# Node type — CollisionShape3D  (box, sphere, capsule)
# ---------------------------------------------------------------------------

class TestExportColliderNode:
    def test_box_collider_class_id(self, tmp_path):
        node = _base_node("Wall", "StaticBody3D")
        node["components"] = {
            "colliders": [{"collider_type": "box", "is_trigger": False,
                           "center": [0, 0, 0], "size": [1, 1, 1]}]
        }
        content = _export_content(_ir([node]), tmp_path)
        assert "!u!65 &" in content  # BoxCollider

    def test_sphere_collider_class_id(self, tmp_path):
        node = _base_node("Ball", "StaticBody3D")
        node["components"] = {
            "colliders": [{"collider_type": "sphere", "is_trigger": False,
                           "center": [0, 0, 0], "radius": 0.5}]
        }
        content = _export_content(_ir([node]), tmp_path / "r2")
        assert "!u!135 &" in content  # SphereCollider

    def test_capsule_collider_class_id(self, tmp_path):
        node = _base_node("Capsule", "CharacterBody3D")
        node["components"] = {
            "colliders": [{"collider_type": "capsule", "is_trigger": False,
                           "center": [0, 0, 0], "radius": 0.3, "height": 1.8,
                           "direction": 1}]
        }
        content = _export_content(_ir([node]), tmp_path / "r3")
        assert "!u!136 &" in content  # CapsuleCollider


# ---------------------------------------------------------------------------
# Node type — AudioStreamPlayer3D  (exercises AudioSource component)
# ---------------------------------------------------------------------------

class TestExportAudioNode:
    def test_audio_source_class_id_present(self, tmp_path):
        node = _base_node("SFX", "AudioStreamPlayer3D")
        node["components"] = {
            "audio_source": {
                "volume": 1.0,
                "autoplay": False,
                "loop": False,
                "max_distance": 20.0,
            }
        }
        content = _export_content(_ir([node]), tmp_path)
        assert "!u!82 &" in content  # AudioSource

    def test_audio_source_component_present(self, tmp_path):
        node = _base_node("Music", "AudioStreamPlayer3D")
        node["components"] = {
            "audio_source": {"volume": 0.8, "autoplay": True,
                             "loop": True, "max_distance": 50.0}
        }
        content = _export_content(_ir([node]), tmp_path)
        assert "AudioSource:" in content


# ---------------------------------------------------------------------------
# Node with children  (exercises recursive _emit_node)
# ---------------------------------------------------------------------------

class TestExportNestedNodes:
    def test_child_game_object_present(self, tmp_path):
        child = _base_node("Child", "Node3D")
        parent = _base_node("Parent", "Node3D")
        parent["children"] = [child]
        content = _export_content(_ir([parent]), tmp_path)
        assert "Child" in content
        assert "Parent" in content

    def test_deeply_nested_child(self, tmp_path):
        grandchild = _base_node("GrandChild", "Node3D")
        child = _base_node("Child", "Node3D")
        child["children"] = [grandchild]
        parent = _base_node("Parent", "Node3D")
        parent["children"] = [child]
        content = _export_content(_ir([parent]), tmp_path)
        assert "GrandChild" in content


# ---------------------------------------------------------------------------
# Multiple nodes at root  (exercises loop over nodes)
# ---------------------------------------------------------------------------

class TestExportMultipleRootNodes:
    def test_two_root_nodes(self, tmp_path):
        n1 = _base_node("Alpha", "Node3D")
        n2 = _base_node("Beta", "Node3D")
        content = _export_content(_ir([n1, n2]), tmp_path)
        assert "Alpha" in content
        assert "Beta" in content


# ---------------------------------------------------------------------------
# export_instance  (exercises export_instance method)
# ---------------------------------------------------------------------------

class TestExportPrefab:
    def test_prefab_file_created(self, tmp_path):
        node = _base_node("Root")
        ir = _ir([node], scene_name="my_prefab")
        prefab_path, guid, _ = UnitySceneExporter().export_instance(ir, tmp_path)
        assert prefab_path.exists()
        assert prefab_path.suffix == ".prefab"

    def test_prefab_yaml_header(self, tmp_path):
        node = _base_node("Root")
        ir = _ir([node], scene_name="prefab_scene")
        prefab_path, *_ = UnitySceneExporter().export_instance(ir, tmp_path)
        assert prefab_path.read_text(encoding="utf-8").startswith("%YAML 1.1")

    def test_prefab_guid_is_stable(self, tmp_path):
        node = _base_node("Root")
        ir = _ir([node], scene_name="stable_prefab")
        _, guid1, _ = UnitySceneExporter().export_instance(ir, tmp_path / "run1")
        _, guid2, _ = UnitySceneExporter().export_instance(ir, tmp_path / "run2")
        assert guid1 == guid2

    def test_prefab_meta_created(self, tmp_path):
        node = _base_node("Root")
        ir = _ir([node], scene_name="pref")
        prefab_path, *_ = UnitySceneExporter().export_instance(ir, tmp_path)
        meta = prefab_path.parent / (prefab_path.name + ".meta")
        assert meta.exists()

    def test_prefab_multi_root_wraps_nodes(self, tmp_path):
        n1 = _base_node("A")
        n2 = _base_node("B")
        ir = _ir([n1, n2], scene_name="multi_root")
        prefab_path, *_ = UnitySceneExporter().export_instance(ir, tmp_path)
        content = prefab_path.read_text(encoding="utf-8")
        assert "multi_rootRoot" in content

    def test_prefab_with_relative_dir(self, tmp_path):
        node = _base_node("Root")
        ir = _ir([node], scene_name="enemy")
        prefab_path, *_ = UnitySceneExporter().export_instance(
            ir, tmp_path, godot_relative_dir=Path("Enemies")
        )
        assert "Enemies" in str(prefab_path)


# ---------------------------------------------------------------------------
# export_scene_from_prefab  (exercises scene referencing a prefab)
# ---------------------------------------------------------------------------

class TestExportSceneFromPrefab:
    def test_creates_unity_scene_file(self, tmp_path):
        node = _base_node("Root")
        ir = _ir([node], scene_name="main_level")
        exporter = UnitySceneExporter()
        scene_path = exporter.export_scene_from_prefab(
            ir, tmp_path, prefab_guid="abc123", project_name="TestProject"
        )
        assert scene_path.exists()
        assert scene_path.suffix == ".unity"

    def test_scene_references_prefab_guid(self, tmp_path):
        node = _base_node("Root")
        ir = _ir([node], scene_name="scene_with_prefab")
        exporter = UnitySceneExporter()
        scene_path = exporter.export_scene_from_prefab(
            ir, tmp_path, prefab_guid="deadbeef12345678"
        )
        content = scene_path.read_text(encoding="utf-8")
        assert "deadbeef12345678" in content


# ---------------------------------------------------------------------------
# _emit_node_with_ids — child nodes with components (lines 1083-1122)
# These tests place components on CHILD nodes so they go through
# _emit_node_with_ids rather than the root-node _emit_node path.
# ---------------------------------------------------------------------------

class TestChildNodeComponents:
    """Exercises _emit_node_with_ids component paths (lines 1083-1122)."""

    def _export_with_child(self, child: dict, tmp_path: Path) -> str:
        parent = _base_node("Parent", "Node3D")
        parent["children"] = [child]
        return _export_content(_ir([parent]), tmp_path)

    def test_child_rigidbody_emits_cid_54(self, tmp_path):
        child = _base_node("PhysBox", "RigidBody3D")
        child["components"] = {
            "rigidbody": {
                "mass": 2.0, "drag": 0.1, "angular_drag": 0.05,
                "is_kinematic": False, "use_gravity": True,
            }
        }
        content = self._export_with_child(child, tmp_path)
        assert "!u!54 &" in content  # _CID_RIGIDBODY

    def test_child_rigidbody_mass_written(self, tmp_path):
        child = _base_node("Heavy", "RigidBody3D")
        child["components"] = {"rigidbody": {"mass": 9.5}}
        content = self._export_with_child(child, tmp_path)
        assert "m_Mass: 9.5" in content

    def test_child_box_collider_emits_cid_65(self, tmp_path):
        child = _base_node("Wall", "StaticBody3D")
        child["components"] = {
            "colliders": [{"collider_type": "box", "size": [2, 2, 2], "center": [0, 0, 0]}]
        }
        content = self._export_with_child(child, tmp_path)
        assert "!u!65 &" in content  # _CID_BOX_COLLIDER

    def test_child_sphere_collider_emits_sphere(self, tmp_path):
        child = _base_node("Ball", "StaticBody3D")
        child["components"] = {
            "colliders": [{"collider_type": "sphere", "radius": 0.5, "center": [0, 0, 0]}]
        }
        content = self._export_with_child(child, tmp_path)
        assert "SphereCollider:" in content

    def test_child_capsule_collider_emits_capsule(self, tmp_path):
        child = _base_node("Character", "CharacterBody3D")
        child["components"] = {
            "colliders": [{"collider_type": "capsule", "radius": 0.4, "height": 1.8,
                           "direction": 1, "center": [0, 0, 0]}]
        }
        content = self._export_with_child(child, tmp_path)
        assert "CapsuleCollider:" in content

    def test_child_camera_emits_cid_20(self, tmp_path):
        child = _base_node("MainCam", "Camera3D")
        child["components"] = {
            "camera": {"fov": 75.0, "near_clip": 0.1, "far_clip": 500.0}
        }
        content = self._export_with_child(child, tmp_path)
        assert "!u!20 &" in content  # _CID_CAMERA

    def test_child_camera_fov_written(self, tmp_path):
        child = _base_node("Cam", "Camera3D")
        child["components"] = {"camera": {"fov": 90.0}}
        content = self._export_with_child(child, tmp_path)
        assert "m_FieldOfView: 90.0" in content

    def test_child_directional_light_emits_light(self, tmp_path):
        child = _base_node("Sun", "DirectionalLight3D")
        child["components"] = {
            "light": {
                "light_type": "directional",
                "color": {"r": 1.0, "g": 0.9, "b": 0.8, "a": 1.0},
                "intensity": 1.5,
                "cast_shadows": True,
            }
        }
        content = self._export_with_child(child, tmp_path)
        assert "Light:" in content

    def test_child_omni_light_emits_light(self, tmp_path):
        child = _base_node("Lamp", "OmniLight3D")
        child["components"] = {
            "light": {
                "light_type": "point",
                "color": {"r": 1.0, "g": 1.0, "b": 0.5, "a": 1.0},
                "intensity": 2.0,
                "cast_shadows": False,
                "range": 10.0,
            }
        }
        content = self._export_with_child(child, tmp_path)
        assert "Light:" in content

    def test_child_audio_source_emits_cid_82(self, tmp_path):
        child = _base_node("SFX", "AudioStreamPlayer3D")
        child["components"] = {
            "audio_source": {"volume": -6.0, "autoplay": False, "loop": True,
                             "max_distance": 20.0}
        }
        content = self._export_with_child(child, tmp_path)
        assert "!u!82 &" in content  # _CID_AUDIO_SOURCE

    def test_child_audio_loop_written(self, tmp_path):
        child = _base_node("Music", "AudioStreamPlayer3D")
        child["components"] = {"audio_source": {"loop": True}}
        content = self._export_with_child(child, tmp_path)
        assert "Loop: 1" in content

    def test_child_character_controller_emits_cid_97(self, tmp_path):
        child = _base_node("Hero", "CharacterBody3D")
        child["components"] = {
            "physics_body": {"type": "character", "height": 2.0, "radius": 0.5}
        }
        content = self._export_with_child(child, tmp_path)
        assert "!u!97 &" in content  # _CID_CHARACTER_CONTROLLER

    def test_child_character_controller_emits_component_name(self, tmp_path):
        child = _base_node("Player", "CharacterBody3D")
        child["components"] = {"physics_body": {"type": "character"}}
        content = self._export_with_child(child, tmp_path)
        assert "CharacterController:" in content

    def test_child_trigger_emits_box_collider(self, tmp_path):
        child = _base_node("Zone", "Area3D")
        child["components"] = {
            "trigger": {"size": [3, 3, 3], "center": [0, 0, 0]}
        }
        content = self._export_with_child(child, tmp_path)
        assert "m_IsTrigger: 1" in content

    def test_child_trigger_emits_cid_65(self, tmp_path):
        child = _base_node("Trigger", "Area3D")
        child["components"] = {"trigger": {"size": [1, 1, 1], "center": [0, 0, 0]}}
        content = self._export_with_child(child, tmp_path)
        assert "!u!65 &" in content  # _CID_BOX_COLLIDER (trigger uses box)

    def test_child_multiple_colliders(self, tmp_path):
        child = _base_node("Multi", "StaticBody3D")
        child["components"] = {
            "colliders": [
                {"collider_type": "box", "size": [1, 1, 1], "center": [0, 0, 0]},
                {"collider_type": "sphere", "radius": 0.5, "center": [0, 1, 0]},
            ]
        }
        content = self._export_with_child(child, tmp_path)
        assert "BoxCollider:" in content
        assert "SphereCollider:" in content

    def test_deeply_nested_child_still_exports(self, tmp_path):
        grandchild = _base_node("GC", "Node3D")
        grandchild["components"] = {"rigidbody": {"mass": 1.0}}
        child = _base_node("Child", "Node3D")
        child["children"] = [grandchild]
        parent = _base_node("Parent", "Node3D")
        parent["children"] = [child]
        content = _export_content(_ir([parent]), tmp_path)
        assert "Rigidbody:" in content
