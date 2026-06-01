"""
Tier 3 — GodotSceneParser._extract_components().

Tests verify:
  • CollisionShape3D — BoxShape3D / SphereShape3D / CapsuleShape3D extents
  • MeshInstance3D   — mesh reference component
  • Area3D           — trigger component with correct fields
  • CharacterBody3D  — physics_body component
  • signal event_bindings — [connection] sections → event_bindings on source node
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from godot_to_unity.godot_scene_parser import GodotSceneParser


def _parser():
    return GodotSceneParser()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _components(node_type: str, props: dict, sub_resources: dict = None, ext_resources: dict = None):
    """Call _extract_components and return (components, extra_fields)."""
    p = _parser()
    return p._extract_components(
        node_type,
        props,
        ext_resources or {},
        sub_resources or {},
    )


# ---------------------------------------------------------------------------
# CollisionShape3D — BoxShape3D
# ---------------------------------------------------------------------------

class TestBoxShape3D:
    def _box_components(self, size_str: str = "Vector3(2, 4, 6)"):
        sub_resources = {
            "shape_1": {"type": "BoxShape3D", "props": {"size": size_str}},
        }
        props = {"shape": 'SubResource("shape_1")'}
        comps, _ = _components("CollisionShape3D", props, sub_resources)
        return comps

    def test_colliders_key_present(self):
        assert "colliders" in self._box_components()

    def test_colliders_is_list(self):
        assert isinstance(self._box_components()["colliders"], list)

    def test_colliders_has_one_entry(self):
        assert len(self._box_components()["colliders"]) == 1

    def test_box_collider_type(self):
        c = self._box_components()["colliders"][0]
        assert c["collider_type"] == "box"

    def test_box_size_x(self):
        c = self._box_components("Vector3(2, 4, 6)")["colliders"][0]
        assert abs(c["size"][0] - 2.0) < 1e-5

    def test_box_size_y(self):
        c = self._box_components("Vector3(2, 4, 6)")["colliders"][0]
        assert abs(c["size"][1] - 4.0) < 1e-5

    def test_box_size_z(self):
        c = self._box_components("Vector3(2, 4, 6)")["colliders"][0]
        assert abs(c["size"][2] - 6.0) < 1e-5

    def test_box_is_trigger_false(self):
        c = self._box_components()["colliders"][0]
        assert c["is_trigger"] is False

    def test_box_center_zero(self):
        c = self._box_components()["colliders"][0]
        assert c["center"] == [0.0, 0.0, 0.0]

    def test_box_missing_size_defaults(self):
        sub_resources = {"s1": {"type": "BoxShape3D", "props": {}}}
        props = {"shape": 'SubResource("s1")'}
        comps, _ = _components("CollisionShape3D", props, sub_resources)
        c = comps["colliders"][0]
        assert c["collider_type"] == "box"
        assert c["size"][0] == 1.0


# ---------------------------------------------------------------------------
# CollisionShape3D — SphereShape3D
# ---------------------------------------------------------------------------

class TestSphereShape3D:
    def _sphere_components(self, radius: float = 1.5):
        sub_resources = {
            "s1": {"type": "SphereShape3D", "props": {"radius": str(radius)}},
        }
        props = {"shape": 'SubResource("s1")'}
        comps, _ = _components("CollisionShape3D", props, sub_resources)
        return comps

    def test_collider_type_sphere(self):
        c = self._sphere_components()["colliders"][0]
        assert c["collider_type"] == "sphere"

    def test_sphere_radius(self):
        c = self._sphere_components(2.0)["colliders"][0]
        assert abs(c["radius"] - 2.0) < 1e-5

    def test_sphere_is_trigger_false(self):
        c = self._sphere_components()["colliders"][0]
        assert c["is_trigger"] is False

    def test_sphere_center_zero(self):
        c = self._sphere_components()["colliders"][0]
        assert c["center"] == [0.0, 0.0, 0.0]

    def test_sphere_default_radius(self):
        sub_resources = {"s1": {"type": "SphereShape3D", "props": {}}}
        props = {"shape": 'SubResource("s1")'}
        comps, _ = _components("CollisionShape3D", props, sub_resources)
        c = comps["colliders"][0]
        assert c["radius"] == 0.5


# ---------------------------------------------------------------------------
# CollisionShape3D — CapsuleShape3D
# ---------------------------------------------------------------------------

class TestCapsuleShape3D:
    def _capsule_components(self, radius: float = 0.5, height: float = 2.0):
        sub_resources = {
            "c1": {
                "type": "CapsuleShape3D",
                "props": {"radius": str(radius), "height": str(height)},
            },
        }
        props = {"shape": 'SubResource("c1")'}
        comps, _ = _components("CollisionShape3D", props, sub_resources)
        return comps

    def test_collider_type_capsule(self):
        c = self._capsule_components()["colliders"][0]
        assert c["collider_type"] == "capsule"

    def test_capsule_radius(self):
        c = self._capsule_components(0.4, 1.8)["colliders"][0]
        assert abs(c["radius"] - 0.4) < 1e-5

    def test_capsule_height(self):
        c = self._capsule_components(0.5, 3.0)["colliders"][0]
        assert abs(c["height"] - 3.0) < 1e-5

    def test_capsule_direction_y_axis(self):
        c = self._capsule_components()["colliders"][0]
        assert c["direction"] == 1

    def test_capsule_is_trigger_false(self):
        c = self._capsule_components()["colliders"][0]
        assert c["is_trigger"] is False


# ---------------------------------------------------------------------------
# MeshInstance3D — mesh reference
# ---------------------------------------------------------------------------

class TestMeshInstance3DComponents:
    # MeshInstance3D.mesh component comes from an ExtResource with a non-model extension
    # (e.g. .tres, .res). Model extensions (.fbx/.obj/.gltf/.glb/.dae) become asset_reference.

    def _ext_mesh_resources(self, path: str = "res://meshes/cube.tres"):
        return {"1": {"type": "Mesh", "path": path, "uid": ""}}

    def test_mesh_component_present_for_non_model_ext_resource(self):
        ext = self._ext_mesh_resources("res://meshes/cube.tres")
        props = {"mesh": 'ExtResource("1")'}
        comps, _ = _components("MeshInstance3D", props, ext_resources=ext)
        assert "mesh" in comps

    def test_mesh_type_is_static(self):
        ext = self._ext_mesh_resources("res://meshes/cube.tres")
        props = {"mesh": 'ExtResource("1")'}
        comps, _ = _components("MeshInstance3D", props, ext_resources=ext)
        assert comps["mesh"]["mesh_type"] == "static"

    def test_mesh_cast_shadows_true(self):
        ext = self._ext_mesh_resources("res://meshes/cube.tres")
        props = {"mesh": 'ExtResource("1")'}
        comps, _ = _components("MeshInstance3D", props, ext_resources=ext)
        assert comps["mesh"]["cast_shadows"] is True

    def test_mesh_receive_shadows_true(self):
        ext = self._ext_mesh_resources("res://meshes/cube.tres")
        props = {"mesh": 'ExtResource("1")'}
        comps, _ = _components("MeshInstance3D", props, ext_resources=ext)
        assert comps["mesh"]["receive_shadows"] is True

    def test_fbx_mesh_goes_to_asset_reference(self):
        ext = {"1": {"type": "Mesh", "path": "res://models/hero.fbx", "uid": ""}}
        props = {"mesh": 'ExtResource("1")'}
        comps, extra = _components("MeshInstance3D", props, ext_resources=ext)
        assert "asset_reference" in extra
        assert extra["asset_reference"]["format"] == ".fbx"

    def test_obj_mesh_goes_to_asset_reference(self):
        ext = {"1": {"type": "Mesh", "path": "res://models/wall.obj", "uid": ""}}
        props = {"mesh": 'ExtResource("1")'}
        comps, extra = _components("MeshInstance3D", props, ext_resources=ext)
        assert "asset_reference" in extra
        assert extra["asset_reference"]["format"] == ".obj"

    def test_no_mesh_prop_gives_no_mesh_component(self):
        comps, _ = _components("MeshInstance3D", {})
        assert "mesh" not in comps

    def test_mesh_source_path_stored(self):
        ext = self._ext_mesh_resources("res://meshes/cube.tres")
        props = {"mesh": 'ExtResource("1")'}
        comps, _ = _components("MeshInstance3D", props, ext_resources=ext)
        assert "mesh_source_path" in comps["mesh"]

    def test_mesh_source_path_value(self):
        ext = self._ext_mesh_resources("res://meshes/cube.tres")
        props = {"mesh": 'ExtResource("1")'}
        comps, _ = _components("MeshInstance3D", props, ext_resources=ext)
        assert comps["mesh"]["mesh_source_path"] == "res://meshes/cube.tres"


# ---------------------------------------------------------------------------
# Area3D — trigger component
# ---------------------------------------------------------------------------

class TestArea3DComponents:
    def test_trigger_key_present(self):
        comps, _ = _components("Area3D", {})
        assert "trigger" in comps

    def test_trigger_is_trigger_true(self):
        comps, _ = _components("Area3D", {})
        assert comps["trigger"]["is_trigger"] is True

    def test_trigger_default_collision_layer(self):
        comps, _ = _components("Area3D", {})
        assert comps["trigger"]["collision_layer"] == 1

    def test_trigger_default_collision_mask(self):
        comps, _ = _components("Area3D", {})
        assert comps["trigger"]["collision_mask"] == 1

    def test_trigger_monitoring_default_true(self):
        comps, _ = _components("Area3D", {})
        assert comps["trigger"]["monitoring"] is True

    def test_trigger_monitorable_default_true(self):
        comps, _ = _components("Area3D", {})
        assert comps["trigger"]["monitorable"] is True

    def test_trigger_custom_collision_layer(self):
        comps, _ = _components("Area3D", {"collision_layer": "4"})
        assert comps["trigger"]["collision_layer"] == 4

    def test_trigger_monitoring_false_from_props(self):
        comps, _ = _components("Area3D", {"monitoring": "false"})
        assert comps["trigger"]["monitoring"] is False


# ---------------------------------------------------------------------------
# CharacterBody3D — physics_body component
# ---------------------------------------------------------------------------

class TestCharacterBody3DComponents:
    def test_physics_body_key_present(self):
        comps, _ = _components("CharacterBody3D", {})
        assert "physics_body" in comps

    def test_physics_body_type_character(self):
        comps, _ = _components("CharacterBody3D", {})
        assert comps["physics_body"]["type"] == "character"

    def test_physics_body_default_collision_layer(self):
        comps, _ = _components("CharacterBody3D", {})
        assert comps["physics_body"]["collision_layer"] == 1

    def test_physics_body_default_collision_mask(self):
        comps, _ = _components("CharacterBody3D", {})
        assert comps["physics_body"]["collision_mask"] == 1

    def test_physics_body_custom_collision_layer(self):
        comps, _ = _components("CharacterBody3D", {"collision_layer": "3"})
        assert comps["physics_body"]["collision_layer"] == 3

    def test_physics_body_default_motion_mode(self):
        comps, _ = _components("CharacterBody3D", {})
        assert "motion_mode" in comps["physics_body"]


# ---------------------------------------------------------------------------
# Signal event_bindings — via parse_text (attached by _attach_connections)
# ---------------------------------------------------------------------------

class TestSignalEventBindings:
    def _parse_with_connection(self, signal: str = "body_entered",
                                from_: str = "Area", to: str = "Root",
                                method: str = "_on_body_entered"):
        tscn = (
            "[gd_scene format=3]\n\n"
            "[node name=\"Root\" type=\"Node3D\"]\n\n"
            f"[node name=\"{from_}\" type=\"Area3D\" parent=\".\"]\n\n"
            f"[connection signal=\"{signal}\" from=\"{from_}\" "
            f"to=\"{to}\" method=\"{method}\"]\n"
        )
        return GodotSceneParser().parse_text(tscn, source_file="Test.tscn")

    def _find_node(self, ir, name):
        def search(nodes):
            for n in nodes:
                if n.get("node_name") == name:
                    return n
                found = search(n.get("children", []))
                if found:
                    return found
            return None
        return search(ir["nodes"])

    def test_event_bindings_added_to_source_node(self):
        ir = self._parse_with_connection()
        area = self._find_node(ir, "Area")
        assert area is not None
        assert "event_bindings" in area.get("components", {})

    def test_event_binding_signal_name_correct(self):
        ir = self._parse_with_connection(signal="body_entered")
        area = self._find_node(ir, "Area")
        bindings = area["components"]["event_bindings"]
        assert any(b["event_name"] == "body_entered" for b in bindings)

    def test_event_binding_target_correct(self):
        ir = self._parse_with_connection(to="Root")
        area = self._find_node(ir, "Area")
        bindings = area["components"]["event_bindings"]
        assert any(b["target"] == "Root" for b in bindings)

    def test_event_binding_method_correct(self):
        ir = self._parse_with_connection(method="_on_body_entered")
        area = self._find_node(ir, "Area")
        bindings = area["components"]["event_bindings"]
        assert any(b["method"] == "_on_body_entered" for b in bindings)

    def test_connections_list_in_ir(self):
        ir = self._parse_with_connection()
        assert "connections" in ir
        assert isinstance(ir["connections"], list)

    def test_no_connection_means_no_event_bindings(self):
        tscn = (
            "[gd_scene format=3]\n\n"
            "[node name=\"Root\" type=\"Node3D\"]\n\n"
            "[node name=\"Area\" type=\"Area3D\" parent=\".\"]\n"
        )
        ir = GodotSceneParser().parse_text(tscn, source_file="Test.tscn")
        area = self._find_node(ir, "Area")
        assert "event_bindings" not in area.get("components", {})
