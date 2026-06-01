"""
Tier 1 — Reference graph construction and scene/prefab classification.

Correct classification is critical: a wrong result silently drops scenes
from output (INSTANCE-only files never call export()) or exports instances
as full scenes (SCENE-only calls export() instead of export_instance()).
"""

import sys
import os

import pytest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from godot_to_unity.godot_scene_parser import build_reference_graph, classify_by_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph(scene_file: Path, res_path: str = None, **overrides):
    """
    Build a minimal ref_graph for a single file.
    All sets/maps start empty; callers add evidence as needed.
    """
    rp = res_path or f"res://{scene_file.name}"
    graph = {
        "referenced_res_paths": set(),
        "main_scene_res_path": "",
        "file_to_res": {str(scene_file): rp},
        "script_loaded_res_paths": set(),
        "script_instantiate_paths": set(),
        "script_scene_change_paths": set(),
        "dynamic_scene_prefixes": set(),
        "dynamic_prefab_prefixes": set(),
        "usage_map": {},
    }
    graph.update(overrides)
    return graph, rp


def _classify(scene_file: Path, **graph_kwargs):
    graph, rp = _make_graph(scene_file, **graph_kwargs)
    return classify_by_graph(scene_file, graph), rp


# ---------------------------------------------------------------------------
# classify_by_graph — unreferenced defaults
# ---------------------------------------------------------------------------

class TestClassifyUnreferenced:
    def test_unreferenced_file_is_scene(self, tmp_path):
        f = tmp_path / "Level.tscn"
        f.touch()
        result, _ = _classify(f)
        assert result["is_scene"] is True
        assert result["is_instance"] is False

    def test_unreferenced_file_not_prefab(self, tmp_path):
        f = tmp_path / "Menu.tscn"
        f.touch()
        result, _ = _classify(f)
        assert result["is_instance"] is False

    def test_file_not_in_file_to_res_defaults_to_scene(self, tmp_path):
        """When the file has no res:// mapping, is_prefab stays False → is_scene True."""
        f = tmp_path / "Unknown.tscn"
        f.touch()
        graph = {
            "referenced_res_paths": set(),
            "main_scene_res_path": "",
            "file_to_res": {},  # file not in map
            "script_loaded_res_paths": set(),
            "script_instantiate_paths": set(),
            "script_scene_change_paths": set(),
            "dynamic_scene_prefixes": set(),
            "dynamic_prefab_prefixes": set(),
            "usage_map": {},
        }
        result = classify_by_graph(f, graph)
        assert result["is_scene"] is True
        assert result["is_instance"] is False


# ---------------------------------------------------------------------------
# classify_by_graph — PREFAB evidence
# ---------------------------------------------------------------------------

class TestClassifyPrefab:
    def test_embedded_in_another_tscn_is_prefab(self, tmp_path):
        f = tmp_path / "Crate.tscn"
        f.touch()
        graph, rp = _make_graph(f)
        graph["referenced_res_paths"].add(rp)
        result = classify_by_graph(f, graph)
        assert result["is_instance"] is True

    def test_embedded_prefab_is_not_scene(self, tmp_path):
        """Embedded-only file has no scene evidence → is_scene False."""
        f = tmp_path / "Crate.tscn"
        f.touch()
        graph, rp = _make_graph(f)
        graph["referenced_res_paths"].add(rp)
        result = classify_by_graph(f, graph)
        assert result["is_scene"] is False

    def test_script_instantiated_is_prefab(self, tmp_path):
        f = tmp_path / "Enemy.tscn"
        f.touch()
        graph, rp = _make_graph(f)
        graph["script_instantiate_paths"].add(rp)
        result = classify_by_graph(f, graph)
        assert result["is_instance"] is True
        assert result["is_scene"] is False

    def test_dynamic_prefab_prefix_is_prefab(self, tmp_path):
        d = tmp_path / "objects"
        d.mkdir()
        f = d / "Barrel.tscn"
        f.touch()
        graph, _ = _make_graph(f, res_path="res://objects/Barrel.tscn")
        graph["dynamic_prefab_prefixes"].add("res://objects/")
        result = classify_by_graph(f, graph)
        assert result["is_instance"] is True


# ---------------------------------------------------------------------------
# classify_by_graph — SCENE evidence
# ---------------------------------------------------------------------------

class TestClassifyScene:
    def test_main_scene_is_scene(self, tmp_path):
        f = tmp_path / "Main.tscn"
        f.touch()
        graph, rp = _make_graph(f)
        graph["main_scene_res_path"] = rp
        result = classify_by_graph(f, graph)
        assert result["is_scene"] is True
        assert result["is_instance"] is False

    def test_scene_change_to_is_scene(self, tmp_path):
        f = tmp_path / "GameOver.tscn"
        f.touch()
        graph, rp = _make_graph(f)
        graph["script_scene_change_paths"].add(rp)
        result = classify_by_graph(f, graph)
        assert result["is_scene"] is True
        assert result["is_instance"] is False

    def test_ambiguous_load_treated_as_scene(self, tmp_path):
        """A load() without .instantiate() is ambiguous → scene evidence."""
        f = tmp_path / "Ambiguous.tscn"
        f.touch()
        graph, rp = _make_graph(f)
        graph["script_loaded_res_paths"].add(rp)
        # NOT added to script_instantiate_paths → ambiguous
        result = classify_by_graph(f, graph)
        assert result["is_scene"] is True

    def test_instantiated_load_not_ambiguous(self, tmp_path):
        """load().instantiate() is explicit prefab use, not ambiguous scene use."""
        f = tmp_path / "Widget.tscn"
        f.touch()
        graph, rp = _make_graph(f)
        graph["script_loaded_res_paths"].add(rp)
        graph["script_instantiate_paths"].add(rp)  # explicit prefab
        result = classify_by_graph(f, graph)
        assert result["is_instance"] is True
        assert result["is_scene"] is False  # ambiguous_load condition not met

    def test_dynamic_scene_prefix_is_scene(self, tmp_path):
        d = tmp_path / "levels"
        d.mkdir()
        f = d / "Forest.tscn"
        f.touch()
        graph, _ = _make_graph(f, res_path="res://levels/Forest.tscn")
        graph["dynamic_scene_prefixes"].add("res://levels/")
        result = classify_by_graph(f, graph)
        assert result["is_scene"] is True


# ---------------------------------------------------------------------------
# classify_by_graph — BOTH case
# ---------------------------------------------------------------------------

class TestClassifyBoth:
    def test_embedded_and_main_scene_is_both(self, tmp_path):
        f = tmp_path / "Level.tscn"
        f.touch()
        graph, rp = _make_graph(f)
        graph["referenced_res_paths"].add(rp)
        graph["main_scene_res_path"] = rp
        result = classify_by_graph(f, graph)
        assert result["is_instance"] is True
        assert result["is_scene"] is True

    def test_embedded_and_scene_change_is_both(self, tmp_path):
        f = tmp_path / "Hub.tscn"
        f.touch()
        graph, rp = _make_graph(f)
        graph["referenced_res_paths"].add(rp)
        graph["script_scene_change_paths"].add(rp)
        result = classify_by_graph(f, graph)
        assert result["is_instance"] is True
        assert result["is_scene"] is True

    def test_script_instantiated_and_main_is_both(self, tmp_path):
        f = tmp_path / "Boss.tscn"
        f.touch()
        graph, rp = _make_graph(f)
        graph["script_instantiate_paths"].add(rp)
        graph["main_scene_res_path"] = rp
        result = classify_by_graph(f, graph)
        assert result["is_instance"] is True
        assert result["is_scene"] is True


# ---------------------------------------------------------------------------
# build_reference_graph — integration tests using real temp files
# ---------------------------------------------------------------------------

class TestBuildReferenceGraph:
    def test_empty_scene_list_returns_empty_graph(self):
        graph = build_reference_graph([])
        assert graph["referenced_res_paths"] == set()
        assert graph["main_scene_res_path"] == ""
        assert graph["usage_map"] == {}

    def test_single_unreferenced_scene_not_in_referenced(self, tmp_path):
        (tmp_path / "project.godot").write_text("[application]\n")
        scene = tmp_path / "Level.tscn"
        scene.write_text('[gd_scene format=3]\n[node name="Root" type="Node3D"]\n')

        graph = build_reference_graph([scene], godot_root=tmp_path)

        assert "res://Level.tscn" not in graph["referenced_res_paths"]

    def test_instance_reference_added_to_referenced(self, tmp_path):
        (tmp_path / "project.godot").write_text("[application]\n")
        crate = tmp_path / "crate.tscn"
        crate.write_text('[gd_scene format=3]\n[node name="Crate" type="Node3D"]\n')
        level = tmp_path / "Level.tscn"
        level.write_text(
            '[gd_scene load_steps=2 format=3]\n\n'
            '[ext_resource type="PackedScene" path="res://crate.tscn" id="1"]\n\n'
            '[node name="Level" type="Node3D"]\n'
            '[node name="CrateInst" parent="." instance=ExtResource("1")]\n'
        )

        graph = build_reference_graph([level, crate], godot_root=tmp_path)

        assert "res://crate.tscn" in graph["referenced_res_paths"]
        assert "res://Level.tscn" not in graph["referenced_res_paths"]

    def test_embedded_scene_recorded_in_usage_map(self, tmp_path):
        (tmp_path / "project.godot").write_text("[application]\n")
        crate = tmp_path / "crate.tscn"
        crate.write_text('[gd_scene format=3]\n[node name="Crate" type="Node3D"]\n')
        level = tmp_path / "Level.tscn"
        level.write_text(
            '[gd_scene load_steps=2 format=3]\n\n'
            '[ext_resource type="PackedScene" path="res://crate.tscn" id="1"]\n\n'
            '[node name="Level" type="Node3D"]\n'
            '[node name="CrateInst" parent="." instance=ExtResource("1")]\n'
        )

        graph = build_reference_graph([level, crate], godot_root=tmp_path)

        assert "res://crate.tscn" in graph["usage_map"]
        assert "res://Level.tscn" in graph["usage_map"]["res://crate.tscn"]["embedded_in"]

    def test_main_scene_read_from_project_godot(self, tmp_path):
        (tmp_path / "project.godot").write_text(
            '[application]\nrun/main_scene="res://Main.tscn"\n'
        )
        main = tmp_path / "Main.tscn"
        main.write_text('[gd_scene format=3]\n[node name="Root" type="Node3D"]\n')

        graph = build_reference_graph([main], godot_root=tmp_path)

        assert graph["main_scene_res_path"] == "res://Main.tscn"

    def test_no_project_godot_main_scene_empty(self, tmp_path):
        scene = tmp_path / "Level.tscn"
        scene.write_text('[gd_scene format=3]\n[node name="Root" type="Node3D"]\n')

        graph = build_reference_graph([scene], godot_root=tmp_path)

        assert graph["main_scene_res_path"] == ""

    def test_script_instantiate_detected(self, tmp_path):
        (tmp_path / "project.godot").write_text("[application]\n")
        enemy = tmp_path / "enemy.tscn"
        enemy.write_text('[gd_scene format=3]\n[node name="Enemy" type="Node3D"]\n')
        script = tmp_path / "spawner.cs"
        script.write_text(
            'var e = GD.Load<PackedScene>("res://enemy.tscn").Instantiate();\n'
        )

        graph = build_reference_graph(
            [enemy], godot_root=tmp_path, script_files=[script]
        )

        assert "res://enemy.tscn" in graph["script_instantiate_paths"]
        assert "res://enemy.tscn" not in graph["script_scene_change_paths"]

    def test_scene_change_to_file_detected(self, tmp_path):
        (tmp_path / "project.godot").write_text("[application]\n")
        over = tmp_path / "gameover.tscn"
        over.write_text('[gd_scene format=3]\n[node name="Root" type="Node3D"]\n')
        script = tmp_path / "game.cs"
        script.write_text('GetTree().ChangeSceneToFile("res://gameover.tscn");\n')

        graph = build_reference_graph(
            [over], godot_root=tmp_path, script_files=[script]
        )

        assert "res://gameover.tscn" in graph["script_scene_change_paths"]
        assert "res://gameover.tscn" not in graph["script_instantiate_paths"]

    def test_file_to_res_mapping_populated(self, tmp_path):
        (tmp_path / "project.godot").write_text("[application]\n")
        scene = tmp_path / "Level.tscn"
        scene.write_text('[gd_scene format=3]\n[node name="Root" type="Node3D"]\n')

        graph = build_reference_graph([scene], godot_root=tmp_path)

        assert str(scene) in graph["file_to_res"]
        assert graph["file_to_res"][str(scene)] == "res://Level.tscn"

    def test_multiple_instances_all_recorded(self, tmp_path):
        (tmp_path / "project.godot").write_text("[application]\n")
        crate = tmp_path / "crate.tscn"
        crate.write_text('[gd_scene format=3]\n[node name="Crate" type="Node3D"]\n')
        barrel = tmp_path / "barrel.tscn"
        barrel.write_text('[gd_scene format=3]\n[node name="Barrel" type="Node3D"]\n')
        level = tmp_path / "Level.tscn"
        level.write_text(
            '[gd_scene load_steps=3 format=3]\n\n'
            '[ext_resource type="PackedScene" path="res://crate.tscn" id="1"]\n'
            '[ext_resource type="PackedScene" path="res://barrel.tscn" id="2"]\n\n'
            '[node name="Level" type="Node3D"]\n'
            '[node name="C" parent="." instance=ExtResource("1")]\n'
            '[node name="B" parent="." instance=ExtResource("2")]\n'
        )

        graph = build_reference_graph([level, crate, barrel], godot_root=tmp_path)

        assert "res://crate.tscn" in graph["referenced_res_paths"]
        assert "res://barrel.tscn" in graph["referenced_res_paths"]


# ---------------------------------------------------------------------------
# Round-trip: build_reference_graph → classify_by_graph
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_unreferenced_scene_classified_as_scene(self, tmp_path):
        (tmp_path / "project.godot").write_text("[application]\n")
        scene = tmp_path / "Level.tscn"
        scene.write_text('[gd_scene format=3]\n[node name="Root" type="Node3D"]\n')

        graph = build_reference_graph([scene], godot_root=tmp_path)
        result = classify_by_graph(scene, graph)

        assert result["is_scene"] is True
        assert result["is_instance"] is False

    def test_instanced_scene_classified_as_prefab(self, tmp_path):
        (tmp_path / "project.godot").write_text("[application]\n")
        crate = tmp_path / "crate.tscn"
        crate.write_text('[gd_scene format=3]\n[node name="Crate" type="Node3D"]\n')
        level = tmp_path / "Level.tscn"
        level.write_text(
            '[gd_scene load_steps=2 format=3]\n\n'
            '[ext_resource type="PackedScene" path="res://crate.tscn" id="1"]\n\n'
            '[node name="Level" type="Node3D"]\n'
            '[node name="C" parent="." instance=ExtResource("1")]\n'
        )

        graph = build_reference_graph([level, crate], godot_root=tmp_path)
        crate_result = classify_by_graph(crate, graph)
        level_result = classify_by_graph(level, graph)

        assert crate_result["is_instance"] is True
        assert crate_result["is_scene"] is False
        assert level_result["is_scene"] is True
        assert level_result["is_instance"] is False

    def test_main_scene_classified_as_scene(self, tmp_path):
        (tmp_path / "project.godot").write_text(
            '[application]\nrun/main_scene="res://Main.tscn"\n'
        )
        main = tmp_path / "Main.tscn"
        main.write_text('[gd_scene format=3]\n[node name="Root" type="Node3D"]\n')

        graph = build_reference_graph([main], godot_root=tmp_path)
        result = classify_by_graph(main, graph)

        assert result["is_scene"] is True
        assert result["is_instance"] is False
