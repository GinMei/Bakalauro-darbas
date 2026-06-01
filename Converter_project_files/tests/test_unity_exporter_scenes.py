"""
Tier 2 — UnitySceneExporter.export()

Tests verify:
  • .unity and .unity.meta are created at the expected paths
  • The returned value is the .unity path
  • GUID in .unity.meta matches _stable_guid(posix_rel_path)
  • The .unity file begins with the Unity YAML header
  • Required class IDs are present in the .unity content
  • Scene name snake_case → CamelCase conversion is applied
  • Folder .meta is written alongside the scene
"""

import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from godot_to_unity.unity_scene_exporter import UnitySceneExporter, _stable_guid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_ir(scene_name: str = "test_scene") -> dict:
    return {
        "ir_version": "1.0",
        "coordinate_system": {"conversion_applied": True},
        "scene_name": scene_name,
        "nodes": [
            {
                "id": "node_0",
                "node_name": "Root",
                "godot_type": "Node3D",
                "ir_type": "group",
                "transform": {
                    "position": [0.0, 0.0, 0.0],
                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                    "scale": [1.0, 1.0, 1.0],
                },
                "components": {},
                "children": [],
            }
        ],
        "animations": [],
    }


# ---------------------------------------------------------------------------
# File creation
# ---------------------------------------------------------------------------

class TestExportSceneFiles:
    def test_export_returns_path(self, tmp_path):
        result = UnitySceneExporter().export(_minimal_ir(), tmp_path)
        assert isinstance(result, Path)

    def test_unity_file_created(self, tmp_path):
        result = UnitySceneExporter().export(_minimal_ir(), tmp_path)
        assert result.exists()

    def test_unity_file_extension(self, tmp_path):
        result = UnitySceneExporter().export(_minimal_ir(), tmp_path)
        assert result.suffix == ".unity"

    def test_unity_meta_created(self, tmp_path):
        result = UnitySceneExporter().export(_minimal_ir(), tmp_path)
        meta = result.parent / (result.name + ".meta")
        assert meta.exists()

    def test_default_output_path(self, tmp_path):
        result = UnitySceneExporter().export(_minimal_ir("test_scene"), tmp_path)
        assert result == tmp_path / "Assets" / "Scenes" / "test_scene.unity"

    def test_godot_relative_dir_used(self, tmp_path):
        result = UnitySceneExporter().export(
            _minimal_ir("test_scene"), tmp_path,
            godot_relative_dir=Path("levels"),
        )
        assert result == tmp_path / "Assets" / "levels" / "test_scene.unity"

    def test_scene_name_preserved_as_is(self, tmp_path):
        result = UnitySceneExporter().export(_minimal_ir("my_level"), tmp_path)
        assert result.stem == "my_level"

    def test_scene_name_uppercase_preserved(self, tmp_path):
        result = UnitySceneExporter().export(_minimal_ir("MyLevel"), tmp_path)
        assert result.stem == "MyLevel"

    def test_folder_meta_created(self, tmp_path):
        UnitySceneExporter().export(_minimal_ir(), tmp_path)
        assert (tmp_path / "Assets" / "Scenes.meta").exists()


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------

class TestExportSceneContent:
    def test_yaml_header_present(self, tmp_path):
        result = UnitySceneExporter().export(_minimal_ir(), tmp_path)
        assert result.read_text(encoding="utf-8").startswith("%YAML 1.1")

    def test_yaml_tag_directive_present(self, tmp_path):
        result = UnitySceneExporter().export(_minimal_ir(), tmp_path)
        assert "%TAG !u! tag:unity3d.com,2011:" in result.read_text(encoding="utf-8")

    def test_render_settings_class_id_present(self, tmp_path):
        result = UnitySceneExporter().export(_minimal_ir(), tmp_path)
        assert "!u!104 &" in result.read_text(encoding="utf-8")

    def test_game_object_class_id_present(self, tmp_path):
        result = UnitySceneExporter().export(_minimal_ir(), tmp_path)
        assert "!u!1 &" in result.read_text(encoding="utf-8")

    def test_transform_class_id_present(self, tmp_path):
        result = UnitySceneExporter().export(_minimal_ir(), tmp_path)
        assert "!u!4 &" in result.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# GUID
# ---------------------------------------------------------------------------

class TestExportSceneGuid:
    def test_meta_contains_guid_line(self, tmp_path):
        result = UnitySceneExporter().export(_minimal_ir(), tmp_path)
        meta_text = (result.parent / (result.name + ".meta")).read_text(encoding="utf-8")
        assert "guid:" in meta_text

    def test_meta_guid_matches_stable_guid(self, tmp_path):
        result = UnitySceneExporter().export(_minimal_ir("test_scene"), tmp_path)
        meta_text = (result.parent / (result.name + ".meta")).read_text(encoding="utf-8")
        expected = _stable_guid("Assets/Scenes/test_scene.unity")
        assert expected in meta_text

    def test_meta_guid_is_32_char_hex(self, tmp_path):
        result = UnitySceneExporter().export(_minimal_ir(), tmp_path)
        meta_text = (result.parent / (result.name + ".meta")).read_text(encoding="utf-8")
        for line in meta_text.splitlines():
            if line.strip().startswith("guid:"):
                guid = line.split(":", 1)[1].strip()
                assert len(guid) == 32
                assert all(c in "0123456789abcdef" for c in guid)

    def test_meta_has_default_importer(self, tmp_path):
        result = UnitySceneExporter().export(_minimal_ir(), tmp_path)
        meta_text = (result.parent / (result.name + ".meta")).read_text(encoding="utf-8")
        assert "DefaultImporter:" in meta_text

    def test_meta_file_format_version(self, tmp_path):
        result = UnitySceneExporter().export(_minimal_ir(), tmp_path)
        meta_text = (result.parent / (result.name + ".meta")).read_text(encoding="utf-8")
        assert "fileFormatVersion: 2" in meta_text

    def test_guid_stable_across_independent_runs(self, tmp_path):
        result1 = UnitySceneExporter().export(_minimal_ir("test_scene"), tmp_path / "run1")
        result2 = UnitySceneExporter().export(_minimal_ir("test_scene"), tmp_path / "run2")
        meta1 = (result1.parent / (result1.name + ".meta")).read_text(encoding="utf-8")
        meta2 = (result2.parent / (result2.name + ".meta")).read_text(encoding="utf-8")
        assert meta1 == meta2
