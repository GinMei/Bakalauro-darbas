"""
Tier 2 — UnitySceneExporter.export_instance() and export_scene_from_prefab()

Tests verify:
  • export_instance() returns (Path, str, dict) where str is a 32-char hex GUID
  • .prefab and .prefab.meta are created at the expected locations
  • The returned GUID matches _stable_guid("Assets/Prefabs/<Name>.prefab")
  • The returned GUID appears in .prefab.meta
  • The .prefab content contains the fixed root IDs (100000000, 100000001)
  • .prefab.meta uses PrefabImporter (not DefaultImporter)
  • export_scene_from_prefab() creates .unity + .unity.meta
  • The generated scene references the prefab GUID via !u!1001 (PrefabInstance)
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
# export_instance — return type and files
# ---------------------------------------------------------------------------

class TestExportPrefabReturnType:
    def test_returns_tuple(self, tmp_path):
        result = UnitySceneExporter().export_instance(_minimal_ir(), tmp_path)
        assert isinstance(result, tuple) and len(result) == 3

    def test_first_element_is_path(self, tmp_path):
        path, *_ = UnitySceneExporter().export_instance(_minimal_ir(), tmp_path)
        assert isinstance(path, Path)

    def test_second_element_is_string(self, tmp_path):
        _, guid, _ = UnitySceneExporter().export_instance(_minimal_ir(), tmp_path)
        assert isinstance(guid, str)

    def test_returned_path_exists(self, tmp_path):
        path, *_ = UnitySceneExporter().export_instance(_minimal_ir(), tmp_path)
        assert path.exists()

    def test_returned_path_suffix(self, tmp_path):
        path, *_ = UnitySceneExporter().export_instance(_minimal_ir(), tmp_path)
        assert path.suffix == ".prefab"


class TestExportPrefabFiles:
    def test_default_output_path(self, tmp_path):
        path, *_ = UnitySceneExporter().export_instance(_minimal_ir("test_scene"), tmp_path)
        assert path == tmp_path / "Assets" / "Prefabs" / "test_scene.prefab"

    def test_godot_relative_dir_used(self, tmp_path):
        path, *_ = UnitySceneExporter().export_instance(
            _minimal_ir("test_scene"), tmp_path,
            godot_relative_dir=Path("objects"),
        )
        assert path == tmp_path / "Assets" / "objects" / "test_scene.prefab"

    def test_prefab_meta_created(self, tmp_path):
        path, *_ = UnitySceneExporter().export_instance(_minimal_ir(), tmp_path)
        assert (path.parent / (path.name + ".meta")).exists()

    def test_folder_meta_created(self, tmp_path):
        UnitySceneExporter().export_instance(_minimal_ir(), tmp_path)
        assert (tmp_path / "Assets" / "Prefabs.meta").exists()


class TestExportPrefabGuid:
    def test_returned_guid_is_32_char_hex(self, tmp_path):
        _, guid, _ = UnitySceneExporter().export_instance(_minimal_ir(), tmp_path)
        assert len(guid) == 32
        assert all(c in "0123456789abcdef" for c in guid)

    def test_returned_guid_matches_stable_formula(self, tmp_path):
        _, guid, _ = UnitySceneExporter().export_instance(_minimal_ir("test_scene"), tmp_path)
        assert guid == _stable_guid("Assets/Prefabs/test_scene.prefab")

    def test_returned_guid_appears_in_meta(self, tmp_path):
        path, guid, _ = UnitySceneExporter().export_instance(_minimal_ir(), tmp_path)
        meta_text = (path.parent / (path.name + ".meta")).read_text(encoding="utf-8")
        assert guid in meta_text

    def test_meta_has_prefab_importer(self, tmp_path):
        path, *_ = UnitySceneExporter().export_instance(_minimal_ir(), tmp_path)
        meta_text = (path.parent / (path.name + ".meta")).read_text(encoding="utf-8")
        assert "PrefabImporter:" in meta_text

    def test_meta_no_default_importer(self, tmp_path):
        path, *_ = UnitySceneExporter().export_instance(_minimal_ir(), tmp_path)
        meta_text = (path.parent / (path.name + ".meta")).read_text(encoding="utf-8")
        assert "DefaultImporter:" not in meta_text

    def test_guid_stable_across_independent_runs(self, tmp_path):
        _, guid1, _ = UnitySceneExporter().export_instance(_minimal_ir("test_scene"), tmp_path / "run1")
        _, guid2, _ = UnitySceneExporter().export_instance(_minimal_ir("test_scene"), tmp_path / "run2")
        assert guid1 == guid2


class TestExportPrefabContent:
    def test_yaml_header_present(self, tmp_path):
        path, *_ = UnitySceneExporter().export_instance(_minimal_ir(), tmp_path)
        assert path.read_text(encoding="utf-8").startswith("%YAML 1.1")

    def test_root_go_id_present(self, tmp_path):
        path, *_ = UnitySceneExporter().export_instance(_minimal_ir(), tmp_path)
        assert "&100000000" in path.read_text(encoding="utf-8")

    def test_root_transform_id_present(self, tmp_path):
        path, *_ = UnitySceneExporter().export_instance(_minimal_ir(), tmp_path)
        assert "&100000001" in path.read_text(encoding="utf-8")

    def test_game_object_class_id_present(self, tmp_path):
        path, *_ = UnitySceneExporter().export_instance(_minimal_ir(), tmp_path)
        assert "!u!1 &" in path.read_text(encoding="utf-8")

    def test_transform_class_id_present(self, tmp_path):
        path, *_ = UnitySceneExporter().export_instance(_minimal_ir(), tmp_path)
        assert "!u!4 &" in path.read_text(encoding="utf-8")

    def test_meta_file_format_version(self, tmp_path):
        path, *_ = UnitySceneExporter().export_instance(_minimal_ir(), tmp_path)
        meta_text = (path.parent / (path.name + ".meta")).read_text(encoding="utf-8")
        assert "fileFormatVersion: 2" in meta_text


# ---------------------------------------------------------------------------
# export_scene_from_prefab
# ---------------------------------------------------------------------------

class TestExportSceneFromPrefab:
    _PREFAB_GUID = _stable_guid("Assets/Prefabs/test_scene.prefab")

    def _export(self, tmp_path, scene_name: str = "test_scene") -> Path:
        return UnitySceneExporter().export_scene_from_prefab(
            _minimal_ir(scene_name), tmp_path, prefab_guid=self._PREFAB_GUID
        )

    def test_returns_path(self, tmp_path):
        assert isinstance(self._export(tmp_path), Path)

    def test_unity_file_created(self, tmp_path):
        assert self._export(tmp_path).exists()

    def test_unity_meta_created(self, tmp_path):
        result = self._export(tmp_path)
        assert (result.parent / (result.name + ".meta")).exists()

    def test_default_output_path(self, tmp_path):
        result = self._export(tmp_path, "test_scene")
        assert result == tmp_path / "Assets" / "Scenes" / "test_scene.unity"

    def test_yaml_header_present(self, tmp_path):
        result = self._export(tmp_path)
        assert result.read_text(encoding="utf-8").startswith("%YAML 1.1")

    def test_prefab_instance_class_id(self, tmp_path):
        result = self._export(tmp_path)
        assert "!u!1001 &" in result.read_text(encoding="utf-8")

    def test_prefab_guid_referenced_in_scene(self, tmp_path):
        result = self._export(tmp_path)
        assert self._PREFAB_GUID in result.read_text(encoding="utf-8")

    def test_meta_has_default_importer(self, tmp_path):
        result = self._export(tmp_path)
        meta_text = (result.parent / (result.name + ".meta")).read_text(encoding="utf-8")
        assert "DefaultImporter:" in meta_text

    def test_meta_guid_matches_stable_formula(self, tmp_path):
        result = self._export(tmp_path, "test_scene")
        meta_text = (result.parent / (result.name + ".meta")).read_text(encoding="utf-8")
        assert _stable_guid("Assets/Scenes/test_scene.unity") in meta_text
