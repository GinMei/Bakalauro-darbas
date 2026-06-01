"""
Tests for conversion_pipeline.py — ConversionResult dataclass.

The heavy imports (godot_project_builder, project_scanner, unity_parser) are
stubbed out before importing the module so no external dependencies are needed.
"""

import sys
import os
import types
from pathlib import Path
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Stub heavy imports before importing conversion_pipeline.py
# ---------------------------------------------------------------------------

def _stub_pipeline_imports():
    # Only stub modules that have heavy/unavailable dependencies.
    # ir_builder, script_converter, hierarchy_validator are pure Python
    # and are left as real modules so they don't break other test files.
    _heavy = {
        "godot_project_builder": {
            "build_godot_project": MagicMock(),
            "ProjectBuildResult": MagicMock(),
            "zip_project": MagicMock(),
        },
        "project_scanner": {
            "ProjectScanResult": MagicMock(),
            "scan_unity_project": MagicMock(),
            "extract_zip_to_dir": MagicMock(),
            "detect_project_engine": MagicMock(),
            "detect_godot_version": MagicMock(),
            "read_unity_version": MagicMock(),
        },
        "unity_parser": {
            "UnityParseError": Exception,
            "load_unity_scene_debug": MagicMock(),
        },
    }
    for mod_name, attrs in _heavy.items():
        if mod_name not in sys.modules:
            mod = types.ModuleType(mod_name)
            for attr, val in attrs.items():
                setattr(mod, attr, val)
            sys.modules[mod_name] = mod


_stub_pipeline_imports()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conversion_pipeline import ConversionResult, ConversionController, _strip_assets_prefix, _build_script_res_map, _copy_assets


# ---------------------------------------------------------------------------
# ConversionResult dataclass
# ---------------------------------------------------------------------------

class TestConversionResult:
    def test_success_true(self):
        r = ConversionResult(
            success=True,
            primary_ir={},
            all_scenes_ir={},
            scene_names=[],
            warnings=[],
            prefab_count=0,
            project_available=True,
        )
        assert r.success is True

    def test_success_false(self):
        r = ConversionResult(
            success=False,
            primary_ir={},
            all_scenes_ir={},
            scene_names=[],
            warnings=[],
            prefab_count=0,
            project_available=False,
        )
        assert r.success is False

    def test_error_default_empty(self):
        r = ConversionResult(
            success=True,
            primary_ir={},
            all_scenes_ir={},
            scene_names=[],
            warnings=[],
            prefab_count=0,
            project_available=True,
        )
        assert r.error == ""

    def test_scripts_converted_default_zero(self):
        r = ConversionResult(
            success=True,
            primary_ir={},
            all_scenes_ir={},
            scene_names=[],
            warnings=[],
            prefab_count=0,
            project_available=True,
        )
        assert r.scripts_converted == 0

    def test_scripts_failed_default_zero(self):
        r = ConversionResult(
            success=True,
            primary_ir={},
            all_scenes_ir={},
            scene_names=[],
            warnings=[],
            prefab_count=0,
            project_available=True,
        )
        assert r.scripts_failed == 0

    def test_scene_names_stored(self):
        r = ConversionResult(
            success=True,
            primary_ir={},
            all_scenes_ir={},
            scene_names=["Level1", "Level2"],
            warnings=[],
            prefab_count=0,
            project_available=True,
        )
        assert r.scene_names == ["Level1", "Level2"]

    def test_warnings_stored(self):
        r = ConversionResult(
            success=True,
            primary_ir={},
            all_scenes_ir={},
            scene_names=[],
            warnings=["warn1"],
            prefab_count=0,
            project_available=True,
        )
        assert "warn1" in r.warnings

    def test_prefab_count_stored(self):
        r = ConversionResult(
            success=True,
            primary_ir={},
            all_scenes_ir={},
            scene_names=[],
            warnings=[],
            prefab_count=5,
            project_available=True,
        )
        assert r.prefab_count == 5

    def test_custom_error(self):
        r = ConversionResult(
            success=False,
            primary_ir={},
            all_scenes_ir={},
            scene_names=[],
            warnings=[],
            prefab_count=0,
            project_available=False,
            error="Parsing failed",
        )
        assert r.error == "Parsing failed"

    def test_primary_ir_stored(self):
        ir = {"nodes": [{"node_name": "Root"}]}
        r = ConversionResult(
            success=True,
            primary_ir=ir,
            all_scenes_ir={},
            scene_names=[],
            warnings=[],
            prefab_count=0,
            project_available=True,
        )
        assert r.primary_ir is ir

    def test_all_scenes_ir_stored(self):
        all_ir = {"Level1": {"nodes": []}, "Level2": {"nodes": []}}
        r = ConversionResult(
            success=True,
            primary_ir={},
            all_scenes_ir=all_ir,
            scene_names=["Level1", "Level2"],
            warnings=[],
            prefab_count=0,
            project_available=True,
        )
        assert len(r.all_scenes_ir) == 2


# ---------------------------------------------------------------------------
# ConversionController — construction and defaults
# ---------------------------------------------------------------------------

class TestConversionController:
    def test_default_source_engine(self):
        c = ConversionController()
        assert c.source_engine == "Unity 6000.3.9f1"

    def test_default_target_engine(self):
        c = ConversionController()
        assert c.target_engine == "Godot 4.5"

    def test_custom_source_engine(self):
        c = ConversionController(source_engine="Unity 2022.3.0f1")
        assert c.source_engine == "Unity 2022.3.0f1"

    def test_custom_target_engine(self):
        c = ConversionController(target_engine="Godot 3.5")
        assert c.target_engine == "Godot 3.5"

    def test_default_unity_version(self):
        c = ConversionController()
        assert c.unity_version == "6000.3.9f1"

    def test_default_target_version(self):
        c = ConversionController()
        assert c.target_version == "4.5"

    def test_custom_unity_version(self):
        c = ConversionController(unity_version="2022.3")
        assert c.unity_version == "2022.3"

    def test_multiple_instances_independent(self):
        c1 = ConversionController(source_engine="A")
        c2 = ConversionController(source_engine="B")
        assert c1.source_engine != c2.source_engine


# ---------------------------------------------------------------------------
# _strip_assets_prefix
# ---------------------------------------------------------------------------

class TestStripAssetsPrefix:
    def test_strips_assets_prefix(self):
        result = _strip_assets_prefix(Path("Assets/Scenes/Level1.unity"))
        assert result == Path("Scenes/Level1.unity")

    def test_strips_assets_from_prefab(self):
        result = _strip_assets_prefix(Path("Assets/Prefabs/Enemy.prefab"))
        assert result == Path("Prefabs/Enemy.prefab")

    def test_no_assets_prefix_unchanged(self):
        p = Path("Scenes/Level1.unity")
        result = _strip_assets_prefix(p)
        assert result == p

    def test_only_assets_returns_dot(self):
        result = _strip_assets_prefix(Path("Assets"))
        assert result == Path(".")

    def test_nested_assets_prefix(self):
        result = _strip_assets_prefix(Path("Assets/Models/Chars/Hero.fbx"))
        assert result == Path("Models/Chars/Hero.fbx")


# ---------------------------------------------------------------------------
# _build_script_res_map
# ---------------------------------------------------------------------------

class TestBuildScriptResMap:
    def test_cs_file_included(self, tmp_path):
        cs = tmp_path / "Assets" / "Scripts" / "Player.cs"
        cs.parent.mkdir(parents=True)
        cs.touch()
        guid_map = {"guid1": cs}
        result = _build_script_res_map(guid_map, tmp_path)
        assert "guid1" in result

    def test_res_path_format(self, tmp_path):
        cs = tmp_path / "Assets" / "Scripts" / "Player.cs"
        cs.parent.mkdir(parents=True)
        cs.touch()
        guid_map = {"guid1": cs}
        result = _build_script_res_map(guid_map, tmp_path)
        assert result["guid1"].startswith("res://")

    def test_non_cs_file_excluded(self, tmp_path):
        png = tmp_path / "Assets" / "Textures" / "hero.png"
        png.parent.mkdir(parents=True)
        png.touch()
        guid_map = {"guid2": png}
        result = _build_script_res_map(guid_map, tmp_path)
        assert "guid2" not in result

    def test_assets_stripped_from_path(self, tmp_path):
        cs = tmp_path / "Assets" / "Scripts" / "Player.cs"
        cs.parent.mkdir(parents=True)
        cs.touch()
        guid_map = {"guid1": cs}
        result = _build_script_res_map(guid_map, tmp_path)
        assert "Assets" not in result["guid1"]

    def test_empty_guid_map(self, tmp_path):
        result = _build_script_res_map({}, tmp_path)
        assert result == {}

    def test_path_outside_project_root_excluded(self, tmp_path, tmp_path_factory):
        # Use a completely separate tmp directory as the "external" path
        other_dir = tmp_path_factory.mktemp("other")
        cs = other_dir / "Script.cs"
        cs.touch()
        guid_map = {"guid3": cs}
        # The path is under other_dir, not tmp_path, so relative_to(tmp_path) fails
        result = _build_script_res_map(guid_map, tmp_path)
        assert "guid3" not in result

    def test_assets_only_path_uses_filename(self, tmp_path):
        # file is directly in Assets/ with no subdirectory
        assets = tmp_path / "Assets"
        assets.mkdir()
        cs = assets / "Player.cs"
        cs.touch()
        guid_map = {"guid4": cs}
        result = _build_script_res_map(guid_map, tmp_path)
        assert "guid4" in result
        assert "Player.cs" in result["guid4"]


# ---------------------------------------------------------------------------
# _copy_assets
# ---------------------------------------------------------------------------

class TestCopyAssets:
    def test_copies_file_to_project(self, tmp_path):
        project_root = tmp_path / "project"
        project_root.mkdir()
        assets = project_root / "Assets" / "Textures"
        assets.mkdir(parents=True)
        src = assets / "tex.png"
        src.write_bytes(b"png data")

        project_dir = tmp_path / "godot_project"
        project_dir.mkdir()

        warnings = []
        count = _copy_assets(project_root, project_dir, [src], warnings)
        assert count == 1
        assert (project_dir / "Textures" / "tex.png").exists()

    def test_skips_path_outside_root(self, tmp_path, tmp_path_factory):
        project_root = tmp_path / "project"
        project_root.mkdir()
        project_dir = tmp_path / "out"
        project_dir.mkdir()

        external = tmp_path_factory.mktemp("ext") / "file.png"
        external.touch()

        warnings = []
        count = _copy_assets(project_root, project_dir, [external], warnings)
        assert count == 0

    def test_skips_existing_file(self, tmp_path):
        project_root = tmp_path / "project"
        (project_root / "Assets").mkdir(parents=True)
        src = project_root / "Assets" / "tex.png"
        src.write_bytes(b"png")

        project_dir = tmp_path / "out"
        (project_dir / "tex.png").parent.mkdir(parents=True, exist_ok=True)
        (project_dir / "tex.png").write_bytes(b"existing")

        warnings = []
        count = _copy_assets(project_root, project_dir, [src], warnings)
        assert count == 0
        assert (project_dir / "tex.png").read_bytes() == b"existing"

    def test_returns_copied_count(self, tmp_path):
        project_root = tmp_path / "proj"
        assets = project_root / "Assets"
        assets.mkdir(parents=True)
        f1 = assets / "a.png"
        f2 = assets / "b.wav"
        f1.write_bytes(b"a")
        f2.write_bytes(b"b")
        project_dir = tmp_path / "out"
        project_dir.mkdir()

        warnings = []
        count = _copy_assets(project_root, project_dir, [f1, f2], warnings)
        assert count == 2
