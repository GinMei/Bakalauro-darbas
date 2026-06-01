"""
Tier 4 — GodotToUnityPipeline.convert() end-to-end integration test.

Creates a minimal in-memory Godot project on disk (tmp_path), runs the full
pipeline with convert_scripts=False (no Gemini/Ollama needed), and checks that
the output directory contains expected Unity project artefacts.

What is checked:
  • result.success is True
  • At least one .unity or .prefab was exported
  • output_dir/Assets/ exists
  • output_dir/ProjectSettings/ProjectVersion.txt exists
  • Every exported .unity file starts with "%YAML 1.1"
  • Every exported .prefab file starts with "%YAML 1.1"
  • .meta files exist alongside every .unity and .prefab
"""

import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from godot_to_unity.godot_to_unity_pipeline import GodotToUnityPipeline


# ---------------------------------------------------------------------------
# Minimal Godot project fixture helpers
# ---------------------------------------------------------------------------

_MINIMAL_TSCN = """\
[gd_scene format=3]

[node name="Level" type="Node3D"]
"""

_MINIMAL_PROJECT_GODOT = """\
[application]
config/name="TestProject"
config/features=PackedStringArray("4.5")
"""


def _make_godot_project(root: Path, tscn_name: str = "Level.tscn") -> Path:
    """Write a minimal valid Godot project to *root*."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "project.godot").write_text(_MINIMAL_PROJECT_GODOT, encoding="utf-8")
    (root / tscn_name).write_text(_MINIMAL_TSCN, encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    @pytest.fixture
    def project(self, tmp_path) -> tuple:
        godot_root = tmp_path / "godot"
        output_dir = tmp_path / "unity"
        _make_godot_project(godot_root)
        result = GodotToUnityPipeline().convert(
            godot_root=godot_root,
            output_dir=output_dir,
            project_name="TestProject",
            convert_scripts=False,
        )
        return result, output_dir

    def test_result_success(self, project):
        result, _ = project
        assert result.success is True

    def test_at_least_one_scene_or_prefab_exported(self, project):
        result, _ = project
        total = len(result.scenes_exported) + len(result.prefabs_exported)
        assert total >= 1

    def test_assets_dir_created(self, project):
        _, output_dir = project
        assert (output_dir / "Assets").is_dir()

    def test_project_settings_dir_created(self, project):
        _, output_dir = project
        assert (output_dir / "ProjectSettings").is_dir()

    def test_project_version_txt_exists(self, project):
        _, output_dir = project
        assert (output_dir / "ProjectSettings" / "ProjectVersion.txt").exists()

    def test_unity_files_have_yaml_header(self, project):
        result, _ = project
        for p in result.scenes_exported:
            assert p.read_text(encoding="utf-8").startswith("%YAML 1.1"), (
                f"{p.name} does not start with YAML header"
            )

    def test_prefab_files_have_yaml_header(self, project):
        result, _ = project
        for p in result.prefabs_exported:
            assert p.read_text(encoding="utf-8").startswith("%YAML 1.1"), (
                f"{p.name} does not start with YAML header"
            )

    def test_meta_files_alongside_unity(self, project):
        result, _ = project
        for p in result.scenes_exported:
            meta = p.parent / (p.name + ".meta")
            assert meta.exists(), f"Missing .meta for {p.name}"

    def test_meta_files_alongside_prefab(self, project):
        result, _ = project
        for p in result.prefabs_exported:
            meta = p.parent / (p.name + ".meta")
            assert meta.exists(), f"Missing .meta for {p.name}"

    def test_no_scenes_failed(self, project):
        result, _ = project
        assert result.scenes_failed == []

    def test_result_has_scene_irs(self, project):
        result, _ = project
        assert len(result.scene_irs) >= 1


class TestPipelineNoScenes:
    def test_failure_when_no_tscn_files(self, tmp_path):
        godot_root = tmp_path / "empty"
        godot_root.mkdir()
        (godot_root / "project.godot").write_text(_MINIMAL_PROJECT_GODOT, encoding="utf-8")

        result = GodotToUnityPipeline().convert(
            godot_root=godot_root,
            output_dir=tmp_path / "unity",
            convert_scripts=False,
        )
        assert result.success is False
        assert result.error


class TestPipelineMultipleScenes:
    def test_all_scenes_processed(self, tmp_path):
        godot_root = tmp_path / "godot"
        godot_root.mkdir()
        (godot_root / "project.godot").write_text(_MINIMAL_PROJECT_GODOT, encoding="utf-8")
        (godot_root / "Level1.tscn").write_text(_MINIMAL_TSCN, encoding="utf-8")
        (godot_root / "Level2.tscn").write_text(_MINIMAL_TSCN, encoding="utf-8")

        output_dir = tmp_path / "unity"
        result = GodotToUnityPipeline().convert(
            godot_root=godot_root,
            output_dir=output_dir,
            convert_scripts=False,
        )
        total = len(result.scenes_exported) + len(result.prefabs_exported)
        assert total >= 2
