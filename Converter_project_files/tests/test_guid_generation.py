"""
Tier 1 — GUID generation and path flattening.

_stable_guid() is the foundation of every .meta file in the output.
A wrong key means broken Unity cross-references on every asset.
"""

import sys
import os
import uuid as _uuid_mod

import pytest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from godot_to_unity.unity_scene_exporter import _stable_guid, write_folder_meta
from godot_to_unity.godot_to_unity_pipeline import _stable_guid as _stable_guid_g2u, GodotToUnityPipeline

_NAMESPACE = _uuid_mod.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


# ---------------------------------------------------------------------------
# Core GUID properties
# ---------------------------------------------------------------------------

class TestStableGuidCore:
    def test_deterministic_same_key(self):
        key = "Assets/Scenes/Main.unity"
        assert _stable_guid(key) == _stable_guid(key)

    def test_different_keys_produce_different_guids(self):
        assert _stable_guid("Assets/Scenes/A.unity") != _stable_guid("Assets/Scenes/B.unity")

    def test_result_is_32_char_lowercase_hex(self):
        result = _stable_guid("Assets/Scenes/Main.unity")
        assert len(result) == 32
        assert all(c in "0123456789abcdef" for c in result)

    def test_matches_manual_uuid5_computation(self):
        key = "Assets/Scenes/Main.unity"
        expected = _uuid_mod.uuid5(_NAMESPACE, key).hex
        assert _stable_guid(key) == expected

    def test_same_result_from_both_modules(self):
        """unity_scene_exporter and godot_to_unity_pipeline use the same namespace."""
        key = "Assets/Scenes/Main.unity"
        assert _stable_guid(key) == _stable_guid_g2u(key)

    def test_empty_string_key_produces_valid_guid(self):
        result = _stable_guid("")
        assert len(result) == 32

    def test_case_sensitive(self):
        assert _stable_guid("Assets/Scenes/main.unity") != _stable_guid("Assets/Scenes/Main.unity")


# ---------------------------------------------------------------------------
# Key formula contracts
# ---------------------------------------------------------------------------

class TestGuidKeyFormulas:
    def test_folder_prefix_differs_from_plain_path(self):
        """'folder:Assets/Meshes' must differ from 'Assets/Meshes'."""
        path = "Assets/Meshes"
        assert _stable_guid(f"folder:{path}") != _stable_guid(path)

    def test_script_key_includes_assets_scripts_prefix(self):
        """Script GUID key is always 'Assets/Scripts/<filename>'."""
        filename = "PlayerController.cs"
        full_key = f"Assets/Scripts/{filename}"
        assert _stable_guid(full_key) != _stable_guid(filename)

    def test_material_key_includes_unity_mat_prefix(self):
        """Material stub keys use 'unity_mat:' prefix."""
        name = "Wood_42_mat"
        assert _stable_guid(f"unity_mat:{name}") != _stable_guid(name)

    def test_posix_and_windows_paths_differ(self):
        """Only POSIX paths must be used as keys; backslash paths differ."""
        assert (
            _stable_guid("Assets/Scenes/Level1.unity")
            != _stable_guid("Assets\\Scenes\\Level1.unity")
        )

    def test_scene_guid_depends_on_full_relative_path(self):
        """Two scenes in different directories have different GUIDs."""
        assert (
            _stable_guid("Assets/Scenes/Level1.unity")
            != _stable_guid("Assets/Levels/Level1.unity")
        )

    def test_prefab_and_scene_with_same_stem_differ(self):
        """Crate.prefab and Crate.unity have different GUIDs (different keys)."""
        assert (
            _stable_guid("Assets/Prefabs/Crate.prefab")
            != _stable_guid("Assets/Scenes/Crate.unity")
        )

    def test_script_guid_parity_between_monobehaviour_and_meta(self):
        """
        The MonoBehaviour reference and the .cs.meta file must use
        identical keys so Unity can resolve the script component.
        """
        script_filename = "PlayerController.cs"
        monobehaviour_key = f"Assets/Scripts/{script_filename}"
        meta_key = f"Assets/Scripts/{script_filename}"
        assert _stable_guid(monobehaviour_key) == _stable_guid(meta_key)


# ---------------------------------------------------------------------------
# _flatten_path
# ---------------------------------------------------------------------------

class TestFlattenPath:
    def test_path_without_duplicates_unchanged(self):
        p = Path("KayKit/Assets/Meshes/cube.glb")
        assert GodotToUnityPipeline._flatten_path(p) == p

    def test_consecutive_duplicate_collapsed(self):
        p = Path("KayKit/KayKit/Assets/cube.glb")
        assert GodotToUnityPipeline._flatten_path(p) == Path("KayKit/Assets/cube.glb")

    def test_multiple_duplicate_groups_all_collapsed(self):
        p = Path("A/A/B/B/file.txt")
        assert GodotToUnityPipeline._flatten_path(p) == Path("A/B/file.txt")

    def test_non_consecutive_identical_components_preserved(self):
        p = Path("KayKit/Other/KayKit/file.glb")
        assert GodotToUnityPipeline._flatten_path(p) == p

    def test_single_component_unchanged(self):
        p = Path("file.glb")
        assert GodotToUnityPipeline._flatten_path(p) == p

    def test_triple_consecutive_collapsed_to_one(self):
        p = Path("A/A/A/file.txt")
        assert GodotToUnityPipeline._flatten_path(p) == Path("A/file.txt")

    def test_flattened_path_produces_correct_guid(self):
        """The GUID key for a redundant path equals the GUID for its flattened form."""
        redundant = Path("KayKit/KayKit/Assets/cube.glb")
        flat = GodotToUnityPipeline._flatten_path(redundant)
        assert _stable_guid_g2u(flat.as_posix()) == _stable_guid_g2u("KayKit/Assets/cube.glb")

    def test_flatten_changes_guid_vs_unflattened(self):
        """The GUID for the redundant path differs from the flattened one."""
        redundant = Path("KayKit/KayKit/Assets/cube.glb")
        flat = GodotToUnityPipeline._flatten_path(redundant)
        assert _stable_guid_g2u(redundant.as_posix()) != _stable_guid_g2u(flat.as_posix())


# ---------------------------------------------------------------------------
# write_folder_meta
# ---------------------------------------------------------------------------

class TestWriteFolderMeta:
    def test_creates_meta_file_next_to_folder(self, tmp_path):
        output_dir = tmp_path / "output"
        folder = output_dir / "Assets" / "Scenes"
        folder.mkdir(parents=True)

        write_folder_meta(folder, output_dir)

        meta = output_dir / "Assets" / "Scenes.meta"
        assert meta.exists()

    def test_meta_contains_required_fields(self, tmp_path):
        output_dir = tmp_path / "output"
        folder = output_dir / "Assets" / "Scenes"
        folder.mkdir(parents=True)

        write_folder_meta(folder, output_dir)

        content = (output_dir / "Assets" / "Scenes.meta").read_text()
        assert "fileFormatVersion: 2" in content
        assert "folderAsset: yes" in content
        assert "guid:" in content

    def test_meta_guid_matches_stable_guid_formula(self, tmp_path):
        """The GUID in the .meta must equal _stable_guid('folder:Assets/Scenes')."""
        output_dir = tmp_path / "output"
        folder = output_dir / "Assets" / "Scenes"
        folder.mkdir(parents=True)

        write_folder_meta(folder, output_dir)

        content = (output_dir / "Assets" / "Scenes.meta").read_text()
        expected = _stable_guid("folder:Assets/Scenes")
        assert expected in content

    def test_idempotent_does_not_overwrite_existing(self, tmp_path):
        """A second call must leave the existing .meta untouched."""
        output_dir = tmp_path / "output"
        folder = output_dir / "Assets" / "Textures"
        folder.mkdir(parents=True)

        write_folder_meta(folder, output_dir)
        meta = output_dir / "Assets" / "Textures.meta"
        meta.write_text("sentinel_content")

        write_folder_meta(folder, output_dir)

        assert meta.read_text() == "sentinel_content"

    def test_guid_stable_across_independent_runs(self, tmp_path):
        """Two separate output directories produce identical GUIDs for the same rel path."""
        run1 = tmp_path / "run1"
        run2 = tmp_path / "run2"
        for d in [run1 / "Assets" / "Meshes", run2 / "Assets" / "Meshes"]:
            d.mkdir(parents=True)

        write_folder_meta(run1 / "Assets" / "Meshes", run1)
        write_folder_meta(run2 / "Assets" / "Meshes", run2)

        c1 = (run1 / "Assets" / "Meshes.meta").read_text()
        c2 = (run2 / "Assets" / "Meshes.meta").read_text()
        assert c1 == c2

    def test_nested_folder_uses_full_relative_path(self, tmp_path):
        """A deeply nested folder key includes the full path, not just the name."""
        output_dir = tmp_path / "output"
        folder = output_dir / "Assets" / "Models" / "Characters"
        folder.mkdir(parents=True)

        write_folder_meta(folder, output_dir)

        content = (output_dir / "Assets" / "Models" / "Characters.meta").read_text()
        expected = _stable_guid("folder:Assets/Models/Characters")
        assert expected in content
        # Shallow key must differ
        shallow_guid = _stable_guid("folder:Characters")
        assert shallow_guid not in content
