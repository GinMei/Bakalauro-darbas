"""
Tier 2 — Unity .meta file content helpers.

Tests cover the three static methods on UnitySceneExporter:
  • _meta_scene_content   — DefaultImporter, required YAML fields
  • _meta_prefab_content  — PrefabImporter, required YAML fields
  • _meta_native_content  — NativeFormatImporter + mainObjectFileID
  • _meta_content         — legacy dispatcher; also verifies auto-GUID fallback
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from godot_to_unity.unity_scene_exporter import UnitySceneExporter, _stable_guid

_SAMPLE_GUID = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4"


# ---------------------------------------------------------------------------
# _meta_scene_content
# ---------------------------------------------------------------------------

class TestMetaSceneContent:
    def test_has_file_format_version(self):
        content = UnitySceneExporter._meta_scene_content(_SAMPLE_GUID)
        assert "fileFormatVersion: 2" in content

    def test_guid_present(self):
        content = UnitySceneExporter._meta_scene_content(_SAMPLE_GUID)
        assert f"guid: {_SAMPLE_GUID}" in content

    def test_default_importer_present(self):
        content = UnitySceneExporter._meta_scene_content(_SAMPLE_GUID)
        assert "DefaultImporter:" in content

    def test_no_prefab_importer(self):
        content = UnitySceneExporter._meta_scene_content(_SAMPLE_GUID)
        assert "PrefabImporter:" not in content

    def test_no_native_format_importer(self):
        content = UnitySceneExporter._meta_scene_content(_SAMPLE_GUID)
        assert "NativeFormatImporter:" not in content

    def test_different_guids_produce_different_content(self):
        c1 = UnitySceneExporter._meta_scene_content("aaaabbbbccccddddaaaabbbbccccdddd")
        c2 = UnitySceneExporter._meta_scene_content("11112222333344441111222233334444")
        assert c1 != c2


# ---------------------------------------------------------------------------
# _meta_prefab_content
# ---------------------------------------------------------------------------

class TestMetaPrefabContent:
    def test_has_file_format_version(self):
        content = UnitySceneExporter._meta_prefab_content(_SAMPLE_GUID)
        assert "fileFormatVersion: 2" in content

    def test_guid_present(self):
        content = UnitySceneExporter._meta_prefab_content(_SAMPLE_GUID)
        assert f"guid: {_SAMPLE_GUID}" in content

    def test_prefab_importer_present(self):
        content = UnitySceneExporter._meta_prefab_content(_SAMPLE_GUID)
        assert "PrefabImporter:" in content

    def test_no_default_importer(self):
        content = UnitySceneExporter._meta_prefab_content(_SAMPLE_GUID)
        assert "DefaultImporter:" not in content

    def test_no_native_format_importer(self):
        content = UnitySceneExporter._meta_prefab_content(_SAMPLE_GUID)
        assert "NativeFormatImporter:" not in content


# ---------------------------------------------------------------------------
# _meta_native_content
# ---------------------------------------------------------------------------

class TestMetaNativeContent:
    def test_has_file_format_version(self):
        content = UnitySceneExporter._meta_native_content(_SAMPLE_GUID, 74)
        assert "fileFormatVersion: 2" in content

    def test_guid_present(self):
        content = UnitySceneExporter._meta_native_content(_SAMPLE_GUID, 74)
        assert f"guid: {_SAMPLE_GUID}" in content

    def test_native_format_importer_present(self):
        content = UnitySceneExporter._meta_native_content(_SAMPLE_GUID, 74)
        assert "NativeFormatImporter:" in content

    def test_main_object_file_id_anim(self):
        content = UnitySceneExporter._meta_native_content(_SAMPLE_GUID, 74)
        assert "mainObjectFileID: 74" in content

    def test_main_object_file_id_controller(self):
        content = UnitySceneExporter._meta_native_content(_SAMPLE_GUID, 91)
        assert "mainObjectFileID: 91" in content

    def test_no_default_importer(self):
        content = UnitySceneExporter._meta_native_content(_SAMPLE_GUID, 74)
        assert "DefaultImporter:" not in content

    def test_no_prefab_importer(self):
        content = UnitySceneExporter._meta_native_content(_SAMPLE_GUID, 74)
        assert "PrefabImporter:" not in content


# ---------------------------------------------------------------------------
# _meta_content (legacy dispatcher)
# ---------------------------------------------------------------------------

class TestMetaContentLegacyHelper:
    def test_scene_variant_uses_default_importer(self):
        content = UnitySceneExporter._meta_content(
            "MyScene.unity", prefab=False, guid=_SAMPLE_GUID
        )
        assert "DefaultImporter:" in content

    def test_prefab_variant_uses_prefab_importer(self):
        content = UnitySceneExporter._meta_content(
            "MyPrefab.prefab", prefab=True, guid=_SAMPLE_GUID
        )
        assert "PrefabImporter:" in content

    def test_explicit_guid_passed_through(self):
        content = UnitySceneExporter._meta_content(
            "Any.unity", prefab=False, guid=_SAMPLE_GUID
        )
        assert _SAMPLE_GUID in content

    def test_no_guid_generates_valid_guid(self):
        content = UnitySceneExporter._meta_content("AutoScene.unity")
        for line in content.splitlines():
            if line.strip().startswith("guid:"):
                guid = line.split(":", 1)[1].strip()
                assert len(guid) == 32
                assert all(c in "0123456789abcdef" for c in guid)
                break
        else:
            pytest.fail("No guid: line found in meta content")

    def test_auto_guid_is_deterministic(self):
        c1 = UnitySceneExporter._meta_content("AutoScene.unity")
        c2 = UnitySceneExporter._meta_content("AutoScene.unity")
        assert c1 == c2


# ---------------------------------------------------------------------------
# Importer type distinctions
# ---------------------------------------------------------------------------

class TestMetaImporterDistinction:
    def test_scene_and_prefab_meta_differ(self):
        scene = UnitySceneExporter._meta_scene_content(_SAMPLE_GUID)
        prefab = UnitySceneExporter._meta_prefab_content(_SAMPLE_GUID)
        assert scene != prefab

    def test_scene_and_native_meta_differ(self):
        scene = UnitySceneExporter._meta_scene_content(_SAMPLE_GUID)
        native = UnitySceneExporter._meta_native_content(_SAMPLE_GUID, 74)
        assert scene != native

    def test_prefab_and_native_meta_differ(self):
        prefab = UnitySceneExporter._meta_prefab_content(_SAMPLE_GUID)
        native = UnitySceneExporter._meta_native_content(_SAMPLE_GUID, 74)
        assert prefab != native
