"""
Tests for godot_to_unity_pipeline.py — pure utility functions,
G2UConversionResult dataclass, and GodotToUnityPipeline internals.
"""

import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from godot_to_unity.godot_to_unity_pipeline import (
    _has_scene_constructs,
    _stable_guid,
    parse_meta_file,
    _build_layer_map,
    _meta_for_asset,
    _mat_stub_content,
    _mat_meta_content,
    _script_meta_content,
    G2UConversionResult,
    GodotToUnityPipeline,
)


# ---------------------------------------------------------------------------
# _has_scene_constructs
# ---------------------------------------------------------------------------

class TestHasSceneConstructs:
    def test_detects_tscn_extension(self):
        assert _has_scene_constructs('var scene = load("level.tscn")')

    def test_detects_AddChild(self):
        assert _has_scene_constructs("AddChild(node)")

    def test_detects_Instantiate(self):
        assert _has_scene_constructs("var inst = scene.Instantiate()")

    def test_detects_ChangeSceneToFile(self):
        assert _has_scene_constructs('ChangeSceneToFile("res://main.tscn")')

    def test_detects_Load(self):
        assert _has_scene_constructs('var res = Load("res://Player.tscn")')

    def test_detects_PackedScene(self):
        assert _has_scene_constructs("PackedScene packedScene;")

    def test_pure_logic_returns_false(self):
        assert not _has_scene_constructs("var x = 1 + 2;\nprint(x);")

    def test_empty_string_returns_false(self):
        assert not _has_scene_constructs("")

    def test_strips_line_comments(self):
        # Keyword in a comment should not match
        source = "// AddChild(node)\nvar x = 1;"
        assert not _has_scene_constructs(source)

    def test_strips_block_comments(self):
        source = "/* ChangeSceneToFile */\nvar x = 1;"
        assert not _has_scene_constructs(source)

    def test_case_insensitive(self):
        assert _has_scene_constructs("addchild(node)")


# ---------------------------------------------------------------------------
# _stable_guid
# ---------------------------------------------------------------------------

class TestStableGuid:
    def test_returns_32_char_hex(self):
        g = _stable_guid("Assets/Scripts/Player.cs")
        assert len(g) == 32
        assert all(c in "0123456789abcdef" for c in g)

    def test_deterministic(self):
        g1 = _stable_guid("Assets/Scenes/Level.tscn")
        g2 = _stable_guid("Assets/Scenes/Level.tscn")
        assert g1 == g2

    def test_different_keys_different_guids(self):
        g1 = _stable_guid("Assets/A.cs")
        g2 = _stable_guid("Assets/B.cs")
        assert g1 != g2

    def test_empty_string_returns_valid_guid(self):
        g = _stable_guid("")
        assert len(g) == 32


# ---------------------------------------------------------------------------
# parse_meta_file
# ---------------------------------------------------------------------------

class TestParseMetaFile:
    def test_returns_guid_from_meta_file(self, tmp_path):
        meta = tmp_path / "Texture.png.meta"
        meta.write_text("fileFormatVersion: 2\nguid: abc123def456\n", encoding="utf-8")
        assert parse_meta_file(meta) == "abc123def456"

    def test_returns_none_when_no_guid_key(self, tmp_path):
        meta = tmp_path / "file.meta"
        meta.write_text("fileFormatVersion: 2\n", encoding="utf-8")
        assert parse_meta_file(meta) is None

    def test_returns_none_for_missing_file(self, tmp_path):
        result = parse_meta_file(tmp_path / "nonexistent.meta")
        assert result is None

    def test_handles_whitespace_around_guid(self, tmp_path):
        meta = tmp_path / "a.meta"
        meta.write_text("  guid:   deadbeef12345678  \n", encoding="utf-8")
        assert parse_meta_file(meta) == "deadbeef12345678"

    def test_returns_none_for_empty_guid_value(self, tmp_path):
        meta = tmp_path / "b.meta"
        meta.write_text("guid:\n", encoding="utf-8")
        assert parse_meta_file(meta) is None


# ---------------------------------------------------------------------------
# _build_layer_map
# ---------------------------------------------------------------------------

class TestBuildLayerMap:
    def test_empty_list_returns_empty(self):
        assert _build_layer_map([]) == {}

    def test_zero_mask_maps_to_zero(self):
        result = _build_layer_map([0])
        assert result[0] == 0

    def test_single_bit_gets_layer_1(self):
        result = _build_layer_map([1])   # bit 0 → layer 1
        assert result[1] == 1

    def test_two_masks_get_different_layers(self):
        result = _build_layer_map([1, 2])   # bits 0,1
        assert result[1] != result[2]

    def test_result_keys_are_input_masks(self):
        masks = [1, 4, 16]
        result = _build_layer_map(masks)
        for m in masks:
            assert m in result

    def test_large_bitmask_covered(self):
        result = _build_layer_map([0b1111])
        # Lowest bit of 0b1111 is 1 (bit 0)
        assert 0b1111 in result

    def test_layers_start_at_1(self):
        result = _build_layer_map([1])
        assert result[1] >= 1


# ---------------------------------------------------------------------------
# _meta_for_asset
# ---------------------------------------------------------------------------

class TestMetaForAsset:
    def test_fbx_model_uses_ModelImporter(self):
        content = _meta_for_asset(Path("Assets/Models/hero.fbx"), "abc123")
        assert "ModelImporter" in content
        assert "abc123" in content

    def test_obj_model_uses_ModelImporter(self):
        content = _meta_for_asset(Path("Models/thing.obj"), "guid1")
        assert "ModelImporter" in content

    def test_png_uses_TextureImporter(self):
        content = _meta_for_asset(Path("Assets/Textures/tex.png"), "tex_guid")
        assert "TextureImporter" in content
        assert "tex_guid" in content

    def test_jpg_uses_TextureImporter(self):
        content = _meta_for_asset(Path("Assets/Textures/photo.jpg"), "g2")
        assert "TextureImporter" in content

    def test_wav_uses_AudioImporter(self):
        content = _meta_for_asset(Path("Sounds/music.wav"), "aud1")
        assert "AudioImporter" in content
        assert "aud1" in content

    def test_mp3_uses_AudioImporter(self):
        content = _meta_for_asset(Path("Sounds/sound.mp3"), "aud2")
        assert "AudioImporter" in content

    def test_ogg_uses_AudioImporter(self):
        content = _meta_for_asset(Path("Sounds/fx.ogg"), "aud3")
        assert "AudioImporter" in content

    def test_unknown_ext_uses_DefaultImporter(self):
        content = _meta_for_asset(Path("Data/file.bin"), "def1")
        assert "DefaultImporter" in content
        assert "def1" in content

    def test_model_meta_has_file_format_version(self):
        content = _meta_for_asset(Path("hero.glb"), "g")
        assert "fileFormatVersion: 2" in content

    def test_texture_meta_has_file_format_version(self):
        content = _meta_for_asset(Path("tex.jpeg"), "g")
        assert "fileFormatVersion: 2" in content

    def test_model_meta_includes_stem_as_root_name(self):
        content = _meta_for_asset(Path("Assets/soldier.fbx"), "g")
        assert "soldier" in content


# ---------------------------------------------------------------------------
# _mat_stub_content
# ---------------------------------------------------------------------------

class TestMatStubContent:
    def test_contains_yaml_header(self):
        content = _mat_stub_content("MyMaterial")
        assert "%YAML 1.1" in content

    def test_contains_material_name(self):
        content = _mat_stub_content("TestMat")
        assert "TestMat" in content

    def test_contains_material_class(self):
        content = _mat_stub_content("Mat")
        assert "Material:" in content

    def test_contains_white_albedo(self):
        content = _mat_stub_content("Mat")
        assert "r: 1" in content


# ---------------------------------------------------------------------------
# _mat_meta_content
# ---------------------------------------------------------------------------

class TestMatMetaContent:
    def test_contains_guid(self):
        content = _mat_meta_content("mat_guid_123")
        assert "mat_guid_123" in content

    def test_contains_native_format_importer(self):
        content = _mat_meta_content("g")
        assert "NativeFormatImporter" in content

    def test_contains_file_format_version(self):
        content = _mat_meta_content("g")
        assert "fileFormatVersion: 2" in content


# ---------------------------------------------------------------------------
# _script_meta_content
# ---------------------------------------------------------------------------

class TestScriptMetaContent:
    def test_contains_guid(self):
        content = _script_meta_content("sc_guid_999")
        assert "sc_guid_999" in content

    def test_contains_file_format_version(self):
        content = _script_meta_content("g")
        assert "fileFormatVersion: 2" in content


# ---------------------------------------------------------------------------
# G2UConversionResult — dataclass defaults and field storage
# ---------------------------------------------------------------------------

class TestG2UConversionResult:
    def test_success_true(self):
        r = G2UConversionResult(success=True)
        assert r.success is True

    def test_success_false(self):
        r = G2UConversionResult(success=False)
        assert r.success is False

    def test_scenes_exported_default_empty(self):
        r = G2UConversionResult(success=True)
        assert r.scenes_exported == []

    def test_prefabs_exported_default_empty(self):
        r = G2UConversionResult(success=True)
        assert r.prefabs_exported == []

    def test_scenes_failed_default_empty(self):
        r = G2UConversionResult(success=True)
        assert r.scenes_failed == []

    def test_scenes_unresolved_default_empty(self):
        r = G2UConversionResult(success=True)
        assert r.scenes_unresolved == []

    def test_scripts_converted_default_zero(self):
        r = G2UConversionResult(success=True)
        assert r.scripts_converted == 0

    def test_scripts_failed_default_zero(self):
        r = G2UConversionResult(success=True)
        assert r.scripts_failed == 0

    def test_assets_copied_default_zero(self):
        r = G2UConversionResult(success=True)
        assert r.assets_copied == 0

    def test_warnings_default_empty(self):
        r = G2UConversionResult(success=True)
        assert r.warnings == []

    def test_error_default_empty(self):
        r = G2UConversionResult(success=True)
        assert r.error == ""

    def test_scene_irs_default_empty(self):
        r = G2UConversionResult(success=True)
        assert r.scene_irs == {}

    def test_lists_independent_per_instance(self):
        r1 = G2UConversionResult(success=True)
        r2 = G2UConversionResult(success=True)
        r1.scenes_exported.append(Path("a.unity"))
        assert r2.scenes_exported == []

    def test_set_scripts_converted(self):
        r = G2UConversionResult(success=True, scripts_converted=5)
        assert r.scripts_converted == 5

    def test_set_error(self):
        r = G2UConversionResult(success=False, error="Conversion failed")
        assert r.error == "Conversion failed"


# ---------------------------------------------------------------------------
# GodotToUnityPipeline — private scan helpers and _validate_ir
# ---------------------------------------------------------------------------

class TestGodotToUnityPipelineScans:
    def setup_method(self):
        self.pipeline = GodotToUnityPipeline()

    def test_scan_scenes_finds_tscn(self, tmp_path):
        (tmp_path / "Level.tscn").write_text("[gd_scene]", encoding="utf-8")
        scenes = self.pipeline._scan_scenes(tmp_path)
        assert any(p.name == "Level.tscn" for p in scenes)

    def test_scan_scenes_skips_godot_cache(self, tmp_path):
        cache = tmp_path / ".godot"
        cache.mkdir()
        (cache / "hidden.tscn").write_text("", encoding="utf-8")
        scenes = self.pipeline._scan_scenes(tmp_path)
        assert not any(".godot" in str(p) for p in scenes)

    def test_scan_scripts_finds_cs(self, tmp_path):
        (tmp_path / "Player.cs").write_text("class Player {}", encoding="utf-8")
        scripts = self.pipeline._scan_scripts(tmp_path)
        assert any(p.name == "Player.cs" for p in scripts)

    def test_scan_gd_scripts_finds_gd(self, tmp_path):
        (tmp_path / "enemy.gd").write_text("extends Node", encoding="utf-8")
        gds = self.pipeline._scan_gd_scripts(tmp_path)
        assert any(p.name == "enemy.gd" for p in gds)

    def test_scan_assets_finds_fbx(self, tmp_path):
        (tmp_path / "hero.fbx").write_bytes(b"fbx data")
        assets = self.pipeline._scan_assets(tmp_path)
        assert any(p.name == "hero.fbx" for p in assets)

    def test_scan_assets_skips_godot_dir(self, tmp_path):
        godot_dir = tmp_path / ".godot"
        godot_dir.mkdir()
        (godot_dir / "mesh.fbx").write_bytes(b"data")
        assets = self.pipeline._scan_assets(tmp_path)
        assert not any(".godot" in str(p) for p in assets)


class TestValidateIR:
    def setup_method(self):
        self.pipeline = GodotToUnityPipeline()

    def _minimal_ir(self, **extra):
        base = {
            "nodes": [{"node_name": "Root", "ir_node_kind": "spatial"}],
            "ir_version": "1.0",
            "coordinate_system": {"handedness": "right", "up_axis": "Y"},
            "scene_name": "TestScene",
            "classification": "SCENE",
        }
        base.update(extra)
        return base

    def test_valid_ir_no_warnings(self):
        warnings = self.pipeline._validate_ir(self._minimal_ir())
        assert not any("BLOCKING" in w for w in warnings)

    def test_empty_nodes_returns_blocking(self):
        ir = self._minimal_ir()
        ir["nodes"] = []
        warnings = self.pipeline._validate_ir(ir)
        assert any("BLOCKING" in w for w in warnings)

    def test_wrong_ir_version_warns(self):
        ir = self._minimal_ir(ir_version="0.9")
        warnings = self.pipeline._validate_ir(ir)
        assert any("version" in w.lower() for w in warnings)

    def test_wrong_coordinate_system_warns(self):
        ir = self._minimal_ir()
        ir["coordinate_system"] = {"handedness": "left", "up_axis": "Z"}
        warnings = self.pipeline._validate_ir(ir)
        assert any("coordinate" in w.lower() for w in warnings)


# ---------------------------------------------------------------------------
# GodotToUnityPipeline._validate_nodes
# ---------------------------------------------------------------------------

class TestValidateNodes:
    def setup_method(self):
        self.pipeline = GodotToUnityPipeline()

    def test_empty_nodes_no_warnings(self):
        warnings = []
        self.pipeline._validate_nodes([], "Scene", warnings)
        assert warnings == []

    def test_mesh_instance_no_mesh_warns(self):
        nodes = [{"node_name": "Cube", "godot_type": "MeshInstance3D",
                  "components": {"mesh": {}}}]
        warnings = []
        self.pipeline._validate_nodes(nodes, "Scene", warnings)
        assert any("mesh" in w.lower() for w in warnings)

    def test_mesh_instance_no_source_path_warns(self):
        nodes = [{"node_name": "Cube", "godot_type": "MeshInstance3D",
                  "components": {"mesh": {"mesh_type": "Box"}}}]
        warnings = []
        self.pipeline._validate_nodes(nodes, "Scene", warnings)
        assert any("mesh_source_path" in w for w in warnings)

    def test_label3d_warns(self):
        nodes = [{"node_name": "Txt", "godot_type": "Label3D", "components": {}}]
        warnings = []
        self.pipeline._validate_nodes(nodes, "Scene", warnings)
        assert any("Label3D" in w for w in warnings)

    def test_children_validated_recursively(self):
        child = {"node_name": "Child", "godot_type": "Label3D", "components": {}}
        parent = {"node_name": "Parent", "godot_type": "Node3D",
                  "components": {}, "children": [child]}
        warnings = []
        self.pipeline._validate_nodes([parent], "Scene", warnings)
        assert any("Label3D" in w for w in warnings)

    def test_good_mesh_instance_no_warning(self):
        nodes = [{"node_name": "Hero", "godot_type": "MeshInstance3D",
                  "components": {"mesh": {"mesh_source_path": "res://hero.fbx"}}}]
        warnings = []
        self.pipeline._validate_nodes(nodes, "Scene", warnings)
        mesh_warns = [w for w in warnings if "mesh" in w.lower()]
        assert mesh_warns == []


# ---------------------------------------------------------------------------
# GodotToUnityPipeline._collect_unsupported_types
# ---------------------------------------------------------------------------

class TestCollectUnsupportedTypes:
    def setup_method(self):
        self.pipeline = GodotToUnityPipeline()

    def test_empty_nodes_returns_empty(self):
        assert self.pipeline._collect_unsupported_types([]) == {}

    def test_label3d_counted(self):
        nodes = [{"godot_type": "Label3D", "children": []}]
        counts = self.pipeline._collect_unsupported_types(nodes)
        assert counts.get("Label3D") == 1

    def test_supported_type_not_counted(self):
        nodes = [{"godot_type": "Node3D", "children": []}]
        assert self.pipeline._collect_unsupported_types(nodes) == {}

    def test_recursive_counts(self):
        child = {"godot_type": "Label3D", "children": []}
        parent = {"godot_type": "Node3D", "children": [child]}
        counts = self.pipeline._collect_unsupported_types([parent])
        assert counts.get("Label3D") == 1


# ---------------------------------------------------------------------------
# GodotToUnityPipeline._flatten_path
# ---------------------------------------------------------------------------

class TestFlattenPath:
    def test_no_duplicates_unchanged(self):
        p = Path("Assets/Models/hero.fbx")
        assert GodotToUnityPipeline._flatten_path(p) == p

    def test_consecutive_duplicates_collapsed(self):
        p = Path("Pack/Pack/Assets/mesh.fbx")
        result = GodotToUnityPipeline._flatten_path(p)
        assert str(result) == str(Path("Pack/Assets/mesh.fbx"))

    def test_non_consecutive_duplicates_kept(self):
        p = Path("A/B/A/file.fbx")
        result = GodotToUnityPipeline._flatten_path(p)
        assert result == p

    def test_single_component_unchanged(self):
        p = Path("file.png")
        assert GodotToUnityPipeline._flatten_path(p) == p

    def test_triple_consecutive_collapsed(self):
        p = Path("X/X/X/file.bin")
        result = GodotToUnityPipeline._flatten_path(p)
        assert str(result) == str(Path("X/file.bin"))


# ---------------------------------------------------------------------------
# GodotToUnityPipeline._build_asset_guid_map
# ---------------------------------------------------------------------------

class TestBuildAssetGuidMap:
    def test_returns_res_path_keys(self, tmp_path):
        src = tmp_path / "models" / "hero.fbx"
        src.parent.mkdir()
        src.touch()
        result = GodotToUnityPipeline._build_asset_guid_map([src], tmp_path)
        assert any(k.startswith("res://") for k in result)

    def test_guid_is_32_char_hex(self, tmp_path):
        src = tmp_path / "tex.png"
        src.touch()
        result = GodotToUnityPipeline._build_asset_guid_map([src], tmp_path)
        for guid in result.values():
            assert len(guid) == 32

    def test_empty_list_returns_empty(self, tmp_path):
        assert GodotToUnityPipeline._build_asset_guid_map([], tmp_path) == {}

    def test_path_outside_root_uses_name(self, tmp_path, tmp_path_factory):
        other = tmp_path_factory.mktemp("other") / "mesh.obj"
        other.touch()
        result = GodotToUnityPipeline._build_asset_guid_map([other], tmp_path)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# GodotToUnityPipeline._scan_output_metas
# ---------------------------------------------------------------------------

class TestScanOutputMetas:
    def test_returns_guid_for_existing_asset(self, tmp_path):
        (tmp_path / "tex.png").write_bytes(b"data")
        meta = tmp_path / "tex.png.meta"
        meta.write_text("fileFormatVersion: 2\nguid: abc123\n", encoding="utf-8")
        result = GodotToUnityPipeline._scan_output_metas(tmp_path)
        assert "tex.png" in result
        assert result["tex.png"] == "abc123"

    def test_meta_without_asset_not_included(self, tmp_path):
        meta = tmp_path / "ghost.png.meta"
        meta.write_text("fileFormatVersion: 2\nguid: dead\n", encoding="utf-8")
        result = GodotToUnityPipeline._scan_output_metas(tmp_path)
        assert "ghost.png" not in result

    def test_empty_dir_returns_empty(self, tmp_path):
        assert GodotToUnityPipeline._scan_output_metas(tmp_path) == {}


# ---------------------------------------------------------------------------
# GodotToUnityPipeline._copy_assets
# ---------------------------------------------------------------------------

class TestCopyAssets:
    def setup_method(self):
        self.pipeline = GodotToUnityPipeline()

    def test_copies_fbx_to_assets(self, tmp_path):
        src = tmp_path / "godot" / "models" / "hero.fbx"
        src.parent.mkdir(parents=True)
        src.write_bytes(b"fbx data")
        out = tmp_path / "unity"
        out.mkdir()

        warnings = []
        count = self.pipeline._copy_assets([src], tmp_path / "godot", out, warnings)
        assert count == 1
        assert (out / "Assets" / "models" / "hero.fbx").exists()

    def test_creates_meta_sidecar(self, tmp_path):
        src = tmp_path / "godot" / "tex.png"
        src.parent.mkdir(parents=True)
        src.write_bytes(b"png")
        out = tmp_path / "unity"
        out.mkdir()

        warnings = []
        self.pipeline._copy_assets([src], tmp_path / "godot", out, warnings)
        meta = out / "Assets" / "tex.png.meta"
        assert meta.exists()

    def test_skips_existing_file(self, tmp_path):
        src = tmp_path / "godot" / "img.png"
        src.parent.mkdir(parents=True)
        src.write_bytes(b"original")
        out = tmp_path / "unity"
        (out / "Assets").mkdir(parents=True)
        dst = out / "Assets" / "img.png"
        dst.write_bytes(b"already there")

        warnings = []
        count = self.pipeline._copy_assets([src], tmp_path / "godot", out, warnings)
        assert count == 0
        assert dst.read_bytes() == b"already there"

    def test_material_converted_to_mat_stub(self, tmp_path):
        src = tmp_path / "godot" / "rock.tres"
        src.parent.mkdir(parents=True)
        src.write_text("some godot material", encoding="utf-8")
        out = tmp_path / "unity"
        out.mkdir()

        warnings = []
        count = self.pipeline._copy_assets([src], tmp_path / "godot", out, warnings)
        assert count == 1
        mat_file = out / "Assets" / "rock.mat"
        assert mat_file.exists()
        assert "%YAML 1.1" in mat_file.read_text(encoding="utf-8")

    def test_empty_list_returns_zero(self, tmp_path):
        out = tmp_path / "unity"
        out.mkdir()
        warnings = []
        assert self.pipeline._copy_assets([], tmp_path, out, warnings) == 0


# ---------------------------------------------------------------------------
# GodotToUnityPipeline._validate_output
# ---------------------------------------------------------------------------

class TestValidateOutput:
    def setup_method(self):
        self.pipeline = GodotToUnityPipeline()

    def test_no_issues_in_clean_output(self, tmp_path):
        assets = tmp_path / "Assets"
        assets.mkdir()
        result = G2UConversionResult(success=True)
        self.pipeline._validate_output(tmp_path, result)
        assert result.warnings == []

    def test_placeholder_in_cs_file_warns(self, tmp_path):
        assets = tmp_path / "Assets"
        assets.mkdir()
        cs = assets / "Player.cs"
        cs.write_text("var x = __RAYCAST_PLACEHOLDER__;", encoding="utf-8")
        result = G2UConversionResult(success=True)
        self.pipeline._validate_output(tmp_path, result)
        assert any("RAYCAST_PLACEHOLDER" in w for w in result.warnings)

    def test_prefab_with_multiple_roots_warns(self, tmp_path):
        assets = tmp_path / "Assets"
        assets.mkdir()
        prefab = assets / "Scene.prefab"
        prefab.write_text(
            "m_Father: {fileID: 0}\nsome data\nm_Father: {fileID: 0}\n",
            encoding="utf-8",
        )
        result = G2UConversionResult(success=True)
        self.pipeline._validate_output(tmp_path, result)
        assert any("root objects" in w for w in result.warnings)
