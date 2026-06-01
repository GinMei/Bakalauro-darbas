"""
Tests for copy_assets.py — iter_allowed_files, copy_assets, _check_no_self_copy.
"""

import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from copy_assets import (
    iter_allowed_files,
    copy_assets,
    _check_no_self_copy,
    ALLOWED_EXTENSIONS,
    ASSETS_EXCLUDED_SUBDIRS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_unity_project(root: Path) -> Path:
    """Create a minimal Unity project structure."""
    assets = root / "Assets"
    assets.mkdir(parents=True)
    return root


def _make_file(directory: Path, name: str, content: str = "data") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# _check_no_self_copy
# ---------------------------------------------------------------------------

class TestCheckNoSelfCopy:
    def test_target_outside_assets_is_safe(self, tmp_path):
        assets_dir = tmp_path / "project" / "Assets"
        target_dir = tmp_path / "output"
        assert _check_no_self_copy(assets_dir, target_dir) is False

    def test_target_inside_assets_is_detected(self, tmp_path):
        assets_dir = tmp_path / "project" / "Assets"
        target_dir = tmp_path / "project" / "Assets" / "CopiedOutput"
        assert _check_no_self_copy(assets_dir, target_dir) is True

    def test_target_equals_assets_dir_is_detected(self, tmp_path):
        assets_dir = tmp_path / "project" / "Assets"
        assert _check_no_self_copy(assets_dir, assets_dir) is True


# ---------------------------------------------------------------------------
# iter_allowed_files — basic scanning
# ---------------------------------------------------------------------------

class TestIterAllowedFiles:
    def test_yields_allowed_png(self, tmp_path):
        root = _make_unity_project(tmp_path)
        _make_file(root / "Assets", "texture.png")
        files = list(iter_allowed_files(root))
        names = [src.name for src, _ in files]
        assert "texture.png" in names

    def test_yields_allowed_fbx(self, tmp_path):
        root = _make_unity_project(tmp_path)
        _make_file(root / "Assets", "model.fbx")
        files = list(iter_allowed_files(root))
        names = [src.name for src, _ in files]
        assert "model.fbx" in names

    def test_skips_meta_files(self, tmp_path):
        root = _make_unity_project(tmp_path)
        _make_file(root / "Assets", "texture.png.meta")
        files = list(iter_allowed_files(root))
        names = [src.name for src, _ in files]
        assert "texture.png.meta" not in names

    def test_skips_cs_files(self, tmp_path):
        root = _make_unity_project(tmp_path)
        _make_file(root / "Assets", "PlayerController.cs")
        files = list(iter_allowed_files(root))
        names = [src.name for src, _ in files]
        assert "PlayerController.cs" not in names

    def test_skips_hidden_files(self, tmp_path):
        root = _make_unity_project(tmp_path)
        _make_file(root / "Assets", ".hidden_file")
        files = list(iter_allowed_files(root))
        names = [src.name for src, _ in files]
        assert ".hidden_file" not in names

    def test_skips_excluded_settings_subdir(self, tmp_path):
        root = _make_unity_project(tmp_path)
        _make_file(root / "Assets" / "Settings", "config.json")
        files = list(iter_allowed_files(root))
        names = [src.name for src, _ in files]
        assert "config.json" not in names

    def test_skips_excluded_tutorialinfo_subdir(self, tmp_path):
        root = _make_unity_project(tmp_path)
        _make_file(root / "Assets" / "TutorialInfo", "readme.txt")
        files = list(iter_allowed_files(root))
        names = [src.name for src, _ in files]
        assert "readme.txt" not in names

    def test_no_assets_dir_yields_nothing(self, tmp_path):
        empty = tmp_path / "no_assets"
        empty.mkdir()
        files = list(iter_allowed_files(empty))
        assert files == []

    def test_rel_path_starts_with_assets(self, tmp_path):
        root = _make_unity_project(tmp_path)
        _make_file(root / "Assets", "audio.wav")
        files = list(iter_allowed_files(root))
        rels = [str(rel).replace("\\", "/") for _, rel in files]
        assert all(r.startswith("Assets/") for r in rels)

    def test_yields_json_files(self, tmp_path):
        root = _make_unity_project(tmp_path)
        _make_file(root / "Assets", "data.json")
        files = list(iter_allowed_files(root))
        names = [src.name for src, _ in files]
        assert "data.json" in names

    def test_yields_ogg_files(self, tmp_path):
        root = _make_unity_project(tmp_path)
        _make_file(root / "Assets", "music.ogg")
        files = list(iter_allowed_files(root))
        names = [src.name for src, _ in files]
        assert "music.ogg" in names

    def test_nested_file_yielded(self, tmp_path):
        root = _make_unity_project(tmp_path)
        _make_file(root / "Assets" / "SubDir", "texture.jpg")
        files = list(iter_allowed_files(root))
        names = [src.name for src, _ in files]
        assert "texture.jpg" in names

    def test_all_allowed_extensions_covered(self):
        for ext in ALLOWED_EXTENSIONS:
            assert ext.startswith(".")


# ---------------------------------------------------------------------------
# copy_assets
# ---------------------------------------------------------------------------

class TestCopyAssets:
    def test_copies_allowed_file(self, tmp_path):
        root = _make_unity_project(tmp_path)
        _make_file(root / "Assets", "texture.png", "png_data")
        target = tmp_path / "target"
        copied, skipped, errors = copy_assets(root, target)
        assert copied == 1
        assert errors == []
        assert (target / "Assets" / "texture.png").exists()

    def test_skips_disallowed_files(self, tmp_path):
        root = _make_unity_project(tmp_path)
        _make_file(root / "Assets", "script.cs")
        target = tmp_path / "target"
        copied, _, errors = copy_assets(root, target)
        assert copied == 0
        assert errors == []

    def test_skip_existing_without_overwrite(self, tmp_path):
        root = _make_unity_project(tmp_path)
        _make_file(root / "Assets", "texture.png", "original")
        target = tmp_path / "target"
        (target / "Assets").mkdir(parents=True)
        (target / "Assets" / "texture.png").write_text("existing", encoding="utf-8")
        _, skipped, _ = copy_assets(root, target, overwrite=False)
        assert skipped == 1

    def test_overwrite_replaces_file(self, tmp_path):
        root = _make_unity_project(tmp_path)
        _make_file(root / "Assets", "texture.png", "new_content")
        target = tmp_path / "target"
        (target / "Assets").mkdir(parents=True)
        (target / "Assets" / "texture.png").write_text("old", encoding="utf-8")
        copy_assets(root, target, overwrite=True)
        assert (target / "Assets" / "texture.png").read_text(encoding="utf-8") == "new_content"

    def test_self_copy_returns_error(self, tmp_path):
        root = _make_unity_project(tmp_path)
        target = root / "Assets" / "CopiedHere"
        copied, skipped, errors = copy_assets(root, target)
        assert copied == 0
        assert len(errors) == 1
        assert "FATAL" in errors[0]

    def test_creates_target_subdirectories(self, tmp_path):
        root = _make_unity_project(tmp_path)
        _make_file(root / "Assets" / "SubA" / "SubB", "img.png")
        target = tmp_path / "target"
        copy_assets(root, target)
        assert (target / "Assets" / "SubA" / "SubB" / "img.png").exists()

    def test_multiple_files_copied(self, tmp_path):
        root = _make_unity_project(tmp_path)
        _make_file(root / "Assets", "a.png")
        _make_file(root / "Assets", "b.wav")
        _make_file(root / "Assets", "c.fbx")
        target = tmp_path / "target"
        copied, _, errors = copy_assets(root, target)
        assert copied == 3
        assert errors == []


# ---------------------------------------------------------------------------
# ALLOWED_EXTENSIONS and ASSETS_EXCLUDED_SUBDIRS constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_allowed_extensions_nonempty(self):
        assert len(ALLOWED_EXTENSIONS) > 0

    def test_excluded_subdirs_contains_settings(self):
        assert "Settings" in ASSETS_EXCLUDED_SUBDIRS

    def test_excluded_subdirs_contains_tutorialinfo(self):
        assert "TutorialInfo" in ASSETS_EXCLUDED_SUBDIRS
