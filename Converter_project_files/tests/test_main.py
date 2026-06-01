"""
Tests for main.py — build_parser() and _cmd_convert() with mocked imports.
"""

import sys
import os
import types
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub heavy imports before importing main.py
# ---------------------------------------------------------------------------

_mock_unity_parser = None
_mock_godot_exporter = None


def _stub_main_imports():
    global _mock_unity_parser, _mock_godot_exporter

    mock_godot_exporter = types.ModuleType("unity_to_godot.godot_exporter")
    mock_godot_exporter.save_tscn = MagicMock()
    sys.modules["unity_to_godot.godot_exporter"] = mock_godot_exporter
    _mock_godot_exporter = mock_godot_exporter

    mock_unity_parser = types.ModuleType("unity_to_godot.unity_parser")

    class _FakeUnityParseError(Exception):
        pass

    mock_unity_parser.UnityParseError = _FakeUnityParseError
    mock_unity_parser.load_unity_scene = MagicMock(return_value={"nodes": []})
    sys.modules["unity_to_godot.unity_parser"] = mock_unity_parser
    _mock_unity_parser = mock_unity_parser

    # Force main.py to be re-imported so its top-level `from .unity_parser import ...`
    # picks up the mock rather than whatever was cached from earlier test files.
    sys.modules.pop("unity_to_godot.main", None)

_stub_main_imports()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unity_to_godot import main as _main_module
from unity_to_godot.main import build_parser, main


# ---------------------------------------------------------------------------
# build_parser
# ---------------------------------------------------------------------------

class TestBuildParser:
    def test_returns_argparse_parser(self):
        import argparse
        p = build_parser()
        assert isinstance(p, argparse.ArgumentParser)

    def test_convert_subcommand_exists(self):
        p = build_parser()
        args = p.parse_args(["convert", "scene.unity"])
        assert args.command == "convert"

    def test_convert_input_parsed(self):
        p = build_parser()
        args = p.parse_args(["convert", "scene.unity"])
        assert args.input == "scene.unity"

    def test_convert_output_default(self):
        p = build_parser()
        args = p.parse_args(["convert", "scene.unity"])
        assert args.output == "scene.tscn"

    def test_convert_output_override(self):
        p = build_parser()
        args = p.parse_args(["convert", "scene.unity", "-o", "out.tscn"])
        assert args.output == "out.tscn"

    def test_convert_output_long_flag(self):
        p = build_parser()
        args = p.parse_args(["convert", "scene.unity", "--output", "result.tscn"])
        assert args.output == "result.tscn"


# ---------------------------------------------------------------------------
# main() — via mocked imports
# ---------------------------------------------------------------------------

class TestMainFunction:
    def test_convert_calls_save_tscn(self, tmp_path):
        unity_file = tmp_path / "scene.unity"
        unity_file.write_text("data")
        out_file = tmp_path / "scene.tscn"

        exit_code = main(["convert", str(unity_file), "-o", str(out_file)])
        assert exit_code == 0
        sys.modules["unity_to_godot.godot_exporter"].save_tscn.assert_called()

    def test_convert_returns_1_on_parse_error(self, tmp_path):
        unity_file = tmp_path / "bad.unity"
        unity_file.write_text("data")

        # Use the saved mock reference directly — sys.modules may have been
        # overwritten by later test-file imports (e.g. test_job_lifecycle.py).
        UnityParseError = _mock_unity_parser.UnityParseError
        _mock_unity_parser.load_unity_scene.side_effect = UnityParseError("bad file")
        try:
            exit_code = main(["convert", str(unity_file)])
            assert exit_code == 1
        finally:
            _mock_unity_parser.load_unity_scene.side_effect = None
            _mock_unity_parser.load_unity_scene.return_value = {"nodes": []}
