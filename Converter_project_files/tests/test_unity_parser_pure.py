"""
Tests for unity_parser.py — pure Python functions that do not invoke yaml
at runtime: _decode_unity_bytes, _split_unity_blocks, HEADER_RE, _fid,
_mesh_ref, _material_refs, _vec3, _rgba_to_rgb, and the exception class.

yaml is stubbed at module level so no real PyYAML install is needed.
"""

import sys
import os
import types
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Stub yaml + version_profiles before importing unity_parser
# ---------------------------------------------------------------------------

def _stub_yaml():
    if "yaml" not in sys.modules:
        yaml_mod = types.ModuleType("yaml")

        class SafeLoader:
            @classmethod
            def add_multi_constructor(cls, tag, fn):
                pass

        yaml_mod.SafeLoader = SafeLoader
        yaml_mod.load = MagicMock(return_value={})
        yaml_mod.YAMLError = Exception

        class MappingNode:
            pass
        class SequenceNode:
            pass
        class Node:
            pass

        yaml_mod.MappingNode  = MappingNode
        yaml_mod.SequenceNode = SequenceNode
        yaml_mod.Node         = Node
        sys.modules["yaml"]   = yaml_mod

_stub_yaml()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Remove any cached module from earlier stubs so we get the real one
sys.modules.pop("unity_to_godot.unity_parser", None)

from unity_to_godot.unity_parser import (
    UnityParseError,
    HEADER_RE,
    _decode_unity_bytes,
    _split_unity_blocks,
    _fid,
    _mesh_ref,
    _material_refs,
    _vec3,
    _rgba_to_rgb,
    _UNITY_BUILTIN_GUID,
)


# ---------------------------------------------------------------------------
# UnityParseError
# ---------------------------------------------------------------------------

class TestUnityParseError:
    def test_is_exception(self):
        with __import__("pytest").raises(UnityParseError):
            raise UnityParseError("bad file")

    def test_stores_message(self):
        exc = UnityParseError("msg")
        assert "msg" in str(exc)


# ---------------------------------------------------------------------------
# HEADER_RE
# ---------------------------------------------------------------------------

class TestHeaderRE:
    def test_matches_standard_header(self):
        m = HEADER_RE.match("--- !u!4 &123456")
        assert m is not None
        assert m.group(1) == "4"
        assert m.group(2) == "123456"

    def test_matches_negative_file_id(self):
        m = HEADER_RE.match("--- !u!1 &-100")
        assert m is not None
        assert m.group(2) == "-100"

    def test_matches_header_with_trailing_word(self):
        m = HEADER_RE.match("--- !u!1001480554 &100100000 stripped")
        assert m is not None

    def test_does_not_match_plain_line(self):
        assert HEADER_RE.match("m_Name: Player") is None

    def test_does_not_match_yaml_directive(self):
        assert HEADER_RE.match("%YAML 1.1") is None


# ---------------------------------------------------------------------------
# _decode_unity_bytes
# ---------------------------------------------------------------------------

class TestDecodeUnityBytes:
    def test_plain_utf8(self):
        assert _decode_unity_bytes(b"hello") == "hello"

    def test_strips_control_chars(self):
        result = _decode_unity_bytes(b"\x00hello\x01world")
        assert "\x00" not in result
        assert "\x01" not in result
        assert "hello" in result
        assert "world" in result

    def test_preserves_newline(self):
        result = _decode_unity_bytes(b"line1\nline2")
        assert "\n" in result

    def test_preserves_tab(self):
        result = _decode_unity_bytes(b"col1\tcol2")
        assert "\t" in result

    def test_latin1_fallback(self):
        result = _decode_unity_bytes(b"\xe9")
        assert isinstance(result, str)

    def test_utf8_bom_stripped(self):
        result = _decode_unity_bytes(b"\xef\xbb\xbfhello")
        assert result == "hello"

    def test_empty_bytes(self):
        assert _decode_unity_bytes(b"") == ""


# ---------------------------------------------------------------------------
# _split_unity_blocks
# ---------------------------------------------------------------------------

class TestSplitUnityBlocks:
    def _header(self, type_id: int, file_id: int) -> str:
        return f"--- !u!{type_id} &{file_id}"

    def test_single_block_parsed(self):
        text = f"{self._header(1, 100)}\nm_Name: Player\n"
        blocks = _split_unity_blocks(text)
        assert len(blocks) == 1
        t, fid, body = blocks[0]
        assert t == 1
        assert fid == 100
        assert "m_Name" in body

    def test_two_blocks_split(self):
        text = (
            f"{self._header(1, 100)}\nm_Name: A\n"
            f"{self._header(4, 200)}\nm_LocalPosition: {{x: 0}}\n"
        )
        blocks = _split_unity_blocks(text)
        assert len(blocks) == 2

    def test_empty_text_returns_empty(self):
        assert _split_unity_blocks("") == []

    def test_directive_lines_skipped(self):
        text = "%YAML 1.1\n%TAG !u! tag:unity3d.com,2011:\n--- !u!1 &5\ndata: x\n"
        blocks = _split_unity_blocks(text)
        assert len(blocks) == 1

    def test_negative_file_id_parsed(self):
        text = "--- !u!4 &-999\nm_Name: neg\n"
        blocks = _split_unity_blocks(text)
        assert blocks[0][1] == -999

    def test_body_ends_with_newline(self):
        text = f"{self._header(1, 1)}\nkey: value\n"
        blocks = _split_unity_blocks(text)
        assert blocks[0][2].endswith("\n")


# ---------------------------------------------------------------------------
# _fid
# ---------------------------------------------------------------------------

class TestFid:
    def test_returns_file_id_from_dict(self):
        assert _fid({"fileID": 100}) == 100

    def test_returns_zero_for_non_dict(self):
        assert _fid("not a dict") == 0
        assert _fid(None) == 0

    def test_returns_zero_when_no_fileID_key(self):
        assert _fid({"guid": "abc"}) == 0

    def test_handles_string_file_id(self):
        assert _fid({"fileID": "42"}) == 42

    def test_negative_file_id(self):
        assert _fid({"fileID": -1}) == -1


# ---------------------------------------------------------------------------
# _mesh_ref
# ---------------------------------------------------------------------------

class TestMeshRef:
    def test_builtin_guid_with_fileid(self):
        d = {"guid": _UNITY_BUILTIN_GUID, "fileID": 10202}
        assert _mesh_ref(d) == "builtin:10202"

    def test_external_guid(self):
        d = {"guid": "abc123", "fileID": 0}
        assert _mesh_ref(d) == "guid:abc123"

    def test_fileid_only(self):
        d = {"fileID": 99}
        assert _mesh_ref(d) == "fileID:99"

    def test_empty_dict_returns_empty(self):
        assert _mesh_ref({}) == ""

    def test_non_dict_returns_empty(self):
        assert _mesh_ref("not a dict") == ""
        assert _mesh_ref(None) == ""


# ---------------------------------------------------------------------------
# _material_refs
# ---------------------------------------------------------------------------

class TestMaterialRefs:
    def test_extracts_guid(self):
        mats = [{"guid": "abc", "fileID": 0}]
        refs = _material_refs(mats)
        assert "guid:abc" in refs

    def test_skips_entries_without_guid(self):
        mats = [{"fileID": 10}]
        refs = _material_refs(mats)
        assert refs == []

    def test_multiple_materials(self):
        mats = [{"guid": "a"}, {"guid": "b"}]
        refs = _material_refs(mats)
        assert len(refs) == 2

    def test_non_list_returns_empty(self):
        assert _material_refs(None) == []
        assert _material_refs("str") == []

    def test_empty_list_returns_empty(self):
        assert _material_refs([]) == []


# ---------------------------------------------------------------------------
# _vec3
# ---------------------------------------------------------------------------

class TestVec3:
    def test_extracts_xyz(self):
        result = _vec3({"x": 1.0, "y": 2.0, "z": 3.0})
        assert result == [1.0, 2.0, 3.0]

    def test_default_for_non_dict(self):
        result = _vec3(None)
        assert result == [0.0, 0.0, 0.0]

    def test_default_value_used_for_missing_keys(self):
        result = _vec3({})
        assert result == [0.0, 0.0, 0.0]

    def test_custom_default(self):
        result = _vec3({}, default=1.0)
        assert result == [1.0, 1.0, 1.0]


# ---------------------------------------------------------------------------
# _rgba_to_rgb
# ---------------------------------------------------------------------------

class TestRgbaToRgb:
    def test_extracts_rgb(self):
        result = _rgba_to_rgb({"r": 0.5, "g": 0.2, "b": 0.8, "a": 1.0})
        assert len(result) == 3
        assert abs(result[0] - 0.5) < 1e-6

    def test_default_for_non_dict(self):
        result = _rgba_to_rgb(None)
        assert result == [1.0, 1.0, 1.0]

    def test_missing_keys_use_default(self):
        result = _rgba_to_rgb({})
        assert result == [1.0, 1.0, 1.0]
