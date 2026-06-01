"""
Tier 4 — GodotToUnityCSharpConverter.convert_file()

Tests verify:
  • Stage 1 / 2 / 3 deterministic substitutions (_preprocess, _apply_type_mapping,
    _apply_api_mapping, _apply_rules) without any AI calls
  • Stage 5 _validate() catches invalid Godot patterns / missing UnityEngine
  • GodotToUnityCSharpConverter._apply_type_mapping  — TYPE_MAP substitutions
  • GodotToUnityCSharpConverter._apply_api_mapping   — API_MAP substitutions
  • GodotToUnityCSharpConverter._validate            — INVALID_PATTERNS + UnityEngine check
  • convert_file() writes output even when Gemini is mocked to return None
    (the file is written with whatever deterministic stage outputs)
  • _is_valid_arch_output() structural gating

Gemini is mocked via monkeypatch on the converter's .gemini attribute to prevent
any network calls and to control stage 4 output.
"""

import sys
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from script_converter import (
    GodotToUnityCSharpConverter,
    TYPE_MAP,
    API_MAP,
    INVALID_PATTERNS,
    _validate_godot_csharp,
    CSharpStructuralParser,
)


# ---------------------------------------------------------------------------
# Minimal valid Unity and Godot C# snippets
# ---------------------------------------------------------------------------

_MINIMAL_UNITY_CS = """\
using UnityEngine;

public class Player : MonoBehaviour
{
    void Start() { }
    void Update() { }
}
"""

_MINIMAL_GODOT_CS = """\
using Godot;

public partial class Player : Node3D
{
    public override void _Ready() { }
    public override void _Process(double delta) { }
}
"""


def _make_converter_with_mock_gemini(return_value=None):
    """Return a GodotToUnityCSharpConverter with Gemini mocked out."""
    converter = GodotToUnityCSharpConverter()
    mock_gemini = MagicMock()
    mock_gemini.generate_with_agent.return_value = return_value
    converter.gemini = mock_gemini
    return converter


# ---------------------------------------------------------------------------
# _apply_type_mapping — deterministic TYPE_MAP substitutions
# ---------------------------------------------------------------------------

class TestApplyTypeMapping:
    def test_node3d_replaced_with_monobehaviour(self):
        result = GodotToUnityCSharpConverter._apply_type_mapping(
            "public partial class Foo : Node3D { }"
        )
        assert "MonoBehaviour" in result
        assert "Node3D" not in result

    def test_characterbody3d_replaced(self):
        result = GodotToUnityCSharpConverter._apply_type_mapping(
            "public class Foo : CharacterBody3D { }"
        )
        assert "CharacterBody3D" not in result

    def test_rigidbody3d_replaced_with_rigidbody(self):
        result = GodotToUnityCSharpConverter._apply_type_mapping(
            "RigidBody3D rb;"
        )
        assert "Rigidbody" in result
        assert "RigidBody3D" not in result

    def test_animationplayer_replaced_with_animator(self):
        result = GodotToUnityCSharpConverter._apply_type_mapping(
            "AnimationPlayer anim;"
        )
        assert "Animator" in result

    def test_all_type_map_keys_replaced(self):
        for godot_type in TYPE_MAP:
            code = f"SomeClass c = new {godot_type}();"
            result = GodotToUnityCSharpConverter._apply_type_mapping(code)
            assert godot_type not in result or TYPE_MAP[godot_type] in result

    def test_unknown_type_unchanged(self):
        code = "MyCustomType foo;"
        result = GodotToUnityCSharpConverter._apply_type_mapping(code)
        assert result == code


# ---------------------------------------------------------------------------
# _apply_api_mapping — deterministic API_MAP substitutions
# ---------------------------------------------------------------------------

class TestApplyApiMapping:
    def test_ready_replaced_with_start(self):
        result = GodotToUnityCSharpConverter._apply_api_mapping(
            "public override void _Ready() { }"
        )
        assert "Start" in result
        assert "_Ready" not in result

    def test_process_replaced_with_update(self):
        result = GodotToUnityCSharpConverter._apply_api_mapping(
            "public override void _Process(double delta) { }"
        )
        assert "Update" in result
        assert "_Process" not in result

    def test_physics_process_replaced_with_fixedupdate(self):
        result = GodotToUnityCSharpConverter._apply_api_mapping(
            "public override void _PhysicsProcess(double delta) { }"
        )
        assert "FixedUpdate" in result

    def test_gd_print_replaced_with_debug_log(self):
        result = GodotToUnityCSharpConverter._apply_api_mapping(
            'GD.Print("hello");'
        )
        assert "Debug.Log" in result
        assert "GD.Print" not in result

    def test_exit_tree_replaced_with_on_destroy(self):
        result = GodotToUnityCSharpConverter._apply_api_mapping(
            "public override void _ExitTree() { }"
        )
        assert "OnDestroy" in result

    def test_all_api_map_keys_replaced(self):
        for godot_api in API_MAP:
            code = f"void method() {{ {godot_api}(); }}"
            result = GodotToUnityCSharpConverter._apply_api_mapping(code)
            assert API_MAP[godot_api] in result or godot_api not in result


# ---------------------------------------------------------------------------
# _validate — stage 5: INVALID_PATTERNS + UnityEngine check
# ---------------------------------------------------------------------------

class TestValidate:
    def test_clean_unity_code_has_no_errors(self):
        errors = GodotToUnityCSharpConverter._validate(_MINIMAL_UNITY_CS)
        assert errors == []

    def test_godot_code_flagged_as_invalid(self):
        errors = GodotToUnityCSharpConverter._validate(_MINIMAL_GODOT_CS)
        assert len(errors) > 0

    def test_missing_unity_engine_flagged(self):
        code = "public class Foo : MonoBehaviour { void Start() { } }"
        errors = GodotToUnityCSharpConverter._validate(code)
        assert any("UnityEngine" in e for e in errors)

    def test_node3d_in_code_flagged(self):
        code = "using UnityEngine;\npublic class Foo : Node3D { }"
        errors = GodotToUnityCSharpConverter._validate(code)
        assert any("Node3D" in e for e in errors)

    def test_using_godot_flagged(self):
        code = "using UnityEngine;\nusing Godot;\npublic class Foo : MonoBehaviour { }"
        errors = GodotToUnityCSharpConverter._validate(code)
        assert any("Godot" in e for e in errors)

    def test_queuefree_flagged(self):
        code = "using UnityEngine;\npublic class Foo : MonoBehaviour { void F() { QueueFree(); } }"
        errors = GodotToUnityCSharpConverter._validate(code)
        assert any("QueueFree" in e for e in errors)

    def test_each_invalid_pattern_caught(self):
        for pattern in INVALID_PATTERNS:
            code = f"using UnityEngine;\n// {pattern}\n"
            errors = GodotToUnityCSharpConverter._validate(code)
            assert any(pattern in e for e in errors), (
                f"INVALID_PATTERN '{pattern}' not caught by _validate"
            )


# ---------------------------------------------------------------------------
# _is_valid_arch_output — stage 4 structural gating
# ---------------------------------------------------------------------------

class TestIsValidArchOutput:
    def test_valid_unity_code_passes(self):
        assert GodotToUnityCSharpConverter._is_valid_arch_output(_MINIMAL_UNITY_CS)

    def test_missing_class_fails(self):
        code = "using UnityEngine;\nvoid Start() { }\n}"
        assert not GodotToUnityCSharpConverter._is_valid_arch_output(code)

    def test_missing_closing_brace_fails(self):
        code = "using UnityEngine;\npublic class Foo : MonoBehaviour { void Start() {"
        assert not GodotToUnityCSharpConverter._is_valid_arch_output(code)

    def test_missing_unity_engine_fails(self):
        code = "public class Foo : Node { void Start() { } }"
        assert not GodotToUnityCSharpConverter._is_valid_arch_output(code)

    def test_using_inside_class_fails(self):
        code = (
            "using UnityEngine;\n"
            "public class Foo : MonoBehaviour {\n"
            "    using System;\n"
            "    void Start() { }\n"
            "}\n"
        )
        assert not GodotToUnityCSharpConverter._is_valid_arch_output(code)


# ---------------------------------------------------------------------------
# convert_file — mocked Gemini
# ---------------------------------------------------------------------------

class TestConvertFileWithMockedGemini:
    def test_file_written_even_when_gemini_returns_none(self, tmp_path):
        cs_path = tmp_path / "Player.cs"
        cs_path.write_text(_MINIMAL_GODOT_CS, encoding="utf-8")
        output_path = tmp_path / "PlayerOut.cs"

        converter = _make_converter_with_mock_gemini(return_value=None)
        result = converter.convert_file(cs_path, output_path)

        assert output_path.exists()
        assert result.success is True

    def test_result_success_true_when_gemini_returns_valid_code(self, tmp_path):
        cs_path = tmp_path / "Player.cs"
        cs_path.write_text(_MINIMAL_GODOT_CS, encoding="utf-8")
        output_path = tmp_path / "PlayerOut.cs"

        converter = _make_converter_with_mock_gemini(return_value=_MINIMAL_UNITY_CS)
        result = converter.convert_file(cs_path, output_path)

        assert result.success is True

    def test_output_path_matches_result(self, tmp_path):
        cs_path = tmp_path / "Foo.cs"
        cs_path.write_text(_MINIMAL_GODOT_CS, encoding="utf-8")
        output_path = tmp_path / "FooOut.cs"

        converter = _make_converter_with_mock_gemini(return_value=_MINIMAL_UNITY_CS)
        result = converter.convert_file(cs_path, output_path)

        assert result.output_path == output_path

    def test_source_path_matches_result(self, tmp_path):
        cs_path = tmp_path / "Bar.cs"
        cs_path.write_text(_MINIMAL_GODOT_CS, encoding="utf-8")
        output_path = tmp_path / "BarOut.cs"

        converter = _make_converter_with_mock_gemini(return_value=_MINIMAL_UNITY_CS)
        result = converter.convert_file(cs_path, output_path)

        assert result.source_path == cs_path

    def test_empty_file_does_not_write_output(self, tmp_path):
        cs_path = tmp_path / "Empty.cs"
        cs_path.write_text("", encoding="utf-8")
        output_path = tmp_path / "EmptyOut.cs"

        converter = _make_converter_with_mock_gemini()
        result = converter.convert_file(cs_path, output_path)

        assert result.success is False
        assert result.error

    def test_missing_file_returns_error(self, tmp_path):
        cs_path = tmp_path / "nonexistent.cs"
        output_path = tmp_path / "out.cs"

        converter = _make_converter_with_mock_gemini()
        result = converter.convert_file(cs_path, output_path)

        assert result.success is False
        assert result.error

    def test_gemini_called_with_architecture_agent(self, tmp_path):
        cs_path = tmp_path / "Test.cs"
        cs_path.write_text(_MINIMAL_GODOT_CS, encoding="utf-8")
        output_path = tmp_path / "TestOut.cs"

        converter = _make_converter_with_mock_gemini(return_value=_MINIMAL_UNITY_CS)
        converter.convert_file(cs_path, output_path)

        calls = [call[0][0] for call in converter.gemini.generate_with_agent.call_args_list]
        assert "architecture" in calls


# ---------------------------------------------------------------------------
# GodotScriptConversionResult — dataclass
# ---------------------------------------------------------------------------

from script_converter import GodotScriptConversionResult, ModelUsageTracker, GeminiClient
from script_converter import _clean_placeholders
from script_converter import OllamaClient, _safe_generate, OLLAMA_MODEL


class TestGodotScriptConversionResult:
    def test_success_stored(self):
        r = GodotScriptConversionResult(
            source_path=Path("a.cs"), output_path=Path("b.cs"), success=True
        )
        assert r.success is True

    def test_failure_stored(self):
        r = GodotScriptConversionResult(
            source_path=Path("a.cs"), output_path=Path("b.cs"), success=False,
            error="oops"
        )
        assert r.error == "oops"

    def test_csharp_default_empty(self):
        r = GodotScriptConversionResult(
            source_path=Path("a.cs"), output_path=Path("b.cs"), success=True
        )
        assert r.csharp == ""

    def test_error_default_empty(self):
        r = GodotScriptConversionResult(
            source_path=Path("a.cs"), output_path=Path("b.cs"), success=True
        )
        assert r.error == ""


# ---------------------------------------------------------------------------
# ModelUsageTracker
# ---------------------------------------------------------------------------

class TestModelUsageTracker:
    def test_initial_state_clean(self):
        t = ModelUsageTracker()
        assert t.can_use("some-model") is True

    def test_mark_failed_blacklists_model(self):
        t = ModelUsageTracker()
        t.mark_failed("gemini-x")
        assert t.can_use("gemini-x") is False

    def test_mark_failed_twice_no_error(self):
        t = ModelUsageTracker()
        t.mark_failed("m1")
        t.mark_failed("m1")
        assert t.can_use("m1") is False

    def test_record_increments_today_counter(self):
        t = ModelUsageTracker()
        t.record("model-a")
        assert t._calls_today["model-a"] == 1

    def test_record_twice_increments_twice(self):
        t = ModelUsageTracker()
        t.record("model-a")
        t.record("model-a")
        assert t._calls_today["model-a"] == 2

    def test_can_use_unknown_model_returns_true(self):
        t = ModelUsageTracker()
        assert t.can_use("totally-unknown-model-xyz") is True

    def test_daily_limit_blocks_model(self):
        from script_converter import MODEL_LIMITS
        t = ModelUsageTracker()
        model = "gemini-2.5-flash"
        if model in MODEL_LIMITS:
            rpd = MODEL_LIMITS[model]["rpd"]
            t._calls_today[model] = rpd
            assert t.can_use(model) is False


# ---------------------------------------------------------------------------
# GeminiClient — no-key path
# ---------------------------------------------------------------------------

class TestGeminiClientNoKey:
    def test_returns_none_when_no_api_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        client = GeminiClient()
        client.api_key = ""
        result = client.generate("any prompt")
        assert result is None

    def test_generate_routes_to_architecture(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        client = GeminiClient()
        client.api_key = ""
        result = client.generate_with_agent("architecture", "any prompt")
        assert result is None

    def test_generate_with_role_shim_balanced(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        client = GeminiClient()
        client.api_key = ""
        result = client.generate_with_role("any prompt", role="balanced")
        assert result is None

    def test_generate_with_role_shim_fast(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        client = GeminiClient()
        client.api_key = ""
        result = client.generate_with_role("any prompt", role="fast")
        assert result is None


# ---------------------------------------------------------------------------
# _clean_placeholders
# ---------------------------------------------------------------------------

class TestCleanPlaceholders:
    def test_raycast_replaced(self):
        code = "var x = __RAYCAST_PLACEHOLDER__;"
        result = _clean_placeholders(code)
        assert "RaycastHit" in result
        assert "__RAYCAST_PLACEHOLDER__" not in result

    def test_scenetree_replaced(self):
        code = "var tree = __SCENETREE_PLACEHOLDER__;"
        result = _clean_placeholders(code)
        assert "SceneManager" in result
        assert "__SCENETREE_PLACEHOLDER__" not in result

    def test_scene_manager_using_injected(self):
        code = "using UnityEngine;\npublic class Foo : MonoBehaviour { var x = __SCENETREE_PLACEHOLDER__; }"
        result = _clean_placeholders(code)
        assert "using UnityEngine.SceneManagement;" in result

    def test_no_using_injected_when_scene_manager_absent(self):
        code = "using UnityEngine;\nvar x = 1;"
        result = _clean_placeholders(code)
        assert "SceneManagement" not in result

    def test_no_change_when_no_placeholders(self):
        code = "using UnityEngine;\npublic class X : MonoBehaviour {}"
        assert _clean_placeholders(code) == code


# ---------------------------------------------------------------------------
# CSharpStructuralParser
# ---------------------------------------------------------------------------

class TestCSharpStructuralParser:
    def _parser(self):
        return CSharpStructuralParser()

    def test_parse_simple_class(self):
        src = (
            "using UnityEngine;\n"
            "public class Player : MonoBehaviour {\n"
            "    void Start() { }\n"
            "}\n"
        )
        parsed = self._parser().parse(src)
        assert parsed.class_name == "Player"

    def test_parse_using_block_extracted(self):
        src = (
            "using UnityEngine;\n"
            "using System.Collections;\n"
            "public class Foo : MonoBehaviour { }\n"
        )
        parsed = self._parser().parse(src)
        assert "UnityEngine" in parsed.using_block
        assert "System.Collections" in parsed.using_block

    def test_parse_base_class_extracted(self):
        src = "using UnityEngine;\npublic class Enemy : MonoBehaviour { }\n"
        parsed = self._parser().parse(src)
        assert parsed.base_class == "MonoBehaviour"

    def test_parse_namespace_extracted(self):
        src = (
            "namespace MyGame {\n"
            "public class Hero : MonoBehaviour { }\n"
            "}\n"
        )
        parsed = self._parser().parse(src)
        assert "MyGame" in parsed.namespace_open

    def test_parse_no_class_raises(self):
        import pytest
        with pytest.raises(ValueError, match="No class declaration"):
            CSharpStructuralParser().parse("using UnityEngine;\nint x = 1;\n")

    def test_parse_attribute_before_class_captured(self):
        src = (
            "using UnityEngine;\n"
            "[RequireComponent(typeof(Rigidbody))]\n"
            "public class Foo : MonoBehaviour { }\n"
        )
        parsed = self._parser().parse(src)
        assert "RequireComponent" in parsed.attributes

    def test_parse_method_extracted(self):
        src = (
            "using UnityEngine;\n"
            "public class Foo : MonoBehaviour {\n"
            "    void Start() { Debug.Log(\"hi\"); }\n"
            "}\n"
        )
        parsed = self._parser().parse(src)
        assert any(m.name == "Start" for m in parsed.methods)

    def test_parse_multiple_methods(self):
        src = (
            "using UnityEngine;\n"
            "public class Foo : MonoBehaviour {\n"
            "    void Start() { }\n"
            "    void Update() { }\n"
            "}\n"
        )
        parsed = self._parser().parse(src)
        names = [m.name for m in parsed.methods]
        assert "Start" in names
        assert "Update" in names

    def test_parse_fields_block_before_methods(self):
        src = (
            "using UnityEngine;\n"
            "public class Foo : MonoBehaviour {\n"
            "    private int _hp = 100;\n"
            "    void Start() { }\n"
            "}\n"
        )
        parsed = self._parser().parse(src)
        assert "_hp" in parsed.fields_block

    def test_parse_inline_comment_not_breaking_parse(self):
        src = (
            "using UnityEngine; // engine\n"
            "public class Foo : MonoBehaviour {\n"
            "    void Start() { // init\n"
            "    }\n"
            "}\n"
        )
        parsed = self._parser().parse(src)
        assert parsed.class_name == "Foo"

    def test_parse_block_comment_skipped(self):
        src = (
            "/* header comment */\n"
            "using UnityEngine;\n"
            "public class Foo : MonoBehaviour {\n"
            "    void Start() { }\n"
            "}\n"
        )
        parsed = self._parser().parse(src)
        assert parsed.class_name == "Foo"

    def test_parse_string_literal_braces_ignored(self):
        src = (
            "using UnityEngine;\n"
            "public class Foo : MonoBehaviour {\n"
            '    void Start() { Debug.Log("{not a brace}"); }\n'
            "}\n"
        )
        parsed = self._parser().parse(src)
        assert any(m.name == "Start" for m in parsed.methods)

    def test_parse_verbatim_string(self):
        src = (
            "using UnityEngine;\n"
            "public class Foo : MonoBehaviour {\n"
            '    void Start() { string s = @"line1\nline2"; }\n'
            "}\n"
        )
        parsed = self._parser().parse(src)
        assert any(m.name == "Start" for m in parsed.methods)

    def test_parse_char_literal(self):
        src = (
            "using UnityEngine;\n"
            "public class Foo : MonoBehaviour {\n"
            "    void Start() { char c = '}'; }\n"
            "}\n"
        )
        parsed = self._parser().parse(src)
        assert any(m.name == "Start" for m in parsed.methods)


# ---------------------------------------------------------------------------
# OllamaClient — construction and mocked HTTP
# ---------------------------------------------------------------------------

class TestOllamaClientScriptConverter:
    def test_default_model(self):
        c = OllamaClient()
        assert c.model == OLLAMA_MODEL

    def test_custom_model(self):
        c = OllamaClient(model="mistral")
        assert c.model == "mistral"

    def test_health_url_property(self):
        c = OllamaClient(base_url="http://localhost:11434")
        assert c.health_url == "http://localhost:11434"

    def test_generate_returns_response(self, monkeypatch):
        from unittest.mock import MagicMock
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "converted code"}
        monkeypatch.setattr("script_converter.requests.post", lambda *a, **kw: mock_resp)
        result = OllamaClient().generate("some prompt")
        assert result == "converted code"

    def test_generate_raises_on_error_status(self, monkeypatch):
        from unittest.mock import MagicMock
        import pytest
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Server Error"
        monkeypatch.setattr("script_converter.requests.post", lambda *a, **kw: mock_resp)
        with pytest.raises(RuntimeError, match="Ollama error"):
            OllamaClient().generate("prompt")

    def test_is_running_true(self, monkeypatch):
        from unittest.mock import MagicMock
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        monkeypatch.setattr("script_converter.requests.get", lambda *a, **kw: mock_resp)
        assert OllamaClient().is_running() is True

    def test_is_running_false_on_exception(self, monkeypatch):
        import requests as req_mod
        monkeypatch.setattr(
            "script_converter.requests.get",
            lambda *a, **kw: (_ for _ in ()).throw(req_mod.RequestException("down"))
        )
        assert OllamaClient().is_running() is False


# ---------------------------------------------------------------------------
# _safe_generate — retry wrapper
# ---------------------------------------------------------------------------

class TestSafeGenerate:
    def test_returns_on_success(self, monkeypatch):
        from unittest.mock import MagicMock
        client = MagicMock()
        client.generate.return_value = "ok"
        result = _safe_generate(client, "prompt", retries=1)
        assert result == "ok"

    def test_reraises_after_all_retries(self, monkeypatch):
        import pytest
        from unittest.mock import MagicMock
        client = MagicMock()
        client.generate.side_effect = RuntimeError("always fails")
        monkeypatch.setattr("script_converter.time.sleep", lambda s: None)
        with pytest.raises(RuntimeError, match="always fails"):
            _safe_generate(client, "prompt", retries=2)

    def test_retries_on_failure_then_succeeds(self, monkeypatch):
        from unittest.mock import MagicMock
        client = MagicMock()
        client.generate.side_effect = [RuntimeError("fail"), "success"]
        monkeypatch.setattr("script_converter.time.sleep", lambda s: None)
        result = _safe_generate(client, "prompt", retries=3)
        assert result == "success"
