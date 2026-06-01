"""
Tests for unity_to_godot_script_converter.py
— U2GConversionResult and U2GBatchResult dataclasses,
  _UNITY_NAMESPACES / _UNITY_CLASSES / _UNITY_LIFECYCLE regex patterns.
"""

import sys
import os
import re
import pytest

# Remove any stub injected by other test modules (e.g. test_job_lifecycle.py)
# so we import the real module here.
sys.modules.pop("unity_to_godot_script_converter", None)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unity_to_godot.unity_to_godot_script_converter import (
    U2GConversionResult,
    U2GBatchResult,
    _UNITY_NAMESPACES,
    _UNITY_CLASSES,
    _UNITY_LIFECYCLE,
    _OllamaClient,
    _GeminiClient,
    _RateLimitError,
    _GeminiError,
)


# ---------------------------------------------------------------------------
# U2GConversionResult
# ---------------------------------------------------------------------------

class TestU2GConversionResult:
    def test_success_true(self):
        r = U2GConversionResult(success=True)
        assert r.success is True

    def test_success_false(self):
        r = U2GConversionResult(success=False)
        assert r.success is False

    def test_source_path_default_empty(self):
        r = U2GConversionResult(success=True)
        assert r.source_path == ""

    def test_output_path_default_empty(self):
        r = U2GConversionResult(success=True)
        assert r.output_path == ""

    def test_godot_code_default_empty(self):
        r = U2GConversionResult(success=True)
        assert r.godot_code == ""

    def test_extracted_structure_default_empty_dict(self):
        r = U2GConversionResult(success=True)
        assert r.extracted_structure == {}

    def test_semantic_summary_default_empty(self):
        r = U2GConversionResult(success=True)
        assert r.semantic_summary == ""

    def test_ir_default_empty_dict(self):
        r = U2GConversionResult(success=True)
        assert r.ir == {}

    def test_todos_default_empty_list(self):
        r = U2GConversionResult(success=True)
        assert r.todos == []

    def test_warnings_default_empty_list(self):
        r = U2GConversionResult(success=True)
        assert r.warnings == []

    def test_error_default_empty(self):
        r = U2GConversionResult(success=True)
        assert r.error == ""

    def test_stages_completed_default_empty(self):
        r = U2GConversionResult(success=True)
        assert r.stages_completed == []

    def test_used_ollama_fallback_default_false(self):
        r = U2GConversionResult(success=True)
        assert r.used_ollama_fallback is False

    def test_set_source_path(self):
        r = U2GConversionResult(success=True, source_path="/src/Player.cs")
        assert r.source_path == "/src/Player.cs"

    def test_set_error(self):
        r = U2GConversionResult(success=False, error="Something went wrong")
        assert r.error == "Something went wrong"

    def test_lists_independent_per_instance(self):
        r1 = U2GConversionResult(success=True)
        r2 = U2GConversionResult(success=True)
        r1.todos.append("TODO 1")
        assert r2.todos == []

    def test_dicts_independent_per_instance(self):
        r1 = U2GConversionResult(success=True)
        r2 = U2GConversionResult(success=True)
        r1.ir["key"] = "value"
        assert r2.ir == {}


# ---------------------------------------------------------------------------
# U2GBatchResult
# ---------------------------------------------------------------------------

class TestU2GBatchResult:
    def test_success_true(self):
        r = U2GBatchResult(success=True)
        assert r.success is True

    def test_success_false(self):
        r = U2GBatchResult(success=False)
        assert r.success is False

    def test_converted_default_empty(self):
        r = U2GBatchResult(success=True)
        assert r.converted == []

    def test_failed_default_empty(self):
        r = U2GBatchResult(success=True)
        assert r.failed == []

    def test_total_files_default_zero(self):
        r = U2GBatchResult(success=True)
        assert r.total_files == 0

    def test_total_todos_default_zero(self):
        r = U2GBatchResult(success=True)
        assert r.total_todos == 0

    def test_total_warnings_default_zero(self):
        r = U2GBatchResult(success=True)
        assert r.total_warnings == 0

    def test_set_total_files(self):
        r = U2GBatchResult(success=True, total_files=5)
        assert r.total_files == 5

    def test_converted_and_failed_independent(self):
        r1 = U2GBatchResult(success=True)
        r2 = U2GBatchResult(success=True)
        r1.converted.append(U2GConversionResult(success=True))
        assert r2.converted == []

    def test_add_conversion_result(self):
        res = U2GConversionResult(success=True, source_path="Foo.cs")
        batch = U2GBatchResult(success=True)
        batch.converted.append(res)
        assert len(batch.converted) == 1
        assert batch.converted[0].source_path == "Foo.cs"


# ---------------------------------------------------------------------------
# Regex pattern tests
# ---------------------------------------------------------------------------

class TestUnityNamespacesRegex:
    def test_matches_using_unity_engine(self):
        assert _UNITY_NAMESPACES.search("using UnityEngine;")

    def test_matches_using_unity_editor(self):
        assert _UNITY_NAMESPACES.search("using UnityEditor;")

    def test_matches_unity_dotted_namespace(self):
        assert _UNITY_NAMESPACES.search("using Unity.SomeModule;")

    def test_matches_tmpro(self):
        assert _UNITY_NAMESPACES.search("using TMPro;")

    def test_does_not_match_using_godot(self):
        assert not _UNITY_NAMESPACES.search("using Godot;")

    def test_does_not_match_using_system(self):
        assert not _UNITY_NAMESPACES.search("using System;")


class TestUnityClassesRegex:
    def test_matches_monobehaviour(self):
        assert _UNITY_CLASSES.search("public class Foo : MonoBehaviour")

    def test_matches_scriptableobject(self):
        assert _UNITY_CLASSES.search("public class Data : ScriptableObject")

    def test_matches_editor(self):
        assert _UNITY_CLASSES.search("public class MyEditor : Editor")

    def test_does_not_match_node3d(self):
        assert not _UNITY_CLASSES.search("public class Foo : Node3D")


class TestUnityLifecycleRegex:
    def test_matches_awake(self):
        assert _UNITY_LIFECYCLE.search("void Awake()")

    def test_matches_start(self):
        assert _UNITY_LIFECYCLE.search("void Start()")

    def test_matches_update(self):
        assert _UNITY_LIFECYCLE.search("void Update()")

    def test_matches_fixed_update(self):
        assert _UNITY_LIFECYCLE.search("void FixedUpdate()")

    def test_matches_on_destroy(self):
        assert _UNITY_LIFECYCLE.search("void OnDestroy()")

    def test_matches_on_trigger_enter(self):
        assert _UNITY_LIFECYCLE.search("void OnTriggerEnter(")

    def test_does_not_match_ready(self):
        assert not _UNITY_LIFECYCLE.search("void _Ready()")

    def test_does_not_match_process(self):
        assert not _UNITY_LIFECYCLE.search("void _Process(double delta)")


# ---------------------------------------------------------------------------
# _OllamaClient — construction (no network calls)
# ---------------------------------------------------------------------------

class TestOllamaClient:
    def test_default_model(self):
        client = _OllamaClient()
        assert client._model == "qwen3"

    def test_custom_model(self):
        client = _OllamaClient(model="mistral")
        assert client._model == "mistral"

    def test_is_available_returns_bool(self):
        import unittest.mock as mock
        client = _OllamaClient()
        with mock.patch("unity_to_godot.unity_to_godot_script_converter.requests.get") as m:
            m.return_value.status_code = 200
            result = client.is_available()
            assert result is True

    def test_is_available_false_when_unreachable(self):
        import unittest.mock as mock
        import requests as req_module
        client = _OllamaClient()
        with mock.patch("unity_to_godot.unity_to_godot_script_converter.requests.get",
                        side_effect=req_module.RequestException("connection refused")):
            result = client.is_available()
            assert result is False

    def test_generate_raises_on_error_status(self):
        import unittest.mock as mock
        client = _OllamaClient()
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        with mock.patch("unity_to_godot.unity_to_godot_script_converter.requests.post",
                        return_value=mock_resp):
            with pytest.raises(RuntimeError):
                client.generate("test prompt")

    def test_generate_returns_response_text(self):
        import unittest.mock as mock
        client = _OllamaClient()
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "converted code"}
        with mock.patch("unity_to_godot.unity_to_godot_script_converter.requests.post",
                        return_value=mock_resp):
            result = client.generate("prompt")
            assert result == "converted code"

    def test_generate_raises_on_request_exception(self):
        import unittest.mock as mock
        import requests as req_module
        client = _OllamaClient()
        with mock.patch("unity_to_godot.unity_to_godot_script_converter.requests.post",
                        side_effect=req_module.RequestException("connection refused")):
            with pytest.raises(RuntimeError, match="Ollama unreachable"):
                client.generate("prompt")


# ---------------------------------------------------------------------------
# _GeminiClient — construction and key management
# ---------------------------------------------------------------------------

class TestGeminiClient:
    def test_single_key_stored(self):
        c = _GeminiClient(primary_key="key1")
        assert c._keys == ["key1"]

    def test_fallback_key_included_when_provided(self):
        c = _GeminiClient(primary_key="key1", fallback_key="key2")
        assert c._keys == ["key1", "key2"]

    def test_empty_fallback_not_stored(self):
        c = _GeminiClient(primary_key="key1", fallback_key="")
        assert c._keys == ["key1"]

    def test_initial_index_zero(self):
        c = _GeminiClient(primary_key="key1")
        assert c._idx == 0

    def test_total_waited_starts_zero(self):
        c = _GeminiClient(primary_key="key1")
        assert c._total_waited == 0

    def test_current_key_returns_primary(self):
        c = _GeminiClient(primary_key="mykey")
        assert c._current_key() == "mykey"

    def test_current_key_raises_when_no_keys(self):
        c = _GeminiClient(primary_key="")
        with pytest.raises(RuntimeError, match="No Gemini API key"):
            c._current_key()

    def test_rotate_key_advances_index(self):
        c = _GeminiClient(primary_key="k1", fallback_key="k2")
        c._rotate_key()
        assert c._idx == 1

    def test_rotate_key_wraps_around(self):
        c = _GeminiClient(primary_key="k1", fallback_key="k2")
        c._idx = 1
        c._rotate_key()
        assert c._idx == 0

    def test_rotate_key_single_key_stays(self):
        c = _GeminiClient(primary_key="k1")
        c._rotate_key()
        assert c._idx == 0


# ---------------------------------------------------------------------------
# _RateLimitError and _GeminiError — are proper Exception subclasses
# ---------------------------------------------------------------------------

class TestExceptionClasses:
    def test_rate_limit_error_is_exception(self):
        with pytest.raises(_RateLimitError):
            raise _RateLimitError("rate limited")

    def test_gemini_error_is_exception(self):
        with pytest.raises(_GeminiError):
            raise _GeminiError("api error")

    def test_rate_limit_stores_message(self):
        exc = _RateLimitError("msg")
        assert "msg" in str(exc)

    def test_gemini_error_stores_message(self):
        exc = _GeminiError("oops")
        assert "oops" in str(exc)
