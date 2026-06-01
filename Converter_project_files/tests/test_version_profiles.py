"""
Tier 3 — version_profiles module.

Tests cover:
  • get_profile()               — returns correct data for supported version,
                                  raises UnsupportedVersionError for unknown
  • is_supported()              — True for known, False for unknown
  • parse_engine_option()       — splits "Unity 6000.3.9f1" → ("unity", "6000.3.9f1")
  • UnsupportedVersionError     — is a ValueError subclass
  • SUPPORTED_VERSIONS          — list, contains at least one entry
  • SOURCE/TARGET_ENGINE_OPTIONS — lists, each entry parseable
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from version_profiles import (
    PROFILES,
    SUPPORTED_VERSIONS,
    SOURCE_ENGINE_OPTIONS,
    TARGET_ENGINE_OPTIONS,
    UnsupportedVersionError,
    get_profile,
    is_supported,
    parse_engine_option,
)

_KNOWN_VERSION = "6000.3.9f1"


# ---------------------------------------------------------------------------
# UnsupportedVersionError
# ---------------------------------------------------------------------------

class TestUnsupportedVersionError:
    def test_is_value_error_subclass(self):
        assert issubclass(UnsupportedVersionError, ValueError)

    def test_can_be_raised_and_caught_as_value_error(self):
        with pytest.raises(ValueError):
            raise UnsupportedVersionError("bad version")

    def test_message_preserved(self):
        with pytest.raises(UnsupportedVersionError, match="test msg"):
            raise UnsupportedVersionError("test msg")


# ---------------------------------------------------------------------------
# get_profile
# ---------------------------------------------------------------------------

class TestGetProfile:
    def test_returns_dict_for_known_version(self):
        profile = get_profile(_KNOWN_VERSION)
        assert isinstance(profile, dict)

    def test_has_display_name(self):
        profile = get_profile(_KNOWN_VERSION)
        assert "display_name" in profile

    def test_has_has_prefab_instance(self):
        profile = get_profile(_KNOWN_VERSION)
        assert "has_prefab_instance" in profile

    def test_has_has_constrain_scale(self):
        profile = get_profile(_KNOWN_VERSION)
        assert "has_constrain_scale" in profile

    def test_has_scene_roots_class(self):
        profile = get_profile(_KNOWN_VERSION)
        assert "scene_roots_class" in profile

    def test_has_light_serial_ver(self):
        profile = get_profile(_KNOWN_VERSION)
        assert "light_serial_ver" in profile
        assert isinstance(profile["light_serial_ver"], int)

    def test_raises_on_unknown_version(self):
        with pytest.raises(UnsupportedVersionError):
            get_profile("9999.0.0f1")

    def test_raises_on_empty_string(self):
        with pytest.raises(UnsupportedVersionError):
            get_profile("")

    def test_error_message_contains_supported_list(self):
        with pytest.raises(UnsupportedVersionError, match=_KNOWN_VERSION):
            get_profile("nope")

    def test_known_version_in_profiles(self):
        assert _KNOWN_VERSION in PROFILES


# ---------------------------------------------------------------------------
# is_supported
# ---------------------------------------------------------------------------

class TestIsSupported:
    def test_true_for_known_version(self):
        assert is_supported(_KNOWN_VERSION) is True

    def test_false_for_unknown_version(self):
        assert is_supported("1.0.0f1") is False

    def test_false_for_empty_string(self):
        assert is_supported("") is False

    def test_false_for_partial_version(self):
        assert is_supported("6000") is False

    def test_consistent_with_supported_versions_list(self):
        for v in SUPPORTED_VERSIONS:
            assert is_supported(v) is True


# ---------------------------------------------------------------------------
# SUPPORTED_VERSIONS
# ---------------------------------------------------------------------------

class TestSupportedVersionsList:
    def test_is_list(self):
        assert isinstance(SUPPORTED_VERSIONS, list)

    def test_non_empty(self):
        assert len(SUPPORTED_VERSIONS) >= 1

    def test_all_versions_have_profiles(self):
        for v in SUPPORTED_VERSIONS:
            assert v in PROFILES

    def test_all_profiles_in_supported_versions(self):
        for v in PROFILES:
            assert v in SUPPORTED_VERSIONS


# ---------------------------------------------------------------------------
# parse_engine_option
# ---------------------------------------------------------------------------

class TestParseEngineOption:
    def test_unity_version_parsed(self):
        engine, version = parse_engine_option("Unity 6000.3.9f1")
        assert engine == "unity"
        assert version == "6000.3.9f1"

    def test_godot_version_parsed(self):
        engine, version = parse_engine_option("Godot 4.5")
        assert engine == "godot"
        assert version == "4.5"

    def test_engine_is_lowercased(self):
        engine, _ = parse_engine_option("Unity 6000.3.9f1")
        assert engine == engine.lower()

    def test_version_preserved_verbatim(self):
        _, version = parse_engine_option("Unity 6000.3.9f1")
        assert version == "6000.3.9f1"

    def test_unknown_engine_not_crash(self):
        engine, version = parse_engine_option("Unreal 5.4")
        assert engine == "unreal"
        assert version == "5.4"

    def test_no_space_returns_engine_only(self):
        engine, version = parse_engine_option("Unity")
        assert engine == "unity"
        assert version == ""

    def test_source_options_all_parseable(self):
        for opt in SOURCE_ENGINE_OPTIONS:
            engine, version = parse_engine_option(opt)
            assert engine
            assert version

    def test_target_options_all_parseable(self):
        for opt in TARGET_ENGINE_OPTIONS:
            engine, version = parse_engine_option(opt)
            assert engine
            assert version


# ---------------------------------------------------------------------------
# SOURCE / TARGET engine option lists
# ---------------------------------------------------------------------------

class TestEngineOptionLists:
    def test_source_engine_options_non_empty(self):
        assert len(SOURCE_ENGINE_OPTIONS) >= 1

    def test_target_engine_options_non_empty(self):
        assert len(TARGET_ENGINE_OPTIONS) >= 1

    def test_unity_in_source_options(self):
        assert any("Unity" in opt for opt in SOURCE_ENGINE_OPTIONS)

    def test_godot_in_target_options(self):
        assert any("Godot" in opt for opt in TARGET_ENGINE_OPTIONS)
