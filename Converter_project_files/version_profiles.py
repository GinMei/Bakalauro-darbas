"""version_profiles.py

Version-aware configuration for the Unity parser.

Unity source: only "6000.3.9f1" is supported.
Godot target: only "4.5" is supported.
"""

from __future__ import annotations
from typing import Any, Dict

# ── Unity source versions ─────────────────────────────────────────────────────

SUPPORTED_VERSIONS: list[str] = [
    "6000.3.9f1",
]

PROFILES: Dict[str, Dict[str, Any]] = {
    "6000.3.9f1": {
        "display_name":        "Unity 6 (6000.3.9f1)",
        "light_serial_ver":    12,
        "has_prefab_instance": True,
        "has_constrain_scale": True,
        "scene_roots_class":   "SceneRoots",
    },
}

class UnsupportedVersionError(ValueError):
    """Raised when an unknown or unsupported Unity version is requested."""


def get_profile(unity_version: str) -> Dict[str, Any]:
    if unity_version not in PROFILES:
        supported = ", ".join(SUPPORTED_VERSIONS)
        raise UnsupportedVersionError(
            f"Unity version '{unity_version}' is not supported. "
            f"Supported versions: {supported}"
        )
    return PROFILES[unity_version]


def is_supported(unity_version: str) -> bool:
    return unity_version in PROFILES


# ── Godot target (fixed at 4.5) ───────────────────────────────────────────────

GODOT_VERSION  = "4.5"
GODOT_FEATURES = 'PackedStringArray("4.5", "C#", "Forward Plus")'

# ── Supported source and target engines ──────────────────────────────────────

SOURCE_ENGINES: list[str] = ["Unity", "Godot"]
TARGET_ENGINES: list[str] = ["Godot", "Unity"]

# ── Combined engine+version dropdown options ──────────────────────────────────
# These are the strings shown in the UI dropdowns.
# Format: "<EngineName> <version>"

SOURCE_ENGINE_OPTIONS: list[str] = [
    "Unity 6000.3.9f1",
    "Godot 4.5",
]

TARGET_ENGINE_OPTIONS: list[str] = [
    "Godot 4.5",
    "Unity 6000.3.9f1",
]


def parse_engine_option(option: str) -> tuple[str, str]:
    """Split "Unity 6000.3.9f1" → ("unity", "6000.3.9f1").
       Split "Godot 4.5"       → ("godot", "4.5").
    """
    parts = option.split(" ", 1)
    engine  = parts[0].lower() if parts else "unknown"
    version = parts[1] if len(parts) > 1 else ""
    return engine, version
