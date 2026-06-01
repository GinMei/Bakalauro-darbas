"""ir_builder.py

Converts the raw parser output (unity_parser) into a spec-compliant,
engine-agnostic IR bundle.

IR design principles:
  - ``node_type`` is an engine-agnostic label (e.g. "entity", "camera").
  - ``components`` carries engine-agnostic sub-type data (mesh refs, light
    sub-type, material refs, prefab_instance).  Target builders read
    ``components``, not raw Unity or Godot structures.
  - ``original_data`` preserves raw source-engine fields for debugging and
    round-trip fidelity; it must not be relied upon by target generators.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Scalar normalisation — Unity YAML sometimes serialises floats as {value: x}
# ---------------------------------------------------------------------------

def _unwrap_scalar(v: Any, *, context: str = "", warnings: Optional[List[str]] = None) -> Any:
    """Unwrap Unity YAML scalar wrapper dicts like {"value": 1.0} → 1.0.

    Unity YAML (and some intermediate parsers) can represent a numeric scalar
    as a single-key dict ``{"value": x}``.  This function normalises that
    pattern so that ``float(_unwrap_scalar(v))`` is always safe.

    * If *v* is not a dict, it is returned unchanged.
    * If *v* is ``{"value": x}``, *x* is returned.
    * If *v* is any other dict, a warning is appended (if *warnings* is given)
      and 0.0 is returned as a safe fallback.
    """
    if not isinstance(v, dict):
        return v
    if "value" in v:
        return v["value"]
    # Malformed: dict without "value" key
    msg = f"Unexpected dict for scalar{(' at ' + context) if context else ''}: {v!r}"
    if warnings is not None:
        warnings.append(msg)
    assert not isinstance(v, dict), msg  # fires in dev/test; removed with -O
    return 0.0  # production fallback (assert is a no-op with -O)


# ---------------------------------------------------------------------------
# Node type mapping: Unity component combination → engine-agnostic IR type
# ---------------------------------------------------------------------------

# IR node types (engine-agnostic labels used in the spec)
_IR_TYPE_ENTITY       = "entity"
_IR_TYPE_CAMERA       = "camera"
_IR_TYPE_LIGHT        = "light"
_IR_TYPE_AUDIO_SOURCE = "audio_source"
_IR_TYPE_UI_CANVAS    = "ui_canvas"
_IR_TYPE_UI_ELEMENT   = "ui_element"
_IR_TYPE_PARTICLES    = "particle_system"
_IR_TYPE_SPRITE       = "sprite"
_IR_TYPE_GROUP        = "group"   # Node3D with no special component
_IR_TYPE_WORLD_ENV    = "world_environment"  # WorldEnvironment (URP Volume)
_IR_TYPE_TERRAIN      = "terrain_object"     # Unity Terrain (heightmap + mesh)
_IR_TYPE_WIND_ZONE    = "wind_zone"          # Unity WindZone
_IR_TYPE_TEXT_3D      = "text_3d"            # Unity TextMesh (legacy 3D text)
_IR_TYPE_TREE         = "tree"               # Unity Tree (SpeedTree / Tree Creator)
_IR_TYPE_NAVIGATION   = "navigation"         # NavigationRegion3D / baked NavMesh

# IR type → engine-agnostic light sub-type label.
# Used when building the ``components`` dict so target generators can emit
# the correct light class without reading Godot strings from the IR.
_LIGHT_SUBTYPE: Dict[str, str] = {
    "DirectionalLight3D": "directional",
    "OmniLight3D":        "point",
    "SpotLight3D":        "spot",
    "ReflectionProbe":    "probe",
}


def _make_meta(
    status: str = "mapped",
    confidence: float = 1.0,
    notes: str = "",
) -> Dict[str, Any]:
    """Returns a minimal meta block. Extended fields added in later steps."""
    return {
        "conversion_status": status,   # "mapped" | "partial" | "unknown"
        "confidence": confidence,
        "ai_generated": False,
        "user_edited": False,
        "requires_review": confidence < 0.75 or status == "unknown",
        "notes": notes,
        "warnings": [],
        "errors": [],
    }


def _safe_id(text: str) -> str:
    """Turns any string into a safe identifier segment (lowercase, underscores)."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "scene"


def _build_node_id(scene_prefix: str, counter: List[int]) -> str:
    """Generates the next prefixed node ID and increments the counter."""
    counter[0] += 1
    return f"{scene_prefix}_node_{counter[0]:04d}"


# ---------------------------------------------------------------------------
# Internal helpers for reading the parser's raw node dict
# ---------------------------------------------------------------------------

def _pos(transform: Dict[str, Any]) -> List[float]:
    p = transform.get("position") or [0.0, 0.0, 0.0]
    if isinstance(p, (list, tuple)) and len(p) >= 3:
        return [float(_unwrap_scalar(p[0])), float(_unwrap_scalar(p[1])), float(_unwrap_scalar(p[2]))]
    return [0.0, 0.0, 0.0]


def _rot(transform: Dict[str, Any]) -> List[float]:
    """Return rotation as a [x, y, z, w] quaternion array."""
    r = transform.get("rotation") or {}
    if isinstance(r, dict):
        return [
            float(_unwrap_scalar(r.get("x", 0.0))),
            float(_unwrap_scalar(r.get("y", 0.0))),
            float(_unwrap_scalar(r.get("z", 0.0))),
            float(_unwrap_scalar(r.get("w", 1.0))),
        ]
    if isinstance(r, (list, tuple)) and len(r) >= 4:
        return [float(_unwrap_scalar(r[i])) for i in range(4)]
    return [0.0, 0.0, 0.0, 1.0]


def _scl(transform: Dict[str, Any]) -> List[float]:
    s = transform.get("scale") or [1.0, 1.0, 1.0]
    if isinstance(s, (list, tuple)) and len(s) >= 3:
        return [float(_unwrap_scalar(s[0])), float(_unwrap_scalar(s[1])), float(_unwrap_scalar(s[2]))]
    return [1.0, 1.0, 1.0]


# Valid Unity/Godot physics combine modes — anything outside this set is invalid.
_VALID_COMBINE_MODES: frozenset = frozenset({"average", "minimum", "maximum", "multiply"})


def _build_physics_material_ir(
    pm_ref: Optional[str],
    pm_map: Optional[Dict[str, Any]],
    col_id: str,
    warnings: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Build an engine-agnostic physics_material IR from a raw guid ref.

    Returns None when the ref is absent or the map doesn't contain it.

    Schema:
      pm_id             str   — unique identifier for SubResource dedup
      friction          float — dynamic friction (the value active during motion)
      static_friction   float — friction when stationary (stored as metadata in Godot)
      restitution       float — bounce coefficient [0, 1]
      friction_combine  str   — "average"|"minimum"|"maximum"|"multiply"
      bounce_combine    str   — "average"|"minimum"|"maximum"|"multiply"
    """
    if not pm_ref or not pm_map:
        return None
    # Fix 6: validate GUID ref — "guid:" with an empty part must not silently produce None.
    if ":" not in pm_ref or not pm_ref.split(":", 1)[1]:
        if warnings is not None:
            warnings.append(
                f"Malformed physics material GUID ref '{pm_ref}' on {col_id}; skipping."
            )
        return None
    guid = pm_ref.split(":", 1)[1] if pm_ref.startswith("guid:") else pm_ref
    raw = pm_map.get(guid)
    if not raw:
        return None
    # Fix 9: validate combine modes against the whitelist; fallback to "average" with a warning.
    fc = raw.get("friction_combine", "average")
    if fc not in _VALID_COMBINE_MODES:
        if warnings is not None:
            warnings.append(
                f"Unknown friction_combine mode '{fc}' on {col_id}; using 'average'."
            )
        fc = "average"
    bc = raw.get("bounce_combine", "average")
    if bc not in _VALID_COMBINE_MODES:
        if warnings is not None:
            warnings.append(
                f"Unknown bounce_combine mode '{bc}' on {col_id}; using 'average'."
            )
        bc = "average"
    return {
        "pm_id":            f"pm_{guid[:12]}",
        "friction":         float(_unwrap_scalar(raw.get("dynamic_friction", 0.6))),
        "static_friction":  float(_unwrap_scalar(raw.get("static_friction",  0.6))),
        "restitution":      float(_unwrap_scalar(raw.get("restitution",       0.0))),
        "friction_combine": fc,
        "bounce_combine":   bc,
    }


def _build_collider_ir(
    raw_col: Dict[str, Any],
    col_index: int,
    node_id: str,
    pm_map: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Convert a raw parser collider dict to engine-agnostic IR.

    Shape keys per type:
      box:          size [x,y,z], center [x,y,z]
      sphere:       radius, center
      capsule:      radius, height, axis (0=X 1=Y 2=Z), center
      mesh_convex / mesh_concave:  mesh_ref (string), center
    All dimensions are clamped to a safe minimum (0.001) so zero-sized shapes
    never reach the emitter.  A warning is emitted whenever clamping occurs.

    ``physics_material`` is a nested IR dict (or None) built from the collider's
    physics material GUID reference and the provided pm_map.
    """
    shape_type = raw_col.get("type", "box")
    center     = raw_col.get("center") or [0.0, 0.0, 0.0]
    is_trigger = bool(raw_col.get("is_trigger", False))

    shape: Dict[str, Any] = {"type": shape_type, "center": [float(_unwrap_scalar(v)) for v in center]}
    col_id = f"{node_id}_col_{col_index}"

    # Fix 8: warn whenever a dimension is clamped rather than silently rounding up.
    if shape_type == "box":
        raw_size = raw_col.get("size") or [1.0, 1.0, 1.0]
        clamped = [max(0.001, float(_unwrap_scalar(v))) for v in raw_size]
        if warnings is not None and any(float(_unwrap_scalar(raw_size[i])) < 0.001 for i in range(len(raw_size))):
            warnings.append(
                f"Box collider {col_id} had a dimension < 0.001; clamped to 0.001."
            )
        shape["size"] = clamped
    elif shape_type == "sphere":
        r = float(_unwrap_scalar(raw_col.get("radius", 0.5)))
        if warnings is not None and r < 0.001:
            warnings.append(
                f"Sphere collider {col_id} radius {r} < 0.001; clamped to 0.001."
            )
        shape["radius"] = max(0.001, r)
    elif shape_type == "capsule":
        r = float(_unwrap_scalar(raw_col.get("radius", 0.5)))
        h = float(_unwrap_scalar(raw_col.get("height", 2.0)))
        if warnings is not None and (r < 0.001 or h < 0.001):
            warnings.append(
                f"Capsule collider {col_id} dimension(s) < 0.001; clamped to 0.001."
            )
        r = max(0.001, r)
        h = max(0.001, h)
        # Godot 4 requires CapsuleShape3D height >= 2 * radius (hemispherical caps).
        min_h = 2.0 * r
        if h < min_h:
            if warnings is not None:
                warnings.append(
                    f"Capsule collider {col_id} height {h:.6g} < 2 * radius "
                    f"({min_h:.6g}); clamped to {min_h:.6g}."
                )
            h = min_h
        shape["radius"] = r
        shape["height"] = h
        shape["axis"]   = int(raw_col.get("axis", 1))   # 0=X  1=Y  2=Z
    elif shape_type in ("mesh_convex", "mesh_concave"):
        shape["mesh_ref"] = raw_col.get("mesh_ref", "")
    elif shape_type == "wheel":
        # Preserve all wheel-specific data so _detect_vehicles and the emitter can read it.
        shape["radius"]              = max(0.001, float(_unwrap_scalar(raw_col.get("radius", 0.5))))
        shape["suspension_distance"] = max(0.0,   float(_unwrap_scalar(raw_col.get("suspension_distance", 0.3))))
        shape["mass"]                = max(0.001, float(_unwrap_scalar(raw_col.get("mass", 20.0))))
        shape["spring_stiffness"]    = float(_unwrap_scalar(raw_col.get("spring_stiffness", 35000.0)))
        shape["spring_damper"]       = float(_unwrap_scalar(raw_col.get("spring_damper", 4500.0)))
        shape["sideways_stiffness"]  = float(_unwrap_scalar(raw_col.get("sideways_stiffness", 1.0)))

    pm_ref = raw_col.get("physics_material_ref")
    physics_material = _build_physics_material_ir(pm_ref, pm_map, col_id, warnings=warnings)

    return {
        "collider_id":     col_id,
        "shape":           shape,
        "is_trigger":      is_trigger,
        "physics_material": physics_material,   # None when not assigned
    }


def _build_rigidbody_ir(raw_rb: Dict[str, Any], node_id: str) -> Dict[str, Any]:
    """Convert a raw parser rigidbody dict to engine-agnostic IR.

    type: "dynamic" | "kinematic" | "static"
    Constraint booleans preserve Unity's per-axis freeze flags.

    Extended fields:
      interpolation       "none"|"interpolate"|"extrapolate"
                          Godot 4 has no per-body interpolation; stored for reference.
      collision_detection "discrete"|"continuous"|"continuous_dynamic"|"continuous_speculative"
                          Maps to Godot RigidBody3D.continuous_cd (true for any non-discrete).
      center_of_mass      [x,y,z] when Unity m_AutomaticCenterOfMass==0, else None.
                          Maps to Godot center_of_mass_mode=1 + center_of_mass vector.
      sleep_threshold     float — Unity energy threshold; no Godot per-body equivalent,
                          stored as metadata/sleep_threshold.
    """
    constraints = raw_rb.get("constraints") or {}
    return {
        "rigidbody_id":      f"rb_{node_id}",
        "type":              raw_rb.get("type", "dynamic"),
        "mass":              max(0.001, float(_unwrap_scalar(raw_rb.get("mass", 1.0)))),
        "linear_damping":    max(0.0, float(_unwrap_scalar(raw_rb.get("linear_damping", 0.0)))),
        "angular_damping":   max(0.0, float(_unwrap_scalar(raw_rb.get("angular_damping", 0.05)))),
        "gravity_enabled":   bool(raw_rb.get("gravity_enabled", True)),
        "constraints": {
            "freeze_position": list(constraints.get("freeze_position", [False, False, False])),
            "freeze_rotation": list(constraints.get("freeze_rotation", [False, False, False])),
        },
        "interpolation":       raw_rb.get("interpolation", "none"),
        "collision_detection": raw_rb.get("collision_detection", "discrete"),
        "center_of_mass":      raw_rb.get("center_of_mass"),  # [x,y,z] or None
        "sleep_threshold":     float(_unwrap_scalar(raw_rb.get("sleep_threshold", 0.005))),
    }


# ---------------------------------------------------------------------------
# Camera IR builder
# ---------------------------------------------------------------------------

_DEFAULT_CAMERA: Dict[str, Any] = {
    "projection":       "perspective",
    "fov":              60.0,
    "ortho_size":       5.0,     # Unity half-height; doubled when emitting Godot 'size'
    "near_clip":        0.1,
    "far_clip":         1000.0,
    "clear_flags":      "skybox",
    "background_color": [0.1921, 0.3020, 0.4745, 0.0],
    "culling_mask":     0xFFFFFFFF,
}


def _build_camera_ir(raw_camera: Dict[str, Any], node_id: str) -> Dict[str, Any]:
    """Convert a raw parser camera dict to an engine-agnostic camera IR.

    IR schema:
      camera_id:               str
      projection:              "perspective" | "orthographic"
      fov:                     float (vertical degrees — perspective only)
      ortho_size:              float (Unity half-height — orthographic only; builders double it)
      near_clip:               float
      far_clip:                float
      clear_flags:             "skybox" | "solid_color" | "depth_only" | "dont_clear"
                               NOTE: Godot 4 does not expose clear flags on Camera3D; stored as
                               metadata for reference. Background is set via Environment/Viewport.
      background_color:        [r, g, b, a]  (used when clear_flags == "solid_color")
      culling_mask:            int (32-bit layer bitmask — maps directly to Godot cull_mask)
      is_main_camera:          bool — True when the Unity GameObject tag is "MainCamera".
                               Used to prioritise which camera receives per-camera
                               CameraAttributesPhysical (DOF + auto-exposure).
      render_post_processing:  bool — True by default; False when
                               UniversalAdditionalCameraData.renderPostProcessing is
                               explicitly disabled.  Cameras with False should NOT
                               receive CameraAttributesPhysical from the global volume.
    """
    d = raw_camera
    return {
        "camera_id":               f"camera_{node_id}",
        "projection":              d.get("projection",       _DEFAULT_CAMERA["projection"]),
        "fov":                     float(_unwrap_scalar(d.get("fov",         _DEFAULT_CAMERA["fov"]))),
        "ortho_size":              float(_unwrap_scalar(d.get("ortho_size",  _DEFAULT_CAMERA["ortho_size"]))),
        "near_clip":               float(_unwrap_scalar(d.get("near_clip",   _DEFAULT_CAMERA["near_clip"]))),
        "far_clip":                float(_unwrap_scalar(d.get("far_clip",    _DEFAULT_CAMERA["far_clip"]))),
        "clear_flags":             d.get("clear_flags",       _DEFAULT_CAMERA["clear_flags"]),
        "background_color":        d.get("background_color",  _DEFAULT_CAMERA["background_color"]),
        "culling_mask":            int(d.get("culling_mask",  _DEFAULT_CAMERA["culling_mask"])),
        "is_main_camera":          bool(d.get("is_main_camera",         False)),
        "render_post_processing":  bool(d.get("render_post_processing", True)),
    }


# ---------------------------------------------------------------------------
# Lighting IR builders
# ---------------------------------------------------------------------------

# Unity intensity → physical unit scale factors per light type.
# These define the IR's physical interpretation; Godot export inverts them.
#   directional: 1.0 Unity intensity ≈ 100,000 lux (full sunlight)
#   point / spot: 1.0 Unity intensity ≈ 800 candela (a 100W tungsten bulb)
_LIGHT_INTENSITY_SCALE: Dict[str, float] = {
    "directional": 100_000.0,
    "point":       800.0,
    "spot":        800.0,
}
_LIGHT_INTENSITY_UNIT: Dict[str, str] = {
    "directional": "lux",
    "point":       "candela",
    "spot":        "lumen",
}

# Fallback light IR used when a light node has no raw data
_DEFAULT_LIGHT: Dict[str, Any] = {
    "type":                "directional",
    "color":               [1.0, 1.0, 1.0],
    "intensity":           1.0,
    "range":               None,
    "spot_angle":          None,
    "shadows_enabled":     False,
    "shadow_strength":     1.0,
    "shadow_bias":         0.05,
    "shadow_normal_bias":  0.4,
    "shadow_near_plane":   0.2,
    "shadow_resolution":   "from_quality",
    "indirect_multiplier": 1.0,
    "light_mode":          "realtime",
    "color_temperature":   None,   # float Kelvin when active, else None
    "cookie_ref":          None,   # "guid:..." or None
    "render_mode":         "auto", # "auto" | "important" | "not_important"
}


def _build_light_ir(raw_light: Dict[str, Any], node_id: str) -> Dict[str, Any]:
    """Convert a raw parser light dict to a physically-grounded engine-agnostic IR.

    Intensity is converted from Unity dimensionless units to physical approximations:
      directional → lux    (1.0 Unity ≈ 100,000 lux)
      point       → candela (1.0 Unity ≈ 800 cd)
      spot        → lumen  (1.0 Unity ≈ 800 lm)

    IR → Godot conversion inverts this: lux / 100_000 = Godot energy (so 1:1 pass-through
    for typical values, but the IR retains the physical semantics for future GI systems).
    """
    ltype = raw_light.get("type", "directional")
    scale = _LIGHT_INTENSITY_SCALE.get(ltype, 1.0)
    unit  = _LIGHT_INTENSITY_UNIT.get(ltype, "arbitrary")

    unity_intensity = float(_unwrap_scalar(raw_light.get("intensity", 1.0)))
    phys_value      = round(unity_intensity * scale, 2)

    return {
        "light_id":   f"light_{node_id}",
        "type":       ltype,
        "color":      raw_light.get("color") or [1.0, 1.0, 1.0],
        "intensity":  {"value": phys_value, "unit": unit},
        "range":      raw_light.get("range"),
        "spot_angle": raw_light.get("spot_angle"),
        "shadows": {
            "enabled":      bool(raw_light.get("shadows_enabled", False)),
            "strength":     float(_unwrap_scalar(raw_light.get("shadow_strength", 1.0))),
            "bias":         float(_unwrap_scalar(raw_light.get("shadow_bias", 0.05))),
            "normal_bias":  float(_unwrap_scalar(raw_light.get("shadow_normal_bias", 0.4))),
            "near_plane":   float(_unwrap_scalar(raw_light.get("shadow_near_plane", 0.2))),
            "resolution":   raw_light.get("shadow_resolution", "from_quality"),
            # shadow_cascades is a project-global Quality Setting in Unity, not
            # per-light; stored here only when explicitly read from light data.
            "cascades":     raw_light.get("shadow_cascades"),
        },
        "indirect_multiplier": float(_unwrap_scalar(raw_light.get("indirect_multiplier", 1.0))),
        "light_mode":          raw_light.get("light_mode", "realtime"),
        # Extended fields — preserved in IR even if not fully consumable by
        # all target generators.
        "color_temperature":   raw_light.get("color_temperature"),   # float K or None
        "cookie_ref":          raw_light.get("cookie_ref"),          # "guid:..." or None
        "render_mode":         raw_light.get("render_mode", "auto"),
    }


def _build_reflection_probe_ir(raw_probe: Dict[str, Any], node_id: str) -> Dict[str, Any]:
    """Convert raw parser reflection probe dict to an engine-agnostic IR.

    IR schema:
      probe_id      str           — unique ID for deduplication
      intensity     float         — m_IntensityMultiplier
      size          [x, y, z]     — m_BoxSize (capture volume extents)
      origin_offset [x, y, z]     — m_BoxOffset (center relative to probe position)
      box_projection bool         — parallax-correct box projection
      interior      bool          — interior mode (no sky visible inside)
      update_mode   0 | 1         — 0=ONCE (baked), 1=ALWAYS (realtime)
    """
    def _fv(key: str, default: float) -> float:
        return float(_unwrap_scalar(raw_probe.get(key, default)))

    return {
        "probe_id":      f"probe_{node_id}",
        "intensity":     _fv("intensity", 1.0),
        "size":          raw_probe.get("size")          or [10.0, 10.0, 10.0],
        "origin_offset": raw_probe.get("origin_offset") or [0.0,  0.0,  0.0],
        "box_projection": bool(raw_probe.get("box_projection", False)),
        "interior":       bool(raw_probe.get("interior", False)),
        "update_mode":    int(raw_probe.get("update_mode", 0)),
    }


def _build_nav_mesh_settings_ir(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert raw NavMeshSettings.m_BuildSettings data to engine-agnostic IR.

    Maps Unity field names to the keys consumed by godot_project_builder when
    emitting a NavigationMesh sub_resource on a NavigationRegion3D node.

    Unity defaults:  agentRadius=0.5, agentHeight=2.0, agentSlope=45, agentClimb=0.4
    Godot defaults:  agent_radius=1.0, agent_height=1.5, agent_max_slope=45, agent_max_climb=0.25
    """
    return {
        "agent_radius":    float(raw.get("agent_radius",    0.5)),
        "agent_height":    float(raw.get("agent_height",    2.0)),
        "agent_max_slope": float(raw.get("agent_slope",    45.0)),
        "agent_max_climb": float(raw.get("agent_climb",     0.4)),
        "cell_size":       float(raw.get("cell_size",    0.16667)),
        "min_region_area": float(raw.get("min_region_area", 2.0)),
    }


def _build_environment_ir(render_settings: Dict[str, Any]) -> Dict[str, Any]:
    """Convert raw render_settings from the parser to an engine-agnostic environment IR.

    IR schema (engine-agnostic):
    {
      "ambient": {
        "mode":          "flat" | "skybox" | "gradient",
        "color":         [r, g, b],          # flat/fallback color
        "sky_color":     [r, g, b],          # gradient sky hemisphere
        "equator_color": [r, g, b],          # gradient equator band
        "ground_color":  [r, g, b],          # gradient ground hemisphere
        "intensity":     float
      },
      "sky": {
        "type":         "skybox" | "procedural" | "none",
        "material_ref": "guid:..." | null
      },
      "fog": {
        "enabled": bool,
        "mode":    "linear" | "exponential" | "exponential_squared",
        "color":   [r, g, b],
        "density": float,
        "start":   float,   # linear fog start distance
        "end":     float    # linear fog end distance
      },
      "reflections": { "intensity": float }
    }
    """
    ambient_mode = render_settings.get("ambient_mode", "skybox")
    sky_guid     = render_settings.get("sky_guid")

    # Sky IR: "skybox" when a sky material GUID is present, "procedural" otherwise
    # (we can't import the actual Unity skybox texture, so we fall back to procedural)
    if sky_guid:
        sky = {"type": "skybox", "material_ref": f"guid:{sky_guid}"}
    elif ambient_mode == "skybox":
        sky = {"type": "procedural", "material_ref": None}
    else:
        sky = {"type": "none", "material_ref": None}

    return {
        "ambient": {
            "mode":          ambient_mode,
            "color":         render_settings.get("ambient_color")         or [0.2, 0.2, 0.2],
            "sky_color":     render_settings.get("ambient_sky_color")     or [0.2, 0.2, 0.2],
            "equator_color": render_settings.get("ambient_equator_color") or [0.1, 0.1, 0.1],
            "ground_color":  render_settings.get("ambient_ground_color")  or [0.05, 0.05, 0.05],
            "intensity":     float(_unwrap_scalar(render_settings.get("ambient_intensity", 1.0))),
        },
        "sky": sky,
        "fog": {
            "enabled": bool(render_settings.get("fog_enabled", False)),
            "mode":    render_settings.get("fog_mode", "exponential"),
            "color":   render_settings.get("fog_color")  or [0.5, 0.5, 0.5],
            "density": float(_unwrap_scalar(render_settings.get("fog_density", 0.01))),
            "start":   float(_unwrap_scalar(render_settings.get("fog_start", 0.0))),
            "end":     float(_unwrap_scalar(render_settings.get("fog_end", 300.0))),
        },
        "reflections": {
            "intensity": float(_unwrap_scalar(render_settings.get("reflection_intensity", 1.0))),
        },
    }


# ---------------------------------------------------------------------------
# Volume effects (URP / HDRP VolumeProfile) IR builder + validator
# ---------------------------------------------------------------------------

# Tonemapping mode integer → engine-agnostic name
_TONEMAP_MODE_MAP: Dict[int, str] = {
    0: "none",
    1: "neutral",
    2: "aces",
    3: "custom",
}

# AmbientOcclusion quality integer → engine-agnostic name
_AO_QUALITY_MAP: Dict[int, str] = {
    0: "low",
    1: "medium",
    2: "high",
    3: "ultra",
}


def _build_volume_effects_ir(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a raw VolumeProfile effects dict (from project_scanner) into
    the engine-agnostic volume-effects IR schema.

    All Unity m_* field names are already stripped by project_scanner helpers
    (_vp_val / _vp_bool / _vp_int).  This layer enforces:
      - enum integers converted to named strings
      - scalar values clamped to documented Unity ranges
      - consistent types (bool, float, int, List[float])

    Only keys present in *raw* are emitted; absent effects stay absent so that
    emitters can distinguish "not present in profile" from "active=False".

    IR schema
    ---------
    bloom:               active(bool), intensity(float>=0), threshold(float>=0),
                         scatter(float 0..1), tint(List[float] 0..1 x3)
    tonemapping:         active(bool), mode(str: none|neutral|aces|custom)
    color_adjustments:   active(bool), post_exposure(float),
                         contrast(float -100..100), saturation(float -100..100),
                         hue_shift(float -180..180), color_filter(List[float] 0..1 x4)
    exposure:            active(bool), mode(str: fixed|automatic),
                         fixed_ev100(float), min_ev100(float), max_ev100(float),
                         compensation(float),
                         adaptation_speed_dark(float>0), adaptation_speed_light(float>0)
    ambient_occlusion:   active(bool), intensity(float 0..4), radius(float>0),
                         quality(str: low|medium|high|ultra),
                         direct_lighting_strength(float 0..1)
    white_balance:       active(bool), temperature(float -100..100),
                         tint(float -100..100)
    depth_of_field:      active(bool), focus_mode(str: off|manual|auto),
                         focus_distance(float>0), aperture(float 1..32),
                         focal_length(float>0), blade_count(int 3..11)
    vignette:            active(bool), intensity(float 0..1),
                         smoothness(float 0.01..1), rounded(bool),
                         color(List[float] 0..1 x3)
    chromatic_aberration: active(bool), intensity(float 0..1)
    motion_blur:         active(bool), mode(str: camera|per_object),
                         quality(str: low|medium|high), intensity(float 0..1),
                         clamp(float 0..0.2), sample_count(int>=1)
    lens_distortion:     active(bool), intensity(float -1..1),
                         x_multiplier(float 0..1), y_multiplier(float 0..1),
                         center(List[float] x2), scale(float 0.01..5)
    film_grain:          active(bool), type(str: thin|medium|large|custom),
                         intensity(float 0..1), luminance_response(float 0..1)
    panini_projection:   active(bool), distance(float 0..1), crop_to_fit(float 0..1)
    screen_space_reflections: active(bool), algorithm(str: approximation|pbr),
                         max_ray_length(float>0), max_steps(int>=1),
                         depth_tolerance(float>0), fade_distance(float>=0)
    contact_shadows:     active(bool), length(float>0), opacity(float 0..1),
                         distance_scale(float 0..1), max_distance(float>0),
                         sample_count(int>=1)
    channel_mixer:       active(bool), red(List[float] x3), green(List[float] x3),
                         blue(List[float] x3)   [Unity range -200..200]
    split_toning:        active(bool), shadows(List[float] 0..1 x4),
                         highlights(List[float] 0..1 x4), balance(float -100..100)
    lift_gamma_gain:     active(bool), lift(List[float] 0..1 x4),
                         gamma(List[float] 0..1 x4), gain(List[float] 0..1 x4)
    shadows_midtones_highlights: active(bool), shadows/midtones/highlights(List[float] 0..1 x4),
                         shadows_start/end(float 0..1), highlights_start/end(float 0..1)
    color_curves:        active(bool), master/red/green/blue/hue_vs_sat/hue_vs_hue/
                         sat_vs_sat/lum_vs_sat: List[{time:float, value:float}]
    """
    ir: Dict[str, Any] = {}

    # ── Bloom ──────────────────────────────────────────────────────────────────
    raw_bloom = raw.get("bloom")
    if raw_bloom is not None:
        tint_raw = raw_bloom.get("tint") or [1.0, 1.0, 1.0]
        tint = [max(0.0, min(1.0, float(_unwrap_scalar(c)))) for c in (list(tint_raw) + [1.0, 1.0])[:3]]
        ir["bloom"] = {
            "active":    bool(raw_bloom.get("active", False)),
            "intensity": max(0.0, float(_unwrap_scalar(raw_bloom.get("intensity", 0.25)))),
            "threshold": max(0.0, float(_unwrap_scalar(raw_bloom.get("threshold", 0.9)))),
            "scatter":   max(0.0, min(1.0, float(_unwrap_scalar(raw_bloom.get("scatter", 0.7))))),
            "tint":      tint,
        }

    # ── Tonemapping ────────────────────────────────────────────────────────────
    raw_tm = raw.get("tonemapping")
    if raw_tm is not None:
        mode_raw = raw_tm.get("mode", 0)
        if isinstance(mode_raw, int):
            mode_str = _TONEMAP_MODE_MAP.get(mode_raw, "none")
        else:
            mode_str = str(mode_raw) if mode_raw in _TONEMAP_MODE_MAP.values() else "none"
        ir["tonemapping"] = {
            "active": bool(raw_tm.get("active", False)),
            "mode":   mode_str,
        }

    # ── Color adjustments ──────────────────────────────────────────────────────
    raw_ca = raw.get("color_adjustments")
    if raw_ca is not None:
        cf_raw = raw_ca.get("color_filter") or [1.0, 1.0, 1.0, 1.0]
        cf = [max(0.0, min(1.0, float(_unwrap_scalar(c)))) for c in (list(cf_raw) + [1.0, 1.0, 1.0, 1.0])[:4]]
        ir["color_adjustments"] = {
            "active":        bool(raw_ca.get("active", False)),
            "post_exposure": float(_unwrap_scalar(raw_ca.get("post_exposure", 0.0))),
            "contrast":      max(-100.0, min(100.0, float(_unwrap_scalar(raw_ca.get("contrast",   0.0))))),
            "saturation":    max(-100.0, min(100.0, float(_unwrap_scalar(raw_ca.get("saturation", 0.0))))),
            "hue_shift":     max(-180.0, min(180.0, float(_unwrap_scalar(raw_ca.get("hue_shift",  0.0))))),
            "color_filter":  cf,
        }

    # ── Exposure ───────────────────────────────────────────────────────────────
    raw_exp = raw.get("exposure")
    if raw_exp is not None:
        mode_exp = str(raw_exp.get("mode", "fixed"))
        if mode_exp not in ("fixed", "automatic"):
            mode_exp = "fixed"
        ir["exposure"] = {
            "active":                 bool(raw_exp.get("active", False)),
            "mode":                   mode_exp,
            "fixed_ev100":            float(_unwrap_scalar(raw_exp.get("fixed_ev100", 0.0))),
            "min_ev100":              float(_unwrap_scalar(raw_exp.get("min_ev100",   -9.0))),
            "max_ev100":              float(_unwrap_scalar(raw_exp.get("max_ev100",    9.0))),
            "compensation":           float(_unwrap_scalar(raw_exp.get("compensation", 0.0))),
            "adaptation_speed_dark":  max(0.001, float(_unwrap_scalar(raw_exp.get("adaptation_speed_dark",  3.0)))),
            "adaptation_speed_light": max(0.001, float(_unwrap_scalar(raw_exp.get("adaptation_speed_light", 1.0)))),
        }

    # ── Ambient occlusion ──────────────────────────────────────────────────────
    raw_ao = raw.get("ambient_occlusion")
    if raw_ao is not None:
        quality_raw = raw_ao.get("quality", 1)
        if isinstance(quality_raw, int):
            quality_str = _AO_QUALITY_MAP.get(quality_raw, "medium")
        else:
            quality_str = str(quality_raw) if quality_raw in _AO_QUALITY_MAP.values() else "medium"
        ir["ambient_occlusion"] = {
            "active":                   bool(raw_ao.get("active", False)),
            "intensity":                max(0.0, min(4.0, float(_unwrap_scalar(raw_ao.get("intensity", 0.0))))),
            "radius":                   max(0.001, float(_unwrap_scalar(raw_ao.get("radius", 0.5)))),
            "quality":                  quality_str,
            "direct_lighting_strength": max(0.0, min(1.0, float(_unwrap_scalar(raw_ao.get("direct_lighting_strength", 0.0))))),
        }

    # ── White balance ──────────────────────────────────────────────────────────
    raw_wb = raw.get("white_balance")
    if raw_wb is not None:
        ir["white_balance"] = {
            "active":      bool(raw_wb.get("active", False)),
            "temperature": max(-100.0, min(100.0, float(_unwrap_scalar(raw_wb.get("temperature", 0.0))))),
            "tint":        max(-100.0, min(100.0, float(_unwrap_scalar(raw_wb.get("tint",        0.0))))),
        }

    # ── Depth of field ─────────────────────────────────────────────────────────
    raw_dof = raw.get("depth_of_field")
    if raw_dof is not None:
        focus_mode = str(raw_dof.get("focus_mode", "off"))
        if focus_mode not in ("off", "manual", "auto"):
            focus_mode = "off"
        ir["depth_of_field"] = {
            "active":         bool(raw_dof.get("active", False)),
            "focus_mode":     focus_mode,
            "focus_distance": max(0.001, float(_unwrap_scalar(raw_dof.get("focus_distance", 10.0)))),
            "aperture":       max(1.0, min(32.0, float(_unwrap_scalar(raw_dof.get("aperture",    5.6))))),
            "focal_length":   max(1.0,            float(_unwrap_scalar(raw_dof.get("focal_length", 50.0)))),
            "blade_count":    max(3, min(11,       int(raw_dof.get("blade_count", 5)))),
        }

    # ── Vignette ───────────────────────────────────────────────────────────────
    raw_vig = raw.get("vignette")
    if raw_vig is not None:
        vc_raw = raw_vig.get("color") or [0.0, 0.0, 0.0]
        vc = [max(0.0, min(1.0, float(_unwrap_scalar(c)))) for c in (list(vc_raw) + [0.0, 0.0])[:3]]
        ir["vignette"] = {
            "active":     bool(raw_vig.get("active", False)),
            "intensity":  max(0.0, min(1.0, float(_unwrap_scalar(raw_vig.get("intensity",  0.0))))),
            "smoothness": max(0.01, min(1.0, float(_unwrap_scalar(raw_vig.get("smoothness", 0.2))))),
            "rounded":    bool(raw_vig.get("rounded", False)),
            "color":      vc,
        }

    # ── Chromatic aberration ───────────────────────────────────────────────────
    raw_chrom = raw.get("chromatic_aberration")
    if raw_chrom is not None:
        ir["chromatic_aberration"] = {
            "active":    bool(raw_chrom.get("active", False)),
            "intensity": max(0.0, min(1.0, float(_unwrap_scalar(raw_chrom.get("intensity", 0.0))))),
        }

    # ── Motion blur ────────────────────────────────────────────────────────────
    raw_mb = raw.get("motion_blur")
    if raw_mb is not None:
        mode_mb = str(raw_mb.get("mode", "camera"))
        if mode_mb not in ("camera", "per_object"):
            mode_mb = "camera"
        qual_mb = str(raw_mb.get("quality", "medium"))
        if qual_mb not in ("low", "medium", "high"):
            qual_mb = "medium"
        ir["motion_blur"] = {
            "active":       bool(raw_mb.get("active", False)),
            "mode":         mode_mb,
            "quality":      qual_mb,
            "intensity":    max(0.0, min(1.0,  float(_unwrap_scalar(raw_mb.get("intensity",    0.0))))),
            "clamp":        max(0.0, min(0.2,  float(_unwrap_scalar(raw_mb.get("clamp",       0.05))))),
            "sample_count": max(1,             int(raw_mb.get("sample_count",   8))),
        }

    # ── Lens distortion ────────────────────────────────────────────────────────
    raw_ld = raw.get("lens_distortion")
    if raw_ld is not None:
        ctr_raw = raw_ld.get("center") or [0.0, 0.0]
        ctr = [float(_unwrap_scalar(ctr_raw[0])) if len(ctr_raw) > 0 else 0.0,
               float(_unwrap_scalar(ctr_raw[1])) if len(ctr_raw) > 1 else 0.0]
        ir["lens_distortion"] = {
            "active":       bool(raw_ld.get("active", False)),
            "intensity":    max(-1.0, min(1.0, float(_unwrap_scalar(raw_ld.get("intensity",    0.0))))),
            "x_multiplier": max(0.0,  min(1.0, float(_unwrap_scalar(raw_ld.get("x_multiplier", 1.0))))),
            "y_multiplier": max(0.0,  min(1.0, float(_unwrap_scalar(raw_ld.get("y_multiplier", 1.0))))),
            "center":       ctr,
            "scale":        max(0.01, min(5.0, float(_unwrap_scalar(raw_ld.get("scale", 1.0))))),
        }

    # ── Film grain ─────────────────────────────────────────────────────────────
    raw_fg = raw.get("film_grain")
    if raw_fg is not None:
        fg_type = str(raw_fg.get("type", "medium"))
        if fg_type not in ("thin", "medium", "large", "custom"):
            fg_type = "medium"
        ir["film_grain"] = {
            "active":             bool(raw_fg.get("active", False)),
            "type":               fg_type,
            "intensity":          max(0.0, min(1.0, float(_unwrap_scalar(raw_fg.get("intensity",         0.0))))),
            "luminance_response": max(0.0, min(1.0, float(_unwrap_scalar(raw_fg.get("luminance_response", 0.8))))),
        }

    # ── Panini projection ──────────────────────────────────────────────────────
    raw_pp = raw.get("panini_projection")
    if raw_pp is not None:
        ir["panini_projection"] = {
            "active":      bool(raw_pp.get("active", False)),
            "distance":    max(0.0, min(1.0, float(_unwrap_scalar(raw_pp.get("distance",    0.0))))),
            "crop_to_fit": max(0.0, min(1.0, float(_unwrap_scalar(raw_pp.get("crop_to_fit", 1.0))))),
        }

    # ── Screen-space reflections ───────────────────────────────────────────────
    raw_ssr = raw.get("screen_space_reflections")
    if raw_ssr is not None:
        alg_ssr = str(raw_ssr.get("algorithm", "approximation"))
        if alg_ssr not in ("approximation", "pbr"):
            alg_ssr = "approximation"
        ir["screen_space_reflections"] = {
            "active":          bool(raw_ssr.get("active", False)),
            "algorithm":       alg_ssr,
            "max_ray_length":  max(0.001, float(_unwrap_scalar(raw_ssr.get("max_ray_length",  50.0)))),
            "max_steps":       max(1,     int(raw_ssr.get("max_steps",         32))),
            "depth_tolerance": max(0.001, float(_unwrap_scalar(raw_ssr.get("depth_tolerance",  0.1)))),
            "fade_distance":   max(0.0,   float(_unwrap_scalar(raw_ssr.get("fade_distance",    0.1)))),
        }

    # ── Contact shadows ────────────────────────────────────────────────────────
    raw_cs = raw.get("contact_shadows")
    if raw_cs is not None:
        ir["contact_shadows"] = {
            "active":         bool(raw_cs.get("active", False)),
            "length":         max(0.001, float(_unwrap_scalar(raw_cs.get("length",         0.15)))),
            "opacity":        max(0.0, min(1.0, float(_unwrap_scalar(raw_cs.get("opacity", 1.0))))),
            "distance_scale": max(0.0, min(1.0, float(_unwrap_scalar(raw_cs.get("distance_scale", 0.5))))),
            "max_distance":   max(0.001, float(_unwrap_scalar(raw_cs.get("max_distance",   50.0)))),
            "sample_count":   max(1,     int(raw_cs.get("sample_count",      8))),
        }

    # ── Channel mixer ──────────────────────────────────────────────────────────
    raw_cm = raw.get("channel_mixer")
    if raw_cm is not None:
        def _clamp_mixer(lst: Any) -> List[float]:
            v = list(lst) if isinstance(lst, list) else [0.0, 0.0, 0.0]
            return [max(-200.0, min(200.0, float(_unwrap_scalar(c)))) for c in (v + [0.0, 0.0, 0.0])[:3]]
        ir["channel_mixer"] = {
            "active": bool(raw_cm.get("active", False)),
            "red":    _clamp_mixer(raw_cm.get("red")),
            "green":  _clamp_mixer(raw_cm.get("green")),
            "blue":   _clamp_mixer(raw_cm.get("blue")),
        }

    # ── Split toning ───────────────────────────────────────────────────────────
    raw_st = raw.get("split_toning")
    if raw_st is not None:
        def _clamp_color4(lst: Any) -> List[float]:
            v = list(lst) if isinstance(lst, list) else []
            return [max(0.0, min(1.0, float(_unwrap_scalar(c)))) for c in (v + [0.0, 0.0, 0.0, 1.0])[:4]]
        ir["split_toning"] = {
            "active":     bool(raw_st.get("active", False)),
            "shadows":    _clamp_color4(raw_st.get("shadows")),
            "highlights": _clamp_color4(raw_st.get("highlights")),
            "balance":    max(-100.0, min(100.0, float(_unwrap_scalar(raw_st.get("balance", 0.0))))),
        }

    # ── Lift / Gamma / Gain (HDRP) ─────────────────────────────────────────────
    raw_lgg = raw.get("lift_gamma_gain")
    if raw_lgg is not None:
        def _clamp_c4(lst: Any) -> List[float]:
            v = list(lst) if isinstance(lst, list) else []
            return [max(0.0, min(1.0, float(_unwrap_scalar(c)))) for c in (v + [0.0, 0.0, 0.0, 1.0])[:4]]
        ir["lift_gamma_gain"] = {
            "active": bool(raw_lgg.get("active", False)),
            "lift":   _clamp_c4(raw_lgg.get("lift")),
            "gamma":  _clamp_c4(raw_lgg.get("gamma")),
            "gain":   _clamp_c4(raw_lgg.get("gain")),
        }

    # ── Shadows / Midtones / Highlights ───────────────────────────────────────
    raw_smh = raw.get("shadows_midtones_highlights")
    if raw_smh is not None:
        def _clamp_c4b(lst: Any) -> List[float]:
            v = list(lst) if isinstance(lst, list) else []
            return [max(0.0, min(1.0, float(_unwrap_scalar(c)))) for c in (v + [0.0, 0.0, 0.0, 1.0])[:4]]
        ir["shadows_midtones_highlights"] = {
            "active":           bool(raw_smh.get("active", False)),
            "shadows":          _clamp_c4b(raw_smh.get("shadows")),
            "midtones":         _clamp_c4b(raw_smh.get("midtones")),
            "highlights":       _clamp_c4b(raw_smh.get("highlights")),
            "shadows_start":    max(0.0, min(1.0, float(_unwrap_scalar(raw_smh.get("shadows_start",     0.0))))),
            "shadows_end":      max(0.0, min(1.0, float(_unwrap_scalar(raw_smh.get("shadows_end",       0.3))))),
            "highlights_start": max(0.0, min(1.0, float(_unwrap_scalar(raw_smh.get("highlights_start",  0.55))))),
            "highlights_end":   max(0.0, min(1.0, float(_unwrap_scalar(raw_smh.get("highlights_end",    1.0))))),
        }

    # ── Color curves ───────────────────────────────────────────────────────────
    raw_cc = raw.get("color_curves")
    if raw_cc is not None:
        def _clean_kps(kps: Any) -> List[Dict[str, float]]:
            if not isinstance(kps, list):
                return []
            return [
                {"time": float(_unwrap_scalar(kp.get("time", 0.0))), "value": float(_unwrap_scalar(kp.get("value", 0.0)))}
                for kp in kps if isinstance(kp, dict)
            ]
        ir["color_curves"] = {
            "active":     bool(raw_cc.get("active", False)),
            "master":     _clean_kps(raw_cc.get("master")),
            "red":        _clean_kps(raw_cc.get("red")),
            "green":      _clean_kps(raw_cc.get("green")),
            "blue":       _clean_kps(raw_cc.get("blue")),
            "hue_vs_sat": _clean_kps(raw_cc.get("hue_vs_sat")),
            "hue_vs_hue": _clean_kps(raw_cc.get("hue_vs_hue")),
            "sat_vs_sat": _clean_kps(raw_cc.get("sat_vs_sat")),
            "lum_vs_sat": _clean_kps(raw_cc.get("lum_vs_sat")),
        }

    return ir


def _validate_volume_effects_ir(ve_ir: Dict[str, Any]) -> List[str]:
    """Validate a volume_effects IR dict against the schema.

    Returns a list of violation strings; an empty list means the IR is valid.
    Each effect key is optional (absent = not present in the source profile).
    Present effects must have all required fields with correct types and ranges.
    """
    _VALID_TONEMAP  = {"none", "neutral", "aces", "custom"}
    _VALID_AO_QUAL  = {"low", "medium", "high", "ultra"}
    _VALID_FOCUS    = {"off", "manual", "auto"}
    _VALID_EXP_MODE = {"fixed", "automatic"}

    errors: List[str] = []

    def _chk_float(path: str, val: Any,
                   lo: Optional[float] = None, hi: Optional[float] = None) -> None:
        if not isinstance(val, (int, float)):
            errors.append(f"{path}: expected float, got {type(val).__name__}")
            return
        if lo is not None and float(_unwrap_scalar(val)) < lo:
            errors.append(f"{path}: {val} < min {lo}")
        if hi is not None and float(_unwrap_scalar(val)) > hi:
            errors.append(f"{path}: {val} > max {hi}")

    def _chk_bool(path: str, val: Any) -> None:
        if not isinstance(val, bool):
            errors.append(f"{path}: expected bool, got {type(val).__name__}")

    def _chk_str(path: str, val: Any, allowed: set) -> None:
        if not isinstance(val, str):
            errors.append(f"{path}: expected str, got {type(val).__name__}")
        elif val not in allowed:
            errors.append(f"{path}: '{val}' not in {sorted(allowed)}")

    def _chk_color(path: str, val: Any, n: int) -> None:
        if not isinstance(val, list) or len(val) < n:
            errors.append(f"{path}: expected list[{n}+], got {val!r}")
            return
        for i, c in enumerate(val[:n]):
            _chk_float(f"{path}[{i}]", c, 0.0, 1.0)

    if "bloom" in ve_ir:
        b = ve_ir["bloom"]
        _chk_bool( "bloom.active",    b.get("active"))
        _chk_float("bloom.intensity", b.get("intensity"), 0.0)
        _chk_float("bloom.threshold", b.get("threshold"), 0.0)
        _chk_float("bloom.scatter",   b.get("scatter"),   0.0, 1.0)
        _chk_color("bloom.tint",      b.get("tint"),      3)

    if "tonemapping" in ve_ir:
        tm = ve_ir["tonemapping"]
        _chk_bool("tonemapping.active", tm.get("active"))
        _chk_str( "tonemapping.mode",   tm.get("mode"), _VALID_TONEMAP)

    if "color_adjustments" in ve_ir:
        ca = ve_ir["color_adjustments"]
        _chk_bool( "color_adjustments.active",       ca.get("active"))
        _chk_float("color_adjustments.contrast",     ca.get("contrast"),    -100.0, 100.0)
        _chk_float("color_adjustments.saturation",   ca.get("saturation"),  -100.0, 100.0)
        _chk_float("color_adjustments.hue_shift",    ca.get("hue_shift"),   -180.0, 180.0)
        _chk_color("color_adjustments.color_filter", ca.get("color_filter"), 4)

    if "exposure" in ve_ir:
        ex = ve_ir["exposure"]
        _chk_bool( "exposure.active",                 ex.get("active"))
        _chk_str(  "exposure.mode",                   ex.get("mode"), _VALID_EXP_MODE)
        _chk_float("exposure.adaptation_speed_dark",  ex.get("adaptation_speed_dark"),  0.001)
        _chk_float("exposure.adaptation_speed_light", ex.get("adaptation_speed_light"), 0.001)

    if "ambient_occlusion" in ve_ir:
        ao = ve_ir["ambient_occlusion"]
        _chk_bool( "ambient_occlusion.active",                   ao.get("active"))
        _chk_float("ambient_occlusion.intensity",                 ao.get("intensity"), 0.0, 4.0)
        _chk_float("ambient_occlusion.radius",                    ao.get("radius"),    0.001)
        _chk_str(  "ambient_occlusion.quality",                   ao.get("quality"),   _VALID_AO_QUAL)
        _chk_float("ambient_occlusion.direct_lighting_strength",  ao.get("direct_lighting_strength"), 0.0, 1.0)

    if "white_balance" in ve_ir:
        wb = ve_ir["white_balance"]
        _chk_bool( "white_balance.active",      wb.get("active"))
        _chk_float("white_balance.temperature", wb.get("temperature"), -100.0, 100.0)
        _chk_float("white_balance.tint",        wb.get("tint"),        -100.0, 100.0)

    if "depth_of_field" in ve_ir:
        dof = ve_ir["depth_of_field"]
        _chk_bool( "depth_of_field.active",         dof.get("active"))
        _chk_str(  "depth_of_field.focus_mode",     dof.get("focus_mode"), _VALID_FOCUS)
        _chk_float("depth_of_field.focus_distance", dof.get("focus_distance"), 0.001)
        _chk_float("depth_of_field.aperture",       dof.get("aperture"),       1.0, 32.0)
        _chk_float("depth_of_field.focal_length",   dof.get("focal_length"),   1.0)
        blade = dof.get("blade_count")
        if not isinstance(blade, int):
            errors.append(f"depth_of_field.blade_count: expected int, got {type(blade).__name__}")
        elif not (3 <= blade <= 11):
            errors.append(f"depth_of_field.blade_count: {blade} not in 3..11")

    if "vignette" in ve_ir:
        vig = ve_ir["vignette"]
        _chk_bool( "vignette.active",     vig.get("active"))
        _chk_float("vignette.intensity",  vig.get("intensity"),  0.0, 1.0)
        _chk_float("vignette.smoothness", vig.get("smoothness"), 0.01, 1.0)
        _chk_bool( "vignette.rounded",    vig.get("rounded"))
        _chk_color("vignette.color",      vig.get("color"),      3)

    if "chromatic_aberration" in ve_ir:
        ch = ve_ir["chromatic_aberration"]
        _chk_bool( "chromatic_aberration.active",    ch.get("active"))
        _chk_float("chromatic_aberration.intensity", ch.get("intensity"), 0.0, 1.0)

    _VALID_MB_MODE  = {"camera", "per_object"}
    _VALID_MB_QUAL  = {"low", "medium", "high"}
    _VALID_FG_TYPE  = {"thin", "medium", "large", "custom"}
    _VALID_SSR_ALG  = {"approximation", "pbr"}

    if "motion_blur" in ve_ir:
        mb = ve_ir["motion_blur"]
        _chk_bool( "motion_blur.active",       mb.get("active"))
        _chk_str(  "motion_blur.mode",         mb.get("mode"),    _VALID_MB_MODE)
        _chk_str(  "motion_blur.quality",      mb.get("quality"), _VALID_MB_QUAL)
        _chk_float("motion_blur.intensity",    mb.get("intensity"),    0.0, 1.0)
        _chk_float("motion_blur.clamp",        mb.get("clamp"),        0.0, 0.2)
        sc = mb.get("sample_count")
        if not isinstance(sc, int) or sc < 1:
            errors.append(f"motion_blur.sample_count: expected int>=1, got {sc!r}")

    if "lens_distortion" in ve_ir:
        ld = ve_ir["lens_distortion"]
        _chk_bool( "lens_distortion.active",       ld.get("active"))
        _chk_float("lens_distortion.intensity",    ld.get("intensity"),    -1.0, 1.0)
        _chk_float("lens_distortion.x_multiplier", ld.get("x_multiplier"),  0.0, 1.0)
        _chk_float("lens_distortion.y_multiplier", ld.get("y_multiplier"),  0.0, 1.0)
        _chk_float("lens_distortion.scale",        ld.get("scale"),         0.01, 5.0)

    if "film_grain" in ve_ir:
        fg = ve_ir["film_grain"]
        _chk_bool( "film_grain.active",             fg.get("active"))
        _chk_str(  "film_grain.type",               fg.get("type"), _VALID_FG_TYPE)
        _chk_float("film_grain.intensity",          fg.get("intensity"),         0.0, 1.0)
        _chk_float("film_grain.luminance_response", fg.get("luminance_response"), 0.0, 1.0)

    if "panini_projection" in ve_ir:
        pp = ve_ir["panini_projection"]
        _chk_bool( "panini_projection.active",      pp.get("active"))
        _chk_float("panini_projection.distance",    pp.get("distance"),    0.0, 1.0)
        _chk_float("panini_projection.crop_to_fit", pp.get("crop_to_fit"), 0.0, 1.0)

    if "screen_space_reflections" in ve_ir:
        ssr = ve_ir["screen_space_reflections"]
        _chk_bool( "screen_space_reflections.active",          ssr.get("active"))
        _chk_str(  "screen_space_reflections.algorithm",       ssr.get("algorithm"), _VALID_SSR_ALG)
        _chk_float("screen_space_reflections.max_ray_length",  ssr.get("max_ray_length"),  0.001)
        _chk_float("screen_space_reflections.depth_tolerance", ssr.get("depth_tolerance"), 0.001)
        _chk_float("screen_space_reflections.fade_distance",   ssr.get("fade_distance"),   0.0)
        ms = ssr.get("max_steps")
        if not isinstance(ms, int) or ms < 1:
            errors.append(f"screen_space_reflections.max_steps: expected int>=1, got {ms!r}")

    if "contact_shadows" in ve_ir:
        cs = ve_ir["contact_shadows"]
        _chk_bool( "contact_shadows.active",         cs.get("active"))
        _chk_float("contact_shadows.length",         cs.get("length"),         0.001)
        _chk_float("contact_shadows.opacity",        cs.get("opacity"),        0.0, 1.0)
        _chk_float("contact_shadows.distance_scale", cs.get("distance_scale"), 0.0, 1.0)
        _chk_float("contact_shadows.max_distance",   cs.get("max_distance"),   0.001)

    if "channel_mixer" in ve_ir:
        cm = ve_ir["channel_mixer"]
        _chk_bool("channel_mixer.active", cm.get("active"))
        for ch_name in ("red", "green", "blue"):
            v = cm.get(ch_name)
            if not isinstance(v, list) or len(v) < 3:
                errors.append(f"channel_mixer.{ch_name}: expected list[3+], got {v!r}")
            else:
                for i, c in enumerate(v[:3]):
                    _chk_float(f"channel_mixer.{ch_name}[{i}]", c, -200.0, 200.0)

    if "split_toning" in ve_ir:
        st = ve_ir["split_toning"]
        _chk_bool( "split_toning.active",    st.get("active"))
        _chk_color("split_toning.shadows",    st.get("shadows"),    4)
        _chk_color("split_toning.highlights", st.get("highlights"), 4)
        _chk_float("split_toning.balance",    st.get("balance"), -100.0, 100.0)

    if "lift_gamma_gain" in ve_ir:
        lgg = ve_ir["lift_gamma_gain"]
        _chk_bool( "lift_gamma_gain.active", lgg.get("active"))
        _chk_color("lift_gamma_gain.lift",   lgg.get("lift"),  4)
        _chk_color("lift_gamma_gain.gamma",  lgg.get("gamma"), 4)
        _chk_color("lift_gamma_gain.gain",   lgg.get("gain"),  4)

    if "shadows_midtones_highlights" in ve_ir:
        smh = ve_ir["shadows_midtones_highlights"]
        _chk_bool( "shadows_midtones_highlights.active",     smh.get("active"))
        _chk_color("shadows_midtones_highlights.shadows",    smh.get("shadows"),    4)
        _chk_color("shadows_midtones_highlights.midtones",   smh.get("midtones"),   4)
        _chk_color("shadows_midtones_highlights.highlights", smh.get("highlights"), 4)
        for fld in ("shadows_start", "shadows_end", "highlights_start", "highlights_end"):
            _chk_float(f"shadows_midtones_highlights.{fld}", smh.get(fld), 0.0, 1.0)

    if "color_curves" in ve_ir:
        cc = ve_ir["color_curves"]
        _chk_bool("color_curves.active", cc.get("active"))
        for curve_name in ("master", "red", "green", "blue",
                           "hue_vs_sat", "hue_vs_hue", "sat_vs_sat", "lum_vs_sat"):
            kps = cc.get(curve_name)
            if not isinstance(kps, list):
                errors.append(f"color_curves.{curve_name}: expected list, got {type(kps).__name__}")
            else:
                for i, kp in enumerate(kps):
                    if not isinstance(kp, dict) or "time" not in kp or "value" not in kp:
                        errors.append(
                            f"color_curves.{curve_name}[{i}]: expected {{time,value}} dict"
                        )

    return errors


def _build_collision_component(layer_index: int) -> Dict[str, Any]:
    """Build the collision layer/mask component from a Unity layer index (0-31).

    ``layer`` is a single-bit bitmask for the object's own layer (bit N = layer N+1).
    ``mask``  defaults to 0xFFFFFFFF (collides with all layers).  The real Unity
    collision matrix lives in ProjectSettings/DynamicPhysics.asset which is not
    available at scene-parse time; using all-layers is far safer than the previous
    "same-layer only" default which caused most objects to never collide.

    ``source_layer_index`` preserves the original source-engine layer index (0-31)
    for post-import tooling that can re-apply the correct collision matrix once the
    project's layer configuration is available.
    """
    idx = int(layer_index) % 32
    bit = 1 << idx
    return {
        "layer":              bit,
        "mask":               0xFFFFFFFF,
        "source_layer_index": idx,
    }


def _parser_type_to_ir(parser_type: str) -> str:
    """Maps the parser's Godot-type string to an engine-agnostic IR type.

    The parser already resolved Unity components → Godot type names.
    We translate those to engine-agnostic IR types here.
    """
    mapping: Dict[str, str] = {
        # entity-like (physics bodies and visual meshes all map to entity;
        # the actual Godot node type is derived from components at emit time)
        "MeshInstance3D":    _IR_TYPE_ENTITY,
        "StaticBody3D":      _IR_TYPE_ENTITY,
        "RigidBody3D":       _IR_TYPE_ENTITY,
        "CharacterBody3D":   _IR_TYPE_ENTITY,
        # cameras
        "Camera3D":          _IR_TYPE_CAMERA,
        # lights
        "DirectionalLight3D": _IR_TYPE_LIGHT,
        "OmniLight3D":        _IR_TYPE_LIGHT,
        "SpotLight3D":        _IR_TYPE_LIGHT,
        "ReflectionProbe":    _IR_TYPE_LIGHT,
        # audio
        "AudioStreamPlayer3D": _IR_TYPE_AUDIO_SOURCE,
        # UI — canvas
        "CanvasLayer":       _IR_TYPE_UI_CANVAS,
        # UI — generic / element
        "Control":           _IR_TYPE_UI_ELEMENT,
        "Button":            _IR_TYPE_UI_ELEMENT,
        "Label":             _IR_TYPE_UI_ELEMENT,
        "TextureRect":       _IR_TYPE_UI_ELEMENT,
        "LineEdit":          _IR_TYPE_UI_ELEMENT,
        "HSlider":           _IR_TYPE_UI_ELEMENT,
        "VSlider":           _IR_TYPE_UI_ELEMENT,
        "CheckBox":          _IR_TYPE_UI_ELEMENT,
        "ScrollContainer":   _IR_TYPE_UI_ELEMENT,
        "HScrollBar":        _IR_TYPE_UI_ELEMENT,
        "VScrollBar":        _IR_TYPE_UI_ELEMENT,
        # UI — containers (layout groups)
        "HBoxContainer":     _IR_TYPE_UI_ELEMENT,
        "VBoxContainer":     _IR_TYPE_UI_ELEMENT,
        "GridContainer":     _IR_TYPE_UI_ELEMENT,
        # particles
        "GPUParticles3D":    _IR_TYPE_PARTICLES,
        # sprites
        "Sprite2D":          _IR_TYPE_SPRITE,
        # generic group
        "Node3D":            _IR_TYPE_GROUP,
        # WorldEnvironment (URP Volume → Godot WorldEnvironment)
        "WorldEnvironment":  _IR_TYPE_WORLD_ENV,
        # Terrain / environment objects
        "Terrain":           _IR_TYPE_TERRAIN,
        "WindZone":          _IR_TYPE_WIND_ZONE,
        "TextMesh":          _IR_TYPE_TEXT_3D,
        "Tree":              _IR_TYPE_TREE,
        # Navigation (baked NavMesh region node)
        "NavigationRegion3D": _IR_TYPE_NAVIGATION,
    }
    return mapping.get(parser_type, _IR_TYPE_GROUP)


def _is_spatial(ir_type: str) -> bool:
    """Non-spatial nodes (UI canvas, pure groups with no transform) don't
    require a transform component later. For now everything is spatial
    except UI canvases."""
    return ir_type != _IR_TYPE_UI_CANVAS


# ---------------------------------------------------------------------------
# Mesh + material IR builders
# ---------------------------------------------------------------------------

# Unity built-in primitive fileID → engine-agnostic mesh type label.
# The fileID is encoded in the mesh_ref string as "builtin:<fileID>".
_BUILTIN_MESH_TYPES: Dict[int, str] = {
    10202: "BoxMesh",
    10206: "CylinderMesh",
    10207: "SphereMesh",
    10208: "CapsuleMesh",
    10209: "PlaneMesh",
    10210: "QuadMesh",
}

_DEFAULT_PBR: Dict[str, Any] = {
    "albedo_color":          [1.0, 1.0, 1.0, 1.0],
    "albedo_texture":        None,
    "metallic":              0.0,
    "roughness":             0.5,
    "emission":              [0.0, 0.0, 0.0],
    "emission_enabled":      False,
    "normal_texture":        None,
    "normal_scale":          1.0,
    "occlusion_texture":     None,
    "occlusion_strength":    1.0,
    "detail_albedo_texture": None,
    "detail_normal_texture": None,
    "detail_scale":          1.0,
    "blend_mode":            "opaque",   # "opaque" | "cutout" | "transparent"
    "alpha_cutoff":          0.5,
    "double_sided":          False,
    "shader_type":           "lit",      # "lit" | "unlit"
}


def _build_mesh_ir(
    mesh_ref: str,
    guid_map: Optional[Dict[str, Path]] = None,
    material_count: int = 1,
) -> Optional[Dict[str, Any]]:
    """Convert a raw mesh_ref string into a structured, engine-agnostic mesh IR.

    Supported ref formats:
      ``builtin:<fileID>``  — Unity built-in primitive (Box, Sphere, …)
      ``guid:<guid>``       — External asset (.fbx, .asset, etc.)
      ``fileID:<id>``       — Internal scene reference (rare)

    ``material_count`` determines the submesh list length.  Unity uses one submesh
    per material slot, so a mesh with 3 materials has submesh indices 0, 1, 2.
    Built-in primitives always have exactly one submesh regardless of material_count.

    Returns None when *mesh_ref* is empty or cannot be interpreted.
    """
    if not mesh_ref:
        return None

    # One submesh per material slot; clamp to at least 1.
    n_slots   = max(1, int(material_count))
    submeshes = [{"index": i, "material_slot": i} for i in range(n_slots)]

    if mesh_ref.startswith("builtin:"):
        try:
            fid = int(mesh_ref.split(":", 1)[1])
        except (ValueError, IndexError):
            return None
        builtin_type = _BUILTIN_MESH_TYPES.get(fid, "BoxMesh")
        return {
            "mesh_id": f"builtin_{builtin_type}",
            "source": {
                "type":         "builtin",
                "builtin_type": builtin_type,
            },
            # Built-in primitives always have a single submesh.
            "submeshes": [{"index": 0, "material_slot": 0}],
        }

    if mesh_ref.startswith("guid:"):
        guid = mesh_ref.split(":", 1)[1]
        path = ""
        if guid_map and guid in guid_map:
            path = guid_map[guid].as_posix()
        return {
            "mesh_id": f"mesh_{guid[:12]}",
            "source": {
                "type": "external",
                "guid": guid,
                "path": path,
            },
            "submeshes": submeshes,
        }

    if mesh_ref.startswith("fileID:"):
        return {
            "mesh_id": f"mesh_{mesh_ref.replace(':', '_')}",
            "source": {"type": "internal", "ref": mesh_ref},
            "submeshes": submeshes,
        }

    return None


def _build_material_ir(
    mat_ref: str,
    material_ir_map: Optional[Dict[str, Any]] = None,
    slot_index: int = 0,
) -> Dict[str, Any]:
    """Build a structured, engine-agnostic material IR for one material slot.

    Args:
        mat_ref:          Raw ref string like ``"guid:abc123…"``
        material_ir_map:  guid → PBR property dict from project scanner
        slot_index:       Slot position (used for fallback ID generation)

    The PBR dict uses engine-agnostic keys:
      albedo_color, albedo_texture, metallic, roughness, emission, emission_enabled
    """
    guid = mat_ref.split(":", 1)[1] if mat_ref.startswith("guid:") else ""
    mat_id = f"mat_{guid[:12]}" if guid else f"mat_slot_{slot_index}"

    if guid and material_ir_map and guid in material_ir_map:
        pbr = dict(material_ir_map[guid])
    else:
        pbr = dict(_DEFAULT_PBR)

    return {
        "material_id": mat_id,
        "source_guid": guid,
        "pbr":         pbr,
    }


# ---------------------------------------------------------------------------
# Animation component IR builder
# ---------------------------------------------------------------------------

def _build_animation_component(
    ctrl_guid: Optional[str],
    animation_ir_map: Optional[Dict[str, Any]],
    animator_controller_map: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Build an engine-agnostic animation component IR for an Animator node.

    Looks up the AnimatorController by its GUID, collects all referenced
    AnimationClip data, and returns a structured dict ready for the Godot emitter.

    Returns None when there is no controller GUID or no animation data is found.

    Schema:
      controller_guid  str  — raw GUID of the .controller asset
      controller       dict — parsed controller (name, parameters, layers) or None
      clips            dict — motion_guid → clip_dict  (all referenced clips)
    """
    if not ctrl_guid:
        return None

    controller: Optional[Dict[str, Any]] = None
    if animator_controller_map and ctrl_guid in animator_controller_map:
        controller = animator_controller_map[ctrl_guid]

    # Collect all motion GUIDs referenced in the controller
    clips: Dict[str, Any] = {}
    if controller and animation_ir_map:
        for layer in (controller.get("layers") or []):
            for state in (layer.get("states") or []):
                _collect_state_clips(state, animation_ir_map, clips)

    if not controller and not clips:
        return None

    return {
        "controller_guid": ctrl_guid,
        "controller":      controller,
        "clips":           clips,
    }


def _collect_state_clips(
    state: Dict[str, Any],
    animation_ir_map: Dict[str, Any],
    clips: Dict[str, Any],
    _depth: int = 0,
) -> None:
    """Collect clip data for all motion GUIDs referenced in a state or blend tree.

    Fix 5: Unity BlendTrees can nest arbitrarily deep — each motion inside a
    BlendTree can itself carry a ``blend_tree`` key pointing to another level.
    We recurse until all clips are collected or the depth guard (32) fires to
    prevent infinite loops from malformed assets.
    """
    if _depth >= 32:
        return

    motion_guid_ref = state.get("motion_guid")  # "guid:abc..." or None
    if motion_guid_ref and motion_guid_ref.startswith("guid:"):
        raw_guid = motion_guid_ref[5:]
        if raw_guid not in clips and raw_guid in animation_ir_map:
            clips[raw_guid] = animation_ir_map[raw_guid]

    bt = state.get("blend_tree")
    if bt:
        for motion in (bt.get("motions") or []):
            mg = motion.get("motion_guid")
            if mg and mg.startswith("guid:"):
                raw_guid = mg[5:]
                if raw_guid not in clips and raw_guid in animation_ir_map:
                    clips[raw_guid] = animation_ir_map[raw_guid]
            # Recurse into nested blend trees.
            nested_bt = motion.get("blend_tree")
            if nested_bt:
                _collect_state_clips(
                    {"blend_tree": nested_bt}, animation_ir_map, clips, _depth + 1
                )


# ---------------------------------------------------------------------------
# Audio + Particle IR builders
# ---------------------------------------------------------------------------

def _build_audio_source_ir(
    raw_audio: Optional[Dict[str, Any]],
    node_id: str,
) -> Optional[Dict[str, Any]]:
    """Build engine-agnostic audio_source IR from _extract_audio_source_data output.

    IR schema (all fields match _extract_audio_source_data keys plus audio_id):
      audio_id          str   — unique ID for this source
      audio_clip_ref    str|None — "guid:..." of the attached AudioClip asset
      volume            float  — linear [0, 1]
      pitch             float  — multiplier (1.0 = unchanged)
      loop              bool
      play_on_awake     bool
      spatial_blend     float  — 0.0=2D  1.0=3D
      rolloff_mode      str    — "logarithmic" | "linear" | "custom"
      min_distance      float  — full-volume inner radius
      max_distance      float  — silence outer radius
      doppler_level     float  — Unity doppler scale (0 = disabled)
      bypass_effects    bool
      bypass_reverb     bool
      priority          int    — 0 (highest) … 256 (lowest)
      stereo_pan        float  — -1 … +1
      reverb_zone_mix   float  — reverb wet level [0, 1.1]
    """
    if not raw_audio or not isinstance(raw_audio, dict):
        return None
    result = dict(raw_audio)
    result["audio_id"] = f"audio_{node_id}"
    return result


def _build_particle_system_ir(
    raw_ps: Optional[Dict[str, Any]],
    node_id: str,
) -> Optional[Dict[str, Any]]:
    """Build engine-agnostic particle_system IR from _extract_particle_system_data output.

    Passes all parsed module dicts through verbatim and adds a ``ps_id``.
    All curve / gradient data is preserved without approximation.

    IR schema (top-level keys):
      ps_id                str   — unique ID for SubResource deduplication
      main                 dict  — duration, looping, prewarm, play_on_awake,
                                    start_lifetime, start_speed, start_size,
                                    start_color, gravity_modifier, max_particles,
                                    simulation_space
      emission             dict  — enabled, rate_over_time, rate_over_distance,
                                    bursts: [{time, min_count, max_count,
                                              cycle_count, repeat_interval, probability}]
      shape                dict  — enabled, type, radius, angle, length, arc, scale
      velocity             dict  — enabled, space,
                                    x_curve / y_curve / z_curve: full MinMaxCurve
                                    (mode, scalar, min_scalar, curve, min_curve),
                                    x / y / z: backward-compat scalar summary
      color_over_lifetime  dict  — enabled,
                                    gradient: {mode, color, gradient.stops, min_gradient}
      size_over_lifetime   dict  — enabled, separate_axes,
                                    curve / x_curve / y_curve / z_curve: MinMaxCurve,
                                    size_start / size_end: scalar endpoints
      noise                dict  — enabled, strength, frequency, octaves, scroll_speed,
                                    damping, quality
      renderer             dict  — enabled, material_ref, mesh_ref, render_mode,
                                    render_alignment, sort_mode, sort_fudge,
                                    min_particle_size, max_particle_size,
                                    velocity_scale, length_scale
      texture_sheet        dict  — enabled, mode, tiles_x, tiles_y, animation,
                                    row_mode, row_index, cycle_count, flip_u, flip_v,
                                    frame_over_time / start_frame: MinMaxCurve
    """
    if not raw_ps or not isinstance(raw_ps, dict):
        return None
    result = dict(raw_ps)
    result["ps_id"] = f"ps_{node_id}"
    return result


# ---------------------------------------------------------------------------
# UI IR builders
# ---------------------------------------------------------------------------

def _build_rect_transform_ir(raw_rt: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Convert raw RectTransform data to engine-agnostic layout IR.

    Unity RectTransform coordinate conventions:
      - Y axis: 0 = bottom of parent, 1 = top.  Godot Y: 0 = top, 1 = bottom.
      - anchoredPosition is the offset of the node's *pivot point* from the
        *anchor reference point* (the centre of the anchor rect in parent space).
      - sizeDelta: when anchor_min == anchor_max this is the full rect size;
        when stretched it is the edge inset (positive = shrink from edge).

    Stored as-is so the Godot emitter can apply the correct formulas at emit time
    when it knows the Godot anchor conventions.

    Schema:
      anchor_min          [x, y]  — normalized (0-1) lower-left anchor
      anchor_max          [x, y]  — normalized (0-1) upper-right anchor
      pivot               [x, y]  — normalized pivot within the rect (Unity Y-up)
      anchored_position   [x, y]  — pivot offset from anchor reference point
      size_delta          [w, h]  — rect size (point anchor) or inset (stretched)
    """
    if not raw_rt or not isinstance(raw_rt, dict):
        return None
    return {
        "anchor_min":        list(raw_rt.get("anchor_min",        [0.0, 0.0])),
        "anchor_max":        list(raw_rt.get("anchor_max",        [1.0, 1.0])),
        "pivot":             list(raw_rt.get("pivot",             [0.5, 0.5])),
        "anchored_position": list(raw_rt.get("anchored_position", [0.0, 0.0])),
        "size_delta":        list(raw_rt.get("size_delta",        [0.0, 0.0])),
    }


def _build_ui_element_ir(
    raw_ui: Optional[Dict[str, Any]],
    node_id: str,
) -> Optional[Dict[str, Any]]:
    """Build engine-agnostic UI element IR from a raw parser ui_element dict.

    Passes all parsed fields through verbatim so the emitter has full data.
    Only adds an ``element_id`` for sub-resource deduplication.

    Schema subset (full set matches _extract_ui_element_data output):
      element_id     str
      element_kind   str    — engine-agnostic type ("label"|"button"|"text_input"|
                              "slider_h"|"slider_v"|"scrollbar_h"|"scrollbar_v"|
                              "toggle"|"scroll_view"|"image"|"widget")
      text           str
      font_size      int
      font_ref       str|None
      color          [r,g,b,a]
      interactable   bool
      h_align        str
      v_align        str
      wrap_mode      str
      overflow       str
      image_ref      str|None
      image_type     str
      raycast_target bool
    """
    if not raw_ui or not isinstance(raw_ui, dict):
        return None
    result = dict(raw_ui)
    result["element_id"] = f"ui_{node_id}"
    return result


def _build_ui_layout_ir(
    raw_layout: Optional[Dict[str, Any]],
    node_id: str,
) -> Optional[Dict[str, Any]]:
    """Build engine-agnostic layout group IR from a raw parser ui_layout dict.

    Schema subset (full set matches _extract_layout_group_data output):
      layout_id              str
      kind                   "horizontal" | "vertical" | "grid"
      padding                [left, right, top, bottom]
      spacing                float
      child_alignment        int
      child_force_expand_width  bool
      child_force_expand_height bool
      cell_size              [w, h]  (grid)
      cell_spacing           [x, y]  (grid)
      constraint             "flexible" | "fixed_column" | "fixed_row"
      constraint_count       int     (grid)
    """
    if not raw_layout or not isinstance(raw_layout, dict):
        return None
    result = dict(raw_layout)
    result["layout_id"] = f"layout_{node_id}"
    return result


# ---------------------------------------------------------------------------
# Navigation IR builders
# ---------------------------------------------------------------------------

def _build_nav_agent_ir(raw: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Build engine-agnostic nav_agent IR component from parsed NavMeshAgent data.

    Schema:
      agent_type_id               int   — index into project NavMesh agent types
      base_offset                 float — vertical offset from pivot
      speed                       float — max speed (m/s)
      angular_speed               float — max rotation speed (deg/s)
      acceleration                float — max acceleration (m/s²)
      stopping_distance           float
      auto_braking                bool
      radius                      float — agent cylinder radius
      height                      float — agent cylinder height
      avoidance_quality           str   — "none"|"low"|"medium"|"high"|"very_high"
      avoidance_priority          int   — 0 (highest) – 99 (lowest)
      auto_traverse_off_mesh_link bool
      auto_repath                 bool
      area_mask                   int   — walkable area bitmask
    """
    if not raw or not isinstance(raw, dict):
        return None
    return {
        "agent_type_id":               int(raw.get("agent_type_id", -1)),
        "base_offset":                 float(_unwrap_scalar(raw.get("base_offset", 0.0))),
        "speed":                       float(_unwrap_scalar(raw.get("speed", 3.5))),
        "angular_speed":               float(_unwrap_scalar(raw.get("angular_speed", 120.0))),
        "acceleration":                float(_unwrap_scalar(raw.get("acceleration", 8.0))),
        "stopping_distance":           float(_unwrap_scalar(raw.get("stopping_distance", 0.0))),
        "auto_braking":                bool(raw.get("auto_braking", True)),
        "radius":                      float(_unwrap_scalar(raw.get("radius", 0.5))),
        "height":                      float(_unwrap_scalar(raw.get("height", 2.0))),
        "avoidance_quality":           str(raw.get("avoidance_quality", "very_high")),
        "avoidance_priority":          int(raw.get("avoidance_priority", 50)),
        "auto_traverse_off_mesh_link": bool(raw.get("auto_traverse_off_mesh_link", True)),
        "auto_repath":                 bool(raw.get("auto_repath", True)),
        "area_mask":                   int(raw.get("area_mask", -1)),
    }


def _build_nav_obstacle_ir(raw: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Build engine-agnostic nav_obstacle IR component from parsed NavMeshObstacle data.

    Schema:
      shape                  str   — "capsule" | "box"
      center                 [x,y,z]
      size                   [x,y,z]  — box half-extents
      radius                 float    — capsule radius
      height                 float    — capsule height
      carve                  bool     — carve a hole in the navmesh
      move_threshold         float
      time_to_stationary     float
      carve_only_stationary  bool
    """
    if not raw or not isinstance(raw, dict):
        return None
    center = raw.get("center") or [0.0, 0.0, 0.0]
    size   = raw.get("size")   or [1.0, 1.0, 1.0]
    return {
        "shape":                 str(raw.get("shape", "capsule")),
        "center":                [float(_unwrap_scalar(v)) for v in (center if len(center) >= 3 else [0, 0, 0])],
        "size":                  [float(_unwrap_scalar(v)) for v in (size   if len(size)   >= 3 else [1, 1, 1])],
        "radius":                float(_unwrap_scalar(raw.get("radius", 0.5))),
        "height":                float(_unwrap_scalar(raw.get("height", 2.0))),
        "carve":                 bool(raw.get("carve", False)),
        "move_threshold":        float(_unwrap_scalar(raw.get("move_threshold", 0.1))),
        "time_to_stationary":    float(_unwrap_scalar(raw.get("time_to_stationary", 0.5))),
        "carve_only_stationary": bool(raw.get("carve_only_stationary", True)),
    }


def _build_script_components_ir(
    raw_list: Optional[List[Dict[str, Any]]],
    node_id: str,
) -> Optional[List[Dict[str, Any]]]:
    """Build a list of engine-agnostic script_component IR entries.

    Each entry schema:
      component_id      str  — unique ID for sub-resource tracking
      script_guid       str  — GUID of the source C# script asset
      script_file_id    int  — fileID within the script assembly
      serialized_fields dict — all user-authored serialized fields verbatim
      object_refs       list — [{field, guid, file_id}] object pointer fields
      event_connections list — [{field, calls: [{target_guid, target_file_id,
                                 method_name, call_state}]}]
    """
    if not raw_list:
        return None
    result = []
    for idx, raw in enumerate(raw_list):
        if not isinstance(raw, dict):
            continue
        entry = dict(raw)
        entry["component_id"] = f"script_{node_id}_{idx}"
        result.append(entry)
    return result or None


# ---------------------------------------------------------------------------
# Core recursive node builder
# ---------------------------------------------------------------------------

def _build_ir_node(
    raw_node: Dict[str, Any],
    scene_prefix: str,
    counter: List[int],
    guid_map: Optional[Dict[str, Path]] = None,
    material_ir_map: Optional[Dict[str, Any]] = None,
    volume_profile_map: Optional[Dict[str, Any]] = None,
    physics_material_ir_map: Optional[Dict[str, Any]] = None,
    animation_ir_map: Optional[Dict[str, Any]] = None,
    animator_controller_map: Optional[Dict[str, Any]] = None,
    terrain_ir_map: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Converts one raw parser node (and its children) into an IR node dict."""

    node_id   = _build_node_id(scene_prefix, counter)
    node_name = str(raw_node.get("name") or "Node")
    raw_type  = str(raw_node.get("type") or "Node3D")

    ir_type = _parser_type_to_ir(raw_type)
    spatial = _is_spatial(ir_type)

    t = raw_node.get("transform") or {}

    # Determine meta quality: types we fully understand get confidence 1.0;
    # anything that fell back to "Node3D" because we didn't recognise it
    # gets partial status with a note.
    unity_data = raw_node.get("unity") or {}
    unity_flags = unity_data.get("flags") or {}

    if raw_type == "Node3D" and unity_flags:
        # It had source flags but we couldn't identify a better type — partial.
        meta = _make_meta(
            status="partial",
            confidence=0.6,
            notes="Source type not fully mapped; fell back to group node.",
        )
    elif ir_type == _IR_TYPE_GROUP and raw_type != "Node3D":
        # Parser returned a type string that isn't in the IR type mapping.
        meta = _make_meta(
            status="partial",
            confidence=0.4,
            notes=f"Parser type '{raw_type}' is not in the IR type mapping; node fell back to Node3D.",
        )
        if warnings is not None:
            warnings.append(
                f"Node '{node_name}': unrecognized parser type '{raw_type}' fell back to Node3D"
            )
    else:
        meta = _make_meta()

    # Propagate parser-level warnings (from unity_parser) into IR meta so they
    # survive through to the final report and are traceable via meta.warnings.
    # Also forward them to the pipeline warnings list so they appear in the
    # final user-visible output — without this they were buried in IR metadata.
    parser_warnings = raw_node.get("warnings") or []
    if parser_warnings:
        meta["warnings"].extend(parser_warnings)
        meta["requires_review"] = True
        if warnings is not None:
            warnings.extend(parser_warnings)

    # ── Engine-agnostic components dict ──────────────────────────────────
    # Target generators must read from ``components``, not from
    # ``original_data``.  ``original_data`` exists only for debugging and
    # round-trip traceability.
    components: Dict[str, Any] = {}

    raw_mesh_ref      = raw_node.get("mesh") or ""
    raw_material_refs = raw_node.get("materials") or []

    mesh_ir = _build_mesh_ir(raw_mesh_ref, guid_map, material_count=len(raw_material_refs) or 1)
    if mesh_ir:
        components["mesh"] = mesh_ir

    if raw_material_refs:
        components["materials"] = [
            _build_material_ir(m, material_ir_map, i)
            for i, m in enumerate(raw_material_refs)
        ]

    # ── Camera ───────────────────────────────────────────────────────────────
    if ir_type == _IR_TYPE_CAMERA:
        raw_camera = raw_node.get("camera")
        if raw_camera and isinstance(raw_camera, dict):
            components["camera"] = _build_camera_ir(raw_camera, node_id)
        else:
            components["camera"] = _build_camera_ir(_DEFAULT_CAMERA, node_id)

    if ir_type == _IR_TYPE_LIGHT:
        if raw_type == "ReflectionProbe":
            # ReflectionProbes carry their own component dict; they are not lights.
            raw_probe = raw_node.get("reflection_probe")
            components["reflection_probe"] = _build_reflection_probe_ir(
                raw_probe if isinstance(raw_probe, dict) else {}, node_id
            )
        else:
            raw_light = raw_node.get("light")
            if raw_light and isinstance(raw_light, dict):
                components["light"] = _build_light_ir(raw_light, node_id)
            else:
                # Fallback: derive type from the parser's Godot type string
                fallback_type = _LIGHT_SUBTYPE.get(raw_type, "directional")
                components["light"] = _build_light_ir(
                    {**_DEFAULT_LIGHT, "type": fallback_type}, node_id
                )

    # ── Physics components ────────────────────────────────────────────────
    raw_colliders = raw_node.get("colliders") or []
    if raw_colliders:
        components["colliders"] = [
            _build_collider_ir(c, i, node_id, pm_map=physics_material_ir_map, warnings=warnings)
            for i, c in enumerate(raw_colliders)
        ]

    raw_rb = raw_node.get("rigidbody")
    if raw_rb and isinstance(raw_rb, dict):
        components["rigidbody"] = _build_rigidbody_ir(raw_rb, node_id)

    layer_index = int(raw_node.get("layer") or 0)
    if raw_colliders or raw_rb:
        components["collision"] = _build_collision_component(layer_index)

    # ── WorldEnvironment: attach post-processing effects from VolumeProfile ──
    if ir_type == _IR_TYPE_WORLD_ENV and volume_profile_map:
        profile_guid = raw_node.get("volume_profile_guid", "")
        if profile_guid and profile_guid in volume_profile_map:
            components["volume_effects"] = _build_volume_effects_ir(volume_profile_map[profile_guid])

    # ── Terrain: heightmap/mesh object ────────────────────────────────────────
    if ir_type == _IR_TYPE_TERRAIN:
        terrain_raw = raw_node.get("terrain_data")
        if terrain_raw and isinstance(terrain_raw, dict):
            terrain_data = dict(terrain_raw)
            # Resolve terrain data GUID to rich asset info from TerrainData .asset
            td_ref = terrain_raw.get("terrain_data_ref") or {}
            td_guid = ""
            if isinstance(td_ref, dict):
                td_guid = td_ref.get("guid", "")
            elif isinstance(td_ref, str) and td_ref.startswith("guid:"):
                td_guid = td_ref[5:]
            if td_guid and terrain_ir_map and td_guid in terrain_ir_map:
                terrain_data["asset_data"] = terrain_ir_map[td_guid]
            components["terrain"] = terrain_data

    # ── Wind Zone: volume-based wind force ────────────────────────────────────
    if ir_type == _IR_TYPE_WIND_ZONE:
        wz_raw = raw_node.get("wind_zone_data")
        if wz_raw and isinstance(wz_raw, dict):
            components["wind_zone"] = wz_raw

    # ── Text 3D: legacy TextMesh component ────────────────────────────────────
    if ir_type == _IR_TYPE_TEXT_3D:
        tm_raw = raw_node.get("text_mesh_data")
        if tm_raw and isinstance(tm_raw, dict):
            components["text"] = tm_raw

    # ── UI: RectTransform + element/layout/CSF components ────────────────────
    if ir_type in (_IR_TYPE_UI_CANVAS, _IR_TYPE_UI_ELEMENT):
        raw_rt = raw_node.get("rect_transform")
        rt_ir  = _build_rect_transform_ir(raw_rt)
        if rt_ir:
            components["rect_transform"] = rt_ir

        # Canvas-level metadata (render mode, sort order)
        if ir_type == _IR_TYPE_UI_CANVAS:
            raw_canvas = raw_node.get("canvas_data")
            if raw_canvas and isinstance(raw_canvas, dict):
                components["canvas"] = raw_canvas

        raw_ui     = raw_node.get("ui_element")
        raw_layout = raw_node.get("ui_layout")
        raw_csf    = raw_node.get("ui_csf")

        ui_ir = _build_ui_element_ir(raw_ui, node_id)
        if ui_ir:
            components["ui_element"] = ui_ir

        layout_ir = _build_ui_layout_ir(raw_layout, node_id)
        if layout_ir:
            components["ui_layout"] = layout_ir

        if raw_csf and isinstance(raw_csf, dict):
            components["content_size_fitter"] = raw_csf

        # LayoutElement: per-element size overrides for parent layout groups.
        # Stored verbatim — engine-agnostic (min/preferred/flexible widths and heights).
        raw_le = raw_node.get("layout_element")
        if raw_le and isinstance(raw_le, dict):
            components["layout_element"] = raw_le

    # ── Audio source ──────────────────────────────────────────────────────────
    if ir_type == _IR_TYPE_AUDIO_SOURCE:
        raw_audio = raw_node.get("audio_source_data")
        audio_ir  = _build_audio_source_ir(raw_audio, node_id)
        if audio_ir:
            components["audio_source"] = audio_ir

    # ── Particle system ───────────────────────────────────────────────────────
    if ir_type == _IR_TYPE_PARTICLES:
        raw_ps = raw_node.get("particle_data")
        ps_ir  = _build_particle_system_ir(raw_ps, node_id)
        if ps_ir:
            components["particle_system"] = ps_ir

    # ── Joints: ragdoll / physics joints ──────────────────────────────────────
    raw_joints = raw_node.get("joints") or []
    if raw_joints:
        components["joints"] = raw_joints

    # ── Skinned Mesh: Skeleton3D bone hierarchy ───────────────────────────────
    raw_skm = raw_node.get("skinned_mesh_data")
    if raw_skm and isinstance(raw_skm, dict) and raw_skm.get("bones"):
        components["skinned_mesh"] = raw_skm

    # ── Animation: Animator + AnimationClip data ───────────────────────────────
    if unity_flags.get("has_animator"):
        ctrl_guid = unity_flags.get("animator_controller_guid")
        anim_comp = _build_animation_component(
            ctrl_guid, animation_ir_map, animator_controller_map
        )
        if anim_comp:
            components["animation"] = anim_comp

    if unity_flags.get("is_prefab_instance"):
        components["instance_ref"] = {
            "is_instance": True,
            # The parser stores the source prefab GUID under "source_prefab_guid"
            # (set in unity_parser._objects_to_ir via flags["source_prefab_guid"]).
            "prefab_guid": unity_flags.get("source_prefab_guid", ""),
            "prefab_name": unity_flags.get("prefab_name", ""),
            "prefab_ir":   unity_flags.get("prefab_ir"),
            # True when this instance is itself a Variant Prefab asset.
            "is_variant":  bool(unity_flags.get("is_variant_prefab", False)),
            # Per-instance component property overrides (non-transform).
            # Each entry: {property_path, value, target_file_id, target_guid}
            "component_overrides": list(unity_flags.get("component_overrides") or []),
            # Components removed from / added to this instance vs the source prefab.
            # Each entry: {file_id, guid}
            "removed_components":  list(unity_flags.get("removed_components") or []),
            "added_components":    list(unity_flags.get("added_components")   or []),
        }

    # ── Navigation agent ─────────────────────────────────────────────────────
    raw_nav_agent = raw_node.get("nav_agent_data")
    if raw_nav_agent:
        nav_agent_ir = _build_nav_agent_ir(raw_nav_agent)
        if nav_agent_ir:
            components["nav_agent"] = nav_agent_ir

    # ── Navigation obstacle ───────────────────────────────────────────────────
    raw_nav_obstacle = raw_node.get("nav_obstacle_data")
    if raw_nav_obstacle:
        nav_obstacle_ir = _build_nav_obstacle_ir(raw_nav_obstacle)
        if nav_obstacle_ir:
            components["nav_obstacle"] = nav_obstacle_ir

    # ── Script components (non-UI MonoBehaviours) ─────────────────────────────
    raw_scripts = raw_node.get("script_components") or []
    if raw_scripts:
        scripts_ir = _build_script_components_ir(raw_scripts, node_id)
        if scripts_ir:
            components["script_components"] = scripts_ir

    # ── Assemble node dict ────────────────────────────────────────────────
    node: Dict[str, Any] = {
        "node_id":    node_id,
        "node_name":  node_name,
        "node_type":  ir_type,
        "is_spatial": spatial,
        "transform": {
            "position": _pos(t),
            "rotation": _rot(t),
            "scale":    _scl(t),
        },
        "components": components,
        "original_data": {
            "source_object_id":    unity_data.get("gameObject_fileID"),
            "source_transform_id": unity_data.get("transform_fileID"),
            "source_flags":        unity_flags,
            "mesh_ref":            raw_mesh_ref,
            "material_refs":       raw_material_refs,
        },
        "meta":     meta,
        "children": [],
    }

    for raw_child in raw_node.get("children") or []:
        if isinstance(raw_child, dict):
            node["children"].append(
                _build_ir_node(
                    raw_child, scene_prefix, counter,
                    guid_map, material_ir_map, volume_profile_map,
                    physics_material_ir_map,
                    animation_ir_map, animator_controller_map,
                    terrain_ir_map=terrain_ir_map,
                    warnings=warnings,
                )
            )

    return node


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _deep_normalize_ir(obj: Any) -> Any:
    """Recursively normalise Unity YAML scalar wrapper dicts in an IR tree.

    Unity YAML sometimes serialises numeric scalars as ``{"value": 1.0}``.
    This function walks the entire IR dict/list structure and replaces every
    ``{"value": primitive}`` with the primitive itself, so that downstream
    builders never see a dict where they expect a float.

    Only single-key ``{"value": x}`` dicts whose value is int, float, bool, or
    str are normalised.  All other dicts are recursed into without modification,
    preserving complex nested structures like material dicts or component maps.
    """
    if isinstance(obj, dict):
        if len(obj) == 1 and "value" in obj:
            inner = obj["value"]
            if isinstance(inner, (int, float, bool, str)):
                return inner
        return {k: _deep_normalize_ir(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_normalize_ir(item) for item in obj]
    return obj


_MIN_IR_SCALE = 1e-4   # shared constant; mirrors godot_project_builder._MIN_SCALE


def _validate_ir_nodes(
    nodes: List[Dict[str, Any]],
    warnings: Optional[List[str]],
    path: str = "",
) -> None:
    """Recursively validate IR nodes and auto-fix issues where possible.

    Checks performed:
    * Transform scale — clamps any component whose magnitude is < _MIN_IR_SCALE
      (zero-scale makes the basis matrix singular → Godot affine_invert error).
    * Text font size  — clamps to ≥ 1 (zero size → Godot shaped_text error).
    * Collider shape  — warns when mesh_convex/mesh_concave has no inline points
      (no auto-fix at IR stage; the emission stage does the fallback).
    * Mesh vertex list — warns when a mesh IR has an empty vertices list.
    """
    for node in nodes:
        if not isinstance(node, dict):
            continue
        name = node.get("node_name") or node.get("name") or "?"
        node_path = f"{path}/{name}" if path else name

        # ── Transform scale ───────────────────────────────────────────────────
        t = node.get("transform")
        if isinstance(t, dict):
            scl = t.get("scale")
            if isinstance(scl, list) and len(scl) >= 3:
                fixed = False
                new_scl = list(scl)
                for i in range(3):
                    v = new_scl[i]
                    if isinstance(v, (int, float)) and abs(v) < _MIN_IR_SCALE:
                        new_scl[i] = _MIN_IR_SCALE if v >= 0 else -_MIN_IR_SCALE
                        fixed = True
                if fixed:
                    t["scale"] = new_scl
                    if warnings is not None:
                        warnings.append(
                            f"Zero scale on node '{node_path}': {scl!r} → "
                            f"clamped to {new_scl!r}"
                        )

        # ── Text component — font size ─────────────────────────────────────────
        comps = node.get("components") or {}
        text_comp = comps.get("text")
        if isinstance(text_comp, dict):
            fs = text_comp.get("font_size")
            if isinstance(fs, (int, float)) and fs < 1:
                text_comp["font_size"] = 1
                if warnings is not None:
                    warnings.append(
                        f"Font size <= 0 on node '{node_path}' ({fs!r}); "
                        f"clamped to 1."
                    )

        # ── Collider shapes — warn on unresolvable mesh colliders ─────────────
        for col in (comps.get("colliders") or []):
            if not isinstance(col, dict):
                continue
            shape = col.get("shape") or {}
            st = shape.get("type", "")
            if st in ("mesh_convex", "mesh_concave"):
                pts = shape.get("points") or []
                if not (isinstance(pts, (list, tuple)) and len(pts) >= 4):
                    mesh_ref = shape.get("mesh_ref") or "<unknown>"
                    if warnings is not None:
                        warnings.append(
                            f"Mesh collider '{mesh_ref}' on node '{node_path}' "
                            f"has no inline vertex data; will use BoxShape3D fallback."
                        )

        # ── Mesh IR — warn on empty vertex arrays (external meshes only) ─────
        mesh_ir = comps.get("mesh") or {}
        mesh_src = mesh_ir.get("source") or {}
        # Builtin/procedural meshes (BoxMesh, SphereMesh, …) have no inline vertex
        # data by design — Godot generates them.  Only warn for external (GUID) refs.
        if (mesh_ir
                and mesh_src.get("type") not in ("builtin", "procedural")
                and isinstance(mesh_ir.get("vertices"), list)
                and len(mesh_ir["vertices"]) == 0
                and mesh_ir.get("mesh_id")):
            if warnings is not None:
                warnings.append(
                    f"Empty vertex array in external mesh for node '{node_path}' "
                    f"(mesh_id='{mesh_ir.get('mesh_id', '?')}')."
                )

        # ── Recurse into children ──────────────────────────────────────────────
        _validate_ir_nodes(node.get("children") or [], warnings, path=node_path)


def _detect_vehicles(
    ir_nodes: List[Dict[str, Any]],
    warnings: Optional[List[str]] = None,
) -> None:
    """Post-process pass: detect Rigidbody + WheelCollider vehicle setups.

    Traverses the IR tree. When a dynamic-rigidbody node has direct children
    that each carry a wheel collider, the root is marked as a vehicle body and
    each wheel child is marked with its wheel parameters.

    Mutations (in-place):
      root node: components["vehicle"] = {"wheel_count": N}
      each wheel child: components["vehicle_wheel"] = {radius, suspension_travel,
                          spring_stiffness, spring_damper, sideways_stiffness}
    """

    def _wheel_shape(node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return the first wheel shape dict on *node*, or None."""
        comps = node.get("components") or {}
        for col in (comps.get("colliders") or []):
            shape = col.get("shape") or {}
            if shape.get("type") == "wheel":
                return shape
        return None

    def _process(nodes: List[Dict[str, Any]]) -> None:
        for node in nodes:
            comps    = node.get("components") or {}
            rb       = comps.get("rigidbody")
            children = node.get("children") or []

            if rb and rb.get("type") != "kinematic":
                wheel_children: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
                for child in children:
                    ws = _wheel_shape(child)
                    if ws is not None:
                        wheel_children.append((child, ws))

                if wheel_children:
                    comps["vehicle"] = {"wheel_count": len(wheel_children)}
                    node_name = node.get("node_name", "?")
                    if warnings is not None:
                        warnings.append(
                            f"Vehicle detected: '{node_name}' → VehicleBody3D "
                            f"with {len(wheel_children)} VehicleWheel3D child(ren). "
                            "Suspension/friction values are approximated; tune after import."
                        )
                    for wchild, ws in wheel_children:
                        wcomps = wchild.setdefault("components", {})
                        wcomps["vehicle_wheel"] = {
                            "radius":              float(_unwrap_scalar(ws.get("radius", 0.5))),
                            "suspension_travel":   float(_unwrap_scalar(ws.get("suspension_distance", 0.3))),
                            "spring_stiffness":    float(_unwrap_scalar(ws.get("spring_stiffness", 35000.0))),
                            "spring_damper":       float(_unwrap_scalar(ws.get("spring_damper", 4500.0))),
                            "sideways_stiffness":  float(_unwrap_scalar(ws.get("sideways_stiffness", 1.0))),
                        }

            _process(children)

    _process(ir_nodes)


def build_scene_ir(
    raw_ir: Dict[str, Any],
    scene_name: str = "scene",
    source_file: str = "",
    source_engine: str = "Unity",
    source_engine_version: str = "6000.3.9f1",
    target_engine: str = "Godot",
    target_engine_version: str = "4.5",
    guid_map: Optional[Dict[str, Path]] = None,
    material_ir_map: Optional[Dict[str, Any]] = None,
    volume_profile_map: Optional[Dict[str, Any]] = None,
    physics_material_ir_map: Optional[Dict[str, Any]] = None,
    animation_ir_map: Optional[Dict[str, Any]] = None,
    animator_controller_map: Optional[Dict[str, Any]] = None,
    terrain_ir_map: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Converts the raw parser IR dict into a minimal spec-compliant scene IR.

    Args:
        raw_ir:                  Dict returned by unity_parser._objects_to_ir  {"nodes": [...]}
        scene_name:              Human name for this scene (e.g. "Main", "Level1").
        source_file:             Original .unity file path string, stored for traceability.
        source_engine:           Display name of the source engine (e.g. "Unity").
        source_engine_version:   Version of the source engine (e.g. "6000.3.9f1").
        target_engine:           Display name of the target engine (e.g. "Godot").
        target_engine_version:   Version of the target engine (e.g. "4.5").
        guid_map:                Optional guid → Path map for resolving external mesh paths.
        material_ir_map:         Optional guid → PBR dict for resolving material properties.
        volume_profile_map:      Optional guid → post-processing effects dict.
        physics_material_ir_map: Optional guid → physics material dict (friction/bounce).
        terrain_ir_map:          Optional guid → terrain asset dict (resolution, world size, splatmaps).
        warnings:                Optional list that receives warning strings during IR build.
    """
    scene_id     = _safe_id(scene_name)
    scene_prefix = f"scene_{scene_id}"
    counter      = [0]   # mutable int wrapped in list for closure mutation

    ir_nodes = [
        _build_ir_node(
            raw_node, scene_prefix, counter,
            guid_map, material_ir_map, volume_profile_map,
            physics_material_ir_map,
            animation_ir_map, animator_controller_map,
            terrain_ir_map=terrain_ir_map,
            warnings=warnings,
        )
        for raw_node in (raw_ir.get("nodes") or [])
        if isinstance(raw_node, dict)
    ]

    scene: Dict[str, Any] = {
        # Coordinate system is defined once here; never repeated inside nodes.
        # Transforms are stored in Unity left-handed space; the builder converts
        # at emit time (negate Z position, Y+180 rotation).
        "coordinate_system": {
            "handedness":   "right",
            "up_axis":      "Y",
            "forward_axis": "-Z",
        },
        "ir_version":            "1.0",
        "source_engine":         source_engine,
        "source_engine_version": source_engine_version,
        "target_engine":         target_engine,
        "target_engine_version": target_engine_version,
        "scene_id":              scene_id,
        "scene_name":            scene_name,
        "source_file":           source_file,
        "node_count":            counter[0],
        "nodes":                 ir_nodes,
    }

    raw_render_settings = raw_ir.get("render_settings")
    if raw_render_settings and isinstance(raw_render_settings, dict):
        scene["environment"] = _build_environment_ir(raw_render_settings)

    raw_nav_settings = raw_ir.get("nav_settings")
    if raw_nav_settings and isinstance(raw_nav_settings, dict):
        scene["nav_mesh_settings"] = _build_nav_mesh_settings_ir(raw_nav_settings)

    # Post-build validation: auto-fix zero scales, clamp font sizes, warn on
    # unresolvable mesh colliders and empty vertex arrays.  Runs on the mutable
    # ir_nodes list in-place so that _deep_normalize_ir sees the corrected values.
    _validate_ir_nodes(ir_nodes, warnings)

    # Detect Rigidbody + WheelCollider vehicle setups and mark vehicle nodes.
    # Must run after _validate_ir_nodes (which normalises scalar wrappers).
    _detect_vehicles(ir_nodes, warnings)

    # Normalise any {"value": x} scalar wrappers that came through from Unity YAML
    # parsing before handing the IR off to downstream builders.
    return _deep_normalize_ir(scene)


# ---------------------------------------------------------------------------
# Exporter adapter
# ---------------------------------------------------------------------------
# The godot_exporter reads a flat dict with {"nodes": [...]}, where each node
# has: name, type (Godot class string), transform.position, children.
#
# Our IR uses node_name, node_type (engine-agnostic), components.
# This adapter converts IR → exporter-compatible dict without touching the
# exporter at all.

def ir_to_exporter_dict(scene_ir: Dict[str, Any]) -> Dict[str, Any]:
    """Converts a scene IR dict into the format godot_exporter.save_tscn expects.

    This is the only place that knows about both formats. Keep it thin.
    """

    _ir_to_godot: Dict[str, str] = {
        _IR_TYPE_ENTITY:       "MeshInstance3D",
        _IR_TYPE_CAMERA:       "Camera3D",
        _IR_TYPE_AUDIO_SOURCE: "AudioStreamPlayer3D",
        _IR_TYPE_UI_CANVAS:    "CanvasLayer",
        _IR_TYPE_UI_ELEMENT:   "Control",
        _IR_TYPE_PARTICLES:    "GPUParticles3D",
        _IR_TYPE_SPRITE:       "Sprite2D",
        _IR_TYPE_GROUP:        "Node3D",
        _IR_TYPE_TERRAIN:      "StaticBody3D",
        _IR_TYPE_WIND_ZONE:    "Area3D",
        _IR_TYPE_TEXT_3D:      "Label3D",
        _IR_TYPE_TREE:         "MeshInstance3D",
    }
    _light_subtype_to_godot: Dict[str, str] = {
        "directional": "DirectionalLight3D",
        "point":       "OmniLight3D",
        "spot":        "SpotLight3D",
        "probe":       "ReflectionProbe",
    }

    def _convert_node(ir_node: Dict[str, Any]) -> Dict[str, Any]:
        ir_type = ir_node.get("node_type") or _IR_TYPE_GROUP
        comps   = ir_node.get("components") or {}
        if ir_type == _IR_TYPE_LIGHT:
            godot_type = _light_subtype_to_godot.get(
                comps.get("light_subtype", "directional"), "DirectionalLight3D"
            )
        elif ir_type == _IR_TYPE_AUDIO_SOURCE:
            # spatial_blend == 0 → AudioStreamPlayer (2D); otherwise 3D.
            audio_comp = comps.get("audio_source") or {}
            spatial_blend = float(_unwrap_scalar(audio_comp.get("spatial_blend", 1.0)))
            godot_type = "AudioStreamPlayer" if spatial_blend < 0.5 else "AudioStreamPlayer3D"
        else:
            godot_type = _ir_to_godot.get(ir_type, "Node3D")

        t = ir_node.get("transform") or {}

        mesh_ir   = comps.get("mesh") or {}
        mesh_src  = mesh_ir.get("source") or {}
        # Reconstruct a raw ref string for exporters that still use it.
        if mesh_src.get("type") == "builtin":
            mesh_ref = f"builtin_{mesh_src.get('builtin_type', 'BoxMesh')}"
        elif mesh_src.get("type") == "external":
            mesh_ref = f"guid:{mesh_src.get('guid', '')}"
        else:
            mesh_ref = ""

        return {
            "name":      ir_node.get("node_name", "Node"),
            "type":      godot_type,
            "transform": {
                "position": t.get("position", [0.0, 0.0, 0.0]),
                "rotation": t.get("rotation", {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}),
                "scale":    t.get("scale",    [1.0, 1.0, 1.0]),
            },
            "mesh":      mesh_ref,
            "mesh_ir":   mesh_ir,
            "materials": [m.get("source_guid", "") for m in (comps.get("materials") or [])],
            "materials_ir": comps.get("materials") or [],
            "children":  [_convert_node(ch) for ch in ir_node.get("children") or []],
        }

    return {
        "nodes": [_convert_node(n) for n in scene_ir.get("nodes") or []]
    }


# ---------------------------------------------------------------------------
# Basic validation (inline — full validator comes in a later step)
# ---------------------------------------------------------------------------

class IRValidationError(Exception):
    """Raised when the IR fails a blocking check."""


def validate_scene_ir_basic(scene_ir: Dict[str, Any]) -> List[str]:
    """Runs minimal structural checks. Returns a list of warning strings.

    Raises IRValidationError for blocking errors that would prevent emission.
    """
    warnings: List[str] = []

    # Blocking: must have the required top-level keys
    for key in ("ir_version", "scene_id", "scene_name", "nodes"):
        if key not in scene_ir:
            raise IRValidationError(f"Missing required top-level key: '{key}'")

    if not isinstance(scene_ir["nodes"], list):
        raise IRValidationError("'nodes' must be a list")

    if len(scene_ir["nodes"]) == 0:
        warnings.append("Scene has no nodes — output will be an empty scene.")

    # Walk all nodes and check basic structure
    seen_ids: set[str] = set()

    def _check_node(node: Any, depth: int) -> None:
        if not isinstance(node, dict):
            warnings.append(f"Non-dict node encountered at depth {depth}, skipped.")
            return

        node_id = node.get("node_id", "")
        if not node_id:
            warnings.append(f"Node '{node.get('node_name')}' is missing node_id.")
        elif node_id in seen_ids:
            warnings.append(f"Duplicate node_id '{node_id}' found within scene.")
        else:
            seen_ids.add(node_id)

        if not node.get("node_name"):
            warnings.append(f"Node '{node_id}' has an empty node_name.")

        if node.get("is_spatial") and "transform" not in node:
            warnings.append(f"Spatial node '{node_id}' is missing transform.")

        for child in node.get("children") or []:
            _check_node(child, depth + 1)

    for root in scene_ir["nodes"]:
        _check_node(root, 0)

    return warnings