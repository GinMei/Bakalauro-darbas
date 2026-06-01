"""unity_scene_exporter.py

Stage D of the Godot → Unity pipeline.

Converts an engine-agnostic scene IR into a Unity 6000.3.9f1 .unity scene file
(YAML format) and writes it to disk.  Also generates a minimal .meta file and
a stub C# Editor script that triggers a reimport when the scene is opened.

Supported IR component types:
    transform        — LocalPosition / LocalRotation / LocalScale
    mesh (entity)    — MeshFilter + MeshRenderer
    rigidbody        — Rigidbody component
    colliders        — BoxCollider / SphereCollider / CapsuleCollider
    camera           — Camera component

Explicitly NOT supported:
    scripts, animations, lights, terrain, UI, particles, shaders

Public API:
    UnitySceneExporter     — export a scene IR to a Unity project folder
"""

from __future__ import annotations

import logging
import math
import uuid as _uuid_mod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("unity_scene_exporter")

# ---------------------------------------------------------------------------
# Stable GUID helpers (shared with godot_to_unity_pipeline.py — same namespace)
# ---------------------------------------------------------------------------

_GUID_NAMESPACE = _uuid_mod.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # = NAMESPACE_URL


def _stable_guid(key: str) -> str:
    """Return a deterministic 32-char hex GUID derived from *key*.

    The GUID is stable across re-runs as long as *key* does not change.
    Uses UUID5 with a fixed namespace so collisions are cryptographically
    improbable.
    """
    return _uuid_mod.uuid5(_GUID_NAMESPACE, key).hex


def _stable_local_id(key: str) -> int:
    """Return a deterministic negative int64 local fileID derived from *key*.

    Used for embedded sub-objects within a single asset file (AnimatorController,
    AnimatorStateMachine, AnimatorState).  Negative values are the Unity convention
    for intra-file references.  Range: [-(2^62), -1].
    """
    h = _uuid_mod.uuid5(_GUID_NAMESPACE, key).int
    return -((h % (2 ** 62)) + 1)


# ---------------------------------------------------------------------------
# Asset .meta reader
# ---------------------------------------------------------------------------

def _read_meta_guid(meta_path: Path) -> str:
    """Return the ``guid`` value from a Unity .meta file, or '' on failure."""
    try:
        for line in meta_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if stripped.startswith("guid:"):
                return stripped.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


# ---------------------------------------------------------------------------
# Folder .meta helper
# ---------------------------------------------------------------------------

def write_folder_meta(folder: Path, output_dir: Path) -> None:
    """Write a Unity folder .meta sidecar next to *folder* if absent.

    The meta is placed at ``<folder.parent>/<folder.name>.meta``.
    GUID is derived from the folder's posix path relative to *output_dir*
    so it is stable across pipeline re-runs.
    """
    meta_path = folder.parent / f"{folder.name}.meta"
    if meta_path.exists():
        return
    try:
        rel = folder.relative_to(output_dir).as_posix()
    except ValueError:
        rel = folder.name
    guid = _stable_guid(f"folder:{rel}")
    meta_path.write_text(
        f"fileFormatVersion: 2\n"
        f"guid: {guid}\n"
        f"folderAsset: yes\n"
        f"DefaultImporter:\n"
        f"  externalObjects: {{}}\n"
        f"  userData: \n"
        f"  assetBundleName: \n"
        f"  assetBundleVariant: \n",
        encoding="utf-8",
    )



# ---------------------------------------------------------------------------
# Unity YAML constants
# ---------------------------------------------------------------------------

_YAML_HEADER = """\
%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
"""

# Unity class IDs used in the .unity YAML
_CID_OCCLUSION        = 29
_CID_RENDER_SETTINGS  = 104
_CID_SCENE_SETTINGS   = 1044696467
_CID_GAME_OBJECT      = 1
_CID_TRANSFORM        = 4
_CID_MESH_FILTER      = 33
_CID_MESH_RENDERER    = 23
_CID_RIGIDBODY        = 54
_CID_BOX_COLLIDER     = 65
_CID_SPHERE_COLLIDER  = 135
_CID_CAPSULE_COLLIDER = 136
_CID_CAMERA           = 20
_CID_LIGHT            = 108
_CID_AUDIO_SOURCE     = 82
_CID_ANIMATOR         = 95
_CID_RECT_TRANSFORM        = 224
_CID_CANVAS                = 223
_CID_CHARACTER_CONTROLLER  = 97
_CID_MONOBEHAVIOUR         = 114
_CID_PARTICLE_SYSTEM       = 198
_CID_PREFAB_INSTANCE       = 1001
_CID_SPRITE_RENDERER       = 212
_CID_MESH_COLLIDER         = 64

# Fixed Unity built-in GUIDs for UnityEngine.UI components (invariant across projects)
_CANVAS_SCALER_GUID     = "0cd44c1031e13a943bb63640046fad76"
_GRAPHIC_RAYCASTER_GUID = "dc42784cf147c0c48a680349fa168899"
_UI_IMAGE_GUID          = "fe87c0e1cc204ed48ad3b37840f39efc"
_UI_BUTTON_GUID         = "4e29b1a8efbd4b44bb3f3716e73f07ff"
_TMP_UGUI_GUID          = "f4688fdb7df04437aeb418b961361dc5"
_TMP_FONT_GUID          = "8f586378b4e144a9851e7b34d9b748ee"
_CID_CANVAS_RENDERER    = 222

# Godot anchors_preset → Unity (AnchorMin, AnchorMax, Pivot)
# Y-axis is flipped: Godot Y=0 is top, Unity Y=0 is bottom.
_ANCHORS_PRESET: Dict[int, Tuple[Tuple[float, float], ...]] = {
    0:  ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),  # Top Left
    1:  ((1.0, 1.0), (1.0, 1.0), (1.0, 1.0)),  # Top Right
    2:  ((1.0, 0.0), (1.0, 0.0), (1.0, 0.0)),  # Bottom Right
    3:  ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),  # Bottom Left
    4:  ((0.0, 0.5), (0.0, 0.5), (0.0, 0.5)),  # Center Left
    8:  ((0.5, 0.5), (0.5, 0.5), (0.5, 0.5)),  # Center
    12: ((0.0, 1.0), (1.0, 1.0), (0.5, 1.0)),  # Top Wide
    14: ((0.0, 0.0), (1.0, 0.0), (0.5, 0.0)),  # Bottom Wide
    15: ((0.0, 0.0), (1.0, 1.0), (0.5, 0.5)),  # Full Rect
}

# Standard Unity internal fileIDs for the root objects within an imported
# 3D model asset (OBJ/FBX) using Unity's fileIdsGeneration: 1 scheme.
_MODEL_ROOT_TRANSFORM_ID: int = -4216859302048453862
_MODEL_ROOT_GO_ID:        int = -927199367670048503

_MODEL_ASSET_EXTENSIONS = frozenset({".obj", ".fbx", ".gltf", ".glb", ".dae"})
_CID_ANIMATION_CLIP        = 74
_CID_ANIMATOR_CONTROLLER   = 91
_CID_ANIMATOR_STATE_MACHINE = 1107
_CID_ANIMATOR_STATE        = 1102

# When we generate a prefab, the root node always gets these deterministic IDs
# (_IdCounter starts at 100_000_000; first call → go_id, second → tr_id).
_PREFAB_ROOT_GO_ID        = 100_000_000
_PREFAB_ROOT_TRANSFORM_ID = 100_000_001

_LIGHT_TYPE_INT = {
    "directional": 1,
    "point":       2,
    "spot":        0,
}

# Maps Godot value-track property names to Unity AnimationClip attribute names.
# Unknown properties are passed through unchanged.
_GODOT_PROP_TO_UNITY_ATTR: Dict[str, str] = {
    "visible":      "m_IsActive",
    "modulate:r":   "m_Color.r",
    "modulate:g":   "m_Color.g",
    "modulate:b":   "m_Color.b",
    "modulate:a":   "m_Color.a",
}


# ---------------------------------------------------------------------------
# ID counter
# ---------------------------------------------------------------------------

class _IdCounter:
    """Generates unique monotonically-increasing Unity fileIDs."""

    def __init__(self, start: int = 100_000_000) -> None:
        self._next = start

    def next(self) -> int:
        v = self._next
        self._next += 1
        return v


# ---------------------------------------------------------------------------
# IR pre-processing — structural rules applied before Unity emission
# ---------------------------------------------------------------------------

def _is_identity_rot(rot: Any) -> bool:
    """Return True if a Godot quaternion is approximately identity [0,0,0,1]."""
    if isinstance(rot, (list, tuple)) and len(rot) >= 4:
        x, y, z, w = float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])
    elif isinstance(rot, dict):
        x = float(rot.get("x", 0.0)); y = float(rot.get("y", 0.0))
        z = float(rot.get("z", 0.0)); w = float(rot.get("w", 1.0))
    else:
        return True
    return abs(x) < 1e-6 and abs(y) < 1e-6 and abs(z) < 1e-6 and abs(w - 1.0) < 1e-6


def _rot_as_list(rot: Any) -> List[float]:
    if isinstance(rot, (list, tuple)) and len(rot) >= 4:
        return [float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])]
    if isinstance(rot, dict):
        return [float(rot.get("x", 0.0)), float(rot.get("y", 0.0)),
                float(rot.get("z", 0.0)), float(rot.get("w", 1.0))]
    return [0.0, 0.0, 0.0, 1.0]


def _collect_nested_instances(
    children: List[Dict[str, Any]],
    parent_interior_path: str,
):
    """Yield (interior_parent_path, instance_node) for every _InstanceNode
    that is a descendant of *children*.

    interior_parent_path is the slash-separated path of the PARENT node
    relative to the prefab root's children (empty string = direct child of
    prefab root).  Regular (non-instance) nodes are recursed into; the first
    _InstanceNode found at any level terminates recursion for that branch.
    """
    for child in children:
        if child.get("godot_type") == "_InstanceNode":
            yield parent_interior_path, child
        else:
            child_path = (
                f"{parent_interior_path}/{child['node_name']}"
                if parent_interior_path else child["node_name"]
            )
            yield from _collect_nested_instances(child.get("children", []), child_path)


def _quat_mul(a: List[float], b: List[float]) -> List[float]:
    """Hamilton product of two [x,y,z,w] quaternions (Godot space)."""
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return [
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ]


_PHYSICS_BODY_TYPES = frozenset({
    "StaticBody3D", "RigidBody3D", "AnimatableBody3D", "CharacterBody3D",
})


def _collider_from_shape_node(
    coll_node: Dict[str, Any],
    is_trigger: bool = False,
) -> List[Dict[str, Any]]:
    """Return collider dicts from a CollisionShape3D IR node, applying its
    local position and scale to each collider's centre/size.
    Rotation is intentionally not folded in here (see rule 2).
    """
    t       = coll_node.get("transform", {})
    c_pos   = list(t.get("position", [0.0, 0.0, 0.0]))
    c_scale = list(t.get("scale",    [1.0, 1.0, 1.0]))
    result  = []
    for coll in coll_node.get("components", {}).get("colliders", []):
        coll = dict(coll)
        coll["is_trigger"] = is_trigger
        raw_c = list(coll.get("center", [0.0, 0.0, 0.0]))
        coll["center"] = [
            c_pos[0] + c_scale[0] * raw_c[0],
            c_pos[1] + c_scale[1] * raw_c[1],
            c_pos[2] + c_scale[2] * raw_c[2],
        ]
        if "size" in coll:
            sz = list(coll["size"])
            coll["size"] = [sz[0]*c_scale[0], sz[1]*c_scale[1], sz[2]*c_scale[2]]
        if "radius" in coll:
            coll["radius"] = float(coll["radius"]) * max(c_scale)
        if "height" in coll:
            coll["height"] = float(coll["height"]) * c_scale[1]
        result.append(coll)
    return result


def _apply_mesh_body_collapse_pattern_a(
    node: Dict[str, Any],
    decisions: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Rules 4/5/6 — Pattern A: PhysicsBody3D with MeshInstance3D + CollisionShape3D children.

    StaticBody3D(CollisionShape3D, MeshInstance3D) collapses into one Unity object
    named after the StaticBody3D, with mesh + rigidbody + collider as components.
    """
    children  = node.get("children", [])
    mesh_kids = [c for c in children if c.get("godot_type") == "MeshInstance3D"]
    coll_kids = [c for c in children if c.get("godot_type") == "CollisionShape3D"]
    if not mesh_kids or not coll_kids:
        return node

    node  = dict(node)
    comps = dict(node.get("components", {}))

    mesh_comps = mesh_kids[0].get("components", {})
    if "mesh"     in mesh_comps: comps["mesh"]     = mesh_comps["mesh"]
    if "material" in mesh_comps: comps["material"] = mesh_comps["material"]

    pulled: List[Dict[str, Any]] = []
    for ck in coll_kids:
        pulled.extend(_collider_from_shape_node(ck, is_trigger=False))
    if pulled:
        comps["colliders"] = comps.get("colliders", []) + pulled

    node["components"] = comps
    stolen = {id(c) for c in mesh_kids + coll_kids}
    node["children"]   = [c for c in children if id(c) not in stolen]
    return node


def _apply_mesh_body_collapse_pattern_b(
    node: Dict[str, Any],
    decisions: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Rules 4/5/6 — Pattern B: MeshInstance3D with a PhysicsBody3D child that contains CollisionShape3D.

    MeshInstance3D(StaticBody3D(CollisionShape3D)) collapses into one Unity object
    named after the MeshInstance3D, with mesh + rigidbody + collider as components.
    """
    children  = node.get("children", [])
    body_kids = [c for c in children if c.get("godot_type") in _PHYSICS_BODY_TYPES]
    if not body_kids:
        return node

    comps        = dict(node.get("components", {}))
    stolen_ids   = set()
    all_colliders: List[Dict[str, Any]] = []

    for body in body_kids:
        coll_kids = [c for c in body.get("children", [])
                     if c.get("godot_type") == "CollisionShape3D"]
        if not coll_kids:
            continue
        body_comps = body.get("components", {})
        if "rigidbody" in body_comps and "rigidbody" not in comps:
            comps["rigidbody"] = body_comps["rigidbody"]
        for ck in coll_kids:
            all_colliders.extend(_collider_from_shape_node(ck, is_trigger=False))
        stolen_ids.add(id(body))

    if not stolen_ids:
        return node

    if all_colliders:
        comps["colliders"] = comps.get("colliders", []) + all_colliders
    node = dict(node)
    node["components"] = comps
    node["children"]   = [c for c in children if id(c) not in stolen_ids]
    return node


def _apply_area3d_rules(
    node: Dict[str, Any],
    decisions: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Rules 1, 8 — Area3D always becomes an empty GameObject.

    • 0 CollisionShape3D children  → Rule 1: empty GO, no collider.
    • 1+ CollisionShape3D children → Rule 8: Area3D keeps its own transform;
                                     each child becomes Wrapper(pos+rot) / Leaf(scale+collider,
                                     isTrigger = true).
    """
    node  = dict(node)
    comps = dict(node.get("components", {}))
    node["components"] = comps

    children   = list(node.get("children", []))
    coll_kids  = [c for c in children if c.get("godot_type") == "CollisionShape3D"]
    other_kids = [c for c in children if c.get("godot_type") != "CollisionShape3D"]

    if not coll_kids:
        # Rule 1: no CollisionShape3D children — keep inline trigger (if any) as-is.
        return node

    # CollisionShape3D children take over collision; drop any redundant inline trigger.
    comps.pop("trigger", None)

    # ── Rule 8 ────────────────────────────────────────────────────────────────
    new_children = []
    for coll in coll_kids:
        ct        = coll.get("transform", {})
        c_pos     = list(ct.get("position", [0.0, 0.0, 0.0]))
        c_rot     = _rot_as_list(ct.get("rotation", [0.0, 0.0, 0.0, 1.0]))
        c_scale   = list(ct.get("scale",    [1.0, 1.0, 1.0]))
        coll_name = coll.get("node_name", "Shape")

        coll_dicts = []
        for cd in coll.get("components", {}).get("colliders", []):
            cd = dict(cd); cd["is_trigger"] = True; cd["center"] = [0.0, 0.0, 0.0]
            coll_dicts.append(cd)

        leaf    = {"node_name": coll_name, "godot_type": "_ColliderLeaf",
                   "transform": {"position": [0.0, 0.0, 0.0],
                                 "rotation": [0.0, 0.0, 0.0, 1.0], "scale": c_scale},
                   "components": {"colliders": coll_dicts}, "children": []}
        wrapper = {"node_name": coll_name, "godot_type": "_ColliderWrapper",
                   "transform": {"position": c_pos, "rotation": c_rot,
                                 "scale": [1.0, 1.0, 1.0]},
                   "components": {}, "children": [leaf]}
        new_children.append(wrapper)

    node["children"] = new_children + other_kids
    return node


def _apply_collider_pull_up(
    node: Dict[str, Any],
    decisions: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Rule 2 — remaining CollisionShape3D children on any non-Area3D parent.

    Identity rotation   → pull collider up as a component (centre = shape's local pos).
    Non-identity rotation → keep as a separate Empty GO wrapper; log a DECISION entry.
    """
    children  = node.get("children", [])
    coll_kids = [c for c in children if c.get("godot_type") == "CollisionShape3D"]
    if not coll_kids:
        return node

    node  = dict(node)
    comps = dict(node.get("components", {}))
    keep: List[Dict[str, Any]] = [c for c in children
                                   if c.get("godot_type") != "CollisionShape3D"]

    for ck in coll_kids:
        c_rot = _rot_as_list(ck.get("transform", {}).get("rotation", [0.0,0.0,0.0,1.0]))
        if _is_identity_rot(c_rot):
            comps["colliders"] = comps.get("colliders", []) + _collider_from_shape_node(ck)
        else:
            if decisions is not None:
                decisions.append({
                    "type":      "collider_separate_go",
                    "universal": False,
                    "message":   (
                        f"'{ck.get('node_name','?')}': CollisionShape3D has non-identity "
                        f"rotation — kept as a separate Empty GameObject with collider component"
                    ),
                    "context": {"node": ck.get("node_name", "?")},
                })
            keep.append(ck)

    node["components"] = comps
    node["children"]   = keep
    return node


def _apply_animator_pull_up(
    node: Dict[str, Any],
    decisions: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Rule N2 — AnimationPlayer children contribute their animator component to parent.

    Mirrors CollisionShape3D pull-up: the child node is removed and its
    ``animator`` component is hoisted onto the parent GameObject.
    Only the first AnimationPlayer's animator is taken when multiple exist.
    """
    children  = node.get("children", [])
    anim_kids = [c for c in children if c.get("godot_type") == "AnimationPlayer"]
    if not anim_kids:
        return node

    node  = dict(node)
    comps = dict(node.get("components", {}))
    keep  = [c for c in children if c.get("godot_type") != "AnimationPlayer"]

    for ak in anim_kids:
        ak_comps = ak.get("components", {})
        if "animator" in ak_comps and "animator" not in comps:
            comps["animator"] = ak_comps["animator"]

    node["components"] = comps
    node["children"]   = keep
    return node


def _preprocess_nodes(
    nodes: List[Dict[str, Any]],
    decisions: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Recursively apply structural rules to IR nodes before Unity emission (bottom-up).

    Order per node:
      1. Recurse into children first.
      2. Area3D restructuring       (rules 1, 7, 8).
      3. Mesh+body+collider collapse (rules 4, 5, 6): pattern A then B.
      4. Remaining CollisionShape3D pull-up (rule 2).
    """
    result = []
    for node in nodes:
        node = dict(node)
        node["children"] = _preprocess_nodes(node.get("children", []), decisions)
        gtype = node.get("godot_type", "")
        if gtype == "Area3D":
            node = _apply_area3d_rules(node, decisions)
        elif gtype in _PHYSICS_BODY_TYPES:
            node = _apply_mesh_body_collapse_pattern_a(node, decisions)
        elif gtype == "MeshInstance3D":
            node = _apply_mesh_body_collapse_pattern_b(node, decisions)
        if gtype != "Area3D":
            node = _apply_collider_pull_up(node, decisions)
        node = _apply_animator_pull_up(node, decisions)
        result.append(node)
    return result


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------

class UnitySceneExporter:
    """Convert a scene IR dict into a Unity 6000.3.9f1 .unity scene file."""

    def __init__(self) -> None:
        self._decision_log: Optional[List[Dict[str, Any]]] = None
        self._warning_log:  Optional[List[str]]            = None
        self._current_node_name: str = ""
        self._written_signal_stubs: set = set()

    def _record(self, type_: str, message: str, universal: bool = False, **ctx) -> None:
        """Append a decision entry to the active decision log, if any."""
        if self._decision_log is not None:
            self._decision_log.append({
                "type":      type_,
                "message":   message,
                "universal": universal,
                "context":   ctx,
            })

    def export(
        self,
        scene_ir:            Dict[str, Any],
        output_dir:          Path,
        project_name:        str            = "ConvertedGodotProject",
        godot_relative_dir:  Optional[Path] = None,
        instance_guid_map:     Optional[Dict[str, str]] = None,
        mesh_guid_map:       Optional[Dict[str, str]] = None,
        layer_map:           Optional[Dict[int, int]] = None,
        decisions:           Optional[List[Dict[str, Any]]] = None,
        warnings:            Optional[List[str]]            = None,
        prefab_node_id_map:  Optional[Dict[str, Dict[str, int]]] = None,
    ) -> Path:
        """Write the Unity scene file and return its path.

        If *godot_relative_dir* is given the output mirrors the original
        Godot directory layout::

            <output_dir>/Assets/<godot_relative_dir>/<scene_name>.unity

        Otherwise falls back to::

            <output_dir>/Assets/Scenes/<scene_name>.unity
        """
        self._decision_log = decisions
        self._warning_log  = warnings
        self._written_signal_stubs = set()
        scene_name = scene_ir.get("scene_name", "Scene") or "Scene"
        if godot_relative_dir is not None:
            scenes_dir = output_dir / "Assets" / godot_relative_dir
        else:
            scenes_dir = output_dir / "Assets" / "Scenes"
        scenes_dir.mkdir(parents=True, exist_ok=True)

        scene_path = scenes_dir / f"{scene_name}.unity"
        meta_path  = scenes_dir / f"{scene_name}.unity.meta"

        # Stable GUID derived from output-relative path
        try:
            rel = scene_path.relative_to(output_dir).as_posix()
        except ValueError:
            rel = scene_path.name
        scene_guid = _stable_guid(rel)

        ids      = _IdCounter()
        objects: List[str] = []

        # Pre-compute controller GUID so the Animator component can reference it
        # before the .controller asset is written.
        animations      = scene_ir.get("animations", [])
        controller_guid: Optional[str] = (
            _stable_guid(f"Assets/Animations/{scene_name}/{scene_name}.controller")
            if animations else None
        )

        # Scene settings preamble (minimal)
        objects.append(self._render_settings(ids))
        objects.append(self._occlusion_settings(ids))
        objects.append(self._lightmap_settings(ids))

        # Traverse IR nodes (structural rules applied before emission)
        for node in _preprocess_nodes(scene_ir.get("nodes", []), self._decision_log):
            try:
                self._emit_node(
                    node, parent_transform_id=0, ids=ids, out=objects,
                    output_dir=output_dir, instance_guid_map=instance_guid_map,
                    mesh_guid_map=mesh_guid_map,
                    controller_guid=controller_guid,
                    layer_map=layer_map,
                    prefab_node_id_map=prefab_node_id_map,
                )
            except Exception as exc:
                name = node.get("node_name", "Node")
                t    = node.get("transform", {})
                if node.get("components", {}):
                    log.error(
                        "[G2U] Root node '%s' (%s) failed to emit: %s"
                        " — substituting empty GameObject",
                        name, node.get("godot_type", "?"), exc,
                    )
                go_id = ids.next()
                tr_id = ids.next()
                objects.append(self._transform(
                    tr_id, go_id,
                    t.get("position", [0, 0, 0]),
                    t.get("rotation", [0.0, 0.0, 0.0, 1.0]),
                    t.get("scale",    [1, 1, 1]),
                    0, [],
                ))
                objects.append(self._game_object(go_id, name, [tr_id]))

        scene_text = _YAML_HEADER + "\n".join(objects)
        scene_path.write_text(scene_text, encoding="utf-8")
        meta_path.write_text(self._meta_scene_content(scene_guid), encoding="utf-8")
        write_folder_meta(scenes_dir, output_dir)

        # Export animation assets after node emission
        if animations and output_dir:
            clip_pairs = self._export_animation_clips(animations, output_dir, scene_name)
            self._export_animator_controller(scene_name, clip_pairs, output_dir)

        log.info(
            "exported Unity scene  %s  (%d objects)",
            scene_path.name, len(objects),
        )
        return scene_path

    def export_instance(
        self,
        scene_ir:           Dict[str, Any],
        output_dir:         Path,
        project_name:       str            = "ConvertedGodotProject",
        godot_relative_dir: Optional[Path] = None,
        mesh_guid_map:      Optional[Dict[str, str]] = None,
        layer_map:          Optional[Dict[int, int]] = None,
        decisions:          Optional[List[Dict[str, Any]]] = None,
        warnings:           Optional[List[str]]            = None,
    ) -> Tuple[Path, str]:
        """Write a Unity .prefab file and return (path, guid).

        The GUID is pre-generated so callers can reference the prefab asset
        in a companion .unity scene (BOTH classification case).

        If *godot_relative_dir* is given the output mirrors the original
        Godot directory layout::

            <output_dir>/Assets/<godot_relative_dir>/<scene_name>.prefab

        Otherwise falls back to::

            <output_dir>/Assets/Prefabs/<scene_name>.prefab
        """
        self._decision_log = decisions
        self._warning_log  = warnings
        self._written_signal_stubs = set()
        scene_name  = scene_ir.get("scene_name", "Prefab") or "Prefab"
        if godot_relative_dir is not None:
            prefabs_dir = output_dir / "Assets" / godot_relative_dir
        else:
            prefabs_dir = output_dir / "Assets" / "Prefabs"
        prefabs_dir.mkdir(parents=True, exist_ok=True)

        prefab_path = prefabs_dir / f"{scene_name}.prefab"
        meta_path   = prefabs_dir / f"{scene_name}.prefab.meta"

        # Stable GUID derived from output-relative path
        try:
            rel = prefab_path.relative_to(output_dir).as_posix()
        except ValueError:
            rel = prefab_path.name
        guid = _stable_guid(rel)

        ids      = _IdCounter()
        objects: List[str] = []

        # Pre-compute controller GUID before node emission (same as export())
        animations      = scene_ir.get("animations", [])
        controller_guid: Optional[str] = (
            _stable_guid(f"Assets/Animations/{scene_name}/{scene_name}.controller")
            if animations else None
        )

        # Unity prefabs require exactly one root GameObject.
        # If the IR has multiple top-level nodes, wrap them under a synthetic root
        # so the prefab is never exported with multiple roots (which Unity rejects).
        nodes = _preprocess_nodes(scene_ir.get("nodes", []), self._decision_log)
        if len(nodes) > 1:
            log.warning(
                "[export_instance] %s has %d root nodes — wrapping under '%sRoot'",
                scene_name, len(nodes), scene_name,
            )
            nodes = [{
                "node_name":  scene_name + "Root",
                "godot_type": "Node3D",
                "transform":  {
                    "position": [0.0, 0.0, 0.0],
                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                    "scale":    [1.0, 1.0, 1.0],
                },
                "components": {},
                "children":   nodes,
            }]

        # Collector maps interior_path → transform_fileID for all nodes in
        # this prefab.  Callers (the pipeline) store this so scene export can
        # reference interior nodes via m_AddedGameObjects.
        node_id_map: Dict[str, int] = {}

        for node in nodes:
            try:
                self._emit_node(
                    node, parent_transform_id=0, ids=ids, out=objects,
                    output_dir=output_dir, mesh_guid_map=mesh_guid_map,
                    controller_guid=controller_guid,
                    layer_map=layer_map,
                    current_path="",
                    node_id_collector=node_id_map,
                )
            except Exception as exc:
                name = node.get("node_name", "Node")
                t    = node.get("transform", {})
                if node.get("components", {}):
                    log.error(
                        "[G2U] Root node '%s' (%s) failed to emit: %s"
                        " — substituting empty GameObject",
                        name, node.get("godot_type", "?"), exc,
                    )
                go_id = ids.next()
                tr_id = ids.next()
                objects.append(self._transform(
                    tr_id, go_id,
                    t.get("position", [0, 0, 0]),
                    t.get("rotation", [0.0, 0.0, 0.0, 1.0]),
                    t.get("scale",    [1, 1, 1]),
                    0, [],
                ))
                objects.append(self._game_object(go_id, name, [tr_id]))

        prefab_text = _YAML_HEADER + "\n".join(objects)
        prefab_path.write_text(prefab_text, encoding="utf-8")
        meta_path.write_text(self._meta_prefab_content(guid), encoding="utf-8")
        write_folder_meta(prefabs_dir, output_dir)

        # Export animation assets after node emission
        if animations and output_dir:
            clip_pairs = self._export_animation_clips(animations, output_dir, scene_name)
            self._export_animator_controller(scene_name, clip_pairs, output_dir)

        log.info(
            "exported Unity prefab  %s  (%d objects, %d interior node IDs)",
            prefab_path.name, len(objects), len(node_id_map),
        )
        return prefab_path, guid, node_id_map

    def export_scene_from_prefab(
        self,
        scene_ir:           Dict[str, Any],
        output_dir:         Path,
        prefab_guid:        str,
        project_name:       str            = "ConvertedGodotProject",
        godot_relative_dir: Optional[Path] = None,
    ) -> Path:
        """Write a .unity scene whose root object is a single PrefabInstance.

        Used for the BOTH classification case: the prefab already contains the
        full hierarchy; the scene simply instantiates it rather than duplicating
        the IR nodes.  Generates the three Unity-required objects:
          - PrefabInstance  (the instantiation record)
          - stripped Transform  (root transform placeholder, required by 2019.3+)
          - stripped GameObject (root GO placeholder, required by 2019.3+)

        Output path mirrors *godot_relative_dir* when provided, otherwise falls
        back to Assets/Scenes/.
        """
        scene_name = scene_ir.get("scene_name", "Scene") or "Scene"
        if godot_relative_dir is not None:
            scenes_dir = output_dir / "Assets" / godot_relative_dir
        else:
            scenes_dir = output_dir / "Assets" / "Scenes"
        scenes_dir.mkdir(parents=True, exist_ok=True)

        scene_path = scenes_dir / f"{scene_name}.unity"
        meta_path  = scenes_dir / f"{scene_name}.unity.meta"

        # Stable GUID derived from output-relative path
        try:
            rel = scene_path.relative_to(output_dir).as_posix()
        except ValueError:
            rel = scene_path.name
        scene_guid = _stable_guid(rel)

        ids = _IdCounter()
        objects: List[str] = []

        # Scene settings preamble (IDs 100_000_000 – 100_000_002)
        objects.append(self._render_settings(ids))
        objects.append(self._occlusion_settings(ids))
        objects.append(self._lightmap_settings(ids))

        # Allocate contiguous IDs for the prefab instance + its stripped stubs.
        # Unity 2019.3+ requires the two stripped stubs even for root-level instances.
        pi_id          = ids.next()   # 100_000_003
        stripped_tr_id = ids.next()   # 100_000_004
        stripped_go_id = ids.next()   # 100_000_005

        objects.append(self._prefab_instance_block(prefab_guid, pi_id))
        objects.append(self._stripped_transform_stub(stripped_tr_id, pi_id, prefab_guid))
        objects.append(self._stripped_game_object_stub(stripped_go_id, pi_id, prefab_guid))

        scene_text = _YAML_HEADER + "\n".join(objects)
        scene_path.write_text(scene_text, encoding="utf-8")
        meta_path.write_text(self._meta_scene_content(scene_guid), encoding="utf-8")
        write_folder_meta(scenes_dir, output_dir)

        log.info("exported Unity scene (prefab-root)  %s", scene_path.name)
        return scene_path

    @staticmethod
    def _emit_prefab_instance_block(
        pi_id:              int,
        node:               Dict[str, Any],
        prefab_guid:        str,
        parent_tr_id:       int = 0,
        added_game_objects: Optional[List[tuple]] = None,
    ) -> str:
        """Emit a PrefabInstance YAML block for an _InstanceNode.

        Uses m_Modifications to carry the instance's position/rotation/scale,
        referencing the prefab's root Transform (fileID 100000001) and root
        GameObject (fileID 100000000) as targets.
        """
        t    = node.get("transform", {})
        pos  = t.get("position", [0.0, 0.0, 0.0])
        raw  = t.get("rotation", [0.0, 0.0, 0.0, 1.0])
        if isinstance(raw, (list, tuple)):
            rx, ry, rz, rw = float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])
        else:
            rx, ry, rz, rw = float(raw.get("x",0)), float(raw.get("y",0)), float(raw.get("z",0)), float(raw.get("w",1))
        rot  = {"x": rx, "y": ry, "z": -rz, "w": rw}
        scl  = t.get("scale",    [1.0, 1.0, 1.0])
        name = node.get("node_name", "Node")
        ex, ey, ez = _quat_to_euler_degrees(rot["x"], rot["y"], rot["z"], rot["w"])

        root_tr = _PREFAB_ROOT_TRANSFORM_ID
        root_go = _PREFAB_ROOT_GO_ID

        def mod(fid: int, prop: str, val: Any) -> str:
            return (
                f"    - target: {{fileID: {fid}, guid: {prefab_guid}, type: 3}}\n"
                f"      propertyPath: {prop}\n"
                f"      value: {val}\n"
                f"      objectReference: {{fileID: 0}}"
            )

        mods = "\n".join([
            mod(root_tr, "m_LocalPosition.x", pos[0]),
            mod(root_tr, "m_LocalPosition.y", pos[1]),
            mod(root_tr, "m_LocalPosition.z", -pos[2]),
            mod(root_tr, "m_LocalScale.x",    scl[0]),
            mod(root_tr, "m_LocalScale.y",    scl[1]),
            mod(root_tr, "m_LocalScale.z",    scl[2]),
            mod(root_tr, "m_LocalRotation.x", rot.get("x", 0.0)),
            mod(root_tr, "m_LocalRotation.y", rot.get("y", 0.0)),
            mod(root_tr, "m_LocalRotation.z", rot.get("z", 0.0)),
            mod(root_tr, "m_LocalRotation.w", rot.get("w", 1.0)),
            mod(root_tr, "m_LocalEulerAnglesHint.x", round(ex, 4)),
            mod(root_tr, "m_LocalEulerAnglesHint.y", round(ey, 4)),
            mod(root_tr, "m_LocalEulerAnglesHint.z", round(ez, 4)),
            mod(root_tr, "m_ConstrainProportionsScale", 1),
            mod(root_go, "m_Name", name),
        ])
        parent_ref = f"{{fileID: {parent_tr_id}}}"
        if added_game_objects:
            ago_lines = "\n".join(
                f"    - targetCorrespondingSourceObject: "
                f"{{fileID: {src_fid}, guid: {src_guid}, type: 3}}\n"
                f"      insertIndex: -1\n"
                f"      addedObject: {{fileID: {obj_fid}}}"
                for src_fid, src_guid, obj_fid in added_game_objects
            )
            ago_block = f"    m_AddedGameObjects:\n{ago_lines}\n"
        else:
            ago_block = "    m_AddedGameObjects: []\n"
        return (
            f"--- !u!{_CID_PREFAB_INSTANCE} &{pi_id}\n"
            f"PrefabInstance:\n"
            f"  m_ObjectHideFlags: 0\n"
            f"  serializedVersion: 2\n"
            f"  m_Modification:\n"
            f"    serializedVersion: 3\n"
            f"    m_TransformParent: {parent_ref}\n"
            f"    m_Modifications:\n"
            f"{mods}\n"
            f"    m_RemovedComponents: []\n"
            f"    m_RemovedGameObjects: []\n"
            f"{ago_block}"
            f"    m_AddedComponents: []\n"
            f"  m_SourcePrefab: {{fileID: 100100000, guid: {prefab_guid}, type: 3}}\n"
        )

    @staticmethod
    def _prefab_instance_block(prefab_guid: str, pi_id: int = 100_000_000) -> str:
        """Emit a PrefabInstance YAML block that references *prefab_guid*.

        *pi_id* must be unique within the scene file.  The caller should
        allocate it via ids.next() rather than relying on the default.
        """
        return (
            f"--- !u!{_CID_PREFAB_INSTANCE} &{pi_id}\n"
            f"PrefabInstance:\n"
            f"  m_ObjectHideFlags: 0\n"
            f"  serializedVersion: 2\n"
            f"  m_Modification:\n"
            f"    serializedVersion: 3\n"
            f"    m_TransformParent: {{fileID: 0}}\n"
            f"    m_Modifications: []\n"
            f"    m_RemovedComponents: []\n"
            f"    m_RemovedGameObjects: []\n"
            f"    m_AddedGameObjects: []\n"
            f"    m_AddedComponents: []\n"
            f"  m_SourcePrefab: {{fileID: 100100000, guid: {prefab_guid}, type: 3}}\n"
        )

    @staticmethod
    def _stripped_game_object_stub(
        stripped_go_id: int,
        pi_id:          int,
        prefab_guid:    str,
        root_go_id:     int = _PREFAB_ROOT_GO_ID,
    ) -> str:
        """Emit a 'stripped' GameObject placeholder for a scene-level PrefabInstance.

        Unity 2019.3+ requires this alongside the stripped Transform stub so the
        scene serialiser can resolve the prefab root's GameObject by fileID.
        """
        return (
            f"--- !u!{_CID_GAME_OBJECT} &{stripped_go_id} stripped\n"
            f"GameObject:\n"
            f"  m_CorrespondingSourceObject: "
            f"{{fileID: {root_go_id}, guid: {prefab_guid}, type: 3}}\n"
            f"  m_PrefabInstance: {{fileID: {pi_id}}}\n"
            f"  m_PrefabAsset: {{fileID: 0}}\n"
        )

    # ----------------------------------------------------------------- private

    def _emit_node(
        self,
        node:                Dict[str, Any],
        parent_transform_id: int,
        ids:                 _IdCounter,
        out:                 List[str],
        output_dir:          Optional[Path] = None,
        instance_guid_map:     Optional[Dict[str, str]] = None,
        mesh_guid_map:       Optional[Dict[str, str]] = None,
        controller_guid:     Optional[str] = None,
        layer_map:           Optional[Dict[int, int]] = None,
        prefab_node_id_map:  Optional[Dict[str, Dict[str, int]]] = None,
        current_path:        str = "",
        node_id_collector:   Optional[Dict[str, int]] = None,
    ) -> None:
        self._current_node_name = node.get("node_name", "")
        # Emit a PrefabInstance for an external model asset reference.
        asset_ref = node.get("asset_reference")
        if asset_ref and node.get("instancing", {}).get("mode") == "instance":
            guid = self._resolve_asset_guid(asset_ref["path"], mesh_guid_map, output_dir)
            if guid:
                pi_id          = ids.next()
                stripped_tr_id = ids.next()
                out.append(self._model_prefab_instance_block(
                    pi_id, node, guid, parent_transform_id,
                ))
                out.append(self._stripped_transform_stub(
                    stripped_tr_id, pi_id, guid,
                    root_transform_id=_MODEL_ROOT_TRANSFORM_ID,
                ))
                for child in node.get("children", []):
                    c_tr_id = ids.next()
                    c_go_id = ids.next()
                    self._emit_node_with_ids(
                        child, c_go_id, c_tr_id, stripped_tr_id,
                        ids, out, output_dir=output_dir,
                        instance_guid_map=instance_guid_map,
                        mesh_guid_map=mesh_guid_map,
                        controller_guid=controller_guid,
                        layer_map=layer_map,
                    )
                return
            # GUID unavailable — fall through to regular emit (embed fallback)
            node = dict(node)
            node.pop("asset_reference", None)
            node.pop("instancing", None)
            node["components"] = dict(node.get("components", {}))
            node["components"]["mesh"] = {
                "mesh_type":        "static",
                "mesh_source_path": "res://" + asset_ref["path"],
            }

        # Emit a PrefabInstance block for instanced child scenes
        if node.get("godot_type") == "_InstanceNode" and instance_guid_map:
            pi_info  = node.get("components", {}).get("instance_ref", {})
            res_path = pi_info.get("source_res_path", "")
            guid     = instance_guid_map.get(res_path)
            if guid:
                pi_id = ids.next()
                # Collect any _InstanceNode descendants that need to be added
                # into this prefab's interior via m_AddedGameObjects.
                added_game_objects: List[tuple] = []
                nested_blocks: List[str] = []
                node_ids = (prefab_node_id_map or {}).get(res_path, {})
                for interior_parent_path, child_inst in _collect_nested_instances(
                    node.get("children", []), ""
                ):
                    child_pi_info = child_inst.get("components", {}).get("instance_ref", {})
                    child_res = child_pi_info.get("source_res_path", "")
                    child_guid = instance_guid_map.get(child_res) if child_res else None
                    if not child_guid:
                        continue
                    interior_src_fid = (
                        node_ids.get(interior_parent_path)
                        if interior_parent_path else _PREFAB_ROOT_TRANSFORM_ID
                    )
                    if interior_src_fid is None:
                        log.warning(
                            "nested prefab '%s': interior path '%s' not in node ID map for %s",
                            child_inst.get("node_name"), interior_parent_path, res_path,
                        )
                        continue
                    child_pi_id      = ids.next()
                    int_stripped_id  = ids.next()
                    nested_blocks.append(self._emit_prefab_instance_block(
                        child_pi_id, child_inst, child_guid, int_stripped_id,
                    ))
                    nested_blocks.append(self._stripped_transform_stub(
                        int_stripped_id, pi_id, guid,
                        root_transform_id=interior_src_fid,
                    ))
                    added_game_objects.append((interior_src_fid, guid, child_pi_id))
                out.append(self._emit_prefab_instance_block(
                    pi_id, node, guid, parent_transform_id,
                    added_game_objects=added_game_objects or None,
                ))
                out.extend(nested_blocks)
                return

        go_id   = ids.next()
        tr_id   = ids.next()
        if node_id_collector is not None and current_path:
            node_id_collector[current_path] = tr_id
        t       = node.get("transform", {})
        name    = node.get("node_name", "Node")
        comps   = node.get("components", {})

        component_refs: List[int] = [tr_id]
        deferred_objects: List[str] = []

        # Pre-compute material GUID
        mat_guid = ""
        material = comps.get("material")
        if material:
            if material.get("material_type") == "external":
                mat_path = material.get("material_path", "")
                if mesh_guid_map and mat_path:
                    mat_guid = mesh_guid_map.get(mat_path, "")
            elif output_dir:
                mat_name = f"{name}_mat"
                mat_guid = self._write_material_asset(
                    material, mat_name, output_dir / "Assets" / "Materials", output_dir
                )

        # MeshFilter + MeshRenderer (with optional material)
        mesh = comps.get("mesh")
        if mesh:
            mf_id = ids.next()
            mr_id = ids.next()
            component_refs += [mf_id, mr_id]
            deferred_objects.append(self._mesh_filter(mf_id, go_id, mesh, mesh_guid_map))
            deferred_objects.append(self._mesh_renderer(mr_id, go_id, mat_guid=mat_guid))

        # SpriteRenderer (Sprite2D)
        sprite = comps.get("sprite")
        if sprite:
            sr_id = ids.next()
            component_refs.append(sr_id)
            deferred_objects.append(
                self._sprite_renderer(sr_id, go_id, sprite, mesh_guid_map, output_dir)
            )

        # Rigidbody
        rb = comps.get("rigidbody")
        if rb:
            rb_id = ids.next()
            component_refs.append(rb_id)
            deferred_objects.append(self._rigidbody(rb_id, go_id, rb))

        # Colliders
        for coll in comps.get("colliders", []):
            c_id = ids.next()
            component_refs.append(c_id)
            deferred_objects.append(self._collider(c_id, go_id, coll))

        # Camera
        cam = comps.get("camera")
        if cam:
            cam_id = ids.next()
            component_refs.append(cam_id)
            deferred_objects.append(self._camera(cam_id, go_id, cam))

        # Light
        light = comps.get("light")
        if light:
            l_id = ids.next()
            component_refs.append(l_id)
            deferred_objects.append(self._light(l_id, go_id, light))

        # AudioSource
        audio = comps.get("audio_source")
        if audio:
            a_id = ids.next()
            component_refs.append(a_id)
            deferred_objects.append(self._audio_source(a_id, go_id, audio))

        # CharacterController (from physics_body type=character)
        phys = comps.get("physics_body")
        if phys and phys.get("type") == "character":
            cc_id = ids.next()
            component_refs.append(cc_id)
            deferred_objects.append(self._character_controller(cc_id, go_id, phys))

        # Trigger collider (from Area3D trigger component)
        trigger = comps.get("trigger")
        if trigger:
            t_id = ids.next()
            component_refs.append(t_id)
            deferred_objects.append(self._trigger_collider(t_id, go_id, trigger))

        # Canvas stack for root Control (ui_element) nodes:
        # RectTransform is first (tr_id already in component_refs), then Canvas,
        # CanvasScaler, GraphicRaycaster — BEFORE the user script MonoBehaviour.
        ui = comps.get("ui")
        if ui and ui.get("ui_type") == "ui_element" and parent_transform_id == 0:
            cv_id = ids.next()
            cs_id = ids.next()
            gr_id = ids.next()
            component_refs += [cv_id, cs_id, gr_id]
            deferred_objects.append(self._canvas(cv_id, go_id, ui))
            deferred_objects.append(self._canvas_scaler(cs_id, go_id))
            deferred_objects.append(self._graphic_raycaster(gr_id, go_id))

        # Button: CanvasRenderer + Image (background) + Button MonoBehaviour
        # The synthetic "Text" child is added to child_objects after the loop below.
        _ui_img_id: Optional[int] = None
        _synth_text_tr_id: Optional[int] = None
        if ui and ui.get("element_kind") == "button":
            cr_id  = ids.next()
            img_id = ids.next()
            btn_id = ids.next()
            component_refs += [cr_id, img_id, btn_id]
            _bg = ui.get("bg_color", {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0})
            deferred_objects.append(self._canvas_renderer(cr_id, go_id))
            deferred_objects.append(self._ui_image(img_id, go_id, _bg))
            deferred_objects.append(self._ui_button(btn_id, go_id, img_id))
            _ui_img_id = img_id
            # Reserve IDs for the synthetic Text child now so we can wire
            # its tr_id into this button's RectTransform m_Children list.
            _synth_text_tr_id = ids.next()
            _synth_text_go_id = ids.next()
            _synth_text_cr_id = ids.next()
            _synth_text_tmp_id = ids.next()

        # Label / RichTextLabel: CanvasRenderer + TextMeshProUGUI
        elif ui and ui.get("element_kind") == "label":
            cr_id  = ids.next()
            tmp_id = ids.next()
            component_refs += [cr_id, tmp_id]
            _fc = ui.get("font_color", {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0})
            _fs = float(ui.get("font_size", 14))
            _txt = ui.get("text", "")
            deferred_objects.append(self._canvas_renderer(cr_id, go_id))
            deferred_objects.append(self._ui_text_mesh_pro(tmp_id, go_id, _txt, _fc, _fs))

        # MonoBehaviour script stub
        script = comps.get("script")
        if script:
            mb_id = ids.next()
            component_refs.append(mb_id)
            deferred_objects.append(self._monobehaviour(mb_id, go_id, script))

        # Signal wiring stub (from Godot [connection] sections)
        event_bindings = comps.get("event_bindings", [])
        if event_bindings and output_dir:
            eb_id = ids.next()
            component_refs.append(eb_id)
            sig_guid = self._write_signal_stub(name, event_bindings, output_dir)
            deferred_objects.append(
                self._signal_monobehaviour(eb_id, go_id, name, event_bindings, sig_guid)
            )

        # Animator (from AnimationPlayer)
        animator = comps.get("animator")
        if animator:
            an_id = ids.next()
            component_refs.append(an_id)
            deferred_objects.append(self._animator(an_id, go_id, animator, controller_guid))

        # ParticleSystem
        particle = comps.get("particle_system")
        if particle:
            ps_id = ids.next()
            component_refs.append(ps_id)
            deferred_objects.append(self._particle_system(ps_id, go_id, particle))

        # Canvas (from CanvasLayer)
        if ui and ui.get("ui_type") == "canvas":
            cv_id = ids.next()
            component_refs.append(cv_id)
            deferred_objects.append(self._canvas(cv_id, go_id, ui))

        # Children
        child_transform_ids: List[int] = []
        child_objects: List[str] = []
        for child in node.get("children", []):
            res_path = (child.get("components", {})
                            .get("instance_ref", {})
                            .get("source_res_path", ""))
            child_asset_ref = child.get("asset_reference")
            if (child.get("godot_type") == "_InstanceNode"
                    and instance_guid_map
                    and instance_guid_map.get(res_path)):
                # PrefabInstance child: emit the instance block plus a stripped
                # Transform stub so this child appears in the parent's m_Children.
                pi_id          = ids.next()
                stripped_tr_id = ids.next()
                child_transform_ids.append(stripped_tr_id)
                p_guid = instance_guid_map[res_path]
                child_objects.append(self._emit_prefab_instance_block(
                    pi_id, child, p_guid, tr_id,
                ))
                child_objects.append(self._stripped_transform_stub(
                    stripped_tr_id, pi_id, p_guid,
                ))
            elif (child_asset_ref
                  and child.get("instancing", {}).get("mode") == "instance"):
                # External model asset: emit as PrefabInstance instead of
                # inline MeshFilter/MeshRenderer.
                guid = self._resolve_asset_guid(
                    child_asset_ref["path"], mesh_guid_map, output_dir
                )
                if guid:
                    pi_id          = ids.next()
                    stripped_tr_id = ids.next()
                    child_transform_ids.append(stripped_tr_id)
                    child_objects.append(self._model_prefab_instance_block(
                        pi_id, child, guid, tr_id,
                    ))
                    child_objects.append(self._stripped_transform_stub(
                        stripped_tr_id, pi_id, guid,
                        root_transform_id=_MODEL_ROOT_TRANSFORM_ID,
                    ))
                    # Emit the model node's own children (physics bodies etc.)
                    # as regular GameObjects parented to the stripped transform.
                    for grandchild in child.get("children", []):
                        gc_tr_id = ids.next()
                        gc_go_id = ids.next()
                        self._emit_node_with_ids(
                            grandchild, gc_go_id, gc_tr_id, stripped_tr_id,
                            ids, child_objects, output_dir=output_dir,
                            instance_guid_map=instance_guid_map,
                            mesh_guid_map=mesh_guid_map,
                            controller_guid=controller_guid,
                            layer_map=layer_map,
                        )
                else:
                    # GUID unavailable — fall back to embed mode.
                    child_tr_id = ids.next()
                    child_go_id = ids.next()
                    child_transform_ids.append(child_tr_id)
                    embed_child = dict(child)
                    embed_child.pop("asset_reference", None)
                    embed_child.pop("instancing", None)
                    embed_child["components"] = dict(child.get("components", {}))
                    embed_child["components"]["mesh"] = {
                        "mesh_type":        "static",
                        "mesh_source_path": "res://" + child_asset_ref["path"],
                    }
                    self._emit_node_with_ids(
                        embed_child, child_go_id, child_tr_id, tr_id,
                        ids, child_objects, output_dir=output_dir,
                        instance_guid_map=instance_guid_map,
                        mesh_guid_map=mesh_guid_map,
                        controller_guid=controller_guid,
                        layer_map=layer_map,
                    )
            else:
                child_tr_id = ids.next()
                child_go_id = ids.next()
                child_transform_ids.append(child_tr_id)
                _child_path = (
                    f"{current_path}/{child['node_name']}"
                    if current_path else child["node_name"]
                )
                self._emit_node_with_ids(
                    child, child_go_id, child_tr_id, tr_id,
                    ids, child_objects, output_dir=output_dir,
                    instance_guid_map=instance_guid_map,
                    mesh_guid_map=mesh_guid_map,
                    controller_guid=controller_guid,
                    layer_map=layer_map,
                    current_path=_child_path,
                    node_id_collector=node_id_collector,
                )

        # Synthetic Text child for Button nodes (no Godot source — generated by pipeline).
        # Must be wired into child_transform_ids before _rect_transform is emitted.
        if _synth_text_tr_id is not None:
            child_transform_ids.append(_synth_text_tr_id)
            _fc   = (ui or {}).get("font_color", {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0})
            _fs   = float((ui or {}).get("font_size", 14))
            _txt  = (ui or {}).get("text", "")
            # Text GO: [RectTransform, CanvasRenderer, TextMeshProUGUI]
            child_objects.append(
                self._game_object(_synth_text_go_id, "Text",
                                  [_synth_text_tr_id, _synth_text_cr_id, _synth_text_tmp_id],
                                  layer=5)
            )
            child_objects.append(
                self._rect_transform(_synth_text_tr_id, _synth_text_go_id,
                                     tr_id, [], stretch_fill=True)
            )
            child_objects.append(self._canvas_renderer(_synth_text_cr_id, _synth_text_go_id))
            child_objects.append(
                self._ui_text_mesh_pro(_synth_text_tmp_id, _synth_text_go_id, _txt, _fc, _fs)
            )

        # Resolve Unity layer from collision_layer bitmask (lowest-bit wins)
        node_layer = 0
        if layer_map:
            _comps = node.get("components", {})
            for _ck in ("rigidbody", "physics_body", "trigger"):
                _c = _comps.get(_ck)
                if _c and "collision_layer" in _c:
                    node_layer = layer_map.get(_c["collision_layer"], 0)
                    break

        # Fall back to visual render_layers bitmask when no physics layer set.
        # Unity m_Layer is a single integer; use the lowest set bit (0-indexed).
        # Multiple bits → stays at default 0; flag the node for review.
        if node_layer == 0:
            _rl = comps.get("render_layers")
            if _rl:
                _bitmask = _rl.get("bitmask", 1)
                if _bitmask and _bitmask != 1:
                    _lowest_bit = (_bitmask & -_bitmask).bit_length() - 1
                    if (_bitmask & (_bitmask - 1)) != 0:
                        # Multiple bits set — cannot express in a single Unity layer
                        node.setdefault("meta", {}).setdefault("warnings", []).append(
                            f"render_layers bitmask {_bitmask} has multiple bits; "
                            f"defaulting Unity m_Layer to {_lowest_bit}"
                        )
                    node_layer = _lowest_bit

        # UI nodes use layer 5 and RectTransform instead of regular Transform
        _is_ui_node = (comps.get("ui") is not None) or (comps.get("ui_layout") is not None)
        if _is_ui_node:
            node_layer = 5
            out.append(self._rect_transform(
                tr_id, go_id, parent_transform_id, child_transform_ids, comps.get("ui")
            ))
        else:
            out.append(self._transform(
                tr_id, go_id,
                t.get("position", [0, 0, 0]),
                t.get("rotation", [0.0, 0.0, 0.0, 1.0]),
                t.get("scale",    [1, 1, 1]),
                parent_transform_id,
                child_transform_ids,
            ))
        out.append(self._game_object(go_id, name, component_refs, layer=node_layer))
        out.extend(deferred_objects)
        out.extend(child_objects)

    def _emit_node_with_ids(
        self,
        node:               Dict[str, Any],
        go_id:              int,
        tr_id:              int,
        parent_tr_id:       int,
        ids:                _IdCounter,
        out:                List[str],
        output_dir:         Optional[Path] = None,
        instance_guid_map:    Optional[Dict[str, str]] = None,
        mesh_guid_map:      Optional[Dict[str, str]] = None,
        controller_guid:    Optional[str] = None,
        layer_map:          Optional[Dict[int, int]] = None,
        current_path:       str = "",
        node_id_collector:  Optional[Dict[str, int]] = None,
    ) -> None:
        """Safe wrapper: emits the node or falls back to an empty GameObject."""
        try:
            self._emit_node_with_ids_inner(
                node, go_id, tr_id, parent_tr_id, ids, out,
                output_dir=output_dir,
                instance_guid_map=instance_guid_map,
                mesh_guid_map=mesh_guid_map,
                controller_guid=controller_guid,
                layer_map=layer_map,
                current_path=current_path,
                node_id_collector=node_id_collector,
            )
        except Exception as exc:
            name  = node.get("node_name", "Node")
            t     = node.get("transform", {})
            comps = node.get("components", {})
            if comps:
                log.error(
                    "[G2U] Node '%s' (%s) failed to emit: %s"
                    " — substituting empty GameObject",
                    name, node.get("godot_type", "?"), exc,
                )
            _ui = comps.get("ui")
            _is_ui = (_ui is not None) or (comps.get("ui_layout") is not None)
            if _is_ui:
                out.append(self._rect_transform(tr_id, go_id, parent_tr_id, [], _ui))
                out.append(self._game_object(go_id, name, [tr_id], layer=5))
            else:
                out.append(self._transform(
                    tr_id, go_id,
                    t.get("position", [0, 0, 0]),
                    t.get("rotation", [0.0, 0.0, 0.0, 1.0]),
                    t.get("scale",    [1, 1, 1]),
                    parent_tr_id, [],
                ))
                out.append(self._game_object(go_id, name, [tr_id]))

    def _emit_node_with_ids_inner(
        self,
        node:               Dict[str, Any],
        go_id:              int,
        tr_id:              int,
        parent_tr_id:       int,
        ids:                _IdCounter,
        out:                List[str],
        output_dir:         Optional[Path] = None,
        instance_guid_map:    Optional[Dict[str, str]] = None,
        mesh_guid_map:      Optional[Dict[str, str]] = None,
        controller_guid:    Optional[str] = None,
        layer_map:          Optional[Dict[int, int]] = None,
        current_path:       str = "",
        node_id_collector:  Optional[Dict[str, int]] = None,
    ) -> None:
        """Like _emit_node but uses pre-allocated go_id / tr_id."""
        if node_id_collector is not None and current_path:
            node_id_collector[current_path] = tr_id
        t     = node.get("transform", {})
        name  = node.get("node_name", "Node")
        self._current_node_name = name
        comps = node.get("components", {})

        component_refs: List[int] = [tr_id]
        deferred_objects: List[str] = []

        # Pre-compute material GUID
        mat_guid = ""
        material = comps.get("material")
        if material:
            if material.get("material_type") == "external":
                mat_path = material.get("material_path", "")
                if mesh_guid_map and mat_path:
                    mat_guid = mesh_guid_map.get(mat_path, "")
            elif output_dir:
                mat_name = f"{name}_mat"
                mat_guid = self._write_material_asset(
                    material, mat_name, output_dir / "Assets" / "Materials", output_dir
                )

        mesh = comps.get("mesh")
        if mesh:
            mf_id = ids.next()
            mr_id = ids.next()
            component_refs += [mf_id, mr_id]
            deferred_objects.append(self._mesh_filter(mf_id, go_id, mesh, mesh_guid_map))
            deferred_objects.append(self._mesh_renderer(mr_id, go_id, mat_guid=mat_guid))

        sprite = comps.get("sprite")
        if sprite:
            sr_id = ids.next()
            component_refs.append(sr_id)
            deferred_objects.append(
                self._sprite_renderer(sr_id, go_id, sprite, mesh_guid_map, output_dir)
            )

        rb = comps.get("rigidbody")
        if rb:
            rb_id = ids.next()
            component_refs.append(rb_id)
            deferred_objects.append(self._rigidbody(rb_id, go_id, rb))

        for coll in comps.get("colliders", []):
            c_id = ids.next()
            component_refs.append(c_id)
            deferred_objects.append(self._collider(c_id, go_id, coll))

        cam = comps.get("camera")
        if cam:
            cam_id = ids.next()
            component_refs.append(cam_id)
            deferred_objects.append(self._camera(cam_id, go_id, cam))

        light = comps.get("light")
        if light:
            l_id = ids.next()
            component_refs.append(l_id)
            deferred_objects.append(self._light(l_id, go_id, light))

        audio = comps.get("audio_source")
        if audio:
            a_id = ids.next()
            component_refs.append(a_id)
            deferred_objects.append(self._audio_source(a_id, go_id, audio))

        phys = comps.get("physics_body")
        if phys and phys.get("type") == "character":
            cc_id = ids.next()
            component_refs.append(cc_id)
            deferred_objects.append(self._character_controller(cc_id, go_id, phys))

        trigger = comps.get("trigger")
        if trigger:
            t_id = ids.next()
            component_refs.append(t_id)
            deferred_objects.append(self._trigger_collider(t_id, go_id, trigger))

        # Canvas stack for root Control (ui_element) nodes
        ui = comps.get("ui")
        if ui and ui.get("ui_type") == "ui_element" and parent_tr_id == 0:
            cv_id = ids.next()
            cs_id = ids.next()
            gr_id = ids.next()
            component_refs += [cv_id, cs_id, gr_id]
            deferred_objects.append(self._canvas(cv_id, go_id, ui))
            deferred_objects.append(self._canvas_scaler(cs_id, go_id))
            deferred_objects.append(self._graphic_raycaster(gr_id, go_id))

        # Button: CanvasRenderer + Image (background) + Button MonoBehaviour
        _synth_text_tr_id2: Optional[int] = None
        if ui and ui.get("element_kind") == "button":
            cr_id  = ids.next()
            img_id = ids.next()
            btn_id = ids.next()
            component_refs += [cr_id, img_id, btn_id]
            _bg = ui.get("bg_color", {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0})
            deferred_objects.append(self._canvas_renderer(cr_id, go_id))
            deferred_objects.append(self._ui_image(img_id, go_id, _bg))
            deferred_objects.append(self._ui_button(btn_id, go_id, img_id))
            _synth_text_tr_id2 = ids.next()
            _synth_text_go_id2 = ids.next()
            _synth_text_cr_id2 = ids.next()
            _synth_text_tmp_id2 = ids.next()
        elif ui and ui.get("element_kind") == "label":
            cr_id  = ids.next()
            tmp_id = ids.next()
            component_refs += [cr_id, tmp_id]
            _fc = ui.get("font_color", {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0})
            _fs = float(ui.get("font_size", 14))
            _txt = ui.get("text", "")
            deferred_objects.append(self._canvas_renderer(cr_id, go_id))
            deferred_objects.append(self._ui_text_mesh_pro(tmp_id, go_id, _txt, _fc, _fs))

        script = comps.get("script")
        if script:
            mb_id = ids.next()
            component_refs.append(mb_id)
            deferred_objects.append(self._monobehaviour(mb_id, go_id, script))

        event_bindings = comps.get("event_bindings", [])
        if event_bindings and output_dir:
            eb_id = ids.next()
            component_refs.append(eb_id)
            sig_guid = self._write_signal_stub(name, event_bindings, output_dir)
            deferred_objects.append(
                self._signal_monobehaviour(eb_id, go_id, name, event_bindings, sig_guid)
            )

        animator = comps.get("animator")
        if animator:
            an_id = ids.next()
            component_refs.append(an_id)
            deferred_objects.append(self._animator(an_id, go_id, animator, controller_guid))

        particle = comps.get("particle_system")
        if particle:
            ps_id = ids.next()
            component_refs.append(ps_id)
            deferred_objects.append(self._particle_system(ps_id, go_id, particle))

        if ui and ui.get("ui_type") == "canvas":
            cv_id = ids.next()
            component_refs.append(cv_id)
            deferred_objects.append(self._canvas(cv_id, go_id, ui))

        child_transform_ids: List[int] = []
        for child in node.get("children", []):
            res_path = (child.get("components", {})
                            .get("instance_ref", {})
                            .get("source_res_path", ""))
            child_asset_ref = child.get("asset_reference")
            if (child.get("godot_type") == "_InstanceNode"
                    and instance_guid_map
                    and instance_guid_map.get(res_path)):
                pi_id          = ids.next()
                stripped_tr_id = ids.next()
                child_transform_ids.append(stripped_tr_id)
                p_guid = instance_guid_map[res_path]
                out.append(self._emit_prefab_instance_block(
                    pi_id, child, p_guid, tr_id,
                ))
                out.append(self._stripped_transform_stub(
                    stripped_tr_id, pi_id, p_guid,
                ))
            elif (child_asset_ref
                  and child.get("instancing", {}).get("mode") == "instance"):
                guid = self._resolve_asset_guid(
                    child_asset_ref["path"], mesh_guid_map, output_dir
                )
                if guid:
                    pi_id          = ids.next()
                    stripped_tr_id = ids.next()
                    child_transform_ids.append(stripped_tr_id)
                    out.append(self._model_prefab_instance_block(
                        pi_id, child, guid, tr_id,
                    ))
                    out.append(self._stripped_transform_stub(
                        stripped_tr_id, pi_id, guid,
                        root_transform_id=_MODEL_ROOT_TRANSFORM_ID,
                    ))
                    for grandchild in child.get("children", []):
                        gc_tr_id = ids.next()
                        gc_go_id = ids.next()
                        self._emit_node_with_ids(
                            grandchild, gc_go_id, gc_tr_id, stripped_tr_id,
                            ids, out, output_dir=output_dir,
                            instance_guid_map=instance_guid_map,
                            mesh_guid_map=mesh_guid_map,
                            controller_guid=controller_guid,
                            layer_map=layer_map,
                        )
                else:
                    c_tr_id = ids.next()
                    c_go_id = ids.next()
                    child_transform_ids.append(c_tr_id)
                    embed_child = dict(child)
                    embed_child.pop("asset_reference", None)
                    embed_child.pop("instancing", None)
                    embed_child["components"] = dict(child.get("components", {}))
                    embed_child["components"]["mesh"] = {
                        "mesh_type":        "static",
                        "mesh_source_path": "res://" + child_asset_ref["path"],
                    }
                    self._emit_node_with_ids(
                        embed_child, c_go_id, c_tr_id, tr_id, ids, out,
                        output_dir=output_dir, instance_guid_map=instance_guid_map,
                        mesh_guid_map=mesh_guid_map,
                        controller_guid=controller_guid,
                        layer_map=layer_map,
                    )
            else:
                c_tr_id = ids.next()
                c_go_id = ids.next()
                child_transform_ids.append(c_tr_id)
                _c_path = (
                    f"{current_path}/{child['node_name']}"
                    if current_path else child["node_name"]
                )
                self._emit_node_with_ids(
                    child, c_go_id, c_tr_id, tr_id, ids, out,
                    output_dir=output_dir, instance_guid_map=instance_guid_map,
                    mesh_guid_map=mesh_guid_map,
                    controller_guid=controller_guid,
                    layer_map=layer_map,
                    current_path=_c_path,
                    node_id_collector=node_id_collector,
                )

        # Synthetic Text child for Button nodes
        if _synth_text_tr_id2 is not None:
            child_transform_ids.append(_synth_text_tr_id2)
            _fc   = (ui or {}).get("font_color", {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0})
            _fs   = float((ui or {}).get("font_size", 14))
            _txt  = (ui or {}).get("text", "")
            out.append(
                self._game_object(_synth_text_go_id2, "Text",
                                  [_synth_text_tr_id2, _synth_text_cr_id2, _synth_text_tmp_id2],
                                  layer=5)
            )
            out.append(
                self._rect_transform(_synth_text_tr_id2, _synth_text_go_id2,
                                     tr_id, [], stretch_fill=True)
            )
            out.append(self._canvas_renderer(_synth_text_cr_id2, _synth_text_go_id2))
            out.append(
                self._ui_text_mesh_pro(_synth_text_tmp_id2, _synth_text_go_id2, _txt, _fc, _fs)
            )

        # Resolve Unity layer from collision_layer bitmask (lowest-bit wins)
        node_layer = 0
        if layer_map:
            for _ck in ("rigidbody", "physics_body", "trigger"):
                _c = comps.get(_ck)
                if _c and "collision_layer" in _c:
                    node_layer = layer_map.get(_c["collision_layer"], 0)
                    break

        # UI nodes use layer 5 and RectTransform instead of regular Transform
        _is_ui_node = (comps.get("ui") is not None) or (comps.get("ui_layout") is not None)
        if _is_ui_node:
            node_layer = 5
            out.append(self._rect_transform(
                tr_id, go_id, parent_tr_id, child_transform_ids, comps.get("ui")
            ))
        else:
            out.append(self._transform(
                tr_id, go_id,
                t.get("position", [0, 0, 0]),
                t.get("rotation", [0.0, 0.0, 0.0, 1.0]),
                t.get("scale",    [1, 1, 1]),
                parent_tr_id,
                child_transform_ids,
            ))
        out.append(self._game_object(go_id, name, component_refs, layer=node_layer))
        out.extend(deferred_objects)

    # ------------------------------------------------------ YAML block builders

    @staticmethod
    def _file_id_ref(fid: int) -> str:
        return f"{{fileID: {fid}}}"

    @staticmethod
    def _null_ref() -> str:
        return "{fileID: 0}"

    def _game_object(
        self, go_id: int, name: str, component_refs: List[int], layer: int = 0
    ) -> str:
        comp_lines = "\n".join(
            f"  - component: {self._file_id_ref(r)}" for r in component_refs
        )
        return (
            f"--- !u!{_CID_GAME_OBJECT} &{go_id}\n"
            f"GameObject:\n"
            f"  m_ObjectHideFlags: 0\n"
            f"  m_CorrespondingSourceObject: {self._null_ref()}\n"
            f"  m_PrefabInstance: {self._null_ref()}\n"
            f"  m_PrefabAsset: {self._null_ref()}\n"
            f"  serializedVersion: 6\n"
            f"  m_Component:\n"
            f"{comp_lines}\n"
            f"  m_Layer: {layer}\n"
            f"  m_Name: {name}\n"
            f"  m_TagString: Untagged\n"
            f"  m_Icon: {self._null_ref()}\n"
            f"  m_NavMeshLayer: 0\n"
            f"  m_IsActive: 1\n"
        )

    def _transform(
        self,
        tr_id:              int,
        go_id:              int,
        position:           List[float],
        rotation:           Any,
        scale:              List[float],
        parent_tr_id:       int,
        child_tr_ids:       List[int],
    ) -> str:
        px, py, pz = position[0], position[1], -position[2]
        rot = _godot_rot_to_unity(rotation)
        rx = rot["x"]
        ry = rot["y"]
        rz = rot["z"]
        rw = rot["w"]
        sx, sy, sz = scale[0], scale[1], scale[2]

        # Euler hint (approximation — not used at runtime by Unity)
        ex, ey, ez = _quat_to_euler_degrees(rx, ry, rz, rw)

        if child_tr_ids:
            children_block = (
                "  m_Children:\n"
                + "\n".join(f"  - {self._file_id_ref(c)}" for c in child_tr_ids)
                + "\n"
            )
        else:
            children_block = "  m_Children: []\n"

        parent_ref = (
            self._file_id_ref(parent_tr_id) if parent_tr_id else self._null_ref()
        )

        return (
            f"--- !u!{_CID_TRANSFORM} &{tr_id}\n"
            f"Transform:\n"
            f"  m_ObjectHideFlags: 0\n"
            f"  m_CorrespondingSourceObject: {self._null_ref()}\n"
            f"  m_PrefabInstance: {self._null_ref()}\n"
            f"  m_PrefabAsset: {self._null_ref()}\n"
            f"  m_GameObject: {self._file_id_ref(go_id)}\n"
            f"  serializedVersion: 2\n"
            f"  m_LocalRotation: {{x: {rx:.6f}, y: {ry:.6f}, z: {rz:.6f}, w: {rw:.6f}}}\n"
            f"  m_LocalPosition: {{x: {px:.6f}, y: {py:.6f}, z: {pz:.6f}}}\n"
            f"  m_LocalScale: {{x: {sx:.6f}, y: {sy:.6f}, z: {sz:.6f}}}\n"
            f"  m_ConstrainProportionsScale: 0\n"
            f"{children_block}"
            f"  m_Father: {parent_ref}\n"
            f"  m_LocalEulerAnglesHint: {{x: {ex:.4f}, y: {ey:.4f}, z: {ez:.4f}}}\n"
        )

    def _rect_transform(
        self,
        tr_id:        int,
        go_id:        int,
        parent_tr_id: int,
        child_tr_ids: List[int],
        ui:           Optional[Dict[str, Any]] = None,
        stretch_fill: bool = False,
    ) -> str:
        """Emit a RectTransform block (!u!224) for UI nodes.

        stretch_fill=True emits a full-stretch child rect (anchors 0,0→1,1,
        size 0,0, position 0,0) used for synthetic Text children.
        """
        is_root = (parent_tr_id == 0)
        if is_root:
            anchor_min   = (0.0, 0.0)
            anchor_max   = (0.0, 0.0)
            pivot        = (0.0, 0.0)
            anchored_pos = (0.0, 0.0)
            size_delta   = (0.0, 0.0)
            lscale       = (0.0, 0.0, 0.0)
        elif stretch_fill:
            # Full-stretch fill (used by synthetic text child)
            anchor_min   = (0.0, 0.0)
            anchor_max   = (1.0, 1.0)
            pivot        = (0.5, 0.5)
            anchored_pos = (0.0, 0.0)
            size_delta   = (0.0, 0.0)
            lscale       = (1.0, 1.0, 1.0)
        else:
            ui_dict = ui or {}
            preset = int(ui_dict.get("anchors_preset", -1))
            if preset in _ANCHORS_PRESET:
                preset_data = _ANCHORS_PRESET[preset]
                anchor_min  = preset_data[0]
                anchor_max  = preset_data[1]
                pivot       = preset_data[2]
            else:
                # Individual anchor values with Godot→Unity Y-flip
                al = float(ui_dict.get("anchor_left",   0.0))
                at = float(ui_dict.get("anchor_top",    0.0))
                ar = float(ui_dict.get("anchor_right",  0.0))
                ab = float(ui_dict.get("anchor_bottom", 0.0))
                anchor_min = (al, 1.0 - ab)
                anchor_max = (ar, 1.0 - at)
                pivot      = (0.5, 0.5)

            # Compute SizeDelta and AnchoredPosition from Godot offsets.
            # Godot: offsets are distances from the anchor reference point.
            #   offset_left/right → horizontal extent; offset_top/bottom → vertical extent.
            # Unity AnchoredPosition is the pivot offset from the anchor (Y-up).
            ol = float(ui_dict.get("offset_left",   0.0))
            ot = float(ui_dict.get("offset_top",    0.0))
            or_ = float(ui_dict.get("offset_right", 0.0))
            ob  = float(ui_dict.get("offset_bottom", 0.0))
            width    = or_ - ol
            height   = ob - ot
            center_x = (ol + or_) / 2.0
            center_y = (ot + ob) / 2.0  # Godot Y-down, positive = below anchor
            if width != 0.0 or height != 0.0:
                size_delta   = (width, height)
                anchored_pos = (center_x, -center_y)
            else:
                size_delta   = (0.0, 0.0)
                anchored_pos = (0.0, 0.0)
            lscale = (1.0, 1.0, 1.0)

        if child_tr_ids:
            children_block = (
                "  m_Children:\n"
                + "\n".join(f"  - {self._file_id_ref(c)}" for c in child_tr_ids)
                + "\n"
            )
        else:
            children_block = "  m_Children: []\n"

        parent_ref = (
            self._file_id_ref(parent_tr_id) if parent_tr_id else self._null_ref()
        )
        return (
            f"--- !u!{_CID_RECT_TRANSFORM} &{tr_id}\n"
            f"RectTransform:\n"
            f"  m_ObjectHideFlags: 0\n"
            f"  m_CorrespondingSourceObject: {self._null_ref()}\n"
            f"  m_PrefabInstance: {self._null_ref()}\n"
            f"  m_PrefabAsset: {self._null_ref()}\n"
            f"  m_GameObject: {self._file_id_ref(go_id)}\n"
            f"  m_LocalRotation: {{x: 0, y: 0, z: 0, w: 1}}\n"
            f"  m_LocalPosition: {{x: 0, y: 0, z: 0}}\n"
            f"  m_LocalScale: {{x: {lscale[0]}, y: {lscale[1]}, z: {lscale[2]}}}\n"
            f"  m_ConstrainProportionsScale: 0\n"
            f"{children_block}"
            f"  m_Father: {parent_ref}\n"
            f"  m_LocalEulerAnglesHint: {{x: 0, y: 0, z: 0}}\n"
            f"  m_AnchorMin: {{x: {anchor_min[0]}, y: {anchor_min[1]}}}\n"
            f"  m_AnchorMax: {{x: {anchor_max[0]}, y: {anchor_max[1]}}}\n"
            f"  m_AnchoredPosition: {{x: {anchored_pos[0]}, y: {anchored_pos[1]}}}\n"
            f"  m_SizeDelta: {{x: {size_delta[0]}, y: {size_delta[1]}}}\n"
            f"  m_Pivot: {{x: {pivot[0]}, y: {pivot[1]}}}\n"
        )

    def _mesh_filter(
        self,
        mf_id: int,
        go_id: int,
        mesh: Dict[str, Any],
        mesh_guid_map: Optional[Dict[str, str]] = None,
    ) -> str:
        path = mesh.get("mesh_source_path", "")
        if mesh_guid_map and path:
            guid = mesh_guid_map.get(path, "")
            if not guid:
                self._record(
                    "mesh_unresolved",
                    f"Node '{self._current_node_name}': mesh '{path}' not in asset map "
                    f"→ MeshFilter left empty",
                    node=self._current_node_name, path=path,
                )
            # fileID 4300000 is the standard Unity sub-asset ID for the first mesh
            # in an imported FBX/OBJ file.
            mesh_ref = (f"{{fileID: 4300000, guid: {guid}, type: 3}}"
                        if guid else self._null_ref())
        else:
            mesh_ref = self._null_ref()
        return (
            f"--- !u!{_CID_MESH_FILTER} &{mf_id}\n"
            f"MeshFilter:\n"
            f"  m_ObjectHideFlags: 0\n"
            f"  m_CorrespondingSourceObject: {self._null_ref()}\n"
            f"  m_PrefabInstance: {self._null_ref()}\n"
            f"  m_PrefabAsset: {self._null_ref()}\n"
            f"  m_GameObject: {self._file_id_ref(go_id)}\n"
            f"  m_Mesh: {mesh_ref}\n"
            f"  # source_mesh_path: {path}\n"
        )

    def _mesh_renderer(self, mr_id: int, go_id: int, mat_guid: str = "") -> str:
        if mat_guid:
            mat_ref = f"{{fileID: 2100000, guid: {mat_guid}, type: 2}}"
        else:
            mat_ref = "{fileID: 10303, guid: 0000000000000000f000000000000000, type: 0}"
        return (
            f"--- !u!{_CID_MESH_RENDERER} &{mr_id}\n"
            f"MeshRenderer:\n"
            f"  m_ObjectHideFlags: 0\n"
            f"  m_CorrespondingSourceObject: {self._null_ref()}\n"
            f"  m_PrefabInstance: {self._null_ref()}\n"
            f"  m_PrefabAsset: {self._null_ref()}\n"
            f"  m_GameObject: {self._file_id_ref(go_id)}\n"
            f"  m_Enabled: 1\n"
            f"  m_CastShadows: 1\n"
            f"  m_ReceiveShadows: 1\n"
            f"  m_DynamicOccludee: 1\n"
            f"  m_Materials:\n"
            f"  - {mat_ref}\n"
        )

    def _rigidbody(
        self, rb_id: int, go_id: int, rb: Dict[str, Any]
    ) -> str:
        mass        = rb.get("mass",         1.0)
        drag        = rb.get("drag",         0.0)
        ang_drag    = rb.get("angular_drag", 0.05)
        kinematic   = int(rb.get("is_kinematic", False))
        use_gravity = int(rb.get("use_gravity",  True))
        return (
            f"--- !u!{_CID_RIGIDBODY} &{rb_id}\n"
            f"Rigidbody:\n"
            f"  m_ObjectHideFlags: 0\n"
            f"  m_CorrespondingSourceObject: {self._null_ref()}\n"
            f"  m_PrefabInstance: {self._null_ref()}\n"
            f"  m_PrefabAsset: {self._null_ref()}\n"
            f"  m_GameObject: {self._file_id_ref(go_id)}\n"
            f"  serializedVersion: 2\n"
            f"  m_Mass: {mass}\n"
            f"  m_Drag: {drag}\n"
            f"  m_AngularDrag: {ang_drag}\n"
            f"  m_UseGravity: {use_gravity}\n"
            f"  m_IsKinematic: {kinematic}\n"
            f"  m_Interpolate: 0\n"
            f"  m_CollisionDetection: 0\n"
            f"  m_Constraints: 0\n"
        )

    def _collider(
        self, c_id: int, go_id: int, coll: Dict[str, Any]
    ) -> str:
        ctype = coll.get("collider_type", "box")
        trigger = int(coll.get("is_trigger", False))
        cx, cy, cz = (coll.get("center") or [0, 0, 0])

        if ctype == "sphere":
            r = coll.get("radius", 0.5)
            return (
                f"--- !u!{_CID_SPHERE_COLLIDER} &{c_id}\n"
                f"SphereCollider:\n"
                f"  m_ObjectHideFlags: 0\n"
                f"  m_CorrespondingSourceObject: {self._null_ref()}\n"
                f"  m_PrefabInstance: {self._null_ref()}\n"
                f"  m_PrefabAsset: {self._null_ref()}\n"
                f"  m_GameObject: {self._file_id_ref(go_id)}\n"
                f"  m_Material: {self._null_ref()}\n"
                f"  m_IncludeLayers: {{m_Bits: 0}}\n"
                f"  m_ExcludeLayers: {{m_Bits: 0}}\n"
                f"  m_LayerOverridePriority: 0\n"
                f"  m_IsTrigger: {trigger}\n"
                f"  m_Enabled: 1\n"
                f"  serializedVersion: 3\n"
                f"  m_Radius: {r}\n"
                f"  m_Center: {{x: {cx}, y: {cy}, z: {cz}}}\n"
            )
        elif ctype in ("concave_mesh", "convex_mesh"):
            return self._mesh_collider(c_id, go_id, coll)
        elif ctype == "capsule":
            r = coll.get("radius", 0.5)
            h = coll.get("height", 2.0)
            d = coll.get("direction", 1)
            return (
                f"--- !u!{_CID_CAPSULE_COLLIDER} &{c_id}\n"
                f"CapsuleCollider:\n"
                f"  m_ObjectHideFlags: 0\n"
                f"  m_CorrespondingSourceObject: {self._null_ref()}\n"
                f"  m_PrefabInstance: {self._null_ref()}\n"
                f"  m_PrefabAsset: {self._null_ref()}\n"
                f"  m_GameObject: {self._file_id_ref(go_id)}\n"
                f"  m_Material: {self._null_ref()}\n"
                f"  m_IncludeLayers: {{m_Bits: 0}}\n"
                f"  m_ExcludeLayers: {{m_Bits: 0}}\n"
                f"  m_LayerOverridePriority: 0\n"
                f"  m_IsTrigger: {trigger}\n"
                f"  m_Enabled: 1\n"
                f"  serializedVersion: 2\n"
                f"  m_Radius: {r}\n"
                f"  m_Height: {h}\n"
                f"  m_Direction: {d}\n"
                f"  m_Center: {{x: {cx}, y: {cy}, z: {cz}}}\n"
            )
        else:  # box (default)
            if ctype not in ("box",):
                _msg = (
                    f"Node '{self._current_node_name}': CollisionShape3D type '{ctype}' "
                    f"could not be established — custom shape recreation is not supported; "
                    f"defaulted to BoxCollider"
                )
                log.warning(_msg)
                if self._warning_log is not None:
                    self._warning_log.append(_msg)
            size = coll.get("size", [1, 1, 1])
            hx, hy, hz = size[0]/2, size[1]/2, size[2]/2
            return (
                f"--- !u!{_CID_BOX_COLLIDER} &{c_id}\n"
                f"BoxCollider:\n"
                f"  m_ObjectHideFlags: 0\n"
                f"  m_CorrespondingSourceObject: {self._null_ref()}\n"
                f"  m_PrefabInstance: {self._null_ref()}\n"
                f"  m_PrefabAsset: {self._null_ref()}\n"
                f"  m_GameObject: {self._file_id_ref(go_id)}\n"
                f"  m_Material: {self._null_ref()}\n"
                f"  m_IncludeLayers: {{m_Bits: 0}}\n"
                f"  m_ExcludeLayers: {{m_Bits: 0}}\n"
                f"  m_LayerOverridePriority: 0\n"
                f"  m_IsTrigger: {trigger}\n"
                f"  m_Enabled: 1\n"
                f"  serializedVersion: 2\n"
                f"  m_Size: {{x: {hx*2:.4f}, y: {hy*2:.4f}, z: {hz*2:.4f}}}\n"
                f"  m_Center: {{x: {cx}, y: {cy}, z: {cz}}}\n"
            )

    def _mesh_collider(
        self, c_id: int, go_id: int, coll: Dict[str, Any]
    ) -> str:
        trigger = int(coll.get("is_trigger", False))
        convex  = 1 if coll.get("collider_type") == "convex_mesh" else 0
        cx, cy, cz = (coll.get("center") or [0, 0, 0])
        return (
            f"--- !u!{_CID_MESH_COLLIDER} &{c_id}\n"
            f"MeshCollider:\n"
            f"  m_ObjectHideFlags: 0\n"
            f"  m_CorrespondingSourceObject: {self._null_ref()}\n"
            f"  m_PrefabInstance: {self._null_ref()}\n"
            f"  m_PrefabAsset: {self._null_ref()}\n"
            f"  m_GameObject: {self._file_id_ref(go_id)}\n"
            f"  m_Material: {self._null_ref()}\n"
            f"  m_IncludeLayers: {{m_Bits: 0}}\n"
            f"  m_ExcludeLayers: {{m_Bits: 0}}\n"
            f"  m_LayerOverridePriority: 0\n"
            f"  m_IsTrigger: {trigger}\n"
            f"  m_Enabled: 1\n"
            f"  serializedVersion: 4\n"
            f"  m_Convex: {convex}\n"
            f"  m_CookingOptions: 30\n"
            f"  m_Mesh: {self._null_ref()}\n"
        )

    def _camera(
        self, cam_id: int, go_id: int, cam: Dict[str, Any]
    ) -> str:
        fov   = cam.get("fov", 60.0)
        near  = cam.get("near_clip", 0.3)
        far   = cam.get("far_clip",  1000.0)
        return (
            f"--- !u!{_CID_CAMERA} &{cam_id}\n"
            f"Camera:\n"
            f"  m_ObjectHideFlags: 0\n"
            f"  m_CorrespondingSourceObject: {self._null_ref()}\n"
            f"  m_PrefabInstance: {self._null_ref()}\n"
            f"  m_PrefabAsset: {self._null_ref()}\n"
            f"  m_GameObject: {self._file_id_ref(go_id)}\n"
            f"  m_Enabled: 1\n"
            f"  serializedVersion: 2\n"
            f"  m_ClearFlags: 1\n"
            f"  m_BackGroundColor: {{r: 0.19215, g: 0.30196, b: 0.47451, a: 0}}\n"
            f"  m_projectionMatrixMode: 1\n"
            f"  m_NearClipPlane: {near}\n"
            f"  m_FarClipPlane: {far}\n"
            f"  m_FieldOfView: {fov}\n"
            f"  m_Orthographic: 0\n"
            f"  m_OrthographicSize: 5\n"
            f"  m_Depth: -1\n"
            f"  m_RenderingPath: -1\n"
            f"  m_TargetTexture: {self._null_ref()}\n"
            f"  m_TargetDisplay: 0\n"
        )

    def _light(self, light_id: int, go_id: int, light: Dict[str, Any]) -> str:
        ltype     = _LIGHT_TYPE_INT.get(light.get("light_type", "point"), 2)
        color     = light.get("color", {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0})
        intensity = float(light.get("intensity", 1.0))
        if light.get("light_type", "point") == "point":
            intensity /= 3.0
        range_    = float(light.get("range", 10.0))
        spot_ang  = float(light.get("spot_angle", 30.0))
        shadow_t  = 2 if light.get("cast_shadows") else 0
        return (
            f"--- !u!{_CID_LIGHT} &{light_id}\n"
            f"Light:\n"
            f"  m_ObjectHideFlags: 0\n"
            f"  m_CorrespondingSourceObject: {self._null_ref()}\n"
            f"  m_PrefabInstance: {self._null_ref()}\n"
            f"  m_PrefabAsset: {self._null_ref()}\n"
            f"  m_GameObject: {self._file_id_ref(go_id)}\n"
            f"  m_Enabled: 1\n"
            f"  serializedVersion: 10\n"
            f"  m_Type: {ltype}\n"
            f"  m_Shape: 0\n"
            f"  m_Color: {{r: {color['r']:.4f}, g: {color['g']:.4f}, "
            f"b: {color['b']:.4f}, a: {color['a']:.4f}}}\n"
            f"  m_Intensity: {intensity}\n"
            f"  m_Range: {range_}\n"
            f"  m_SpotAngle: {spot_ang}\n"
            f"  m_InnerSpotAngle: 21.80208\n"
            f"  m_Shadows:\n"
            f"    m_Type: {shadow_t}\n"
            f"    m_Resolution: -1\n"
            f"    m_CustomResolution: -1\n"
            f"    m_Strength: 1\n"
            f"    m_Bias: 0.05\n"
            f"    m_NormalBias: 0.4\n"
            f"    m_NearPlane: 0.2\n"
            f"  m_Lightmapping: 4\n"
            f"  m_AreaSize: {{x: 1, y: 1}}\n"
            f"  m_BounceIntensity: 1\n"
            f"  m_ColorTemperature: 6570\n"
            f"  m_UseColorTemperature: 0\n"
            f"  m_RenderMode: 0\n"
            f"  m_CullingMask:\n"
            f"    serializedVersion: 2\n"
            f"    m_Bits: 4294967295\n"
        )

    def _audio_source(
        self, as_id: int, go_id: int, audio: Dict[str, Any]
    ) -> str:
        # Convert volume_db → linear (clamp to [0, 1])
        vol_db     = float(audio.get("volume", 0.0))
        vol_linear = max(0.0, min(1.0, 10.0 ** (vol_db / 20.0)))
        autoplay   = int(bool(audio.get("autoplay", False)))
        loop_      = int(bool(audio.get("loop",     False)))
        max_dist   = float(audio.get("max_distance", 500.0))
        return (
            f"--- !u!{_CID_AUDIO_SOURCE} &{as_id}\n"
            f"AudioSource:\n"
            f"  m_ObjectHideFlags: 0\n"
            f"  m_CorrespondingSourceObject: {self._null_ref()}\n"
            f"  m_PrefabInstance: {self._null_ref()}\n"
            f"  m_PrefabAsset: {self._null_ref()}\n"
            f"  m_GameObject: {self._file_id_ref(go_id)}\n"
            f"  m_Enabled: 1\n"
            f"  serializedVersion: 4\n"
            f"  OutputAudioMixerGroup: {self._null_ref()}\n"
            f"  m_audioClip: {self._null_ref()}\n"
            f"  m_PlayOnAwake: {autoplay}\n"
            f"  m_Volume: {vol_linear:.4f}\n"
            f"  m_Pitch: 1\n"
            f"  Loop: {loop_}\n"
            f"  Mute: 0\n"
            f"  Spatialize: 1\n"
            f"  SpatializePostEffects: 0\n"
            f"  Priority: 128\n"
            f"  DopplerLevel: 1\n"
            f"  MinDistance: 1\n"
            f"  MaxDistance: {max_dist}\n"
            f"  Pan2D: 0\n"
            f"  rolloffMode: 0\n"
            f"  BypassEffects: 0\n"
            f"  BypassListenerEffects: 0\n"
            f"  BypassReverbZones: 0\n"
        )

    def _animator(
        self, anim_id: int, go_id: int, animator: Dict[str, Any],
        controller_guid: Optional[str] = None,
    ) -> str:
        ctrl_ref = (
            f"{{fileID: 9100000, guid: {controller_guid}, type: 2}}"
            if controller_guid else self._null_ref()
        )
        return (
            f"--- !u!{_CID_ANIMATOR} &{anim_id}\n"
            f"Animator:\n"
            f"  m_ObjectHideFlags: 0\n"
            f"  m_CorrespondingSourceObject: {self._null_ref()}\n"
            f"  m_PrefabInstance: {self._null_ref()}\n"
            f"  m_PrefabAsset: {self._null_ref()}\n"
            f"  m_GameObject: {self._file_id_ref(go_id)}\n"
            f"  m_Enabled: 1\n"
            f"  m_Avatar: {self._null_ref()}\n"
            f"  m_Controller: {ctrl_ref}\n"
            f"  m_CullingMode: 0\n"
            f"  m_UpdateMode: 0\n"
            f"  m_ApplyRootMotion: 0\n"
            f"  m_LinearVelocityBlending: 0\n"
            f"  m_HasTransformHierarchy: 1\n"
            f"  m_AllowConstantClipSamplingOptimization: 1\n"
            f"  m_KeepAnimatorStateOnDisable: 0\n"
            f"  # godot_root_node: {animator.get('root_node', '.')}\n"
            f"  # godot_autoplay: {animator.get('autoplay', '')}\n"
        )

    def _canvas(self, canvas_id: int, go_id: int, ui: Dict[str, Any]) -> str:
        return (
            f"--- !u!{_CID_CANVAS} &{canvas_id}\n"
            f"Canvas:\n"
            f"  m_ObjectHideFlags: 0\n"
            f"  m_CorrespondingSourceObject: {self._null_ref()}\n"
            f"  m_PrefabInstance: {self._null_ref()}\n"
            f"  m_PrefabAsset: {self._null_ref()}\n"
            f"  m_GameObject: {self._file_id_ref(go_id)}\n"
            f"  m_Enabled: 1\n"
            f"  serializedVersion: 3\n"
            f"  m_RenderMode: 0\n"
            f"  m_Camera: {self._null_ref()}\n"
            f"  m_PlaneDistance: 100\n"
            f"  m_PixelPerfect: 0\n"
            f"  m_ReceivesEvents: 1\n"
            f"  m_OverrideSorting: 0\n"
            f"  m_OverridePixelPerfect: 0\n"
            f"  m_SortingBucketNormalizedSize: 0\n"
            f"  m_VertexColorAlwaysGammaSpace: 0\n"
            f"  m_AdditionalShaderChannelsFlag: 25\n"
            f"  m_UpdateRectTransformForStandalone: 0\n"
            f"  m_SortingLayerID: 0\n"
            f"  m_SortingOrder: 0\n"
            f"  m_TargetDisplay: 0\n"
        )

    def _canvas_scaler(self, cs_id: int, go_id: int) -> str:
        """Emit a CanvasScaler MonoBehaviour using the built-in Unity UI GUID."""
        return (
            f"--- !u!{_CID_MONOBEHAVIOUR} &{cs_id}\n"
            f"MonoBehaviour:\n"
            f"  m_ObjectHideFlags: 0\n"
            f"  m_CorrespondingSourceObject: {self._null_ref()}\n"
            f"  m_PrefabInstance: {self._null_ref()}\n"
            f"  m_PrefabAsset: {self._null_ref()}\n"
            f"  m_GameObject: {self._file_id_ref(go_id)}\n"
            f"  m_Enabled: 1\n"
            f"  m_EditorHideFlags: 0\n"
            f"  m_Script: {{fileID: 11500000, guid: {_CANVAS_SCALER_GUID}, type: 3}}\n"
            f"  m_Name: \n"
            f"  m_EditorClassIdentifier: UnityEngine.UI::UnityEngine.UI.CanvasScaler\n"
            f"  m_UiScaleMode: 0\n"
            f"  m_ReferencePixelsPerUnit: 100\n"
            f"  m_ScaleFactor: 1\n"
            f"  m_ReferenceResolution: {{x: 800, y: 600}}\n"
            f"  m_ScreenMatchMode: 0\n"
            f"  m_MatchWidthOrHeight: 0\n"
            f"  m_PhysicalUnit: 3\n"
            f"  m_FallbackScreenDPI: 96\n"
            f"  m_DefaultSpriteDPI: 96\n"
            f"  m_DynamicPixelsPerUnit: 1\n"
            f"  m_PresetInfoIsWorld: 0\n"
        )

    def _graphic_raycaster(self, gr_id: int, go_id: int) -> str:
        """Emit a GraphicRaycaster MonoBehaviour using the built-in Unity UI GUID."""
        return (
            f"--- !u!{_CID_MONOBEHAVIOUR} &{gr_id}\n"
            f"MonoBehaviour:\n"
            f"  m_ObjectHideFlags: 0\n"
            f"  m_CorrespondingSourceObject: {self._null_ref()}\n"
            f"  m_PrefabInstance: {self._null_ref()}\n"
            f"  m_PrefabAsset: {self._null_ref()}\n"
            f"  m_GameObject: {self._file_id_ref(go_id)}\n"
            f"  m_Enabled: 1\n"
            f"  m_EditorHideFlags: 0\n"
            f"  m_Script: {{fileID: 11500000, guid: {_GRAPHIC_RAYCASTER_GUID}, type: 3}}\n"
            f"  m_Name: \n"
            f"  m_EditorClassIdentifier: UnityEngine.UI::UnityEngine.UI.GraphicRaycaster\n"
            f"  m_IgnoreReversedGraphics: 1\n"
            f"  m_BlockingObjects: 0\n"
            f"  m_BlockingMask:\n"
            f"    serializedVersion: 2\n"
            f"    m_Bits: 4294967295\n"
        )

    def _canvas_renderer(self, cr_id: int, go_id: int) -> str:
        return (
            f"--- !u!{_CID_CANVAS_RENDERER} &{cr_id}\n"
            f"CanvasRenderer:\n"
            f"  m_ObjectHideFlags: 0\n"
            f"  m_CorrespondingSourceObject: {self._null_ref()}\n"
            f"  m_PrefabInstance: {self._null_ref()}\n"
            f"  m_PrefabAsset: {self._null_ref()}\n"
            f"  m_GameObject: {self._file_id_ref(go_id)}\n"
            f"  m_CullTransparentMesh: 1\n"
        )

    def _ui_image(self, img_id: int, go_id: int, color: Dict[str, float]) -> str:
        r = color.get("r", 1.0)
        g = color.get("g", 1.0)
        b = color.get("b", 1.0)
        a = color.get("a", 1.0)
        return (
            f"--- !u!{_CID_MONOBEHAVIOUR} &{img_id}\n"
            f"MonoBehaviour:\n"
            f"  m_ObjectHideFlags: 0\n"
            f"  m_CorrespondingSourceObject: {self._null_ref()}\n"
            f"  m_PrefabInstance: {self._null_ref()}\n"
            f"  m_PrefabAsset: {self._null_ref()}\n"
            f"  m_GameObject: {self._file_id_ref(go_id)}\n"
            f"  m_Enabled: 1\n"
            f"  m_EditorHideFlags: 0\n"
            f"  m_Script: {{fileID: 11500000, guid: {_UI_IMAGE_GUID}, type: 3}}\n"
            f"  m_Name: \n"
            f"  m_EditorClassIdentifier: UnityEngine.UI::UnityEngine.UI.Image\n"
            f"  m_Material: {self._null_ref()}\n"
            f"  m_Color: {{r: {r}, g: {g}, b: {b}, a: {a}}}\n"
            f"  m_RaycastTarget: 1\n"
            f"  m_RaycastPadding: {{x: 0, y: 0, z: 0, w: 0}}\n"
            f"  m_Maskable: 1\n"
            f"  m_OnCullStateChanged:\n"
            f"    m_PersistentCalls:\n"
            f"      m_Calls: []\n"
            f"  m_Sprite: {{fileID: 10905, guid: 0000000000000000f000000000000000, type: 0}}\n"
            f"  m_Type: 1\n"
            f"  m_PreserveAspect: 0\n"
            f"  m_FillCenter: 1\n"
            f"  m_FillMethod: 4\n"
            f"  m_FillAmount: 1\n"
            f"  m_FillClockwise: 1\n"
            f"  m_FillOrigin: 0\n"
            f"  m_UseSpriteMesh: 0\n"
            f"  m_PixelsPerUnitMultiplier: 1\n"
        )

    def _ui_button(self, btn_id: int, go_id: int, target_graphic_id: int) -> str:
        return (
            f"--- !u!{_CID_MONOBEHAVIOUR} &{btn_id}\n"
            f"MonoBehaviour:\n"
            f"  m_ObjectHideFlags: 0\n"
            f"  m_CorrespondingSourceObject: {self._null_ref()}\n"
            f"  m_PrefabInstance: {self._null_ref()}\n"
            f"  m_PrefabAsset: {self._null_ref()}\n"
            f"  m_GameObject: {self._file_id_ref(go_id)}\n"
            f"  m_Enabled: 1\n"
            f"  m_EditorHideFlags: 0\n"
            f"  m_Script: {{fileID: 11500000, guid: {_UI_BUTTON_GUID}, type: 3}}\n"
            f"  m_Name: \n"
            f"  m_EditorClassIdentifier: UnityEngine.UI::UnityEngine.UI.Button\n"
            f"  m_Navigation:\n"
            f"    m_Mode: 3\n"
            f"    m_WrapAround: 0\n"
            f"    m_SelectOnUp: {self._null_ref()}\n"
            f"    m_SelectOnDown: {self._null_ref()}\n"
            f"    m_SelectOnLeft: {self._null_ref()}\n"
            f"    m_SelectOnRight: {self._null_ref()}\n"
            f"  m_Transition: 1\n"
            f"  m_Colors:\n"
            f"    m_NormalColor: {{r: 1, g: 1, b: 1, a: 1}}\n"
            f"    m_HighlightedColor: {{r: 0.9607843, g: 0.9607843, b: 0.9607843, a: 1}}\n"
            f"    m_PressedColor: {{r: 0.78431374, g: 0.78431374, b: 0.78431374, a: 1}}\n"
            f"    m_SelectedColor: {{r: 0.9607843, g: 0.9607843, b: 0.9607843, a: 1}}\n"
            f"    m_DisabledColor: {{r: 0.78431374, g: 0.78431374, b: 0.78431374, a: 0.5019608}}\n"
            f"    m_ColorMultiplier: 1\n"
            f"    m_FadeDuration: 0.1\n"
            f"  m_SpriteState:\n"
            f"    m_HighlightedSprite: {self._null_ref()}\n"
            f"    m_PressedSprite: {self._null_ref()}\n"
            f"    m_SelectedSprite: {self._null_ref()}\n"
            f"    m_DisabledSprite: {self._null_ref()}\n"
            f"  m_AnimationTriggers:\n"
            f"    m_NormalTrigger: Normal\n"
            f"    m_HighlightedTrigger: Highlighted\n"
            f"    m_PressedTrigger: Pressed\n"
            f"    m_SelectedTrigger: Selected\n"
            f"    m_DisabledTrigger: Disabled\n"
            f"  m_Interactable: 1\n"
            f"  m_TargetGraphic: {self._file_id_ref(target_graphic_id)}\n"
            f"  m_OnClick:\n"
            f"    m_PersistentCalls:\n"
            f"      m_Calls: []\n"
        )

    def _ui_text_mesh_pro(
        self,
        tmp_id:     int,
        go_id:      int,
        text:       str,
        font_color: Dict[str, float],
        font_size:  float,
    ) -> str:
        r = font_color.get("r", 1.0)
        g = font_color.get("g", 1.0)
        b = font_color.get("b", 1.0)
        a = font_color.get("a", 1.0)
        return (
            f"--- !u!{_CID_MONOBEHAVIOUR} &{tmp_id}\n"
            f"MonoBehaviour:\n"
            f"  m_ObjectHideFlags: 0\n"
            f"  m_CorrespondingSourceObject: {self._null_ref()}\n"
            f"  m_PrefabInstance: {self._null_ref()}\n"
            f"  m_PrefabAsset: {self._null_ref()}\n"
            f"  m_GameObject: {self._file_id_ref(go_id)}\n"
            f"  m_Enabled: 1\n"
            f"  m_EditorHideFlags: 0\n"
            f"  m_Script: {{fileID: 11500000, guid: {_TMP_UGUI_GUID}, type: 3}}\n"
            f"  m_Name: \n"
            f"  m_EditorClassIdentifier: Unity.TextMeshPro::TMPro.TextMeshProUGUI\n"
            f"  m_Material: {self._null_ref()}\n"
            f"  m_Color: {{r: 1, g: 1, b: 1, a: 1}}\n"
            f"  m_RaycastTarget: 1\n"
            f"  m_RaycastPadding: {{x: 0, y: 0, z: 0, w: 0}}\n"
            f"  m_Maskable: 1\n"
            f"  m_OnCullStateChanged:\n"
            f"    m_PersistentCalls:\n"
            f"      m_Calls: []\n"
            f"  m_text: '{text}'\n"
            f"  m_isRightToLeft: 0\n"
            f"  m_fontAsset: {{fileID: 11400000, guid: {_TMP_FONT_GUID}, type: 2}}\n"
            f"  m_sharedMaterial: {{fileID: 2180264, guid: {_TMP_FONT_GUID}, type: 2}}\n"
            f"  m_fontSharedMaterials: []\n"
            f"  m_fontMaterial: {self._null_ref()}\n"
            f"  m_fontMaterials: []\n"
            f"  m_fontColor32:\n"
            f"    serializedVersion: 2\n"
            f"    rgba: 4294967295\n"
            f"  m_fontColor: {{r: {r}, g: {g}, b: {b}, a: {a}}}\n"
            f"  m_enableVertexGradient: 0\n"
            f"  m_colorMode: 3\n"
            f"  m_fontSize: {font_size}\n"
            f"  m_fontSizeBase: {font_size}\n"
            f"  m_fontWeight: 400\n"
            f"  m_enableAutoSizing: 0\n"
            f"  m_fontSizeMin: 18\n"
            f"  m_fontSizeMax: 72\n"
            f"  m_fontStyle: 0\n"
            f"  m_HorizontalAlignment: 2\n"
            f"  m_VerticalAlignment: 512\n"
            f"  m_textAlignment: 65535\n"
            f"  m_TextWrappingMode: 1\n"
            f"  m_overflowMode: 0\n"
            f"  m_isRichText: 1\n"
            f"  m_isOrthographic: 1\n"
            f"  m_margin: {{x: 0, y: 0, z: 0, w: 0}}\n"
        )

    def _character_controller(
        self, cc_id: int, go_id: int, phys: Dict[str, Any]
    ) -> str:
        """Emit a CharacterController component (Unity CID 97)."""
        return (
            f"--- !u!{_CID_CHARACTER_CONTROLLER} &{cc_id}\n"
            f"CharacterController:\n"
            f"  m_ObjectHideFlags: 0\n"
            f"  m_CorrespondingSourceObject: {self._null_ref()}\n"
            f"  m_PrefabInstance: {self._null_ref()}\n"
            f"  m_PrefabAsset: {self._null_ref()}\n"
            f"  m_GameObject: {self._file_id_ref(go_id)}\n"
            f"  m_Material: {self._null_ref()}\n"
            f"  m_IncludeLayers: {{m_Bits: 0}}\n"
            f"  m_ExcludeLayers: {{m_Bits: 0}}\n"
            f"  m_LayerOverridePriority: 0\n"
            f"  m_IsTrigger: 0\n"
            f"  m_Enabled: 1\n"
            f"  serializedVersion: 3\n"
            f"  m_Height: 2\n"
            f"  m_Radius: 0.5\n"
            f"  m_SlopeLimit: 45\n"
            f"  m_StepOffset: 0.3\n"
            f"  m_SkinWidth: 0.08\n"
            f"  m_MinMoveDistance: 0.001\n"
            f"  m_Center: {{x: 0, y: 1, z: 0}}\n"
        )

    def _trigger_collider(
        self, t_id: int, go_id: int, trigger: Dict[str, Any]
    ) -> str:
        """Emit a BoxCollider with m_IsTrigger: 1 (Unity Area3D equivalent)."""
        return (
            f"--- !u!{_CID_BOX_COLLIDER} &{t_id}\n"
            f"BoxCollider:\n"
            f"  m_ObjectHideFlags: 0\n"
            f"  m_CorrespondingSourceObject: {self._null_ref()}\n"
            f"  m_PrefabInstance: {self._null_ref()}\n"
            f"  m_PrefabAsset: {self._null_ref()}\n"
            f"  m_GameObject: {self._file_id_ref(go_id)}\n"
            f"  m_Material: {self._null_ref()}\n"
            f"  m_IncludeLayers: {{m_Bits: 0}}\n"
            f"  m_ExcludeLayers: {{m_Bits: 0}}\n"
            f"  m_LayerOverridePriority: 0\n"
            f"  m_IsTrigger: 1\n"
            f"  m_Enabled: 1\n"
            f"  serializedVersion: 2\n"
            f"  m_Size: {{x: 1, y: 1, z: 1}}\n"
            f"  m_Center: {{x: 0, y: 0, z: 0}}\n"
        )

    # ---------------------------------------------------------------- signal wiring (N3)

    @staticmethod
    def _to_cs_identifier(name: str) -> str:
        """Convert an arbitrary string to a valid PascalCase C# identifier.

        Splits on underscores and non-alphanumeric characters, capitalises each
        segment, and joins them.  ``door_opening`` → ``DoorOpening``.
        """
        import re as _re
        parts  = _re.split(r"[^A-Za-z0-9]+", name)
        result = "".join(p[:1].upper() + p[1:] for p in parts if p)
        if not result:
            return "Signal"
        if result[0].isdigit():
            result = "_" + result
        return result

    def _write_signal_stub(
        self,
        node_name:     str,
        event_bindings: List[Dict[str, Any]],
        output_dir:    Path,
    ) -> str:
        """Write a C# MonoBehaviour stub with UnityEvent fields for each signal.

        Returns the stable GUID for the generated script.  Skips I/O if the
        stub was already written this export run (idempotent).
        """
        class_name   = self._to_cs_identifier(node_name) + "Signals"
        rel_path     = f"Assets/Scripts/Generated/{class_name}.cs"
        guid         = _stable_guid(rel_path)

        if rel_path in self._written_signal_stubs:
            return guid
        self._written_signal_stubs.add(rel_path)

        # Collect unique signal names in declaration order
        seen_signals: List[str] = []
        seen_set: set = set()
        for eb in event_bindings:
            sig = self._to_cs_identifier(eb.get("event_name", "signal"))
            sig = sig[:1].lower() + sig[1:]   # camelCase field name
            if sig not in seen_set:
                seen_signals.append(sig)
                seen_set.add(sig)

        # Build connection comments
        conn_comments = "\n".join(
            f"    // {eb.get('event_name','?')} → {eb.get('target','?')}.{eb.get('method','?')}"
            for eb in event_bindings
        )

        fields = "\n".join(
            f"    [SerializeField] public UnityEvent {sig};"
            for sig in seen_signals
        )

        cs_text = (
            "// Auto-generated by Godot→Unity converter — do not edit manually.\n"
            "// Original Godot signal connections:\n"
            f"{conn_comments}\n"
            "using UnityEngine;\n"
            "using UnityEngine.Events;\n\n"
            f"public class {class_name} : MonoBehaviour\n"
            "{\n"
            f"{fields}\n"
            "}\n"
        )

        scripts_dir = output_dir / "Assets" / "Scripts" / "Generated"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        cs_path   = scripts_dir / f"{class_name}.cs"
        meta_path = scripts_dir / f"{class_name}.cs.meta"

        cs_path.write_text(cs_text, encoding="utf-8")
        meta_path.write_text(
            f"fileFormatVersion: 2\n"
            f"guid: {guid}\n"
            f"MonoImporter:\n"
            f"  externalObjects: {{}}\n"
            f"  serializedVersion: 2\n"
            f"  defaultReferences: []\n"
            f"  executionOrder: 0\n"
            f"  icon: {{instanceID: 0}}\n"
            f"  userData: \n"
            f"  assetBundleName: \n"
            f"  assetBundleVariant: \n",
            encoding="utf-8",
        )
        write_folder_meta(scripts_dir, output_dir)
        return guid

    def _signal_monobehaviour(
        self,
        mb_id:          int,
        go_id:          int,
        node_name:      str,
        event_bindings: List[Dict[str, Any]],
        script_guid:    str,
    ) -> str:
        """Emit a !u!114 MonoBehaviour stub referencing the generated signal script."""
        conn_lines = "".join(
            f"  # godot_signal: {eb.get('event_name','?')} -> "
            f"{eb.get('target','?')}.{eb.get('method','?')}\n"
            for eb in event_bindings
        )
        return (
            f"--- !u!{_CID_MONOBEHAVIOUR} &{mb_id}\n"
            f"MonoBehaviour:\n"
            f"  m_ObjectHideFlags: 0\n"
            f"  m_CorrespondingSourceObject: {self._null_ref()}\n"
            f"  m_PrefabInstance: {self._null_ref()}\n"
            f"  m_PrefabAsset: {self._null_ref()}\n"
            f"  m_GameObject: {self._file_id_ref(go_id)}\n"
            f"  m_Enabled: 1\n"
            f"  m_EditorHideFlags: 0\n"
            f"  m_Script: {{fileID: 11500000, guid: {script_guid}, type: 3}}\n"
            f"  m_Name: \n"
            f"  m_EditorClassIdentifier: \n"
            f"{conn_lines}"
            f"  # godot_node: {node_name}\n"
        )

    def _monobehaviour(
        self, mb_id: int, go_id: int, script: Dict[str, Any]
    ) -> str:
        """Emit a MonoBehaviour component stub referencing the converted script."""
        class_name  = script.get("class_name", "UnknownScript")
        source_path = script.get("source_path", "")
        # Derive GUID from Unity output path so it matches the .cs.meta written
        # by the pipeline's _convert_scripts() — key must be identical.
        script_filename = Path(source_path).name if source_path else f"{class_name}.cs"
        unity_script_rel = f"Assets/Scripts/{script_filename}"
        guid = _stable_guid(unity_script_rel)
        return (
            f"--- !u!{_CID_MONOBEHAVIOUR} &{mb_id}\n"
            f"MonoBehaviour:\n"
            f"  m_ObjectHideFlags: 0\n"
            f"  m_CorrespondingSourceObject: {self._null_ref()}\n"
            f"  m_PrefabInstance: {self._null_ref()}\n"
            f"  m_PrefabAsset: {self._null_ref()}\n"
            f"  m_GameObject: {self._file_id_ref(go_id)}\n"
            f"  m_Enabled: 1\n"
            f"  m_EditorHideFlags: 0\n"
            f"  m_Script: {{fileID: 11500000, guid: {guid}, type: 3}}\n"
            f"  m_Name: \n"
            f"  m_EditorClassIdentifier: \n"
            f"  # godot_class: {class_name}\n"
            f"  # godot_source: {source_path}\n"
        )

    def _particle_system(
        self, ps_id: int, go_id: int, particle: Dict[str, Any]
    ) -> str:
        """Emit a ParticleSystem component (Unity CID 198)."""
        emitting  = int(bool(particle.get("emitting",  True)))
        duration  = float(particle.get("lifetime",     1.0))
        loop_     = int(not particle.get("one_shot",   False))
        start_lt  = float(particle.get("lifetime",     1.0))
        max_parts = int(particle.get("amount",         100))
        return (
            f"--- !u!{_CID_PARTICLE_SYSTEM} &{ps_id}\n"
            f"ParticleSystem:\n"
            f"  m_ObjectHideFlags: 0\n"
            f"  m_CorrespondingSourceObject: {self._null_ref()}\n"
            f"  m_PrefabInstance: {self._null_ref()}\n"
            f"  m_PrefabAsset: {self._null_ref()}\n"
            f"  m_GameObject: {self._file_id_ref(go_id)}\n"
            f"  serializedVersion: 6\n"
            f"  lengthInSec: {duration:.4f}\n"
            f"  simulationSpeed: 1\n"
            f"  stopAction: 0\n"
            f"  looping: {loop_}\n"
            f"  prewarm: 0\n"
            f"  playOnAwake: {emitting}\n"
            f"  maxNumParticles: {max_parts}\n"
            f"  InitialModule:\n"
            f"    startLifetime:\n"
            f"      serializedVersion: 2\n"
            f"      minMaxState: 0\n"
            f"      scalar: {start_lt:.4f}\n"
            f"    startSpeed:\n"
            f"      serializedVersion: 2\n"
            f"      minMaxState: 0\n"
            f"      scalar: 5\n"
            f"    startSize:\n"
            f"      serializedVersion: 2\n"
            f"      minMaxState: 0\n"
            f"      scalar: 1\n"
            f"    gravityModifier:\n"
            f"      serializedVersion: 2\n"
            f"      minMaxState: 0\n"
            f"      scalar: 0\n"
            f"    maxNumParticles: {max_parts}\n"
        )

    def _sprite_renderer(
        self,
        sr_id:         int,
        go_id:         int,
        sprite:        Dict[str, Any],
        mesh_guid_map: Optional[Dict[str, str]] = None,
        output_dir:    Optional[Path] = None,
    ) -> str:
        path = sprite.get("texture_path", "")
        guid = (mesh_guid_map.get(path, "") if mesh_guid_map and path else "")
        sprite_ref = (
            f"{{fileID: 21300000, guid: {guid}, type: 3}}"
            if guid else self._null_ref()
        )
        # Write sprite-mode texture meta before Stage F copies the raw file
        # so Unity imports the texture as a Sprite rather than a plain Texture2D.
        if guid and output_dir and path:
            self._write_sprite_texture_meta(path, guid, output_dir)
        return (
            f"--- !u!{_CID_SPRITE_RENDERER} &{sr_id}\n"
            f"SpriteRenderer:\n"
            f"  m_ObjectHideFlags: 0\n"
            f"  m_CorrespondingSourceObject: {self._null_ref()}\n"
            f"  m_PrefabInstance: {self._null_ref()}\n"
            f"  m_PrefabAsset: {self._null_ref()}\n"
            f"  m_GameObject: {self._file_id_ref(go_id)}\n"
            f"  m_Enabled: 1\n"
            f"  m_CastShadows: 0\n"
            f"  m_ReceiveShadows: 0\n"
            f"  m_DynamicOccludee: 1\n"
            f"  m_MotionVectors: 1\n"
            f"  m_LightProbeUsage: 1\n"
            f"  m_ReflectionProbeUsage: 1\n"
            f"  m_RayTracingMode: 2\n"
            f"  m_RayTraceProcedural: 0\n"
            f"  m_Materials:\n"
            f"  - {{fileID: 10754, guid: 0000000000000000f000000000000000, type: 0}}\n"
            f"  m_Sprite: {sprite_ref}\n"
            f"  m_Color: {{r: 1, g: 1, b: 1, a: 1}}\n"
            f"  m_FlipX: 0\n"
            f"  m_FlipY: 0\n"
            f"  m_DrawMode: 0\n"
            f"  m_Size: {{x: 1, y: 1}}\n"
            f"  m_AdaptiveModeThreshold: 0.5\n"
            f"  m_SpriteTileMode: 0\n"
            f"  m_WasSpriteAssigned: {1 if guid else 0}\n"
            f"  m_MaskInteraction: 0\n"
            f"  m_SpriteSortPoint: 0\n"
        )

    def _write_sprite_texture_meta(
        self, res_path: str, guid: str, output_dir: Path
    ) -> None:
        """Write a TextureImporter meta with spriteMode: 1 for a sprite texture.

        Written during Stage D so Stage F's _copy_assets() does not overwrite it
        (that function skips writing meta if the file already exists).
        """
        rel_str = res_path.removeprefix("res://")
        parts = Path(rel_str).parts
        flat_parts: list = []
        for p in parts:
            if flat_parts and flat_parts[-1] == p:
                continue
            flat_parts.append(p)
        rel_flat = Path(*flat_parts) if flat_parts else Path(rel_str)

        meta_path = output_dir / "Assets" / Path(str(rel_flat) + ".meta")
        if meta_path.exists():
            return
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            f"fileFormatVersion: 2\n"
            f"guid: {guid}\n"
            f"TextureImporter:\n"
            f"  internalIDToNameTable: []\n"
            f"  externalObjects: {{}}\n"
            f"  serializedVersion: 13\n"
            f"  mipmaps:\n"
            f"    mipMapMode: 0\n"
            f"    enableMipMap: 0\n"
            f"    sRGBTexture: 1\n"
            f"    linearTexture: 0\n"
            f"    fadeOut: 0\n"
            f"    borderMipMap: 0\n"
            f"    mipMapsPreserveCoverage: 0\n"
            f"    alphaTestReferenceValue: 0.5\n"
            f"    mipMapFadeDistanceStart: 1\n"
            f"    mipMapFadeDistanceEnd: 3\n"
            f"  isReadable: 0\n"
            f"  textureFormat: 1\n"
            f"  maxTextureSize: 2048\n"
            f"  textureSettings:\n"
            f"    serializedVersion: 2\n"
            f"    filterMode: 1\n"
            f"    aniso: 1\n"
            f"    mipBias: 0\n"
            f"    wrapU: 1\n"
            f"    wrapV: 1\n"
            f"    wrapW: 1\n"
            f"  nPOTScale: 0\n"
            f"  lightmap: 0\n"
            f"  compressionQuality: 50\n"
            f"  spriteMode: 1\n"
            f"  spriteExtrude: 1\n"
            f"  spriteMeshType: 1\n"
            f"  spriteAlignment: 0\n"
            f"  spritePivot: {{x: 0.5, y: 0.5}}\n"
            f"  spritePixelsToUnits: 100\n"
            f"  spriteBorder: {{x: 0, y: 0, z: 0, w: 0}}\n"
            f"  spriteGenerateFallbackPhysicsShape: 1\n"
            f"  alphaUsage: 1\n"
            f"  alphaIsTransparency: 1\n"
            f"  textureType: 8\n"
            f"  textureShape: 1\n"
            f"  userData: \n"
            f"  assetBundleName: \n"
            f"  assetBundleVariant: \n",
            encoding="utf-8",
        )

    def _write_material_asset(
        self,
        mat_ir:    Dict[str, Any],
        mat_name:  str,
        mat_dir:   Path,
        output_dir: Optional[Path] = None,
    ) -> str:
        """Write a Unity Standard .mat file; return its deterministic GUID."""
        mat_dir.mkdir(parents=True, exist_ok=True)
        if output_dir:
            write_folder_meta(mat_dir, output_dir)
        guid = _stable_guid(f"unity_mat:{mat_name}")

        albedo    = mat_ir.get("albedo_color", {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0})
        metallic  = float(mat_ir.get("metallic",  0.0))
        roughness = float(mat_ir.get("roughness", 1.0))
        smoothness = max(0.0, 1.0 - roughness)

        mat_text = (
            "%YAML 1.1\n"
            "%TAG !u! tag:unity3d.com,2011:\n"
            "--- !u!21 &2100000\n"
            "Material:\n"
            "  serializedVersion: 8\n"
            "  m_ObjectHideFlags: 0\n"
            f"  m_CorrespondingSourceObject: {self._null_ref()}\n"
            f"  m_PrefabInstance: {self._null_ref()}\n"
            f"  m_PrefabAsset: {self._null_ref()}\n"
            f"  m_Name: {mat_name}\n"
            "  m_Shader: {fileID: 46, guid: 0000000000000000f000000000000000, type: 0}\n"
            "  m_ShaderKeywords: \n"
            "  m_LightmapFlags: 4\n"
            "  m_EnableInstancingVariants: 0\n"
            "  m_DoubleSidedGI: 0\n"
            "  m_CustomRenderQueue: -1\n"
            "  stringTagMap: {}\n"
            "  disabledShaderPasses: []\n"
            "  m_SavedProperties:\n"
            "    serializedVersion: 3\n"
            "    m_TexEnvs:\n"
            "    - _MainTex:\n"
            "        m_Texture: {fileID: 0}\n"
            "        m_Scale: {x: 1, y: 1}\n"
            "        m_Offset: {x: 0, y: 0}\n"
            "    - _BumpMap:\n"
            "        m_Texture: {fileID: 0}\n"
            "        m_Scale: {x: 1, y: 1}\n"
            "        m_Offset: {x: 0, y: 0}\n"
            "    m_Floats:\n"
            f"    - _Glossiness: {smoothness:.4f}\n"
            f"    - _Metallic: {metallic:.4f}\n"
            "    m_Colors:\n"
            f"    - _Color: {{r: {albedo['r']:.4f}, g: {albedo['g']:.4f}, "
            f"b: {albedo['b']:.4f}, a: {albedo['a']:.4f}}}\n"
        )
        mat_path = mat_dir / f"{mat_name}.mat"
        mat_path.write_text(mat_text, encoding="utf-8")

        meta_path = mat_dir / f"{mat_name}.mat.meta"
        meta_path.write_text(
            f"fileFormatVersion: 2\n"
            f"guid: {guid}\n"
            f"NativeFormatImporter:\n"
            f"  externalObjects: {{}}\n"
            f"  mainObjectFileID: 2100000\n"
            f"  userData:\n"
            f"  assetBundleName:\n"
            f"  assetBundleVariant:\n",
            encoding="utf-8",
        )
        return guid

    @staticmethod
    def _stripped_transform_stub(
        stripped_tr_id: int,
        pi_id: int,
        prefab_guid: str,
        root_transform_id: int = _PREFAB_ROOT_TRANSFORM_ID,
    ) -> str:
        """Emit a 'stripped' Transform placeholder required by Unity 2019.3+ for
        PrefabInstances that are children of a regular GameObject.

        The stub lets the parent Transform's m_Children list reference the instance's
        root Transform by fileID.  *root_transform_id* defaults to the ID used by
        our own exported prefabs; pass _MODEL_ROOT_TRANSFORM_ID for imported models.
        """
        return (
            f"--- !u!{_CID_TRANSFORM} &{stripped_tr_id} stripped\n"
            f"Transform:\n"
            f"  m_CorrespondingSourceObject: "
            f"{{fileID: {root_transform_id}, guid: {prefab_guid}, type: 3}}\n"
            f"  m_PrefabInstance: {{fileID: {pi_id}}}\n"
            f"  m_PrefabAsset: {{fileID: 0}}\n"
        )

    def _model_prefab_instance_block(
        self,
        pi_id:            int,
        node:             Dict[str, Any],
        prefab_guid:      str,
        parent_tr_id:     int,
    ) -> str:
        """Emit a PrefabInstance for a node backed by an imported 3D model asset.

        All formats: C * R * C coordinate conversion (Godot right-hand → Unity left-hand).
        FBX files: additional R(-90° X) post-correction + scale ×100 (Unity cm → m).
        All formats: position Z negated.
        """
        t      = node.get("transform", {})
        pos    = t.get("position", [0.0, 0.0, 0.0])
        raw_q  = t.get("rotation", [0.0, 0.0, 0.0, 1.0])
        scl    = t.get("scale",    [1.0, 1.0, 1.0])
        name   = node.get("node_name", "Node")
        fmt    = node.get("asset_reference", {}).get("format", "")
        is_fbx = fmt == ".fbx"

        rot       = _godot_rot_to_unity(raw_q, is_fbx=is_fbx)
        scale_mul = 100 if is_fbx else 1

        ex, ey, ez = _quat_to_euler_degrees(rot["x"], rot["y"], rot["z"], rot["w"])

        def mod(fid: int, prop: str, val: Any) -> str:
            return (
                f"    - target: {{fileID: {fid}, guid: {prefab_guid}, type: 3}}\n"
                f"      propertyPath: {prop}\n"
                f"      value: {val}\n"
                f"      objectReference: {{fileID: 0}}"
            )

        mods = "\n".join([
            mod(_MODEL_ROOT_TRANSFORM_ID, "m_LocalScale.x",    scl[0] * scale_mul),
            mod(_MODEL_ROOT_TRANSFORM_ID, "m_LocalScale.y",    scl[2] * scale_mul),
            mod(_MODEL_ROOT_TRANSFORM_ID, "m_LocalScale.z",    scl[1] * scale_mul),
            mod(_MODEL_ROOT_TRANSFORM_ID, "m_LocalPosition.x", pos[0]),
            mod(_MODEL_ROOT_TRANSFORM_ID, "m_LocalPosition.y", pos[1]),
            mod(_MODEL_ROOT_TRANSFORM_ID, "m_LocalPosition.z", -pos[2]),
            mod(_MODEL_ROOT_TRANSFORM_ID, "m_LocalRotation.w", rot.get("w", 1.0)),
            mod(_MODEL_ROOT_TRANSFORM_ID, "m_LocalRotation.x", rot.get("x", 0.0)),
            mod(_MODEL_ROOT_TRANSFORM_ID, "m_LocalRotation.y", rot.get("y", 0.0)),
            mod(_MODEL_ROOT_TRANSFORM_ID, "m_LocalRotation.z", rot.get("z", 0.0)),
            mod(_MODEL_ROOT_TRANSFORM_ID, "m_LocalEulerAnglesHint.x", round(ex, 4)),
            mod(_MODEL_ROOT_TRANSFORM_ID, "m_LocalEulerAnglesHint.y", round(ey, 4)),
            mod(_MODEL_ROOT_TRANSFORM_ID, "m_LocalEulerAnglesHint.z", round(ez, 4)),
            mod(_MODEL_ROOT_TRANSFORM_ID, "m_ConstrainProportionsScale", 1),
            mod(_MODEL_ROOT_GO_ID,        "m_Name", name),
        ])
        return (
            f"--- !u!{_CID_PREFAB_INSTANCE} &{pi_id}\n"
            f"PrefabInstance:\n"
            f"  m_ObjectHideFlags: 0\n"
            f"  serializedVersion: 2\n"
            f"  m_Modification:\n"
            f"    serializedVersion: 3\n"
            f"    m_TransformParent: {{fileID: {parent_tr_id}}}\n"
            f"    m_Modifications:\n"
            f"{mods}\n"
            f"    m_RemovedComponents: []\n"
            f"    m_RemovedGameObjects: []\n"
            f"    m_AddedGameObjects: []\n"
            f"    m_AddedComponents: []\n"
            f"  m_SourcePrefab: {{fileID: 100100000, guid: {prefab_guid}, type: 3}}\n"
        )

    def _resolve_asset_guid(
        self,
        rel_path: str,
        mesh_guid_map: Optional[Dict[str, str]],
        output_dir: Optional[Path],
    ) -> str:
        """Return the GUID for an external model asset.

        Tries *mesh_guid_map* first (fast path), then reads the .meta file from
        *output_dir*/Assets if the map misses.  Returns '' with a warning when
        no GUID can be found.
        """
        res_path = "res://" + rel_path
        if mesh_guid_map:
            guid = mesh_guid_map.get(res_path, "")
            if guid:
                return guid
        if output_dir:
            meta_path = output_dir / "Assets" / rel_path
            meta_path = meta_path.parent / (meta_path.name + ".meta")
            guid = _read_meta_guid(meta_path)
            if guid:
                return guid
        log.warning("[exporter] no GUID for model asset %r — will fall back to embed", rel_path)
        return ""

    def _render_settings(self, ids: _IdCounter) -> str:
        fid = ids.next()
        return (
            f"--- !u!{_CID_RENDER_SETTINGS} &{fid}\n"
            f"RenderSettings:\n"
            f"  m_ObjectHideFlags: 0\n"
            f"  serializedVersion: 9\n"
            f"  m_Fog: 0\n"
            f"  m_AmbientMode: 0\n"
            f"  m_AmbientSkyColor: {{r: 0.212, g: 0.227, b: 0.259, a: 1}}\n"
            f"  m_AmbientEquatorColor: {{r: 0.114, g: 0.125, b: 0.133, a: 1}}\n"
            f"  m_AmbientGroundColor: {{r: 0.047, g: 0.043, b: 0.035, a: 1}}\n"
            f"  m_AmbientIntensity: 1\n"
            f"  m_DefaultReflectionMode: 0\n"
            f"  m_DefaultReflectionResolution: 128\n"
            f"  m_Sun: {self._null_ref()}\n"
        )

    def _occlusion_settings(self, ids: _IdCounter) -> str:
        fid = ids.next()
        return (
            f"--- !u!{_CID_OCCLUSION} &{fid}\n"
            f"OcclusionCullingSettings:\n"
            f"  m_ObjectHideFlags: 0\n"
            f"  serializedVersion: 2\n"
            f"  m_OcclusionBakeSettings:\n"
            f"    smallestOccluder: 5\n"
            f"    smallestHole: 0.25\n"
            f"    backfaceThreshold: 100\n"
            f"  m_SceneGUID: 00000000000000000000000000000000\n"
            f"  m_OcclusionCullingData: {self._null_ref()}\n"
        )

    def _lightmap_settings(self, ids: _IdCounter) -> str:
        fid = ids.next()
        return (
            f"--- !u!{_CID_SCENE_SETTINGS} &{fid}\n"
            f"LightmapSettings:\n"
            f"  m_ObjectHideFlags: 0\n"
            f"  serializedVersion: 12\n"
            f"  m_GIWorkflowMode: 1\n"
            f"  m_GISettings:\n"
            f"    realtimeEnvironmentLighting: 1\n"
        )

    @staticmethod
    def _meta_scene_content(guid: str) -> str:
        """Unity .unity scene .meta — uses DefaultImporter."""
        return (
            f"fileFormatVersion: 2\n"
            f"guid: {guid}\n"
            f"DefaultImporter:\n"
            f"  externalObjects: {{}}\n"
            f"  userData: \n"
            f"  assetBundleName: \n"
            f"  assetBundleVariant: \n"
        )

    @staticmethod
    def _meta_prefab_content(guid: str) -> str:
        """Unity .prefab .meta — uses PrefabImporter (not NativeFormatImporter)."""
        return (
            f"fileFormatVersion: 2\n"
            f"guid: {guid}\n"
            f"PrefabImporter:\n"
            f"  externalObjects: {{}}\n"
            f"  userData: \n"
            f"  assetBundleName: \n"
            f"  assetBundleVariant: \n"
        )

    @staticmethod
    def _meta_content(asset_name: str, prefab: bool = False, guid: str = "") -> str:
        """Legacy helper — kept for callers outside the exporter. Prefers stable guid."""
        if not guid:
            guid = _stable_guid(f"legacy_meta:{asset_name}")
        if prefab:
            return UnitySceneExporter._meta_prefab_content(guid)
        return UnitySceneExporter._meta_scene_content(guid)

    @staticmethod
    def _meta_native_content(guid: str, main_object_id: int) -> str:
        """NativeFormatImporter .meta for .anim and .controller assets."""
        return (
            f"fileFormatVersion: 2\n"
            f"guid: {guid}\n"
            f"NativeFormatImporter:\n"
            f"  externalObjects: {{}}\n"
            f"  mainObjectFileID: {main_object_id}\n"
            f"  userData: \n"
            f"  assetBundleName: \n"
            f"  assetBundleVariant: \n"
        )

    # --------------------------------------------------------- animation export

    def _export_animation_clips(
        self,
        animations: List[Dict[str, Any]],
        output_dir: Path,
        scene_name: str,
    ) -> List[Tuple[str, str]]:
        """Write one .anim file per animation; return list of (name, guid) pairs."""
        anim_dir = output_dir / "Assets" / "Animations" / scene_name
        anim_dir.mkdir(parents=True, exist_ok=True)
        write_folder_meta(anim_dir.parent, output_dir)
        write_folder_meta(anim_dir, output_dir)

        result: List[Tuple[str, str]] = []
        for anim in animations:
            name   = anim.get("name", "Clip")
            length = float(anim.get("length", 1.0))
            loop   = bool(anim.get("loop", False))
            tracks = anim.get("tracks", [])

            clip_rel  = f"Assets/Animations/{scene_name}/{name}.anim"
            clip_guid = _stable_guid(clip_rel)
            clip_path = anim_dir / f"{name}.anim"
            meta_path = anim_dir / f"{name}.anim.meta"

            pos_curves:   List[str] = []
            rot_curves:   List[str] = []
            scl_curves:   List[str] = []
            float_curves: List[str] = []

            for track in tracks:
                t_type    = track.get("type", "")
                node_path = track.get("node_path", "")
                prop      = track.get("property", "")
                keys      = track.get("keyframes", [])

                if t_type == "position_3d":
                    pos_curves.append(self._anim_vec3_curve(node_path, keys, negate_z=True))
                elif t_type == "rotation_3d":
                    rot_curves.append(self._anim_quat_curve(node_path, keys))
                elif t_type == "scale_3d":
                    scl_curves.append(self._anim_vec3_curve(node_path, keys))
                elif t_type == "value" and prop:
                    attr = _GODOT_PROP_TO_UNITY_ATTR.get(prop, prop)
                    float_curves.append(self._anim_float_curve(node_path, attr, keys))

            clip_text = self._anim_clip_yaml(
                name, length, loop, pos_curves, rot_curves, scl_curves, float_curves,
            )
            clip_path.write_text(clip_text, encoding="utf-8")
            meta_path.write_text(self._meta_native_content(clip_guid, 7400000), encoding="utf-8")
            result.append((name, clip_guid))
            log.info("exported AnimationClip  %s", clip_path.name)

        return result

    def _export_animator_controller(
        self,
        scene_name: str,
        clip_pairs: List[Tuple[str, str]],
        output_dir: Path,
    ) -> str:
        """Write a .controller asset and return its GUID.

        Creates one AnimatorState per clip.  Default state = first clip.
        All internal fileIDs are derived deterministically from scene+clip names.
        """
        ctrl_rel  = f"Assets/Animations/{scene_name}/{scene_name}.controller"
        ctrl_guid = _stable_guid(ctrl_rel)
        ctrl_path = output_dir / "Assets" / "Animations" / scene_name / f"{scene_name}.controller"
        meta_path = ctrl_path.parent / f"{scene_name}.controller.meta"

        sm_id = _stable_local_id(f"{scene_name}_sm")

        state_ids = [
            _stable_local_id(f"{scene_name}_state_{clip_name}")
            for clip_name, _ in clip_pairs
        ]

        # Child-states block inside AnimatorStateMachine
        child_states = "".join(
            f"  - serializedVersion: 1\n"
            f"    m_State: {{fileID: {sid}}}\n"
            f"    m_Position: {{x: 312, y: {120 + i * 72}, z: 0}}\n"
            for i, (sid, _) in enumerate(zip(state_ids, clip_pairs))
        )
        default_id = state_ids[0] if state_ids else 0

        # One AnimatorState block per clip
        state_blocks = "".join(
            f"--- !u!{_CID_ANIMATOR_STATE} &{sid}\n"
            f"AnimatorState:\n"
            f"  serializedVersion: 6\n"
            f"  m_ObjectHideFlags: 1\n"
            f"  m_CorrespondingSourceObject: {{fileID: 0}}\n"
            f"  m_PrefabInstance: {{fileID: 0}}\n"
            f"  m_PrefabAsset: {{fileID: 0}}\n"
            f"  m_Name: {clip_name}\n"
            f"  m_Speed: 1\n"
            f"  m_CycleOffset: 0\n"
            f"  m_Transitions: []\n"
            f"  m_StateMachineBehaviours: []\n"
            f"  m_Position: {{x: 312, y: {120 + i * 72}, z: 0}}\n"
            f"  m_IKOnFeet: 0\n"
            f"  m_WriteDefaultValues: 1\n"
            f"  m_Mirror: 0\n"
            f"  m_SpeedParameterActive: 0\n"
            f"  m_MirrorParameterActive: 0\n"
            f"  m_CycleOffsetParameterActive: 0\n"
            f"  m_TimeParameterActive: 0\n"
            f"  m_Motion: {{fileID: 7400000, guid: {clip_guid}, type: 2}}\n"
            f"  m_Tag: \n"
            f"  m_SpeedParameter: \n"
            f"  m_MirrorParameter: \n"
            f"  m_CycleOffsetParameter: \n"
            f"  m_TimeParameter: \n"
            for i, (sid, (clip_name, clip_guid)) in enumerate(zip(state_ids, clip_pairs))
        )

        ctrl_text = (
            f"%YAML 1.1\n"
            f"%TAG !u! tag:unity3d.com,2011:\n"
            f"--- !u!{_CID_ANIMATOR_CONTROLLER} &9100000\n"
            f"AnimatorController:\n"
            f"  m_ObjectHideFlags: 0\n"
            f"  m_CorrespondingSourceObject: {{fileID: 0}}\n"
            f"  m_PrefabInstance: {{fileID: 0}}\n"
            f"  m_PrefabAsset: {{fileID: 0}}\n"
            f"  m_Name: {scene_name}\n"
            f"  serializedVersion: 5\n"
            f"  m_AnimatorParameters: []\n"
            f"  m_AnimatorLayers:\n"
            f"  - serializedVersion: 5\n"
            f"    m_Name: Base Layer\n"
            f"    m_StateMachine: {{fileID: {sm_id}}}\n"
            f"    m_Mask: {{fileID: 0}}\n"
            f"    m_Motions: []\n"
            f"    m_Behaviours: []\n"
            f"    m_BlendingMode: 0\n"
            f"    m_SyncedLayerIndex: -1\n"
            f"    m_DefaultWeight: 0\n"
            f"    m_IKPass: 0\n"
            f"    m_SyncedLayerAffectsTiming: 0\n"
            f"    m_Controller: {{fileID: 9100000}}\n"
            f"--- !u!{_CID_ANIMATOR_STATE_MACHINE} &{sm_id}\n"
            f"AnimatorStateMachine:\n"
            f"  serializedVersion: 6\n"
            f"  m_ObjectHideFlags: 1\n"
            f"  m_CorrespondingSourceObject: {{fileID: 0}}\n"
            f"  m_PrefabInstance: {{fileID: 0}}\n"
            f"  m_PrefabAsset: {{fileID: 0}}\n"
            f"  m_Name: Base Layer\n"
            f"  m_ChildStates:\n"
            f"{child_states}"
            f"  m_ChildStateMachines: []\n"
            f"  m_AnyStateTransitions: []\n"
            f"  m_EntryTransitions: []\n"
            f"  m_StateMachineTransitions: {{}}\n"
            f"  m_StateMachineBehaviours: []\n"
            f"  m_AnyStatePosition: {{x: 50, y: 20, z: 0}}\n"
            f"  m_EntryPosition: {{x: 50, y: 120, z: 0}}\n"
            f"  m_ExitPosition: {{x: 800, y: 120, z: 0}}\n"
            f"  m_ParentStateMachinePosition: {{x: 800, y: 20, z: 0}}\n"
            f"  m_DefaultState: {{fileID: {default_id}}}\n"
            + state_blocks
        )

        ctrl_path.write_text(ctrl_text, encoding="utf-8")
        meta_path.write_text(self._meta_native_content(ctrl_guid, 9100000), encoding="utf-8")
        log.info("exported AnimatorController  %s", ctrl_path.name)
        return ctrl_guid

    # ---------------------------------------------------- AnimationClip YAML

    @staticmethod
    def _anim_clip_yaml(
        name: str,
        length: float,
        loop: bool,
        pos_curves: List[str],
        rot_curves: List[str],
        scl_curves: List[str],
        float_curves: List[str],
    ) -> str:
        def _block(items: List[str], key: str) -> str:
            if not items:
                return f"  {key}: []\n"
            return f"  {key}:\n" + "".join(items)

        return (
            f"%YAML 1.1\n"
            f"%TAG !u! tag:unity3d.com,2011:\n"
            f"--- !u!{_CID_ANIMATION_CLIP} &7400000\n"
            f"AnimationClip:\n"
            f"  m_ObjectHideFlags: 0\n"
            f"  m_CorrespondingSourceObject: {{fileID: 0}}\n"
            f"  m_PrefabInstance: {{fileID: 0}}\n"
            f"  m_PrefabAsset: {{fileID: 0}}\n"
            f"  m_Name: {name}\n"
            f"  serializedVersion: 7\n"
            f"  m_Legacy: 0\n"
            f"  m_Compressed: 0\n"
            f"  m_UseHighQualityCurve: 1\n"
            + _block(rot_curves,   "m_RotationCurves")
            + "  m_CompressedRotationCurves: []\n"
            + "  m_EulerCurves: []\n"
            + _block(pos_curves,   "m_PositionCurves")
            + _block(scl_curves,   "m_ScaleCurves")
            + _block(float_curves, "m_FloatCurves")
            + "  m_PPtrCurves: []\n"
            + "  m_SampleRate: 60\n"
            + "  m_WrapMode: 0\n"
            + "  m_Bounds:\n"
            + "    m_Center: {x: 0, y: 0, z: 0}\n"
            + "    m_Extent: {x: 0, y: 0, z: 0}\n"
            + "  m_ClipBindingConstant:\n"
            + "    genericBindings: []\n"
            + "    pptrCurveMapping: []\n"
            + "  m_AnimationClipSettings:\n"
            + "    serializedVersion: 2\n"
            + "    m_AdditiveReferencePoseClip: {fileID: 0}\n"
            + "    m_AdditiveReferencePoseTime: 0\n"
            + "    m_StartTime: 0\n"
            + f"    m_StopTime: {length}\n"
            + "    m_OrientationOffsetY: 0\n"
            + "    m_Level: 0\n"
            + "    m_CycleOffset: 0\n"
            + "    m_HasAdditiveReferencePose: 0\n"
            + f"    m_LoopTime: {1 if loop else 0}\n"
            + "    m_LoopBlend: 0\n"
            + "    m_LoopBlendOrientation: 0\n"
            + "    m_LoopBlendPositionY: 0\n"
            + "    m_LoopBlendPositionXZ: 0\n"
            + "    m_KeepOriginalOrientation: 0\n"
            + "    m_KeepOriginalPositionY: 1\n"
            + "    m_KeepOriginalPositionXZ: 0\n"
            + "    m_HeightFromFeet: 0\n"
            + "    m_Mirror: 0\n"
            + "  m_EditorCurves: []\n"
            + "  m_EulerEditorCurves: []\n"
            + "  m_HasGenericRootTransform: 0\n"
            + "  m_HasMotionFloatCurves: 0\n"
            + "  m_Events: []\n"
        )

    @staticmethod
    def _anim_vec3_curve(
        path: str, keyframes: List[Dict[str, Any]], negate_z: bool = False
    ) -> str:
        """Emit one position or scale curve entry (Vector3 keyframes).

        Pass negate_z=True for position tracks to apply the Godot→Unity Z flip.
        """
        if not keyframes:
            return (
                f"  - curve:\n"
                f"      m_Curve: []\n"
                f"      m_PreInfinity: 2\n"
                f"      m_PostInfinity: 2\n"
                f"      m_RotationOrder: 4\n"
                f"    path: {path}\n"
                f"    classID: 4\n"
                f"    script: {{fileID: 0}}\n"
            )
        kf_lines = "".join(
            f"      - serializedVersion: 3\n"
            f"        time: {kf['time']}\n"
            f"        value: {{x: {kf['x']}, y: {kf['y']}, z: {-kf['z'] if negate_z else kf['z']}}}\n"
            f"        inSlope: {{x: 0, y: 0, z: 0}}\n"
            f"        outSlope: {{x: 0, y: 0, z: 0}}\n"
            f"        tangentMode: 0\n"
            f"        weightedMode: 0\n"
            f"        inWeight: {{x: 0.33333334, y: 0.33333334, z: 0.33333334}}\n"
            f"        outWeight: {{x: 0.33333334, y: 0.33333334, z: 0.33333334}}\n"
            for kf in keyframes
        )
        return (
            f"  - curve:\n"
            f"      m_Curve:\n"
            f"{kf_lines}"
            f"      m_PreInfinity: 2\n"
            f"      m_PostInfinity: 2\n"
            f"      m_RotationOrder: 4\n"
            f"    path: {path}\n"
            f"    classID: 4\n"
            f"    script: {{fileID: 0}}\n"
        )

    @staticmethod
    def _anim_quat_curve(path: str, keyframes: List[Dict[str, Any]]) -> str:
        """Emit one rotation curve entry (Quaternion keyframes)."""
        if not keyframes:
            return (
                f"  - curve:\n"
                f"      m_Curve: []\n"
                f"      m_PreInfinity: 2\n"
                f"      m_PostInfinity: 2\n"
                f"      m_RotationOrder: 4\n"
                f"    path: {path}\n"
                f"    classID: 4\n"
                f"    script: {{fileID: 0}}\n"
            )
        kf_lines = ""
        for kf in keyframes:
            cq = _godot_rot_to_unity({"x": kf["x"], "y": kf["y"], "z": kf["z"], "w": kf["w"]})
            kf_lines += (
                f"      - serializedVersion: 3\n"
                f"        time: {kf['time']}\n"
                f"        value: {{x: {cq['x']:.6f}, y: {cq['y']:.6f}, z: {cq['z']:.6f}, w: {cq['w']:.6f}}}\n"
                f"        inSlope: {{x: 0, y: 0, z: 0, w: 0}}\n"
                f"        outSlope: {{x: 0, y: 0, z: 0, w: 0}}\n"
                f"        tangentMode: 0\n"
                f"        weightedMode: 0\n"
                f"        inWeight: {{x: 0.33333334, y: 0.33333334, z: 0.33333334, w: 0.33333334}}\n"
                f"        outWeight: {{x: 0.33333334, y: 0.33333334, z: 0.33333334, w: 0.33333334}}\n"
            )
        return (
            f"  - curve:\n"
            f"      m_Curve:\n"
            f"{kf_lines}"
            f"      m_PreInfinity: 2\n"
            f"      m_PostInfinity: 2\n"
            f"      m_RotationOrder: 4\n"
            f"    path: {path}\n"
            f"    classID: 4\n"
            f"    script: {{fileID: 0}}\n"
        )

    @staticmethod
    def _anim_float_curve(
        path: str, attribute: str, keyframes: List[Dict[str, Any]]
    ) -> str:
        """Emit one float curve entry (scalar value track)."""
        if not keyframes:
            return (
                f"  - curve:\n"
                f"      m_Curve: []\n"
                f"      m_PreInfinity: 2\n"
                f"      m_PostInfinity: 2\n"
                f"      m_RotationOrder: 4\n"
                f"    attribute: {attribute}\n"
                f"    path: {path}\n"
                f"    classID: 4\n"
                f"    script: {{fileID: 0}}\n"
            )
        kf_lines = "".join(
            f"      - serializedVersion: 3\n"
            f"        time: {kf['time']}\n"
            f"        value: {kf['value']}\n"
            f"        inSlope: 0\n"
            f"        outSlope: 0\n"
            f"        tangentMode: 0\n"
            f"        weightedMode: 0\n"
            f"        inWeight: 0.33333334\n"
            f"        outWeight: 0.33333334\n"
            for kf in keyframes
        )
        return (
            f"  - curve:\n"
            f"      m_Curve:\n"
            f"{kf_lines}"
            f"      m_PreInfinity: 2\n"
            f"      m_PostInfinity: 2\n"
            f"      m_RotationOrder: 4\n"
            f"    attribute: {attribute}\n"
            f"    path: {path}\n"
            f"    classID: 4\n"
            f"    script: {{fileID: 0}}\n"
        )


# ---------------------------------------------------------------------------
# Math helpers — coordinate system conversion (Godot right-hand → Unity left-hand)
# ---------------------------------------------------------------------------

_FBX_CORR: float = 0.7071067811865476  # 1/sqrt(2) — FBX correction quaternion magnitude


def _godot_rot_to_unity(
    q: Any,
    is_fbx: bool = False,
) -> Dict[str, float]:
    """Convert an IR rotation quaternion to Unity space.

    Accepts either a list [x, y, z, w] or a dict {x, y, z, w}.

    Non-FBX:  step 1  temp   = Q * (0,1,0,0)  →  (-z, w, x, -y)
              step 2  result = (-x,-y,z,w)     →  (-z, w, -x, y)
    FBX:      step 1  temp   = (-x, -y, z, w)
              step 2  result = temp * (0, 0.70710678, 0.70710678, 0)
              step 3  final  = (-x, y, z, -w)
    """
    if isinstance(q, (list, tuple)) and len(q) >= 4:
        x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    else:
        x = float(q.get("x", 0.0))
        y = float(q.get("y", 0.0))
        z = float(q.get("z", 0.0))
        w = float(q.get("w", 1.0))
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n > 1e-10:
        x, y, z, w = x/n, y/n, z/n, w/n

    if not is_fbx:
        # Step 1: Q * (0,1,0,0)  →  (-z, w, x, -y)
        # Step 2: (x,y,z,w) → (x,-y,-z,w)  →  (-z, -w, -x, -y)
        return {"x": -z, "y": w, "z": -x, "w": y}

    # temp * correction  where correction = (cx=0, cy=_FBX_CORR, cz=_FBX_CORR, cw=0)
    # final step: (x,y,z,w) -> (x,-y,z,-w)
    c = _FBX_CORR
    return {
            "x":  c * (z + y),
            "y":  c * (w + x),
            "z":  c * (w - x),
            "w":  c * (z - y),
        }


# ---------------------------------------------------------------------------
# Math helper — quaternion → Euler angles (degrees, Unity XYZ order)
# ---------------------------------------------------------------------------

def _quat_to_euler_degrees(
    x: float, y: float, z: float, w: float
) -> Tuple[float, float, float]:
    """Approximate Euler hint (XYZ) from quaternion. Used only for the hint field."""
    # Roll (X)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # Pitch (Y)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    # Yaw (Z)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    deg = math.degrees
    return deg(roll), deg(pitch), deg(yaw)
