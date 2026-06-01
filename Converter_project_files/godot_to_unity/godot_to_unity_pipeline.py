"""godot_to_unity_pipeline.py

Orchestrates the Godot 4.5 → Unity 6000.3.9f1 conversion pipeline.

Pipeline stages:
    A. Scan   — find all .tscn files in the Godot project
    B. Parse  — GodotSceneParser converts each .tscn to engine-agnostic IR
    C. Validate — basic IR sanity checks before export
    D. Export — UnitySceneExporter writes Unity .unity YAML scene files
    E. Assets  — copy passthrough assets (.fbx / .obj) to Unity Assets/

Scope:
    Supported   — Node3D hierarchy, transforms, MeshInstance3D,
                  CollisionShape3D (Box/Sphere/Capsule), RigidBody3D,
                  Camera3D, static meshes (.fbx / .obj),
                  GDScript → Unity C# (via LLM),
                  Godot C# → Unity C# (via LLM)
    Not supported — animations, lights, terrain, UI, particles, shaders,
                    materials beyond stub

Public API:
    GodotToUnityPipeline   — main entry point
    G2UConversionResult    — typed return value
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import uuid as _uuid_mod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Script pre-screening
# ---------------------------------------------------------------------------

_SCENE_KEYWORD_RE = re.compile(
    r'\.tscn|AddChild|Instantiate|ChangeSceneToFile|Load|PackedScene',
    re.IGNORECASE,
)
_LINE_COMMENT_RE  = re.compile(r'//[^\n]*')
_BLOCK_COMMENT_RE = re.compile(r'/\*.*?\*/', re.DOTALL)


def _has_scene_constructs(source: str) -> bool:
    """Return True if source contains any scene-related constructs.

    Strips line and block comments before scanning to reduce false positives.
    """
    stripped = _BLOCK_COMMENT_RE.sub('', _LINE_COMMENT_RE.sub('', source))
    return bool(_SCENE_KEYWORD_RE.search(stripped))


# ---------------------------------------------------------------------------
# .meta helpers
# ---------------------------------------------------------------------------

# Stable UUID namespace used for deterministic asset GUIDs.
_GUID_NAMESPACE = _uuid_mod.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def _stable_guid(key: str) -> str:
    """Return a deterministic 32-char hex GUID derived from *key* (e.g. rel posix path)."""
    return _uuid_mod.uuid5(_GUID_NAMESPACE, key).hex


def parse_meta_file(path: Path) -> Optional[str]:
    """Parse a Unity .meta file and return its guid value, or None."""
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if stripped.startswith("guid:"):
                value = stripped.split(":", 1)[1].strip()
                return value if value else None
    except OSError:
        pass
    return None


def _build_layer_map(collision_layers_used: List[int]) -> Dict[int, int]:
    """Map Godot collision bitmask values to Unity layer indices (1–31).

    Extracts all unique bit positions set across the given bitmask values, sorts
    them, and assigns sequential Unity layer indices starting at 1.  For masks
    with multiple bits set the lowest bit is used as the representative.
    """
    bit_positions: set = set()
    for mask in collision_layers_used:
        m = mask
        while m:
            bit_positions.add((m & -m).bit_length() - 1)
            m &= m - 1

    bit_to_layer: Dict[int, int] = {
        bit_pos: idx + 1
        for idx, bit_pos in enumerate(sorted(bit_positions)[:31])
    }

    result: Dict[int, int] = {}
    for mask in collision_layers_used:
        if mask == 0:
            result[0] = 0
        else:
            lowest_bit_pos = (mask & -mask).bit_length() - 1
            result[mask] = bit_to_layer.get(lowest_bit_pos, 0)
    return result


# File extensions → Unity importer type
_META_MODEL_EXTS    = frozenset({".fbx", ".obj", ".gltf", ".glb"})
_META_TEXTURE_EXTS  = frozenset({".png", ".jpg", ".jpeg", ".tga", ".bmp", ".exr", ".hdr"})
_META_AUDIO_EXTS    = frozenset({".wav", ".mp3", ".ogg"})


# Must stay in sync with the same constants in unity_scene_exporter.py.
_MODEL_ROOT_TRANSFORM_ID: int = -4216859302048453862
_MODEL_ROOT_GO_ID:        int = -927199367670048503


def _meta_for_asset(rel_path: Path, guid: str) -> str:
    """Return the correct .meta content string for a passthrough asset file."""
    ext = rel_path.suffix.lower()
    if ext in _META_MODEL_EXTS:
        root_name = rel_path.stem
        id_table = (
            f"  internalIDToNameTable:\n"
            f"  - first: {{type: 4, fileID: {_MODEL_ROOT_TRANSFORM_ID}}}\n"
            f"    second: {root_name}\n"
            f"  - first: {{type: 1, fileID: {_MODEL_ROOT_GO_ID}}}\n"
            f"    second: {root_name}\n"
        )
        return (
            f"fileFormatVersion: 2\n"
            f"guid: {guid}\n"
            f"ModelImporter:\n"
            f"  serializedVersion: 24200\n"
            f"{id_table}"
            f"  externalObjects: {{}}\n"
            f"  materials:\n"
            f"    materialImportMode: 2\n"
            f"    materialName: 0\n"
            f"    materialSearch: 1\n"
            f"    materialLocation: 1\n"
            f"  animations:\n"
            f"    legacyGenerateAnimations: 4\n"
            f"    bakeSimulation: 0\n"
            f"    resampleCurves: 1\n"
            f"    optimizeGameObjects: 0\n"
            f"    importAnimatedCustomProperties: 0\n"
            f"    importConstraints: 0\n"
            f"    animationCompression: 1\n"
            f"    animationRotationError: 0.5\n"
            f"    animationPositionError: 0.5\n"
            f"    animationScaleError: 0.5\n"
            f"    animationWrapMode: 0\n"
            f"    extraExposedTransformPaths: []\n"
            f"    extraUserProperties: []\n"
            f"    clipAnimations: []\n"
            f"    isReadable: 0\n"
            f"  meshes:\n"
            f"    globalScale: 1\n"
            f"    meshCompression: 0\n"
            f"    addColliders: 0\n"
            f"    useSRGBMaterialColor: 1\n"
            f"    sortHierarchyByName: 1\n"
            f"    importVisibility: 1\n"
            f"    importBlendShapes: 1\n"
            f"    importCameras: 1\n"
            f"    importLights: 1\n"
            f"    fileIdsGeneration: 1\n"
            f"    useFileUnits: 1\n"
            f"    keepQuads: 0\n"
            f"    weldVertices: 1\n"
            f"    bakeAxisConversion: 0\n"
            f"    meshOptimizationFlags: -1\n"
            f"    indexFormat: 0\n"
            f"  importAnimation: 1\n"
            f"  userData: \n"
            f"  assetBundleName: \n"
            f"  assetBundleVariant: \n"
        )
    if ext in _META_TEXTURE_EXTS:
        return (
            f"fileFormatVersion: 2\n"
            f"guid: {guid}\n"
            f"TextureImporter:\n"
            f"  internalIDToNameTable: []\n"
            f"  externalObjects: {{}}\n"
            f"  serializedVersion: 13\n"
            f"  mipmaps:\n"
            f"    mipMapMode: 0\n"
            f"    enableMipMap: 1\n"
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
            f"    wrapU: 0\n"
            f"    wrapV: 0\n"
            f"    wrapW: 0\n"
            f"  nPOTScale: 1\n"
            f"  lightmap: 0\n"
            f"  compressionQuality: 50\n"
            f"  spriteMode: 0\n"
            f"  textureType: 0\n"
            f"  textureShape: 1\n"
            f"  userData: \n"
            f"  assetBundleName: \n"
            f"  assetBundleVariant: \n"
        )
    if ext in _META_AUDIO_EXTS:
        return (
            f"fileFormatVersion: 2\n"
            f"guid: {guid}\n"
            f"AudioImporter:\n"
            f"  externalObjects: {{}}\n"
            f"  serializedVersion: 6\n"
            f"  defaultSettings:\n"
            f"    loadType: 0\n"
            f"    sampleRateSetting: 0\n"
            f"    sampleRateOverride: 44100\n"
            f"    compressionFormat: 1\n"
            f"    quality: 1\n"
            f"    conversionMode: 0\n"
            f"  forceToMono: 0\n"
            f"  normalize: 0\n"
            f"  preloadAudioData: 1\n"
            f"  loadInBackground: 0\n"
            f"  ambisonic: 0\n"
            f"  userData: \n"
            f"  assetBundleName: \n"
            f"  assetBundleVariant: \n"
        )
    # Default: generic binary asset
    return (
        f"fileFormatVersion: 2\n"
        f"guid: {guid}\n"
        f"DefaultImporter:\n"
        f"  externalObjects: {{}}\n"
        f"  userData: \n"
        f"  assetBundleName: \n"
        f"  assetBundleVariant: \n"
    )


def _mat_stub_content(mat_name: str) -> str:
    """Minimal Unity Standard-shader .mat YAML stub (white albedo, no textures)."""
    return (
        "%YAML 1.1\n"
        "%TAG !u! tag:unity3d.com,2011:\n"
        "--- !u!21 &2100000\n"
        "Material:\n"
        "  serializedVersion: 8\n"
        "  m_ObjectHideFlags: 0\n"
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
        "    m_Floats:\n"
        "    - _Glossiness: 0.5\n"
        "    - _Metallic: 0.0\n"
        "    m_Colors:\n"
        "    - _Color: {r: 1, g: 1, b: 1, a: 1}\n"
    )


def _mat_meta_content(guid: str) -> str:
    """NativeFormatImporter .meta for a Unity .mat file."""
    return (
        f"fileFormatVersion: 2\n"
        f"guid: {guid}\n"
        f"NativeFormatImporter:\n"
        f"  externalObjects: {{}}\n"
        f"  mainObjectFileID: 2100000\n"
        f"  userData: \n"
        f"  assetBundleName: \n"
        f"  assetBundleVariant: \n"
    )


def _script_meta_content(guid: str) -> str:
    """Minimal .cs.meta content — matches reference project format."""
    return f"fileFormatVersion: 2\nguid: {guid}\n"

from .godot_scene_parser   import (
    GodotSceneParser, GodotParseError,
    build_reference_graph, SceneClassifier,
)
from .unity_scene_exporter import UnitySceneExporter, write_folder_meta
from script_converter      import (
    GodotToUnityCSharpConverter,
    GDScriptToUnityCSharpConverter,
    _gd_class_name,
)

log = logging.getLogger("godot_to_unity_pipeline")

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class G2UConversionResult:
    """Result of a Godot → Unity project conversion.

    ``success`` is True when at least one scene was exported without a
    blocking error.  ``warnings`` accumulates non-fatal issues throughout
    the pipeline.
    """
    success:           bool
    scenes_exported:   List[Path]          = field(default_factory=list)
    prefabs_exported:  List[Path]          = field(default_factory=list)
    scenes_failed:     List[str]           = field(default_factory=list)
    scenes_unresolved: List[Dict[str, Any]] = field(default_factory=list)
    scripts_converted: int                 = 0
    scripts_failed:    int                 = 0
    assets_copied:     int                 = 0
    warnings:          List[str]           = field(default_factory=list)
    error:             str                 = ""

    # Per-scene IRs — available even when export partially failed
    scene_irs:         Dict[str, Any]      = field(default_factory=dict)

    # All decisions made silently during conversion (scene classification,
    # layer mapping, script skips, collider fallbacks, etc.)
    decisions:         List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class GodotToUnityPipeline:
    """Convert a Godot 4 project folder into a Unity 6000.3.9f1 project.

    Usage::

        pipeline = GodotToUnityPipeline()
        result   = pipeline.convert(
            godot_root  = Path("/path/to/godot_project"),
            output_dir  = Path("/path/to/output_unity_project"),
            project_name = "MyGame",
        )

    The output directory will contain a minimal Unity project layout::

        output_dir/
            Assets/
            ProjectSettings/
                ProjectVersion.txt
    """

    # Passthrough asset extensions (copied verbatim)
    _MESH_EXTENSIONS     = {".fbx", ".obj", ".gltf", ".glb", ".bin"}
    _TEXTURE_EXTENSIONS  = {".png", ".jpg", ".jpeg", ".tga", ".bmp", ".exr", ".hdr"}
    _MATERIAL_EXTENSIONS = {".tres", ".material"}
    _OTHER_EXTENSIONS    = {".mtl", ".txt", ".url"}

    def __init__(self) -> None:
        self._parser          = GodotSceneParser()
        self._exporter        = UnitySceneExporter()
        self._script_converter = GodotToUnityCSharpConverter()
        self._gd_converter     = GDScriptToUnityCSharpConverter()

    # ------------------------------------------------------------------ public

    def convert(
        self,
        godot_root:      Path,
        output_dir:      Path,
        project_name:    str  = "ConvertedProject",
        convert_scripts: bool = True,
        progress_cb            = None,
    ) -> G2UConversionResult:
        """Run the full Godot → Unity pipeline.

        Args:
            godot_root:   Root of the Godot project (contains project.godot).
            output_dir:   Destination Unity project root (created if absent).
            project_name: Written into ProjectSettings/ProjectVersion.txt.

        Returns:
            A :class:`G2UConversionResult` describing what was exported.
        """
        _prog = progress_cb or (lambda *_: None)
        warnings: List[str] = []
        result = G2UConversionResult(success=False, warnings=warnings)

        # ── Stage A: Scan ────────────────────────────────────────────────────
        _prog(18, "Parsing scene graph", "Scanning Godot project")
        scene_files  = self._scan_scenes(godot_root)
        cs_files     = self._scan_scripts(godot_root)
        gd_files     = self._scan_gd_scripts(godot_root)
        script_files = cs_files          # only C# files go to the converter
        all_scripts  = cs_files + gd_files  # all scripts fed to reference graph
        if not scene_files:
            result.error = "No .tscn scene files found in the Godot project."
            return result

        asset_files = self._scan_assets(godot_root)
        log.info(
            "scan complete  scenes=%d  cs_scripts=%d  gd_scripts=%d  assets=%d",
            len(scene_files), len(cs_files), len(gd_files), len(asset_files),
        )

        # ── Stage A½: Build reference graph ──────────────────────────────────
        _prog(28, "Parsing scene graph", "Building reference graph")
        ref_graph = build_reference_graph(
            scene_files, godot_root=godot_root, script_files=all_scripts
        )
        log.info(
            "reference graph built  referenced=%d  main_scene=%s",
            len(ref_graph.get("referenced_res_paths", set())),
            ref_graph.get("main_scene_res_path", "<none>"),
        )

        # ── Stage A¾½: Batch classify all .tscn files ─────────────────────────
        _prog(37, "Mapping instances / nodes", "Classifying scenes")
        # SceneClassifier merges Ollama static analysis with the dependency graph.
        # Flags are monotonic (never reverted) and the default fallback for
        # unresolved files is PREFAB (not SCENE — safer in Unity asset terms).
        _classifier = SceneClassifier()
        classification_map = _classifier.classify_all(
            scene_files, all_scripts, ref_graph, godot_root=godot_root
        )
        log.info(
            "classification complete  scenes=%d  instances=%d  both=%d",
            sum(1 for v in classification_map.values()
                if isinstance(v, dict) and v.get("type") == "SCENE"),
            sum(1 for v in classification_map.values()
                if isinstance(v, dict) and v.get("type") == "INSTANCE"),
            sum(1 for v in classification_map.values()
                if isinstance(v, dict) and v.get("type") == "BOTH"),
        )

        # ── Stages B + C: Parse → Classify → Validate (all scenes first) ───────
        # Maps scene_path → (scene_ir, is_instance, is_scene, godot_rel_dir)
        classified: List[tuple] = []
        file_to_res: Dict[str, str] = ref_graph.get("file_to_res", {})

        _total_scenes = max(len(scene_files), 1)
        for _scene_idx, scene_path in enumerate(scene_files):
            _pct = 50 + int(_scene_idx * 18 / _total_scenes)
            _prog(_pct, "Converting assets", f"Parsing {scene_path.name}")
            scene_name = scene_path.stem
            log.info("processing scene  %s", scene_path.name)

            # Stage B: Parse
            try:
                scene_ir = self._parser.parse_file(scene_path)
            except GodotParseError as exc:
                warn = f"Skipped {scene_path.name}: {exc}"
                warnings.append(warn)
                result.scenes_failed.append(scene_path.name)
                log.warning(warn)
                continue

            result.scene_irs[scene_name] = scene_ir

            # Stage B½: Classify — look up pre-computed result from SceneClassifier.
            # Default for missing entries: INSTANCE (safe fallback per classification rules).
            _res_key = ref_graph.get("file_to_res", {}).get(str(scene_path), "")
            _cls = classification_map.get(_res_key, {"scene": False, "instance": True, "type": "INSTANCE"})
            is_instance = _cls["instance"]
            is_scene    = _cls["scene"]
            cls_str     = _cls.get("type", "BOTH" if is_instance and is_scene else
                                           "INSTANCE" if is_instance else "SCENE")
            scene_ir["classification"] = cls_str
            log.info("classified %s → %s", scene_path.name, cls_str)
            result.decisions.append({
                "type":      "scene_classification",
                "universal": False,
                "message":   f"'{scene_path.name}' classified as {cls_str}",
                "context": {
                    "scene":          scene_path.name,
                    "classification": cls_str,
                    "source":         _cls.get("source", "unknown"),
                },
            })

            # Stage C: Validate IR
            validation_warnings = self._validate_ir(scene_ir)
            warnings.extend(validation_warnings)
            if any("BLOCKING" in w for w in validation_warnings):
                result.scenes_failed.append(scene_path.name)
                continue

            try:
                godot_rel_dir: Optional[Path] = scene_path.relative_to(godot_root).parent
            except ValueError:
                godot_rel_dir = None

            classified.append((scene_path, scene_ir, is_instance, is_scene, godot_rel_dir))

        # Write ir.json and ir_klasifikuotas.json debug artifacts after parsing loop
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            ir_raw = {name: {k: v for k, v in ir.items() if k != "classification"}
                      for name, ir in result.scene_irs.items()}
            (output_dir / "ir.json").write_text(
                json.dumps(ir_raw, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except (OSError, TypeError) as exc:
            log.warning("Could not write ir.json: %s", exc)
        try:
            (output_dir / "ir_klasifikuotas.json").write_text(
                json.dumps(result.scene_irs, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except (OSError, TypeError) as exc:
            log.warning("Could not write ir_klasifikuotas.json: %s", exc)

        # Write skriptu_planas.json before script conversion runs
        def _rel_posix(p: Path) -> str:
            try:
                return p.relative_to(godot_root).as_posix()
            except ValueError:
                return p.name
        try:
            skriptu_planas = {
                "cs_skriptai": [
                    {"saltinis": _rel_posix(p), "tikslas": f"Assets/Scripts/{p.name}"}
                    for p in script_files
                ],
                "gd_skriptai": [
                    {"saltinis": _rel_posix(p), "tikslas": f"Assets/Scripts/{_gd_class_name(p.stem)}.cs"}
                    for p in gd_files
                ],
            }
            (output_dir / "skriptu_planas.json").write_text(
                json.dumps(skriptu_planas, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except (OSError, TypeError) as exc:
            log.warning("Could not write skriptu_planas.json: %s", exc)

        # ── Stage E (scripts first): Convert scripts before scene export ────────
        # C# and GDScript files are converted before scenes/prefabs so .cs files
        # and their .meta GUIDs exist on disk when scene YAML MonoBehaviour refs
        # are written.
        scripts_ok = 0; scripts_fail = 0
        if convert_scripts:
            _prog(70, "Converting assets", "Converting C# scripts")
            scripts_ok, scripts_fail = self._convert_scripts(
                script_files, godot_root, output_dir, warnings, result.decisions
            )
            if gd_files:
                _prog(72, "Converting assets", "Converting GDScript files")
                gd_ok, gd_fail = self._convert_gd_scripts(
                    gd_files, godot_root, output_dir, warnings, result.decisions
                )
                scripts_ok   += gd_ok
                scripts_fail += gd_fail
        result.scripts_converted = scripts_ok
        result.scripts_failed    = scripts_fail

        # ── Prepare output Unity project skeleton ────────────────────────────
        _prog(74, "Mapping instances / nodes", "Setting up Unity project")
        self._write_project_skeleton(output_dir, project_name)

        # ── Pre-compute asset GUIDs for mesh reference resolution ────────────
        # Build res:// → GUID map before Stage D so _mesh_filter can emit proper
        # GUID-based mesh references instead of null fileID: 0 references.
        res_to_mesh_guid = self._build_asset_guid_map(asset_files, godot_root)

        # ── Stage D pass 1: Export prefabs, build res_path → GUID map ────────
        _prog(76, "Mapping instances / nodes", "Exporting prefabs")
        # res_path_to_guid maps "res://path/to/file.tscn" → prefab GUID string
        res_path_to_guid: Dict[str, str] = {}
        # res_path_to_node_ids maps res_path → {interior_path: transform_fileID}
        # captured during prefab export so scene export can wire m_AddedGameObjects
        res_path_to_node_ids: Dict[str, Dict[str, int]] = {}
        # Accumulate per-scene layer maps for layer_map.json debug output
        all_layer_maps: Dict[str, Dict[str, int]] = {}

        for scene_path, scene_ir, is_instance, is_scene, godot_rel_dir in classified:
            if not is_instance:
                continue
            layer_map = _build_layer_map(scene_ir.get("collision_layers_used", []))
            if layer_map:
                all_layer_maps[scene_path.name] = {str(k): v for k, v in layer_map.items()}
            try:
                unity_prefab, prefab_guid, node_ids = self._exporter.export_instance(
                    scene_ir, output_dir, project_name=project_name,
                    godot_relative_dir=godot_rel_dir,
                    mesh_guid_map=res_to_mesh_guid,
                    layer_map=layer_map,
                    decisions=result.decisions,
                    warnings=warnings,
                )
                result.prefabs_exported.append(unity_prefab)
                log.info("exported prefab  %s", unity_prefab.name)
                # Register GUID and node ID map keyed by res:// path
                res_path = file_to_res.get(str(scene_path), "")
                if res_path:
                    res_path_to_guid[res_path]     = prefab_guid
                    res_path_to_node_ids[res_path] = node_ids
            except Exception as exc:
                warn = f"Export failed for {scene_path.name}: {exc}"
                warnings.append(warn)
                result.scenes_failed.append(scene_path.name)
                log.error(warn)

        # Write guid_map.json debug artifact — merges asset GUIDs and prefab GUIDs
        try:
            combined_guid_map = {**res_to_mesh_guid, **res_path_to_guid}
            (output_dir / "guid_map.json").write_text(
                json.dumps(combined_guid_map, indent=2, sort_keys=True), encoding="utf-8"
            )
        except OSError as exc:
            log.warning("Could not write guid_map.json: %s", exc)

        # ── Stage D pass 2: Export scenes with prefab GUID map ────────────────
        _prog(85, "Resolving dependencies", "Exporting scenes")
        for scene_path, scene_ir, is_instance, is_scene, godot_rel_dir in classified:
            if not is_scene:
                continue
            layer_map = _build_layer_map(scene_ir.get("collision_layers_used", []))
            if layer_map:
                all_layer_maps[scene_path.name] = {str(k): v for k, v in layer_map.items()}
            try:
                if is_instance:
                    # BOTH classification: scene references the prefab, not a
                    # duplicate of the hierarchy.  Look up the GUID written in
                    # pass 1; fall back to full export only when the prefab
                    # export failed and no GUID was registered.
                    res_path   = file_to_res.get(str(scene_path), "")
                    prefab_guid = res_path_to_guid.get(res_path, "")
                    if prefab_guid:
                        unity_scene = self._exporter.export_scene_from_prefab(
                            scene_ir, output_dir,
                            prefab_guid=prefab_guid,
                            project_name=project_name,
                            godot_relative_dir=godot_rel_dir,
                        )
                        log.info("exported BOTH (prefab-root scene)  %s", unity_scene.name)
                    else:
                        log.warning(
                            "BOTH-classified %s has no prefab GUID "
                            "(prefab export may have failed) — using full scene export",
                            scene_path.name,
                        )
                        unity_scene = self._exporter.export(
                            scene_ir, output_dir, project_name=project_name,
                            godot_relative_dir=godot_rel_dir,
                            instance_guid_map=res_path_to_guid,
                            mesh_guid_map=res_to_mesh_guid,
                            layer_map=layer_map,
                            decisions=result.decisions,
                            warnings=warnings,
                            prefab_node_id_map=res_path_to_node_ids,
                        )
                        log.info("exported scene (fallback)  %s", unity_scene.name)
                else:
                    unity_scene = self._exporter.export(
                        scene_ir, output_dir, project_name=project_name,
                        godot_relative_dir=godot_rel_dir,
                        instance_guid_map=res_path_to_guid,
                        mesh_guid_map=res_to_mesh_guid,
                        layer_map=layer_map,
                        decisions=result.decisions,
                        warnings=warnings,
                        prefab_node_id_map=res_path_to_node_ids,
                    )
                    log.info("exported scene  %s", unity_scene.name)
                result.scenes_exported.append(unity_scene)
            except Exception as exc:
                warn = f"Export failed for {scene_path.name}: {exc}"
                warnings.append(warn)
                result.scenes_failed.append(scene_path.name)
                log.error(warn)

        # ── Stage F: Copy passthrough assets ─────────────────────────────────
        _prog(92, "Resolving dependencies", "Copying passthrough assets")
        result.assets_copied = self._copy_assets(
            asset_files, godot_root, output_dir, warnings
        )

        # Write layer_map.json debug artifact to output root
        if all_layer_maps:
            result.decisions.insert(0, {
                "type":      "layer_mapping",
                "universal": True,
                "message":   "Godot collision layers remapped to Unity physics layers (0–31)",
                "context": {
                    "layer_count": sum(len(v) for v in all_layer_maps.values()),
                },
            })
            try:
                (output_dir / "layer_map.json").write_text(
                    json.dumps(all_layer_maps, indent=2), encoding="utf-8"
                )
            except OSError as exc:
                log.warning("Could not write layer_map.json: %s", exc)

        log.info(
            "pipeline complete  scenes=%d  prefabs=%d  failed=%d  "
            "unresolved=%d  scripts=%d  assets=%d",
            len(result.scenes_exported),
            len(result.prefabs_exported),
            len(result.scenes_failed),
            len(result.scenes_unresolved),
            result.scripts_converted,
            result.assets_copied,
        )

        # ── Final validation ──────────────────────────────────────────────────
        self._validate_output(output_dir, result)

        result.success = (
            len(result.scenes_exported) > 0 or len(result.prefabs_exported) > 0
        )
        return result

    # ----------------------------------------------------------------- private

    # Godot cache/editor directories — never contain convertible source files
    _GODOT_SKIP_DIRS = {".godot", ".import", ".vs", "addons"}

    def _scan_scenes(self, root: Path) -> List[Path]:
        """Find all .tscn files under the Godot project root, skipping cache dirs."""
        return sorted(
            p for p in root.rglob("*.tscn")
            if not any(part in self._GODOT_SKIP_DIRS for part in p.parts)
        )

    def _scan_scripts(self, root: Path) -> List[Path]:
        """Find all Godot C# (.cs) files under the project root, skipping cache dirs."""
        return sorted(
            p for p in root.rglob("*.cs")
            if not any(part in self._GODOT_SKIP_DIRS for part in p.parts)
        )

    def _scan_gd_scripts(self, root: Path) -> List[Path]:
        """Find all GDScript (.gd) files under the project root, skipping cache dirs."""
        return sorted(
            p for p in root.rglob("*.gd")
            if not any(part in self._GODOT_SKIP_DIRS for part in p.parts)
        )

    def _convert_scripts(
        self,
        script_files: List[Path],
        godot_root:   Path,
        output_dir:   Path,
        warnings:     List[str],
        decisions:    Optional[List[Dict[str, Any]]] = None,
    ) -> tuple:
        """Convert Godot C# scripts to Unity C#. Returns (ok_count, fail_count)."""
        scripts_dir = output_dir / "Assets" / "Scripts"
        ok = fail = 0
        scripts_dir_written = False
        for cs_path in script_files:
            # Pre-screen: skip scripts with no scene-related constructs to
            # avoid unnecessary AI calls.
            try:
                source_text = cs_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                source_text = ""
            if not _has_scene_constructs(source_text):
                log.debug("script pre-screen skip (no scene constructs): %s", cs_path.name)
                if decisions is not None:
                    decisions.append({
                        "type":      "script_skipped",
                        "universal": False,
                        "message":   f"Script '{cs_path.name}' skipped — no scene constructs",
                        "context":   {"script": cs_path.name},
                    })
                continue

            # Use filename only — scripts are placed flat in Assets/Scripts/
            # regardless of their subdirectory depth in the Godot project.
            dst = scripts_dir / cs_path.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not scripts_dir_written:
                write_folder_meta(scripts_dir, output_dir)
                scripts_dir_written = True

            result = self._script_converter.convert_file(cs_path, dst)
            if result.success:
                ok += 1
                log.info("script converted  %s", cs_path.name)
                if decisions is not None:
                    decisions.append({
                        "type":      "script_added",
                        "universal": False,
                        "message":   f"Script '{cs_path.name}' converted to Unity C#",
                        "context":   {"script": cs_path.name, "output": dst.name},
                    })
                if result.error:
                    warnings.append(
                        f"Script converted with unresolved issues ({cs_path.name}): {result.error}"
                    )
                    log.warning("script residual issues  %s: %s", cs_path.name, result.error)
            else:
                fail += 1
                warnings.append(
                    f"Script conversion failed for {cs_path.name}: {result.error}"
                )
                log.warning("script failed  %s: %s", cs_path.name, result.error)
                if decisions is not None:
                    decisions.append({
                        "type":      "script_fallback",
                        "universal": False,
                        "message":   f"Script '{cs_path.name}' conversion failed — stub written",
                        "context":   {"script": cs_path.name, "error": result.error},
                    })

            # Write .cs.meta — must exist even when conversion failed (so Unity
            # does not generate a random GUID on import and break scene references)
            meta_dst = dst.parent / f"{dst.name}.meta"
            if not meta_dst.exists():
                unity_rel = f"Assets/Scripts/{dst.name}"
                guid = _stable_guid(unity_rel)
                meta_dst.write_text(_script_meta_content(guid), encoding="utf-8")

        return ok, fail

    def _convert_gd_scripts(
        self,
        gd_files:   List[Path],
        godot_root: Path,
        output_dir: Path,
        warnings:   List[str],
        decisions:  Optional[List[Dict[str, Any]]] = None,
    ) -> tuple:
        """Convert GDScript (.gd) files to Unity C#. Returns (ok_count, fail_count)."""
        scripts_dir = output_dir / "Assets" / "Scripts"
        ok = fail = 0
        scripts_dir_written = False
        for gd_path in gd_files:
            class_stem = _gd_class_name(gd_path.stem)
            dst = scripts_dir / f"{class_stem}.cs"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not scripts_dir_written:
                write_folder_meta(scripts_dir, output_dir)
                scripts_dir_written = True

            res = self._gd_converter.convert_file(gd_path, dst)
            if res.success:
                ok += 1
                log.info("gdscript converted  %s → %s", gd_path.name, dst.name)
                if decisions is not None:
                    decisions.append({
                        "type":      "script_added",
                        "universal": False,
                        "message":   f"GDScript '{gd_path.name}' converted to Unity C# ({dst.name})",
                        "context":   {"script": gd_path.name, "output": dst.name, "lang": "gdscript"},
                    })
                if res.error:
                    warnings.append(
                        f"GDScript converted with issues ({gd_path.name}): {res.error}"
                    )
            else:
                fail += 1
                warnings.append(
                    f"GDScript conversion failed for {gd_path.name}: {res.error}"
                )
                log.warning("gdscript failed  %s: %s", gd_path.name, res.error)
                if decisions is not None:
                    decisions.append({
                        "type":      "script_fallback",
                        "universal": False,
                        "message":   f"GDScript '{gd_path.name}' conversion failed — stub written",
                        "context":   {"script": gd_path.name, "error": res.error, "lang": "gdscript"},
                    })

            meta_dst = dst.parent / f"{dst.name}.meta"
            if not meta_dst.exists():
                guid = _stable_guid(f"Assets/Scripts/{dst.name}")
                meta_dst.write_text(_script_meta_content(guid), encoding="utf-8")

        return ok, fail

    def _scan_assets(self, root: Path) -> List[Path]:
        """Find all passthrough asset files (meshes, textures, materials, misc), skipping cache dirs."""
        assets: List[Path] = []
        all_extensions = (self._MESH_EXTENSIONS
                          | self._TEXTURE_EXTENSIONS
                          | self._MATERIAL_EXTENSIONS
                          | self._OTHER_EXTENSIONS)
        for ext in all_extensions:
            for p in root.rglob(f"*{ext}"):
                if not any(part in self._GODOT_SKIP_DIRS for part in p.parts):
                    assets.append(p)
        return sorted(set(assets))

    def _validate_ir(self, scene_ir: Dict[str, Any]) -> List[str]:
        """Run sanity checks on a parsed scene IR.

        Returns a list of warning strings.  Warnings containing 'BLOCKING'
        prevent export of that scene.
        """
        warnings: List[str] = []
        nodes      = scene_ir.get("nodes", [])
        scene_name = scene_ir.get("scene_name", "<unknown>")

        if not nodes:
            warnings.append(
                f"BLOCKING: scene '{scene_name}' has no nodes."
            )
            return warnings

        if scene_ir.get("ir_version") != "1.0":
            warnings.append(
                f"IR version mismatch: expected 1.0, got "
                f"{scene_ir.get('ir_version')} — conversion may be inaccurate."
            )

        # Validate coordinate system spec compliance
        coord = scene_ir.get("coordinate_system", {})
        if coord.get("handedness") != "right" or coord.get("up_axis") != "Y":
            warnings.append(
                "Unexpected coordinate system in IR — transforms may be incorrect."
            )

        # Validate each node recursively
        self._validate_nodes(nodes, scene_name, warnings)

        return warnings

    def _validate_nodes(
        self,
        nodes:      List[Dict[str, Any]],
        scene_name: str,
        warnings:   List[str],
    ) -> None:
        """Recursively validate IR nodes, collecting per-node warnings."""
        for node in nodes:
            name  = node.get("node_name", "<unnamed>")
            comps = node.get("components", {})
            gtype = node.get("godot_type", "")
            meta  = node.get("meta", {})

            # Null mesh reference on a mesh entity
            if gtype == "MeshInstance3D":
                mesh = comps.get("mesh", {})
                if not mesh:
                    warnings.append(
                        f"Null mesh reference on MeshInstance3D '{name}' "
                        f"in '{scene_name}' — MeshFilter will have no mesh."
                    )
                elif not mesh.get("mesh_source_path"):
                    warnings.append(
                        f"MeshInstance3D '{name}' has no mesh_source_path "
                        f"— MeshFilter mesh will be unset after import."
                    )

            # Script references (Godot .cs ext_resource not yet attached)
            script_ref = node.get("original_data", {}).get("source_flags", {}).get("script")
            if script_ref:
                warnings.append(
                    f"Node '{name}' references script '{script_ref}' — "
                    f"verify script was converted and attach manually."
                )

            # Warn about node types that had no component mapping
            # (GPUParticles3D is now mapped to particle_system component)
            _unsupported_now = {"Label3D"}
            if gtype in _unsupported_now:
                warnings.append(
                    f"'{gtype}' node '{name}' has no Unity equivalent "
                    f"— exported as empty GameObject."
                )

            self._validate_nodes(node.get("children", []), scene_name, warnings)

    def _collect_unsupported_types(
        self, nodes: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Recursively count node types that have limited export support."""
        _unsupported = {"Label3D"}
        counts: Dict[str, int] = {}
        for node in nodes:
            gtype = node.get("godot_type", "")
            if gtype in _unsupported:
                counts[gtype] = counts.get(gtype, 0) + 1
            counts.update(self._collect_unsupported_types(node.get("children", [])))
        return counts

    @staticmethod
    def _flatten_path(rel: Path) -> Path:
        """Remove consecutive identical directory components.

        Collapses redundant self-named subdirectories that some asset packs create
        (e.g. KayKit_City_Builder_Bits_1.0_FREE/KayKit_City_Builder_Bits_1.0_FREE/Assets
        → KayKit_City_Builder_Bits_1.0_FREE/Assets).
        """
        parts = rel.parts
        out: list[str] = []
        for part in parts:
            if out and out[-1] == part:
                continue
            out.append(part)
        return Path(*out) if out else rel

    @staticmethod
    def _build_asset_guid_map(
        asset_files: List[Path],
        godot_root:  Path,
    ) -> Dict[str, str]:
        """Return a mapping of Godot res:// path → deterministic GUID for every asset.

        GUIDs are derived from the flattened output-relative path so they are
        stable across pipeline re-runs and do not change between invocations.
        """
        res_to_guid: Dict[str, str] = {}
        for src in asset_files:
            try:
                rel_raw  = src.relative_to(godot_root)
            except ValueError:
                rel_raw  = Path(src.name)
            rel_flat = GodotToUnityPipeline._flatten_path(rel_raw)
            res_path = "res://" + rel_raw.as_posix()
            res_to_guid[res_path] = _stable_guid(rel_flat.as_posix())
        return res_to_guid

    @staticmethod
    def _scan_output_metas(output_dir: Path) -> Dict[str, str]:
        """Walk the Unity output for existing *.meta files and return path → guid.

        Key is the asset path relative to *output_dir* (posix, without .meta suffix).
        Used to reuse GUIDs from a previous run so cross-file references stay valid.
        """
        path_to_guid: Dict[str, str] = {}
        for meta_path in output_dir.rglob("*.meta"):
            guid = parse_meta_file(meta_path)
            if not guid:
                continue
            asset_path = meta_path.with_suffix("")  # strip .meta
            if asset_path.exists():
                try:
                    rel = asset_path.relative_to(output_dir)
                    path_to_guid[rel.as_posix()] = guid
                except ValueError:
                    pass
        return path_to_guid

    def _copy_assets(
        self,
        asset_files: List[Path],
        godot_root:  Path,
        output_dir:  Path,
        warnings:    List[str],
    ) -> int:
        """Copy mesh and texture assets, mirroring the Godot project structure.

        Each asset is placed at::

            <output_dir>/Assets/<original_path_relative_to_godot_root>

        so that Unity import paths match the original Godot resource paths.
        Consecutive identical directory components are collapsed (e.g. A/A/B → A/B).
        """
        copied = 0
        seen_dirs: set = set()
        for src in asset_files:
            try:
                rel = src.relative_to(godot_root)
            except ValueError:
                rel = Path(src.name)

            rel = self._flatten_path(rel)
            dst = output_dir / "Assets" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)

            # Write folder .meta for every ancestor directory that was created
            parts = rel.parts[:-1]  # directory components only
            for depth in range(1, len(parts) + 1):
                ancestor = output_dir / "Assets" / Path(*parts[:depth])
                key = ancestor.as_posix()
                if key not in seen_dirs:
                    seen_dirs.add(key)
                    write_folder_meta(ancestor, output_dir)

            # Godot material files (.tres / .material) are converted to Unity .mat
            # stubs rather than copied verbatim — Unity cannot process Godot resources.
            if src.suffix.lower() in self._MATERIAL_EXTENSIONS:
                mat_rel = rel.with_suffix(".mat")
                mat_dst = output_dir / "Assets" / mat_rel
                mat_dst.parent.mkdir(parents=True, exist_ok=True)
                if not mat_dst.exists():
                    guid = _stable_guid(rel.as_posix())
                    mat_dst.write_text(
                        _mat_stub_content(src.stem), encoding="utf-8"
                    )
                    meta_dst = mat_dst.parent / (mat_dst.name + ".meta")
                    if not meta_dst.exists():
                        meta_dst.write_text(
                            _mat_meta_content(guid), encoding="utf-8"
                        )
                    copied += 1
                continue

            if dst.exists():
                continue
            try:
                shutil.copy2(src, dst)
                copied += 1
            except OSError as exc:
                warnings.append(f"Could not copy asset {src.name}: {exc}")
                continue

            # Write a .meta sidecar with the correct importer for the asset type.
            meta_dst = dst.parent / (dst.name + ".meta")
            if not meta_dst.exists():
                guid = _stable_guid(rel.as_posix())
                meta_dst.write_text(
                    _meta_for_asset(rel, guid),
                    encoding="utf-8",
                )
        return copied

    def _validate_output(
        self,
        output_dir: Path,
        result: G2UConversionResult,
    ) -> None:
        """Post-pipeline validation: detect placeholder leakage and prefab root issues."""
        _placeholders = ("__RAYCAST_PLACEHOLDER__", "__SCENETREE_PLACEHOLDER__")
        cs_checked = 0
        for cs_file in (output_dir / "Assets").rglob("*.cs"):
            try:
                text = cs_file.read_text(encoding="utf-8", errors="ignore")
                for p in _placeholders:
                    if p in text:
                        msg = f"Placeholder '{p}' leaked into {cs_file.name} — manual review required"
                        result.warnings.append(msg)
                        log.warning("[validate] %s", msg)
                cs_checked += 1
            except OSError:
                pass

        prefab_checked = 0
        for prefab_file in (output_dir / "Assets").rglob("*.prefab"):
            try:
                text = prefab_file.read_text(encoding="utf-8", errors="ignore")
                root_count = text.count("m_Father: {fileID: 0}")
                if root_count > 1:
                    msg = (
                        f"Prefab '{prefab_file.name}' has {root_count} root objects"
                        f" — may be corrupted"
                    )
                    result.warnings.append(msg)
                    log.warning("[validate] %s", msg)
                prefab_checked += 1
            except OSError:
                pass

        log.info("[validate] checked %d scripts, %d prefabs", cs_checked, prefab_checked)

    @staticmethod
    def _write_project_skeleton(output_dir: Path, project_name: str) -> None:
        """Write the minimal Unity project folder structure."""
        asset_subdirs = ("Assets",)
        for sub in asset_subdirs:
            folder = output_dir / sub
            folder.mkdir(parents=True, exist_ok=True)
            # Walk up from leaf to Assets/ (exclusive) writing folder metas
            parts = Path(sub).parts
            for depth in range(1, len(parts) + 1):
                ancestor = output_dir / Path(*parts[:depth])
                write_folder_meta(ancestor, output_dir)
        for sub in ("ProjectSettings", "Packages"):
            (output_dir / sub).mkdir(parents=True, exist_ok=True)

        # ProjectVersion.txt
        pv = output_dir / "ProjectSettings" / "ProjectVersion.txt"
        if not pv.exists():
            pv.write_text(
                "m_EditorVersion: 6000.3.9f1\n"
                "m_EditorVersionWithRevision: 6000.3.9f1 ()\n",
                encoding="utf-8",
            )

        # Minimal manifest.json for Package Manager
        manifest = output_dir / "Packages" / "manifest.json"
        if not manifest.exists():
            manifest.write_text(
                '{\n  "dependencies": {\n'
                '    "com.unity.ugui": "2.0.0"\n'
                '  }\n}\n',
                encoding="utf-8",
            )

        # Minimal ProjectSettings.asset
        ps = output_dir / "ProjectSettings" / "ProjectSettings.asset"
        if not ps.exists():
            ps.write_text(
                f"%YAML 1.1\n%TAG !u! tag:unity3d.com,2011:\n"
                f"--- !u!129 &1\n"
                f"PlayerSettings:\n"
                f"  productName: {project_name}\n"
                f"  companyName: ConvertedProject\n",
                encoding="utf-8",
            )
