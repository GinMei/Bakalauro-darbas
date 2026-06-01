"""godot_scene_parser.py

Stage A of the Godot → Unity pipeline.

Parses Godot 4 .tscn files into the engine-agnostic IR used by ir_builder.py,
mirroring the structure produced by build_scene_ir() for Unity sources.

Supported node types:
    Node3D, MeshInstance3D, StaticBody3D, RigidBody3D,
    CollisionShape3D (Box / Sphere / Capsule), Camera3D

Explicitly NOT supported (silently preserved as "group" IR nodes):
    GDScript / C# scripts, AnimationPlayer, GPUParticles3D, UI nodes,
    lights, terrain, navigation

Public API:
    GodotSceneParser          — parse a .tscn file into IR
    GodotParseError           — raised on unrecoverable parse failures
"""

from __future__ import annotations

import json
import logging
import math
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests as _requests
    _REQUESTS_OK = True
except ImportError:
    _requests = None  # type: ignore[assignment]
    _REQUESTS_OK = False

log = logging.getLogger("godot_scene_parser")


def _path_to_ir_asset_id(source_path: str) -> str:
    """Derive a stable, deterministic IR asset ID from a res:// path.

    Uses the first 16 hex characters of SHA-256 so the ID is compact,
    collision-resistant, and reproducible across runs.
    """
    import hashlib
    return hashlib.sha256(source_path.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class GodotParseError(ValueError):
    """Raised when a .tscn file cannot be parsed."""


# ---------------------------------------------------------------------------
# Coordinate-system conversion  (Godot right-handed → Unity left-handed)
# ---------------------------------------------------------------------------
# Godot: +X right, +Y up, -Z forward  (right-handed)
# Unity: +X right, +Y up, +Z forward  (left-handed)
#
# Position conversion:  (x, y, z)  →  (x, y, -z)
#
# No coordinate conversion is applied here — the parser stores raw Godot-space
# values in the IR.  All Godot→Unity coordinate conversion (position Z-flip,
# C*R*C rotation transform) is performed by unity_scene_exporter.py so that
# the conversion happens exactly once.  _quat_godot_to_unity only normalises to
# guard against floating-point drift from the 3×3 matrix decomposition step.

def _pos_godot_to_unity(pos: List[float]) -> List[float]:
    return [pos[0], pos[1], pos[2]]


def _quat_godot_to_unity(q: Dict[str, float]) -> Dict[str, float]:
    x, y, z, w = q["x"], q["y"], q["z"], q["w"]
    mag = math.sqrt(x * x + y * y + z * z + w * w)
    if mag > 1e-9:
        x, y, z, w = x / mag, y / mag, z / mag, w / mag
    return {"x": x, "y": y, "z": z, "w": w}


def _scale_godot_to_unity(scale: List[float]) -> List[float]:
    # Scale is axis-magnitude only — no sign flip needed
    return [scale[0], scale[1], scale[2]]


# ---------------------------------------------------------------------------
# Scene classification
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Reference-graph-based scene classification
# ---------------------------------------------------------------------------

# Regex to find any `instance=ExtResource("id")` in a raw .tscn file
_INSTANCE_RES_RE = re.compile(
    r'instance\s*=\s*ExtResource\s*\(\s*"?(\w+)"?\s*\)'
)
# Regex to find `path="res://..."` in an [ext_resource] header
_EXT_PATH_RE = re.compile(r'path\s*=\s*"([^"]+)"')
# Regex for project.godot main_scene
_MAIN_SCENE_RE = re.compile(r'run/main_scene\s*=\s*"([^"]+)"')
# Regex for script-side scene loads with LITERAL full path: GD.Load("res://path.tscn")
_SCRIPT_LOAD_RE = re.compile(
    r'(?:GD\.Load|(?<!\w)load|preload)\s*\(\s*"(res://[^"]+\.tscn)"'
)
# Regex for dynamic string-concat loads: GD.Load("res://prefix/" + ...)
# Only treated as a "scene directory" when the same script also uses SceneTree context.
_LOAD_PREFIX_RE = re.compile(
    r'(?:GD\.Load|(?<!\w)load|preload)\s*\(\s*"(res://[^"]*/)"\s*\+'
)
# Present in scripts that perform scene transitions (vs. object instantiation helpers)
_SCENE_TREE_CONTEXT_RE = re.compile(r'SceneTree|GetTree\s*\(\)|\.ChangeScene')
# Explicit scene-transition call: change_scene_to_file("res://path.tscn") or C# ChangeSceneTo
_CHANGE_SCENE_RE = re.compile(
    r'(?:change_scene_to_file|ChangeSceneTo(?:File)?)\s*\(\s*"(res://[^"]+\.tscn)"'
)
# Explicit prefab-instantiation: load("res://path.tscn").instantiate() / .New() / Instantiate()
_INSTANTIATE_LOAD_RE = re.compile(
    r'(?:GD\.Load(?:<[^>]+>)?|(?<!\w)load|preload)\s*\(\s*"(res://[^"]+\.tscn)"[^)]*\)'
    r'\s*(?:\.\s*(?:instantiate|Instantiate|New)\s*\()'
)


def build_reference_graph(
    scene_files:  List[Path],
    godot_root:   Optional[Path] = None,
    script_files: Optional[List[Path]] = None,
) -> Dict[str, Any]:
    """Scan all .tscn files (and optionally scripts) and return a reference-graph dict.

    Returns::

        {
          "referenced_res_paths":      set of res:// paths instanced by another .tscn,
          "main_scene_res_path":       res:// path of the main scene (project.godot),
          "file_to_res":               dict mapping absolute path → res:// path,
          "script_loaded_res_paths":   union of all loads (backward-compat),
          "script_instantiate_paths":  res:// paths used in load().instantiate() calls,
          "script_scene_change_paths": res:// paths used in change_scene_to_file() calls,
          "dynamic_scene_prefixes":    directory prefixes for dynamic scene transitions,
          "dynamic_prefab_prefixes":   directory prefixes for dynamic prefab instantiation,
          "usage_map": {
              "res://path/to/file.tscn": {
                  "instantiated_by": [script_path, ...],
                  "loaded_as_scene_by": [script_path, ...],
                  "embedded_in": ["res://path/to/parent.tscn", ...],
              },
              ...
          }
        }
    """
    # Build absolute → res:// mapping by detecting godot_root as the
    # common ancestor of all scene files (or the given godot_root).
    root = godot_root
    if root is None and scene_files:
        # Heuristic: walk up from the first file until project.godot exists
        candidate = scene_files[0].parent
        for _ in range(20):
            if (candidate / "project.godot").exists():
                root = candidate
                break
            candidate = candidate.parent

    def to_res(p: Path) -> str:
        if root:
            try:
                rel = p.relative_to(root)
                return "res://" + rel.as_posix()
            except ValueError:
                pass
        return ""

    def _ensure_usage(usage_map: Dict[str, Any], res: str) -> None:
        if res and res not in usage_map:
            usage_map[res] = {"instantiated_by": [], "loaded_as_scene_by": [], "embedded_in": []}

    # Read main_scene from project.godot
    main_scene_res = ""
    if root:
        pg = root / "project.godot"
        if pg.exists():
            try:
                text = pg.read_text(encoding="utf-8", errors="replace")
                m = _MAIN_SCENE_RE.search(text)
                if m:
                    main_scene_res = m.group(1)
            except OSError:
                pass

    # ── Pass 1: scan .tscn files for embedded instance= references ────────────
    referenced: set = set()
    file_to_res: Dict[str, str] = {}
    usage_map: Dict[str, Any] = {}

    _ext_hdr_re = re.compile(
        r'\[ext_resource\s+[^\]]*type\s*=\s*"PackedScene"[^\]]*\]'
    )
    _ext_id_re  = re.compile(r'\bid\s*=\s*"?(\w+)"?')

    for sf in scene_files:
        res_path = to_res(sf)
        if res_path:
            file_to_res[str(sf)] = res_path
        try:
            text = sf.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        # Build id→res path map for PackedScene ext_resources in this file
        id_to_path: Dict[str, str] = {}
        for hdr_m in _ext_hdr_re.finditer(text):
            hdr = hdr_m.group(0)
            id_m   = _ext_id_re.search(hdr)
            path_m = _EXT_PATH_RE.search(hdr)
            if id_m and path_m:
                id_to_path[id_m.group(1)] = path_m.group(1)

        # Find every instance= reference and record both the set and the usage_map
        for inst_m in _INSTANCE_RES_RE.finditer(text):
            res_id = inst_m.group(1)
            if res_id not in id_to_path:
                continue
            instanced_res = id_to_path[res_id]
            referenced.add(instanced_res)
            _ensure_usage(usage_map, instanced_res)
            if res_path and res_path not in usage_map[instanced_res]["embedded_in"]:
                usage_map[instanced_res]["embedded_in"].append(res_path)

    # ── Pass 2: scan scripts for load/instantiate/change_scene calls ──────────
    script_instantiate_paths:  set = set()   # load().instantiate() → PREFAB evidence
    script_scene_change_paths: set = set()   # change_scene_to_file() → SCENE evidence
    script_loaded:             set = set()   # all literal loads (backward-compat union)
    dynamic_scene_prefixes:    set = set()
    dynamic_prefab_prefixes:   set = set()

    for sf in (script_files or []):
        sf_str = str(sf)
        try:
            text = sf.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        # Explicit prefab instantiation: load("res://x.tscn").instantiate()
        for m in _INSTANTIATE_LOAD_RE.finditer(text):
            p = m.group(1)
            script_instantiate_paths.add(p)
            script_loaded.add(p)
            _ensure_usage(usage_map, p)
            if sf_str not in usage_map[p]["instantiated_by"]:
                usage_map[p]["instantiated_by"].append(sf_str)

        # Explicit scene transition: change_scene_to_file("res://x.tscn")
        for m in _CHANGE_SCENE_RE.finditer(text):
            p = m.group(1)
            script_scene_change_paths.add(p)
            script_loaded.add(p)
            _ensure_usage(usage_map, p)
            if sf_str not in usage_map[p]["loaded_as_scene_by"]:
                usage_map[p]["loaded_as_scene_by"].append(sf_str)

        # All remaining literal loads not already classified above
        for m in _SCRIPT_LOAD_RE.finditer(text):
            p = m.group(1)
            script_loaded.add(p)
            if p not in script_instantiate_paths and p not in script_scene_change_paths:
                # Ambiguous load — treat as scene use (conservative)
                _ensure_usage(usage_map, p)
                if sf_str not in usage_map[p]["loaded_as_scene_by"]:
                    usage_map[p]["loaded_as_scene_by"].append(sf_str)

        # Dynamic prefix loads: GD.Load("res://prefix/" + name + ".tscn")
        has_scene_tree = bool(_SCENE_TREE_CONTEXT_RE.search(text))
        for m in _LOAD_PREFIX_RE.finditer(text):
            prefix = m.group(1)
            if has_scene_tree:
                dynamic_scene_prefixes.add(prefix)
            else:
                dynamic_prefab_prefixes.add(prefix)

    return {
        "referenced_res_paths":      referenced,
        "main_scene_res_path":       main_scene_res,
        "file_to_res":               file_to_res,
        "script_loaded_res_paths":   script_loaded,
        "script_instantiate_paths":  script_instantiate_paths,
        "script_scene_change_paths": script_scene_change_paths,
        "dynamic_scene_prefixes":    dynamic_scene_prefixes,
        "dynamic_prefab_prefixes":   dynamic_prefab_prefixes,
        "usage_map":                 usage_map,
    }


def classify_by_graph(
    scene_file: Path,
    ref_graph:  Dict[str, Any],
) -> Dict[str, bool]:
    """Classify a single .tscn as {is_instance, is_scene}.

    Uses the reference graph produced by :func:`build_reference_graph`.

    is_instance — True when ANY of:
        • embedded in another .tscn via instance=ExtResource(...)
        • loaded via load().instantiate() / preload().instantiate() in a script
        • its res:// path is under a dynamic prefab-instantiation directory

    is_scene — True when ANY of:
        • declared as main_scene in project.godot
        • loaded via change_scene_to_file() in a script
        • any non-instantiate load/preload call in a script (ambiguous → scene)
        • its res:// path is under a dynamic scene-transition directory
        • NOT a prefab (unreferenced files default to scene — they are standalone
          entry points that are not reused by any other asset)

    Both flags can be True simultaneously (BOTH case) when a .tscn is both
    embedded/instantiated AND used as a direct entry point.
    """
    file_to_res             = ref_graph.get("file_to_res", {})
    referenced              = ref_graph.get("referenced_res_paths", set())
    main_scene              = ref_graph.get("main_scene_res_path", "")
    script_loaded           = ref_graph.get("script_loaded_res_paths", set())
    script_instantiates     = ref_graph.get("script_instantiate_paths", set())
    script_scene_changes    = ref_graph.get("script_scene_change_paths", set())
    dyn_prefixes            = ref_graph.get("dynamic_scene_prefixes", set())
    dyn_prefab_prefixes     = ref_graph.get("dynamic_prefab_prefixes", set())

    res_path = file_to_res.get(str(scene_file), "")

    is_referenced         = bool(res_path and res_path in referenced)
    is_script_instantiated = bool(res_path and res_path in script_instantiates)
    is_dyn_prefab         = bool(res_path and any(
        res_path.startswith(p) for p in dyn_prefab_prefixes
    ))
    is_main               = bool(res_path and res_path == main_scene)
    is_scene_changed_to   = bool(res_path and res_path in script_scene_changes)
    # Ambiguous loads (not classified as instantiate) are treated as scene evidence
    is_ambiguous_load     = bool(
        res_path and
        res_path in script_loaded and
        res_path not in script_instantiates
    )
    is_dyn_scene          = bool(res_path and any(
        res_path.startswith(p) for p in dyn_prefixes
    ))

    is_instance = is_referenced or is_script_instantiated or is_dyn_prefab

    # is_scene: positive scene evidence OR unreferenced (default to entry scene)
    is_scene = (
        is_main or
        is_scene_changed_to or
        is_ambiguous_load or
        is_dyn_scene or
        not is_instance   # unreferenced → standalone entry scene
    )

    return {"is_instance": is_instance, "is_scene": is_scene}


def classify_tscn(
    scene_ir:    Dict[str, Any],
    source_path: Optional[str] = None,
    ref_graph:   Optional[Dict[str, Any]] = None,
) -> str:
    """Classify a .tscn as 'SCENE', 'INSTANCE', or 'BOTH'.

    When *ref_graph* is provided (built by :func:`build_reference_graph`),
    strict reference-graph classification is used.

    When *ref_graph* is absent a legacy folder-hint fallback is used;
    ambiguous files default to 'INSTANCE'.
    """
    # ── Strict graph-based classification ────────────────────────────────────
    if ref_graph is not None and source_path:
        cls = classify_by_graph(Path(source_path), ref_graph)
        if cls["is_instance"] and cls["is_scene"]:
            return "BOTH"
        if cls["is_instance"]:
            return "INSTANCE"
        return "SCENE"

    # ── Legacy fallback (folder hints only, no keyword heuristics) ───────────
    if source_path:
        parts = [p.lower() for p in Path(source_path).parts]
        if any(p in {"scenes", "levels", "worlds", "maps"} for p in parts):
            return "SCENE"
        if any(p in {"prefabs", "entities", "actors", "characters", "enemies"}
               for p in parts):
            return "INSTANCE"

    return "INSTANCE"


# ---------------------------------------------------------------------------
# Ollama-enhanced batch classifier
# ---------------------------------------------------------------------------

class SceneClassifier:
    """Classify all .tscn files via Ollama static analysis + .tscn dependency graph.

    Signal pipeline (all flags are monotonic — once True, never reverted):

        1. Initialize every entry with scene=False, instance=False.
        2. Ollama LLM analysis of project scripts → set scene/instance True where detected.
        3. .tscn dependency graph (embedded_in / instantiated_by / change_scene) → set True.
        4. Default fallback: if both flags remain False → set instance=True.
        5. Final classification: SCENE | INSTANCE | BOTH.

    Unity output mapping:
        SCENE    → .unity scene file
        INSTANCE → .prefab file
        BOTH     → .prefab + .unity scene referencing the prefab
    """

    _OLLAMA_TIMEOUT          = 120    # seconds — 397B model needs headroom
    _MAX_SCRIPT_CHARS_PER_CHUNK = 75_000  # ~21 K tokens; leaves ~12 K token margin in 40 960-token ctx
    _LAUNCH_POLL_INTERVAL    = 0.5    # seconds between availability checks after launch
    _LAUNCH_TIMEOUT          = 30     # seconds to wait for ollama serve to become ready

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "qwen3",
    ) -> None:
        self._ollama_url  = ollama_url.rstrip("/")
        self._model       = model
        self._proc: Optional[subprocess.Popen] = None  # process we launched, if any

    # ------------------------------------------------------------------ public

    def classify_all(
        self,
        scene_files:  List[Path],
        script_files: List[Path],
        ref_graph:    Dict[str, Any],
        godot_root:   Optional[Path] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Classify all .tscn files and return a res:// → classification map.

        Returns::

            {
              "res://path/to/file.tscn": {
                  "scene":    bool,
                  "instance": bool,
                  "type":     "SCENE" | "INSTANCE" | "BOTH",
              },
              ...
            }

        Also emits a ``"converted"`` list (the spec output format) as a side-effect
        stored in the return dict under the key ``"__report__"``.
        """
        file_to_res: Dict[str, str] = ref_graph.get("file_to_res", {})
        usage_map:   Dict[str, Any] = ref_graph.get("usage_map", {})
        main_scene:  str            = ref_graph.get("main_scene_res_path", "")

        # Build res:// path for every scene file
        res_paths: List[str] = []
        for sf in scene_files:
            rp = file_to_res.get(str(sf), "")
            if not rp:
                if godot_root:
                    try:
                        rp = "res://" + sf.relative_to(godot_root).as_posix()
                    except ValueError:
                        rp = "res://" + sf.name
                else:
                    rp = "res://" + sf.name
                # Back-fill file_to_res so Stage D lookups work
                file_to_res[str(sf)] = rp
            res_paths.append(rp)

        # ── 1. Initialize SceneState (all False) ──────────────────────────────
        state: Dict[str, Dict[str, bool]] = {
            rp: {"scene": False, "instance": False} for rp in res_paths
        }

        # ── 2. Ollama analysis (monotonic merge) ──────────────────────────────
        if _REQUESTS_OK and script_files and self._ensure_running():
            try:
                ollama_items = self._call_ollama(res_paths, script_files)
                for item in ollama_items:
                    rp = item.get("path", "")
                    if rp in state:
                        if _parse_bool(item.get("scene", False)):
                            state[rp]["scene"] = True
                        if _parse_bool(item.get("instance", False)):
                            state[rp]["instance"] = True
                log.info("ollama classification merged  items=%d", len(ollama_items))
            except Exception as exc:
                log.warning("ollama scene analysis failed: %s", exc)
            finally:
                self._stop_ollama()

        # ── 3. Dependency graph signals (monotonic) ───────────────────────────
        referenced          = ref_graph.get("referenced_res_paths", set())
        scene_changes       = ref_graph.get("script_scene_change_paths", set())
        dyn_scene_prefixes  = ref_graph.get("dynamic_scene_prefixes", set())
        dyn_prefab_prefixes = ref_graph.get("dynamic_prefab_prefixes", set())

        for rp in res_paths:
            usage = usage_map.get(rp, {})

            # .tscn embedded in another .tscn → INSTANCE evidence
            if usage.get("embedded_in") or rp in referenced:
                state[rp]["instance"] = True

            # Script-level .instantiate() / Instantiate() call → INSTANCE evidence
            if usage.get("instantiated_by"):
                state[rp]["instance"] = True

            # Dynamic instantiation directory → INSTANCE evidence
            if any(rp.startswith(p) for p in dyn_prefab_prefixes):
                state[rp]["instance"] = True

            # change_scene_to_file() or ambiguous load → SCENE evidence
            if usage.get("loaded_as_scene_by") or rp in scene_changes:
                state[rp]["scene"] = True

            # Dynamic scene-transition directory → SCENE evidence
            # Covers GD.Load("res://prefix/" + name + ".tscn") patterns in scripts.
            if any(rp.startswith(p) for p in dyn_scene_prefixes):
                state[rp]["scene"] = True

            # Declared main scene → SCENE evidence
            if rp == main_scene:
                state[rp]["scene"] = True

        # ── 4. Default fallback: both False → instance = True ────────────────
        for rp in state:
            if not state[rp]["scene"] and not state[rp]["instance"]:
                state[rp]["instance"] = True

        # ── 5. Final classification ────────────────────────────────────────────
        result: Dict[str, Dict[str, Any]] = {}
        report_entries: List[Dict[str, Any]] = []

        for rp, flags in state.items():
            s, p = flags["scene"], flags["instance"]
            cls_type = "BOTH" if s and p else ("SCENE" if s else "INSTANCE")
            result[rp] = {"scene": s, "instance": p, "type": cls_type}
            report_entries.append({
                "scene": rp,
                "type": cls_type,
                "scene_flag": s,
                "instance_flag": p,
            })
            log.debug("classified  %-60s  %s", rp, cls_type)

        result["__report__"] = {"converted": report_entries}  # type: ignore[assignment]
        return result

    # ----------------------------------------------------------------- private

    def _is_available(self) -> bool:
        """Return True only if the Ollama server is reachable."""
        try:
            r = _requests.get(self._ollama_url, timeout=2)  # type: ignore[union-attr]
            return r.status_code < 500
        except Exception:
            return False

    def _ensure_running(self) -> bool:
        """Ensure Ollama is running, launching it if needed.

        Returns True if the server is ready, False if it could not be started.
        Sets self._proc when we launched the process so _stop_ollama() can
        clean it up afterwards.
        """
        if self._is_available():
            return True

        log.info("ollama not running — attempting to launch 'ollama serve'")
        try:
            self._proc = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            log.warning("ollama executable not found on PATH — skipping LLM analysis")
            return False
        except OSError as exc:
            log.warning("could not launch ollama: %s", exc)
            return False

        deadline = time.monotonic() + self._LAUNCH_TIMEOUT
        while time.monotonic() < deadline:
            time.sleep(self._LAUNCH_POLL_INTERVAL)
            if self._is_available():
                log.info("ollama server ready")
                return True

        log.warning("ollama did not become ready within %ds", self._LAUNCH_TIMEOUT)
        self._stop_ollama()
        return False

    def _stop_ollama(self) -> None:
        """Terminate the ollama process we launched, if any."""
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    def _call_ollama(
        self,
        res_paths:    List[str],
        script_files: List[Path],
    ) -> List[Dict[str, Any]]:
        """Classify all scenes across all script chunks; merge signals with OR.

        Splits scripts into chunks that fit within the context window and sends
        one Ollama request per chunk.  Signals are monotonic: once a path is
        marked scene=True or instance=True by any chunk it stays True.
        """
        chunks = self._chunk_scripts(script_files)
        if not chunks:
            chunks = [""]   # single call with no script context

        accumulated: Dict[str, Dict[str, bool]] = {
            rp: {"scene": False, "instance": False} for rp in res_paths
        }

        for i, scripts_text in enumerate(chunks):
            log.info(
                "[SceneClassifier] ollama chunk %d/%d  chars=%d",
                i + 1, len(chunks), len(scripts_text),
            )
            try:
                items = self._call_ollama_chunk(res_paths, scripts_text)
            except Exception as exc:
                log.warning("[SceneClassifier] chunk %d/%d failed: %s — skipping", i + 1, len(chunks), exc)
                continue
            for item in items:
                rp = item.get("path", "")
                if rp not in accumulated:
                    log.warning("[SceneClassifier] unrecognised path %r — ignoring", rp)
                    continue
                if _parse_bool(item.get("scene", False)):
                    accumulated[rp]["scene"] = True
                if _parse_bool(item.get("instance", False)):
                    accumulated[rp]["instance"] = True

        return [
            {"path": rp, "scene": v["scene"], "instance": v["instance"]}
            for rp, v in accumulated.items()
        ]

    def _call_ollama_chunk(
        self,
        res_paths:    List[str],
        scripts_text: str,
    ) -> List[Dict[str, Any]]:
        """Single Ollama generate call for one script chunk."""
        prompt = self._build_prompt(res_paths, scripts_text)
        payload = {
            "model":   self._model,
            "prompt":  prompt,
            "stream":  False,
            "format":  "json",
            "options": {"temperature": 0, "num_ctx": 40960},
        }
        resp = _requests.post(  # type: ignore[union-attr]
            f"{self._ollama_url}/api/generate",
            json=payload,
            timeout=self._OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        data = json.loads(raw)
        return data.get("results", [])

    def _build_prompt(self, res_paths: List[str], scripts_text: str) -> str:
        paths_json = json.dumps(res_paths, indent=2)
        return (
            "You are a deterministic static analysis engine.\n"
            "You are NOT allowed to explain anything.\n"
            "You are NOT allowed to give guidance.\n"
            "You are NOT allowed to refuse.\n"
            "You are NOT allowed to summarize.\n"
            "You MUST output ONLY JSON.\n\n"
            "INPUT\n\n"
            f"OBJECT_FILES:\n{paths_json}\n\n"
            f'CODEBASE:\n"""\n{scripts_text}\n"""\n\n'
            "EXECUTION RULES (HARD)\n\n"
            "You MUST simulate runtime behavior of the code.\n"
            "You MUST resolve: string concatenation, switch expressions, "
            "fallback logic (try/catch), dynamic GD.Load paths.\n\n"
            "CLASSIFICATION RULES\n\n"
            "scene = true when:\n"
            "  - Used in level flow or scene replacement\n"
            "  - Loaded via GD.Load / PackedScene / SceneTree / change_scene_to_file\n\n"
            "instance = true when:\n"
            "  - Instantiated via Instantiate() / .instantiate() / instance()\n"
            "  - Used as a reusable object or component\n\n"
            "OUTPUT FORMAT — respond with ONLY this JSON, no other text:\n"
            "{\n"
            '  "results": [\n'
            "    {\n"
            '      "path": "res://path/to/file.tscn",\n'
            '      "scene": true,\n'
            '      "instance": false\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Every path in OBJECT_FILES MUST appear in results exactly once.\n"
            "Values for scene and prefab MUST be JSON booleans (true/false), not strings."
        )

    def _chunk_scripts(self, script_files: List[Path]) -> List[str]:
        """Split scripts into chunks that each fit within the context window.

        Whole scripts are kept together wherever possible.  A script that alone
        exceeds _MAX_SCRIPT_CHARS_PER_CHUNK is placed in its own chunk and
        truncated only at that boundary — no other script data is lost.
        Returns an empty list when no scripts are readable.
        """
        chunks: List[str] = []
        current_parts: List[str] = []
        current_len = 0

        for sf in script_files:
            try:
                text = sf.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            header = f"\n// === {sf.name} ===\n"
            entry = header + text

            if current_len + len(entry) > self._MAX_SCRIPT_CHARS_PER_CHUNK:
                if current_parts:
                    chunks.append("".join(current_parts))
                    current_parts = []
                    current_len = 0
                # Script larger than one chunk — split across multiple chunks
                if len(entry) > self._MAX_SCRIPT_CHARS_PER_CHUNK:
                    for offset in range(0, len(entry), self._MAX_SCRIPT_CHARS_PER_CHUNK):
                        part = entry[offset : offset + self._MAX_SCRIPT_CHARS_PER_CHUNK]
                        if offset > 0:
                            part = f"\n// === {sf.name} (part {offset // self._MAX_SCRIPT_CHARS_PER_CHUNK + 1}) ===\n" + part
                        chunks.append(part)
                    continue

            current_parts.append(entry)
            current_len += len(entry)

        if current_parts:
            chunks.append("".join(current_parts))

        return chunks


def _parse_bool(val: Any) -> bool:
    """Coerce Ollama output (bool, 'true', '1') to Python bool."""
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# Material IR helper
# ---------------------------------------------------------------------------

def _parse_color(raw: str, default: Dict[str, float]) -> Dict[str, float]:
    """Parse 'Color(r, g, b, a)' string into a dict."""
    m = re.search(
        r'Color\s*\(\s*([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\s*,'
        r'\s*([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\s*\)',
        raw,
    )
    if m:
        return {
            "r": float(m.group(1)), "g": float(m.group(2)),
            "b": float(m.group(3)), "a": float(m.group(4)),
        }
    return default


def _build_material_ir(mat_type: str, props: Dict[str, str]) -> Dict[str, Any]:
    """Convert Godot sub-resource material properties to an IR material dict."""
    white = {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0}
    black = {"r": 0.0, "g": 0.0, "b": 0.0, "a": 1.0}
    return {
        "material_type":  "standard",
        "albedo_color":   _parse_color(props.get("albedo_color", ""), white),
        "metallic":       float(props.get("metallic", 0.0)),
        "roughness":      float(props.get("roughness", 1.0)),
        "emission_color": _parse_color(props.get("emission", ""), black),
        "alpha_mode":     props.get("transparency", "TRANSPARENCY_DISABLED"),
    }


# ---------------------------------------------------------------------------
# Godot node type → engine-agnostic IR node type
# ---------------------------------------------------------------------------

_GODOT_TO_IR_TYPE: Dict[str, str] = {
    # 3D nodes
    "Node3D":              "group",
    "MeshInstance3D":      "entity",
    "StaticBody3D":        "entity",
    "RigidBody3D":         "entity",
    "AnimatableBody3D":    "entity",
    "CharacterBody3D":     "entity",   # physics_body component added
    "CollisionShape3D":    "group",    # handled inline on parent
    "Area3D":              "group",    # trigger component added
    "Camera3D":            "camera",
    "Camera2D":            "camera",
    "DirectionalLight3D":  "light",
    "OmniLight3D":         "light",
    "SpotLight3D":         "light",
    "AudioStreamPlayer3D": "audio_source",
    "AudioStreamPlayer":   "audio_source",
    "GPUParticles3D":      "particle_system",
    "CPUParticles3D":      "particle_system",
    "Label3D":             "text_3d",
    "AnimationPlayer":     "group",    # animator component added
    "AnimationTree":       "group",    # animation_tree component added
    "RayCast3D":           "group",    # raycast component added
    # 2D / UI
    "CanvasLayer":         "ui_canvas",
    "Control":             "ui_element",
    "Panel":               "ui_element",
    "Button":              "ui_element",
    "Label":               "ui_element",
    "RichTextLabel":       "ui_element",
    "ProgressBar":         "ui_element",
    "HSlider":             "ui_element",
    "VSlider":             "ui_element",
    "GridContainer":       "ui_element",
    "HBoxContainer":       "ui_element",
    "VBoxContainer":       "ui_element",
    "Sprite2D":            "sprite",
    "TextureRect":         "ui_element",
    "Node2D":              "group",
}

_DEFAULT_IR_TYPE = "group"


# ---------------------------------------------------------------------------
# Animation parsing helpers (module-level so they can be used by tests too)
# ---------------------------------------------------------------------------

def _extract_dict_value(text: str, key: str) -> Optional[str]:
    """Return the raw value string for *key* from a Godot inline-dict string.

    Handles PackedFloat32Array(...), lists [...], and plain scalars as values.
    Returns None when the key is not present.
    """
    pattern = re.compile(
        r'"' + re.escape(key) + r'"\s*:\s*'
        r'(PackedFloat32Array\s*\([^)]*\)'   # PackedFloat32Array(...)
        r'|\[[^\]]*\]'                         # [...] list
        r'|[^,}\]]+)',                         # plain scalar
        re.DOTALL,
    )
    m = pattern.search(text)
    return m.group(1).strip() if m else None


def _parse_packed_floats(content: str) -> List[float]:
    """Parse comma-separated floats from the body of a PackedFloat32Array(...)."""
    result: List[float] = []
    for tok in content.split(","):
        tok = tok.strip()
        if tok:
            try:
                result.append(float(tok))
            except ValueError:
                pass
    return result


# ---------------------------------------------------------------------------
# UI layout helper
# ---------------------------------------------------------------------------

def _extract_ui_layout(props: Dict[str, str]) -> Dict[str, Any]:
    """Extract anchor/offset layout properties common to all Godot Control nodes."""
    try:
        preset = int(props.get("anchors_preset", -1))
    except (ValueError, TypeError):
        preset = -1
    try:
        al = float(props.get("anchor_left",   0.0))
        at = float(props.get("anchor_top",    0.0))
        ar = float(props.get("anchor_right",  0.0))
        ab = float(props.get("anchor_bottom", 0.0))
    except (ValueError, TypeError):
        al = at = ar = ab = 0.0
    try:
        ol = float(props.get("offset_left",   0.0))
        ot = float(props.get("offset_top",    0.0))
        or_ = float(props.get("offset_right", 0.0))
        ob  = float(props.get("offset_bottom", 0.0))
    except (ValueError, TypeError):
        ol = ot = or_ = ob = 0.0
    return {
        "anchors_preset": preset,
        "anchor_left":    al,
        "anchor_top":     at,
        "anchor_right":   ar,
        "anchor_bottom":  ab,
        "offset_left":    ol,
        "offset_top":     ot,
        "offset_right":   or_,
        "offset_bottom":  ob,
    }


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class GodotSceneParser:
    """Parse a Godot 4 .tscn file into the engine-agnostic scene IR.

    The IR schema mirrors the one produced by ir_builder.build_scene_ir()
    so the Unity exporter can consume both Unity-sourced and Godot-sourced IRs
    without modification.
    """

    # Section header patterns
    _GD_SCENE_RE    = re.compile(r'^\[gd_scene\s+(.+)\]$')
    _EXT_RES_RE     = re.compile(r'^\[ext_resource\s+(.+)\]$')
    _SUB_RES_RE     = re.compile(r'^\[sub_resource\s+(.+)\]$')
    _NODE_RE        = re.compile(r'^\[node\s+(.+)\]$')
    _CONN_RE        = re.compile(r'^\[connection\s+(.+)\]$')

    # Attribute key=value within a section header (unquoted or quoted values)
    _ATTR_RE        = re.compile(r'(\w+)\s*=\s*(?:"([^"]*)"|([\w./:-]+))')

    # ExtResource("id") in a property value or header body
    _EXTRES_VAL_RE  = re.compile(r'ExtResource\s*\(\s*"?(\w+)"?\s*\)')
    # instance=ExtResource("id") in a node header
    _INSTANCE_HDR_RE = re.compile(r'instance\s*=\s*ExtResource\s*\(\s*"?(\w+)"?\s*\)')

    # Transform3D(ax,ay,az, bx,by,bz, cx,cy,cz, ox,oy,oz)
    _TRANSFORM_RE   = re.compile(
        r'Transform3D\s*\(\s*'
        r'([-\d.e]+)\s*,\s*([-\d.e]+)\s*,\s*([-\d.e]+)\s*,\s*'
        r'([-\d.e]+)\s*,\s*([-\d.e]+)\s*,\s*([-\d.e]+)\s*,\s*'
        r'([-\d.e]+)\s*,\s*([-\d.e]+)\s*,\s*([-\d.e]+)\s*,\s*'
        r'([-\d.e]+)\s*,\s*([-\d.e]+)\s*,\s*([-\d.e]+)\s*\)'
    )
    # Vector3(x, y, z)
    _VEC3_RE        = re.compile(
        r'Vector3\s*\(\s*([-\d.e]+)\s*,\s*([-\d.e]+)\s*,\s*([-\d.e]+)\s*\)'
    )
    # BoxShape3D / SphereShape3D / CapsuleShape3D property names
    _SHAPE_TYPE_RE  = re.compile(r'(\w+Shape3D)\s*\(')

    # ── Animation parsing ────────────────────────────────────────────────────
    # NodePath("path") in track path properties
    _NODE_PATH_RE   = re.compile(r'NodePath\s*\(\s*"([^"]*)"\s*\)')
    # PackedFloat32Array(f, f, ...) — captures CSV float content
    _PACKED_F32_RE  = re.compile(r'PackedFloat32Array\s*\(\s*([^)]*)\s*\)')
    # "name": SubResource("id")  inside AnimationLibrary._data
    _ANIM_LIB_RE    = re.compile(r'"([^"]+)"\s*:\s*SubResource\s*\(\s*"?(\w+)"?\s*\)')
    # tracks/N/prop_name  as a standalone dict key (no "=")
    _TRACK_KEY_RE   = re.compile(r'^tracks/(\d+)/(\w+)$')
    # Vector3(x, y, z) in values arrays
    _VEC3_VAL_RE    = re.compile(
        r'Vector3\s*\(\s*([-\d.e]+)\s*,\s*([-\d.e]+)\s*,\s*([-\d.e]+)\s*\)')
    # Quaternion(x, y, z, w) in values arrays
    _QUAT_VAL_RE    = re.compile(
        r'Quaternion\s*\(\s*([-\d.e]+)\s*,\s*([-\d.e]+)\s*,\s*([-\d.e]+)\s*,\s*([-\d.e]+)\s*\)'
    )

    # ------------------------------------------------------------------ public

    def parse_file(self, path: Path) -> Dict[str, Any]:
        """Parse *path* (.tscn) and return a scene IR dict."""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise GodotParseError(f"Cannot read {path}: {exc}") from exc
        return self.parse_text(text, source_file=path.name)

    def parse_text(
        self, text: str, *, source_file: str = "unknown.tscn"
    ) -> Dict[str, Any]:
        """Parse raw .tscn text and return a scene IR dict."""
        sections = self._split_sections(text)
        if not sections:
            raise GodotParseError(f"No sections found in {source_file}")

        ext_resources   = self._collect_ext_resources(sections)
        sub_resources   = self._collect_sub_resources(sections)
        node_sections   = [s for s in sections if self._NODE_RE.match(s[0])]
        conn_sections   = [s for s in sections if self._CONN_RE.match(s[0])]

        if not node_sections:
            raise GodotParseError(f"No [node] sections found in {source_file}")

        scene_name = Path(source_file).stem

        # Build flat node list then wire hierarchy
        flat_nodes  = self._build_flat_nodes(
            node_sections, ext_resources, sub_resources
        )
        root_nodes  = self._wire_hierarchy(flat_nodes)

        # Collect signal connections and attach as event_bindings to source nodes
        connections = self._collect_connections(conn_sections)
        if connections:
            self._attach_connections(connections, flat_nodes)

        # Collect all unique collision_layer bitmask values used in this scene
        _layers_seen: set = set()
        _layers_used: List[int] = []
        for _n in flat_nodes:
            for _ck in ("rigidbody", "physics_body", "trigger"):
                _comp = _n.get("components", {}).get(_ck)
                if _comp and "collision_layer" in _comp:
                    _v = _comp["collision_layer"]
                    if _v not in _layers_seen:
                        _layers_seen.add(_v)
                        _layers_used.append(_v)

        # Build cross-file asset registry from all instance_ref entries.
        _asset_registry: Dict[str, Any] = {}

        def _collect_asset_registry(nodes: List[Dict[str, Any]]) -> None:
            for _n in nodes:
                ref = _n.get("instance_ref")
                if ref:
                    aid = ref.get("ir_asset_id", "")
                    if aid and aid not in _asset_registry:
                        _asset_registry[aid] = {
                            "ir_doc_id":   aid,
                            "source_path": ref.get("source_path", ""),
                            "source_guid": ref.get("source_guid", ""),
                            "source_uid":  ref.get("source_uid", ""),
                        }
                _collect_asset_registry(_n.get("children") or [])

        _collect_asset_registry(root_nodes)

        return {
            "ir_version":            "1.0",
            "ir_doc_kind":           "scene",   # may be overridden by caller after classification
            "source_engine":         "Godot",
            "source_engine_version": "4.5",
            "target_engine":         "Unity",
            "target_engine_version": "6000.3.9f1",
            "scene_id":              re.sub(r"[^a-z0-9]+", "_", scene_name.lower()),
            "scene_name":            scene_name,
            "source_file":           source_file,
            "node_count":            len(flat_nodes),
            "nodes":                 root_nodes,
            "asset_registry":        _asset_registry,
            "connections":           connections,
            "animations":            self._extract_scene_animations(sub_resources),
            "collision_layers_used": _layers_used,
            "coordinate_system": {
                "source":             "godot_right_handed",
                "target":             "unity_left_handed",
                "conversion_applied": True,
            },
        }

    # ----------------------------------------------------------------- private

    def _split_sections(
        self, text: str
    ) -> List[Tuple[str, List[str]]]:
        """Split .tscn into [(header_line, [property_lines]), ...]."""
        sections: List[Tuple[str, List[str]]] = []
        current_header: Optional[str] = None
        current_props:  List[str]     = []

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line.startswith("[") and line.endswith("]"):
                if current_header is not None:
                    sections.append((current_header, current_props))
                current_header = line
                current_props  = []
            elif current_header is not None and line and not line.startswith(";"):
                current_props.append(line)

        if current_header is not None:
            sections.append((current_header, current_props))

        return sections

    def _parse_attrs(self, header_body: str) -> Dict[str, str]:
        """Parse key=value pairs from a section header body."""
        return {
            m.group(1): (m.group(2) if m.group(2) is not None else m.group(3))
            for m in self._ATTR_RE.finditer(header_body)
        }

    def _collect_ext_resources(
        self, sections: List[Tuple[str, List[str]]]
    ) -> Dict[str, Dict[str, str]]:
        """Return id → {type, path, uid} for every [ext_resource]."""
        result: Dict[str, Dict[str, str]] = {}
        for header, _ in sections:
            m = self._EXT_RES_RE.match(header)
            if not m:
                continue
            attrs = self._parse_attrs(m.group(1))
            res_id = attrs.get("id", "")
            if res_id:
                result[res_id] = {
                    "type": attrs.get("type", ""),
                    "path": attrs.get("path", ""),
                    "uid":  attrs.get("uid",  ""),
                }
        return result

    def _collect_sub_resources(
        self, sections: List[Tuple[str, List[str]]]
    ) -> Dict[str, Dict[str, Any]]:
        """Return id → {type, props} for every [sub_resource]."""
        result: Dict[str, Dict[str, Any]] = {}
        for header, props in sections:
            m = self._SUB_RES_RE.match(header)
            if not m:
                continue
            attrs  = self._parse_attrs(m.group(1))
            res_id = attrs.get("id", "")
            if res_id:
                res_type = attrs.get("type", "")
                entry: Dict[str, Any] = {
                    "type":  res_type,
                    "props": self._parse_props(props),
                }
                # Keep raw lines so animation parsers can handle multi-line values
                if res_type in ("Animation", "AnimationLibrary"):
                    entry["raw_text"] = "\n".join(props)
                result[res_id] = entry
        return result

    def _parse_props(self, lines: List[str]) -> Dict[str, str]:
        """Parse 'key = value' property lines into a dict."""
        props: Dict[str, str] = {}
        for ln in lines:
            eq = ln.find("=")
            if eq == -1:
                continue
            key = ln[:eq].strip()
            val = ln[eq + 1:].strip()
            props[key] = val
        return props

    def _build_flat_nodes(
        self,
        node_sections:  List[Tuple[str, List[str]]],
        ext_resources:  Dict[str, Dict[str, str]],
        sub_resources:  Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build one IR-node dict per [node] section (no hierarchy yet)."""
        nodes: List[Dict[str, Any]] = []
        counter = [0]

        for header, prop_lines in node_sections:
            m = self._NODE_RE.match(header)
            if not m:
                continue
            header_body = m.group(1)
            attrs       = self._parse_attrs(header_body)
            props       = self._parse_props(prop_lines)

            # Nodes with `instance=ExtResource(...)` have no explicit type
            instance_m = self._INSTANCE_HDR_RE.search(header_body)
            if instance_m and "type" not in attrs:
                node_type = "_InstanceNode"   # sentinel; handled below
            else:
                node_type = attrs.get("type", "Node3D")

            name   = attrs.get("name", f"Node_{counter[0]}")
            parent = attrs.get("parent", "")

            counter[0] += 1
            node_id = f"godot_node_{counter[0]:04d}"

            transform  = self._extract_transform(props)
            ir_type    = _GODOT_TO_IR_TYPE.get(node_type, _DEFAULT_IR_TYPE)
            components, extra_node_fields = self._extract_components(
                node_type, props, ext_resources, sub_resources
            )

            # ── ir_node_kind determination ───────────────────────────────
            # Assigned before the instance block so both branches can override.
            if instance_m:
                ir_node_kind = "instance_node"
            elif not parent:
                ir_node_kind = "scene_root"   # may be promoted to prefab_root later
            else:
                ir_node_kind = "regular"

            # ── instance_ref / asset_reference ───────────────────────────
            node_instance_ref: Optional[Dict[str, Any]] = None
            if instance_m:
                res_id      = instance_m.group(1)
                res_info    = ext_resources.get(res_id, {})
                source_path = res_info.get("path", "")

                if Path(source_path).suffix.lower() in self._MODEL_EXTENSIONS:
                    # External 3-D model instance → engine-agnostic intent fields
                    rel_path = source_path.removeprefix("res://")
                    extra_node_fields["asset_reference"] = {
                        "path":   rel_path,
                        "type":   "mesh",
                        "format": Path(rel_path).suffix.lower(),
                    }
                    extra_node_fields["instancing"] = {"mode": "instance"}
                else:
                    components["instance_ref"] = {
                        "source_res_id":   res_id,
                        "source_res_path": source_path,
                        "source_res_uid":  res_info.get("uid", ""),
                    }
                    # Unified instance_ref — engine-agnostic pointer to the source file.
                    node_instance_ref = {
                        "ir_asset_id":        _path_to_ir_asset_id(source_path),
                        "source_path":        source_path,
                        "source_guid":        "",
                        "source_uid":         res_info.get("uid", ""),
                        "overrides":          [],
                        "removed_components": [],
                        "added_components":   [],
                        "is_variant_prefab":  False,
                    }
                if ir_type == _DEFAULT_IR_TYPE and node_type == "_InstanceNode":
                    ir_type = "group"   # instance nodes are group containers

            # ── script component ─────────────────────────────────────────
            script_raw = props.get("script", "")
            if script_raw:
                er_m = self._EXTRES_VAL_RE.search(script_raw)
                if er_m:
                    res_id   = er_m.group(1)
                    res_info = ext_resources.get(res_id, {})
                    components["script"] = {
                        "class_name":      Path(res_info.get("path", "")).stem,
                        "source_path":     res_info.get("path", ""),
                        "exported_fields": {},
                    }

            node_entry: Dict[str, Any] = {
                "node_id":      node_id,
                "node_name":    name,
                "node_type":    ir_type,
                "ir_node_kind": ir_node_kind,
                "child_index":  0,   # assigned in _wire_hierarchy
                "godot_type":   node_type,
                "parent_path":  parent,
                "is_spatial":   True,
                "transform":    transform,
                "components":   components,
                **extra_node_fields,
                "original_data": {
                    "source_object_id":    node_id,
                    "source_transform_id": node_id,
                    "source_flags":        {},
                    "mesh_ref":            props.get("mesh", ""),
                    "material_refs":       [],
                },
                "meta": {
                    "conversion_status": "mapped",
                    "confidence":        0.9,
                    "ai_generated":      False,
                    "user_edited":       False,
                    "requires_review":   False,
                    "notes":             "",
                    "warnings":          [],
                    "errors":            [],
                },
                "children": [],
            }
            if node_instance_ref is not None:
                node_entry["instance_ref"] = node_instance_ref

            nodes.append(node_entry)

        return nodes

    def _extract_transform(self, props: Dict[str, str]) -> Dict[str, Any]:
        """Extract a Godot transform into the IR. Values are raw Godot space."""
        default_transform = {
            "position": [0.0, 0.0, 0.0],
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            "scale":    [1.0, 1.0, 1.0],
        }

        raw = props.get("transform", "")
        if not raw:
            return default_transform

        m = self._TRANSFORM_RE.search(raw)
        if not m:
            return default_transform

        vals = [float(m.group(i)) for i in range(1, 13)]
        # Zero Godot epsilon artifacts in the rotation basis only (e.g. sin(0) = 5.1974e-16,
        # sin(-90) = 4.37114e-08).  Origin is left untouched.
        vals[:9] = [0.0 if abs(v) < 1e-7 else v for v in vals[:9]]
        # Basis columns: right=(vals[0..2]), up=(vals[3..5]), back=(vals[6..8])
        # Origin: (vals[9..11])
        ax, ay, az = vals[0], vals[1], vals[2]
        bx, by, bz = vals[3], vals[4], vals[5]
        cx, cy, cz = vals[6], vals[7], vals[8]
        ox, oy, oz = vals[9], vals[10], vals[11]

        # Godot basis = S*R (scale then rotate), stored as column vectors.
        # Scale = row magnitudes: row i = (col0[i], col1[i], col2[i])
        import math
        sx = math.sqrt(ax*ax + bx*bx + cx*cx)  # row 0
        sy = math.sqrt(ay*ay + by*by + cy*cy)  # row 1
        sz = math.sqrt(az*az + bz*bz + cz*cz)  # row 2

        # Rotation columns: R = S^-1 * M, so R[i,j] = M[i,j] / scale_i
        def safe_div(num, den):
            return num / den if den > 1e-9 else 0.0

        right = [safe_div(ax, sx), safe_div(ay, sy), safe_div(az, sz)]
        up    = [safe_div(bx, sx), safe_div(by, sy), safe_div(bz, sz)]
        back  = [safe_div(cx, sx), safe_div(cy, sy), safe_div(cz, sz)]

        # Convert 3x3 rotation matrix → quaternion (Shepperd's method)
        quat = _mat3_to_quat(right, up, back)

        godot_pos   = [ox, oy, oz]
        godot_scale = [sx, sy, sz]

        return {
            "position": _pos_godot_to_unity(godot_pos),
            "rotation": _quat_godot_to_unity(quat),
            "scale":    _scale_godot_to_unity(godot_scale),
        }

    # File extensions that Unity imports as 3D model assets (PrefabInstance).
    _MODEL_EXTENSIONS = frozenset({".obj", ".fbx", ".gltf", ".glb", ".dae"})

    def _extract_components(
        self,
        node_type:     str,
        props:         Dict[str, str],
        ext_resources: Dict[str, Dict[str, str]],
        sub_resources: Dict[str, Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Build the IR components dict and extra top-level node fields.

        Returns ``(components, extra_fields)`` where ``extra_fields`` contains
        engine-agnostic instancing keys (``asset_reference``, ``instancing``)
        that belong at the node level rather than inside ``components``.
        """
        components: Dict[str, Any] = {}
        extra_fields: Dict[str, Any] = {}

        if node_type == "MeshInstance3D":
            mesh_ref_raw = props.get("mesh", "")
            res_id_m     = re.search(r'ExtResource\s*\(\s*"?(\w+)"?\s*\)', mesh_ref_raw)
            if res_id_m:
                res_id   = res_id_m.group(1)
                res_info = ext_resources.get(res_id, {})
                raw_path = res_info.get("path", "")
                rel_path = raw_path.removeprefix("res://")
                if Path(rel_path).suffix.lower() in self._MODEL_EXTENSIONS:
                    # External 3D model file → Unity PrefabInstance
                    extra_fields["asset_reference"] = {
                        "path":   rel_path,
                        "type":   "mesh",
                        "format": Path(rel_path).suffix.lower(),
                    }
                    extra_fields["instancing"] = {"mode": "instance"}
                else:
                    components["mesh"] = {
                        "mesh_type":        "static",
                        "mesh_source_path": raw_path,
                        "mesh_asset_guid":  res_info.get("uid", ""),
                        "cast_shadows":     True,
                        "receive_shadows":  True,
                    }

            # Material: surface_material_override/0, material_override, or material
            mat_raw = props.get(
                "surface_material_override/0",
                props.get("material_override",
                props.get("material", "")),
            )
            if mat_raw:
                sr_m = re.search(r'SubResource\s*\(\s*"?(\w+)"?\s*\)', mat_raw)
                er_m = re.search(r'ExtResource\s*\(\s*"?(\w+)"?\s*\)', mat_raw)
                if sr_m:
                    sr = sub_resources.get(sr_m.group(1), {})
                    components["material"] = _build_material_ir(
                        sr.get("type", ""), sr.get("props", {})
                    )
                elif er_m:
                    res_info = ext_resources.get(er_m.group(1), {})
                    components["material"] = {
                        "material_type": "external",
                        "material_path": res_info.get("path", ""),
                    }

            # Visual render layer bitmask (default = 1 = layer 1 only).
            # Non-default values are stored so the exporter can set m_Layer.
            _raw_layers = props.get("layers", "")
            if _raw_layers:
                try:
                    _layers_int = int(_raw_layers)
                    if _layers_int != 1:
                        components["render_layers"] = {"bitmask": _layers_int}
                except (ValueError, TypeError):
                    pass

            # Skeleton NodePath — the ancestor Skeleton3D used for skinned mesh
            # rendering. Captured for review; no Unity equivalent for static OBJ.
            _skel_raw = props.get("skeleton", "")
            if _skel_raw:
                _np_m = re.search(r'NodePath\s*\(\s*"([^"]*)"\s*\)', _skel_raw)
                if _np_m:
                    extra_fields["skeleton_path"] = _np_m.group(1)

        elif node_type == "CollisionShape3D":
            shape_raw  = props.get("shape", "")
            shape_type = ""
            st_m = self._SHAPE_TYPE_RE.search(shape_raw)
            if st_m:
                shape_type = st_m.group(1)

            # Sub-resource shapes
            sr_m = re.search(r'SubResource\s*\(\s*"?(\w+)"?\s*\)', shape_raw)
            if sr_m:
                sr = sub_resources.get(sr_m.group(1), {})
                shape_type = sr.get("type", shape_type)
                sr_props   = sr.get("props", {})
            else:
                sr_props = {}

            components["colliders"] = [
                self._build_collider_ir(shape_type, sr_props, props)
            ]

        elif node_type == "CharacterBody3D":
            components["physics_body"] = {
                "type":            "character",
                "collision_layer": int(props.get("collision_layer", 1)),
                "collision_mask":  int(props.get("collision_mask",  1)),
                "motion_mode":     props.get("motion_mode", "MOTION_MODE_GROUNDED"),
            }

        elif node_type == "Area3D":
            components["trigger"] = {
                "is_trigger":      True,
                "collision_layer": int(props.get("collision_layer", 1)),
                "collision_mask":  int(props.get("collision_mask",  1)),
                "monitoring":      props.get("monitoring", "true").lower() == "true",
                "monitorable":     props.get("monitorable", "true").lower() == "true",
            }

        elif node_type in ("StaticBody3D", "RigidBody3D", "AnimatableBody3D"):
            is_kinematic = node_type == "AnimatableBody3D"
            is_static    = node_type == "StaticBody3D"
            components["rigidbody"] = {
                "mass":            float(props.get("mass", 1.0)),
                "is_kinematic":    is_kinematic or is_static,
                "type":            "static" if is_static else ("kinematic" if is_kinematic else "dynamic"),
                "use_gravity":     not is_static,
                "drag":            0.0,
                "angular_drag":    0.0,
                "collision_layer": int(props.get("collision_layer", 1)),
                "collision_mask":  int(props.get("collision_mask",  1)),
            }

        elif node_type in ("Camera3D", "Camera2D"):
            components["camera"] = {
                "fov":        float(props.get("fov", 75.0)),
                "near_clip":  float(props.get("near", 0.05)),
                "far_clip":   float(props.get("far", 4000.0)),
                "is_main":    False,
                "projection": "perspective",
            }

        elif node_type in ("DirectionalLight3D", "OmniLight3D", "SpotLight3D"):
            _light_type_map = {
                "DirectionalLight3D": "directional",
                "OmniLight3D":        "point",
                "SpotLight3D":        "spot",
            }
            white = {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0}
            color = _parse_color(props.get("light_color", ""), white)
            range_val = float(props.get(
                "omni_range", props.get("spot_range", 10.0)
            ))
            components["light"] = {
                "light_type":   _light_type_map[node_type],
                "color":        color,
                "intensity":    float(props.get("light_energy", 1.0)),
                "range":        range_val,
                "spot_angle":   float(props.get("spot_angle", 45.0)),
                "cast_shadows": props.get("shadow_enabled", "false").lower() == "true",
            }

        elif node_type in ("AudioStreamPlayer3D", "AudioStreamPlayer"):
            spatial = node_type == "AudioStreamPlayer3D"
            components["audio_source"] = {
                "volume":        float(props.get("volume_db", 0.0)),
                "autoplay":      props.get("autoplay", "false").lower() == "true",
                "loop":          False,
                "spatial":       spatial,
                "spatial_blend": 1.0 if spatial else 0.0,
                "max_distance":  float(props.get("max_distance", 2000.0)),
            }

        elif node_type == "AnimationPlayer":
            components["animator"] = {
                "root_node":      props.get("root_node", "."),
                "autoplay":       props.get("autoplay", ""),
                "playback_speed": float(props.get("speed_scale", 1.0)),
            }

        elif node_type == "AnimationTree":
            components["animation_tree"] = {
                "tree_root":   props.get("tree_root", ""),
                "anim_player": props.get("anim_player", ""),
                "active":      props.get("active", "true").lower() == "true",
            }

        elif node_type in ("GPUParticles3D", "CPUParticles3D"):
            components["particle_system"] = {
                "emitting":   props.get("emitting", "true").lower() == "true",
                "amount":     int(props.get("amount", 8)),
                "lifetime":   float(props.get("lifetime", 1.0)),
                "one_shot":   props.get("one_shot", "false").lower() == "true",
                "explosiveness": float(props.get("explosiveness", 0.0)),
                "preprocess": float(props.get("preprocess", 0.0)),
            }

        elif node_type == "RayCast3D":
            target_raw = props.get("target_position", "")
            tm = self._VEC3_RE.search(target_raw)
            target = [0.0, 0.0, -1.0]
            if tm:
                target = [float(tm.group(1)), float(tm.group(2)), float(tm.group(3))]
            components["raycast"] = {
                "target_position":   target,
                "collision_mask":    int(props.get("collision_mask", 1)),
                "hit_from_inside":   props.get("hit_from_inside",   "false").lower() == "true",
                "collide_with_areas":props.get("collide_with_areas","false").lower() == "true",
                "enabled":           props.get("enabled", "true").lower() == "true",
            }

        elif node_type == "Sprite2D":
            tex_raw = props.get("texture", "")
            er_m = re.search(r'ExtResource\s*\(\s*"?(\w+)"?\s*\)', tex_raw)
            texture_path = ""
            if er_m:
                res_info = ext_resources.get(er_m.group(1), {})
                texture_path = res_info.get("path", "")
            components["sprite"] = {
                "texture_path": texture_path,
            }

        elif node_type == "CanvasLayer":
            components["ui"] = {
                "ui_type":    "canvas",
                "layer":      int(props.get("layer", 1)),
                "anchor_min": [0.0, 0.0],
                "anchor_max": [1.0, 1.0],
            }

        elif node_type in ("Control", "Panel"):
            components["ui"] = {
                "ui_type":      "ui_element",
                "element_kind": "widget",
                **_extract_ui_layout(props),
            }

        elif node_type == "Button":
            _white = {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0}
            _font_color = _parse_color(
                props.get("theme_override_colors/font_color", ""), _white
            )
            try:
                _font_size = float(props.get("theme_override_font_sizes/font_size", 14))
            except (ValueError, TypeError):
                _font_size = 14.0
            _bg_color = {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0}
            _style_raw = props.get("theme_override_styles/normal", "")
            if _style_raw:
                _sr_m = re.search(r'SubResource\s*\(\s*"?(\w+)"?\s*\)', _style_raw)
                if _sr_m:
                    _sr = sub_resources.get(_sr_m.group(1), {})
                    if _sr.get("type") == "StyleBoxFlat":
                        _bg_raw = _sr.get("props", {}).get("bg_color", "")
                        if _bg_raw:
                            _bg_color = _parse_color(_bg_raw, _bg_color)
            _text = props.get("text", "").strip('"')
            components["ui"] = {
                "ui_type":      "ui_element",
                "element_kind": "button",
                "text":         _text,
                "font_color":   _font_color,
                "font_size":    _font_size,
                "bg_color":     _bg_color,
                **_extract_ui_layout(props),
            }

        elif node_type in ("Label", "RichTextLabel"):
            _white = {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0}
            _font_color = _parse_color(
                props.get("theme_override_colors/font_color", ""), _white
            )
            try:
                _font_size = float(props.get("theme_override_font_sizes/font_size", 14))
            except (ValueError, TypeError):
                _font_size = 14.0
            _text = props.get("text", "").strip('"')
            components["ui"] = {
                "ui_type":        "ui_element",
                "element_kind":   "label",
                "text":           _text,
                "bbcode_enabled": node_type == "RichTextLabel",
                "font_color":     _font_color,
                "font_size":      _font_size,
                **_extract_ui_layout(props),
            }

        elif node_type in ("ProgressBar", "HSlider", "VSlider"):
            components["ui"] = {
                "ui_type":      "ui_element",
                "element_kind": "slider_h" if node_type != "VSlider" else "slider_v",
                "min_value":    float(props.get("min_value", 0.0)),
                "max_value":    float(props.get("max_value", 100.0)),
                "value":        float(props.get("value", 0.0)),
                **_extract_ui_layout(props),
            }

        elif node_type in ("GridContainer", "HBoxContainer", "VBoxContainer"):
            _kind_map = {"GridContainer": "grid", "HBoxContainer": "horizontal", "VBoxContainer": "vertical"}
            components["ui_layout"] = {
                "kind":       _kind_map[node_type],
                "columns":    int(props.get("columns", 1)),
                "anchor_min": [0.0, 0.0],
                "anchor_max": [1.0, 1.0],
            }

        else:
            log.error("[G2U] Unrecognized node type '%s' — emitting as empty GameObject", node_type)

        return components, extra_fields

    def _build_collider_ir(
        self,
        shape_type: str,
        sr_props:   Dict[str, str],
        node_props: Dict[str, str],
    ) -> Dict[str, Any]:
        """Build a collider IR dict from a Godot shape type and its properties."""
        if "BoxShape3D" in shape_type:
            size_raw = sr_props.get("size", "")
            vm = self._VEC3_RE.search(size_raw)
            if vm:
                hx, hy, hz = float(vm.group(1))/2, float(vm.group(2))/2, float(vm.group(3))/2
            else:
                hx = hy = hz = 0.5
            return {
                "collider_type": "box",
                "center":   [0.0, 0.0, 0.0],
                "size":     [hx * 2, hy * 2, hz * 2],
                "is_trigger": False,
            }
        elif "SphereShape3D" in shape_type:
            radius = float(sr_props.get("radius", 0.5))
            return {
                "collider_type": "sphere",
                "center":   [0.0, 0.0, 0.0],
                "radius":   radius,
                "is_trigger": False,
            }
        elif "CapsuleShape3D" in shape_type:
            radius = float(sr_props.get("radius", 0.5))
            height = float(sr_props.get("height", 2.0))
            return {
                "collider_type": "capsule",
                "center":   [0.0, 0.0, 0.0],
                "radius":   radius,
                "height":   height,
                "direction": 1,   # Y-axis
                "is_trigger": False,
            }
        elif "ConcavePolygonShape3D" in shape_type:
            return {
                "collider_type": "concave_mesh",
                "center":   [0.0, 0.0, 0.0],
                "is_trigger": False,
            }
        elif "ConvexPolygonShape3D" in shape_type:
            return {
                "collider_type": "convex_mesh",
                "center":   [0.0, 0.0, 0.0],
                "is_trigger": False,
            }
        return {
            "collider_type": "box",
            "center":   [0.0, 0.0, 0.0],
            "size":     [1.0, 1.0, 1.0],
            "is_trigger": False,
        }

    def _wire_hierarchy(
        self, flat_nodes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert parent_path strings into nested children lists.

        After wiring, assigns child_index on every node based on its
        position among siblings (file order = Godot scene order).
        """
        by_name: Dict[str, Dict[str, Any]] = {}
        roots:   List[Dict[str, Any]]      = []

        for node in flat_nodes:
            by_name[node["node_name"]] = node

        for node in flat_nodes:
            parent_path = node.pop("parent_path", "")
            if not parent_path:
                # No parent attribute → scene root
                roots.append(node)
            elif parent_path == ".":
                # "." = direct child of scene root
                if roots:
                    roots[0]["children"].append(node)
                else:
                    roots.append(node)  # graceful fallback
            else:
                # parent_path is like "ParentName" or "A/B/C"
                parent_name = parent_path.split("/")[-1]
                parent_node = by_name.get(parent_name)
                if parent_node and parent_node is not node:
                    parent_node["children"].append(node)
                else:
                    roots.append(node)

        # Assign child_index to every node based on its position among siblings.
        # File order is the canonical order in Godot .tscn files.
        def _assign_indices(nodes: List[Dict[str, Any]]) -> None:
            for idx, n in enumerate(nodes):
                n["child_index"] = idx
                _assign_indices(n.get("children") or [])

        _assign_indices(roots)
        return roots

    def _collect_connections(
        self, conn_sections: List[Tuple[str, List[str]]]
    ) -> List[Dict[str, str]]:
        """Parse [connection] sections into a list of connection dicts."""
        connections: List[Dict[str, str]] = []
        for header, _props in conn_sections:
            m = self._CONN_RE.match(header)
            if not m:
                continue
            attrs = self._parse_attrs(m.group(1))
            signal  = attrs.get("signal", "")
            from_   = attrs.get("from", "")
            to      = attrs.get("to", "")
            method  = attrs.get("method", "")
            if signal and from_ and method:
                connections.append({
                    "signal": signal,
                    "from":   from_,
                    "to":     to,
                    "method": method,
                })
        return connections

    def _attach_connections(
        self,
        connections: List[Dict[str, str]],
        flat_nodes:  List[Dict[str, Any]],
    ) -> None:
        """Add event_bindings to nodes that emit signals."""
        by_name: Dict[str, Dict[str, Any]] = {n["node_name"]: n for n in flat_nodes}
        for conn in connections:
            from_name = conn["from"].split("/")[-1] if "/" in conn["from"] else conn["from"]
            if from_name == ".":
                # "." = scene root
                node = flat_nodes[0] if flat_nodes else None
            else:
                node = by_name.get(from_name)
            if node is not None:
                bindings = node["components"].setdefault("event_bindings", [])
                bindings.append({
                    "event_name": conn["signal"],
                    "target":     conn["to"],
                    "method":     conn["method"],
                })

    # ---------------------------------------------------------------- animation

    def _extract_scene_animations(
        self, sub_resources: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build the scene-level animations list from Animation sub-resources.

        Links AnimationLibrary._data entries to Animation sub-resources to get
        display names, then parses keyframe data from each Animation block.
        """
        # Build sub-resource id → display name via AnimationLibrary._data
        anim_id_to_name: Dict[str, str] = {}
        for res in sub_resources.values():
            if res["type"] == "AnimationLibrary":
                for m in self._ANIM_LIB_RE.finditer(res.get("raw_text", "")):
                    anim_id_to_name[m.group(2)] = m.group(1)

        animations: List[Dict[str, Any]] = []
        for res_id, res in sub_resources.items():
            if res["type"] != "Animation":
                continue
            props    = res["props"]
            raw_text = res.get("raw_text", "")

            # Name priority: library entry > resource_name > sub-resource id
            name = (anim_id_to_name.get(res_id)
                    or props.get("resource_name", "").strip('"')
                    or res_id)

            try:
                length = float(props.get("length", "1.0"))
            except ValueError:
                length = 1.0
            try:
                loop_mode = int(props.get("loop_mode", "0").strip('"'))
            except ValueError:
                loop_mode = 0

            tracks = self._parse_animation_tracks_raw(raw_text)
            animations.append({
                "name":   name,
                "length": length,
                "loop":   loop_mode != 0,
                "tracks": tracks,
            })
        return animations

    def _parse_animation_tracks_raw(self, raw_text: str) -> List[Dict[str, Any]]:
        """Parse tracks/N/* property lines from raw section text.

        Handles multi-line values (e.g. keys = { ... }) by accumulating
        continuation lines until the next tracks/N/ key is found.
        """
        track_data: Dict[int, Dict[str, str]] = {}
        current_key: Optional[Tuple[int, str]] = None
        current_val_lines: List[str] = []

        def flush() -> None:
            if current_key is not None:
                idx, prop = current_key
                track_data.setdefault(idx, {})[prop] = " ".join(current_val_lines)

        for raw_ln in raw_text.split("\n"):
            ln = raw_ln.strip()
            if not ln:
                continue
            eq_pos = ln.find("=")
            if eq_pos != -1:
                key_part = ln[:eq_pos].strip()
                val_part = ln[eq_pos + 1:].strip()
                if self._TRACK_KEY_RE.match(key_part):
                    flush()
                    m = self._TRACK_KEY_RE.match(key_part)
                    current_key = (int(m.group(1)), m.group(2))  # type: ignore[union-attr]
                    current_val_lines = [val_part]
                    continue
            if current_key is not None:
                current_val_lines.append(ln)
        flush()

        tracks: List[Dict[str, Any]] = []
        for idx in sorted(track_data):
            td       = track_data[idx]
            t_type   = td.get("type", "").strip('"')
            path_raw = td.get("path", "")
            keys_raw = td.get("keys", "")

            np_m      = self._NODE_PATH_RE.search(path_raw)
            node_path = np_m.group(1) if np_m else path_raw.strip('"')

            if ":" in node_path:
                node_part, prop_part = node_path.split(":", 1)
            else:
                node_part, prop_part = node_path, ""
            node_part = node_part.lstrip("./")

            keyframes = self._parse_track_keyframes(t_type, keys_raw)
            tracks.append({
                "type":      t_type,
                "node_path": node_part,
                "property":  prop_part,
                "keyframes": keyframes,
            })
        return tracks

    def _parse_track_keyframes(
        self, track_type: str, keys_raw: str
    ) -> List[Dict[str, Any]]:
        """Parse keyframes for *track_type* from the raw *keys* value string.

        Raw Godot-space values are stored — no coordinate conversion.
        Conversion is performed by unity_scene_exporter.py:
          position_3d  — stored as-is; exporter negates Z
          rotation_3d  — stored as-is; exporter applies C*R*C
          scale_3d     — no conversion needed
          value        — raw float pass-through
        """
        keyframes: List[Dict[str, Any]] = []

        times_raw = _extract_dict_value(keys_raw, "times") or ""
        times_m   = self._PACKED_F32_RE.search(times_raw)
        if not times_m:
            return keyframes
        times = _parse_packed_floats(times_m.group(1))
        if not times:
            return keyframes

        if track_type in ("position_3d", "scale_3d"):
            vals_block = _extract_dict_value(keys_raw, "values") or ""
            vectors    = self._VEC3_VAL_RE.findall(vals_block)
            for t, (sx, sy, sz) in zip(times, vectors):
                x, y, z = float(sx), float(sy), float(sz)
                keyframes.append({"time": t, "x": x, "y": y, "z": z})

        elif track_type == "rotation_3d":
            vals_block = _extract_dict_value(keys_raw, "values") or ""
            quats      = self._QUAT_VAL_RE.findall(vals_block)
            for t, (sqx, sqy, sqz, sqw) in zip(times, quats):
                qx, qy, qz, qw = float(sqx), float(sqy), float(sqz), float(sqw)
                keyframes.append({"time": t, "x": qx, "y": qy, "z": qz, "w": qw})

        elif track_type == "value":
            vals_block = _extract_dict_value(keys_raw, "values") or ""
            float_strs = re.findall(r'[-\d.e]+', vals_block)
            for t, v_str in zip(times, float_strs):
                try:
                    keyframes.append({"time": t, "value": float(v_str)})
                except ValueError:
                    keyframes.append({"time": t, "value": 0.0})

        return keyframes


# ---------------------------------------------------------------------------
# Math helper — 3×3 rotation matrix → quaternion  (Shepperd's method)
# ---------------------------------------------------------------------------

def _mat3_to_quat(
    right: List[float], up: List[float], back: List[float]
) -> Dict[str, float]:
    """Convert a normalised 3×3 rotation matrix to a quaternion."""
    import math
    m00, m10, m20 = right[0], right[1], right[2]
    m01, m11, m21 = up[0],    up[1],    up[2]
    m02, m12, m22 = back[0],  back[1],  back[2]

    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    return {"x": x, "y": y, "z": z, "w": w}
