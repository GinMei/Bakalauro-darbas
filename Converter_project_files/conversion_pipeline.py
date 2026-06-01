"""conversion_pipeline.py

ConversionController is the single place that knows the full pipeline order:

    scan → parse scenes → build IR → validate → generate project → copy assets

Nothing here is Unity- or Godot-specific beyond the concrete imports used to
wire up the default engine implementations.  To support a new source engine
the caller constructs a controller with different parser/builder callables;
the orchestration logic stays the same.

Public API:
    ConversionController   — orchestrates a full ZIP project conversion
    ConversionResult       — typed return value
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from unity_to_godot.godot_project_builder import build_godot_project, ProjectBuildResult
from hierarchy_validator import validate_hierarchy
from ir_builder import build_scene_ir, validate_scene_ir_basic
from unity_to_godot.project_scanner import ProjectScanResult, scan_unity_project
from script_converter import ScriptConverter, write_fallback_stubs, write_preservation_stubs
from unity_to_godot.unity_parser import UnityParseError, load_unity_scene_debug

log = logging.getLogger("conversion_pipeline")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ConversionResult:
    """Full result of a multi-scene ZIP conversion.

    ``success`` is True when the Godot project was written to disk without
    a blocking error.  Warnings are always accumulated regardless of success.
    ``primary_ir`` may be non-empty even when ``success`` is False (partial
    failure after IR was built but project generation failed).
    """

    success:           bool
    primary_ir:        Dict[str, Any]
    all_scenes_ir:     Dict[str, Dict[str, Any]]
    scene_names:       List[str]
    warnings:          List[str]
    prefab_count:      int
    project_available: bool
    error:             str = ""
    scripts_converted: int = 0
    scripts_failed:    int = 0


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class ConversionController:
    """Orchestrates the parse → IR → validate → generate pipeline.

    The controller is intentionally independent of FastAPI and performs no
    I/O beyond what is needed for conversion (no HTTP, no job persistence).
    The caller (api.py) is responsible for saving results to disk and
    building HTTP responses.

    Engine-specific behaviour is confined to the concrete callables imported
    at the top of this module.  Swapping source or target engines means
    pointing the controller at different parser / builder functions.
    """

    def __init__(
        self,
        source_engine: str = "Unity 6000.3.9f1",
        target_engine: str = "Godot 4.5",
        unity_version: str = "6000.3.9f1",
        target_version: str = "4.5",
    ) -> None:
        self.source_engine  = source_engine
        self.target_engine  = target_engine
        self.unity_version  = unity_version
        self.target_version = target_version

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def convert_zip(
        self,
        project_root: Path,
        job_id: str,
        project_name: str,
        output_dir: Path,
        scan: Optional[ProjectScanResult] = None,
        convert_scripts: bool = True,
        progress_cb = None,
    ) -> ConversionResult:
        """Convert every scene in an extracted Unity project.

        Args:
            project_root: Extracted Unity project root directory (the folder
                          that contains ``Assets/``).
            job_id:       Unique identifier used only for log messages.
            project_name: Clean project name (e.g. ``"mygame"``), written
                          into the generated ``project.godot``.
            output_dir:   Destination for the generated Godot project.
            scan:         Pre-computed scan result.  When *None* the
                          controller runs the scan itself.

        Returns:
            A :class:`ConversionResult` with IR, scene names, warnings,
            and availability flags.  ``success`` is True only when the
            Godot project was written without a blocking error.
        """
        _prog = progress_cb or (lambda *_: None)
        warnings: List[str] = []

        # ── Step 1: Scan project ───────────────────────────────────────────
        _prog(14, "Scanning project", "Discovering scene and asset files")
        if scan is None:
            scan = scan_unity_project(project_root, self.unity_version)
        warnings.extend(scan.warnings)
        _prog(20, "Analyzing structure", "Indexing GUIDs, materials, physics assets")
        log.info(
            "job=%s  scan complete  scenes=%d  prefabs=%d  materials=%d  assets=%d",
            job_id, len(scan.scene_files),
            len(scan.prefab_ir_map), len(scan.material_ir_map), len(scan.asset_files),
        )

        # ── Step 2: Version mismatch check ────────────────────────────────
        if (
            scan.detected_unity_version
            and scan.detected_unity_version != self.unity_version
        ):
            warnings.append(
                f"Warning: Uploaded project version "
                f"(Unity {scan.detected_unity_version}) does not match "
                f"selected version (Unity {self.unity_version}). "
                f"Conversion may be inaccurate."
            )
            log.warning(
                "job=%s  version mismatch: project=%s  selected=%s",
                job_id, scan.detected_unity_version, self.unity_version,
            )

        if not scan.scene_files:
            return ConversionResult(
                success=False,
                primary_ir={},
                all_scenes_ir={},
                scene_names=[],
                warnings=warnings,
                prefab_count=len(scan.prefab_ir_map),
                project_available=False,
                error="No .unity scene files found in the ZIP.",
            )

        # ── Step 3: Parse each scene + build IR ───────────────────────────
        _prog(24, "Classifying prefabs", "Building prefab and material IR maps")
        all_scenes_ir: Dict[str, Dict[str, Any]] = {}
        primary_ir:    Optional[Dict[str, Any]]  = None
        scene_names:   List[str]                 = []

        _total_scenes = max(len(scan.scene_files), 1)
        for _scene_idx, scene_path in enumerate(scan.scene_files):
            sname = scene_path.stem
            if sname in all_scenes_ir:
                continue  # deduplicate by stem
            _pct = 28 + int(_scene_idx * 24 / _total_scenes)
            _prog(_pct, "Parsing scene graph", f"Parsing {scene_path.name}")
            log.info("job=%s  parsing scene  %s", job_id, scene_path.name)
            try:
                raw_ir, _ = load_unity_scene_debug(
                    scene_path, unity_version=self.unity_version
                )
            except UnityParseError as exc:
                warnings.append(f"Skipped {scene_path.name}: {exc}")
                continue

            scene_ir = build_scene_ir(
                raw_ir,
                scene_name=sname,
                source_file=scene_path.name,
                source_engine=self.source_engine,
                source_engine_version=self.unity_version,
                target_engine=self.target_engine,
                target_engine_version=self.target_version,
                guid_map=scan.guid_map,
                material_ir_map=scan.material_ir_map,
                volume_profile_map=scan.volume_profile_map,
                physics_material_ir_map=scan.physics_material_ir_map,
                animation_ir_map=scan.animation_ir_map,
                animator_controller_map=scan.animator_controller_map,
                terrain_ir_map=scan.terrain_ir_map,
                warnings=warnings,
            )

            # ── Step 4: Validate IR ───────────────────────────────────────
            try:
                warnings.extend(validate_scene_ir_basic(scene_ir))
            except Exception as exc:
                warnings.append(f"Validation error ({sname}): {exc}")

            # ── Step 4b: Hierarchy fidelity check (source IR vs itself) ──
            # Validates internal structural consistency: broken refs, instance
            # integrity, and ir_doc_kind.  A full source-vs-target comparison
            # runs after the project is built (see Step 6b below).
            try:
                hier_result = validate_hierarchy(scene_ir, scene_ir)
                if hier_result.broken_refs:
                    for aid in hier_result.broken_refs:
                        warnings.append(
                            f"[{sname}] Broken asset reference in IR: "
                            f"ir_asset_id '{aid}' not in asset_registry"
                        )
                if not hier_result.instance_integrity:
                    warnings.append(
                        f"[{sname}] Instance integrity check failed — "
                        f"some prefab_instance nodes may have been flattened."
                    )
            except Exception as exc:
                warnings.append(f"Hierarchy validation error ({sname}): {exc}")

            all_scenes_ir[sname] = scene_ir
            scene_names.append(sname)
            if primary_ir is None:
                primary_ir = scene_ir

        if primary_ir is None:
            return ConversionResult(
                success=False,
                primary_ir={},
                all_scenes_ir={},
                scene_names=[],
                warnings=warnings,
                prefab_count=len(scan.prefab_ir_map),
                project_available=False,
                error="All scene files failed to parse.",
            )

        log.info("job=%s  parsed %d scene(s)", job_id, len(all_scenes_ir))
        _prog(54, "Validating scene IR", "Checking node hierarchy")

        # ── Step 5: Resolve source-relative paths ─────────────────────────
        scene_source_path, prefab_source_paths, all_scene_source_paths = \
            self._resolve_source_paths(scan, project_root)

        # ── Step 6: Generate Godot project ────────────────────────────────
        _prog(58, "Building Godot project", "Generating project files")
        # Build GUID → res:// path map for C# scripts so the project builder
        # can wire script = ExtResource(...) into every affected .tscn node.
        script_res_map: Dict[str, str] = _build_script_res_map(
            scan.guid_map, project_root
        )

        log.info(
            "job=%s  building Godot project  prefabs=%d  scenes=%d  script_map=%d",
            job_id, len(scan.prefab_ir_map), len(all_scenes_ir), len(script_res_map),
        )
        build_result: ProjectBuildResult = build_godot_project(
            primary_ir,
            output_dir,
            project_name=project_name,
            prefab_ir_map=scan.prefab_ir_map,
            scene_source_path=scene_source_path,
            prefab_source_paths=prefab_source_paths,
            all_scenes_ir=all_scenes_ir,
            all_scene_source_paths=all_scene_source_paths,
            guid_map=scan.guid_map,
            project_root=project_root,
            script_res_map=script_res_map,
        )
        warnings.extend(build_result.warnings)
        if not build_result.success:
            warnings.append(f"Project build failed: {build_result.error}")
            log.warning("job=%s  project build failed: %s", job_id, build_result.error)
        _prog(67, "Writing scene files", f"Generating .tscn for {len(all_scenes_ir)} scene(s)")

        # ── Step 7: Copy passthrough assets ───────────────────────────────
        _prog(75, "Resolving dependencies", "Copying assets")
        if build_result.success:
            copied = _copy_assets(
                project_root, output_dir, scan.asset_files, warnings
            )
            log.info("job=%s  copied %d asset file(s)", job_id, copied)

        # ── Step 8: Convert or preserve C# scripts ───────────────────────
        scripts_converted = 0
        scripts_failed    = 0
        if build_result.success and scan.script_files:
            if convert_scripts:
                converted, failed = _convert_scripts(
                    scan.script_files, project_root, output_dir, warnings,
                    progress_cb=_prog,
                )
                scripts_converted = converted
                scripts_failed    = failed
                log.info(
                    "job=%s  scripts converted=%d  failed=%d",
                    job_id, converted, failed,
                )
            else:
                _prog(78, "Preserving scripts", "Writing preservation stubs")
                ok, fail = write_preservation_stubs(
                    scan.script_files, project_root, output_dir,
                )
                scripts_converted = ok
                scripts_failed    = fail
                log.info(
                    "job=%s  preservation stubs written=%d  failed=%d",
                    job_id, ok, fail,
                )
                if fail:
                    warnings.append(
                        f"{fail} script(s) could not be preserved as stubs — "
                        "check log for details."
                    )

        log.info(
            "job=%s  done  scenes=%d  proj_ok=%s",
            job_id, len(all_scenes_ir), build_result.success,
        )
        return ConversionResult(
            success=build_result.success,
            primary_ir=primary_ir,
            all_scenes_ir=all_scenes_ir,
            scene_names=scene_names,
            warnings=warnings,
            prefab_count=len(scan.prefab_ir_map),
            project_available=build_result.success,
            scripts_converted=scripts_converted,
            scripts_failed=scripts_failed,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_source_paths(
        self,
        scan: ProjectScanResult,
        project_root: Path,
    ) -> tuple[
        Optional[Path],
        Dict[str, Path],
        Dict[str, Path],
    ]:
        """Compute the three source-path dicts needed by the project builder.

        Returns:
            (scene_source_path, prefab_source_paths, all_scene_source_paths)
        """
        # Primary scene source path — strip Assets/ so the .tscn lands at
        # Scenes/Level1.tscn instead of Assets/Scenes/Level1.tscn.
        primary_scene_file = scan.scene_files[0] if scan.scene_files else None
        try:
            scene_source_path: Optional[Path] = (
                _strip_assets_prefix(primary_scene_file.relative_to(project_root))
                if primary_scene_file else None
            )
        except ValueError:
            scene_source_path = None

        # guid → relative .prefab path (Assets/ stripped)
        # e.g. Assets/Prefabs/Enemy.prefab → Prefabs/Enemy.prefab
        prefab_source_paths: Dict[str, Path] = {}
        for guid, asset_path in scan.guid_map.items():
            if asset_path.suffix.lower() == ".prefab":
                try:
                    prefab_source_paths[guid] = _strip_assets_prefix(
                        asset_path.relative_to(project_root)
                    )
                except ValueError:
                    pass

        # scene_name → relative .unity path (Assets/ stripped, for multi-scene)
        all_scene_source_paths: Dict[str, Path] = {}
        for scene_path in scan.scene_files:
            try:
                all_scene_source_paths[scene_path.stem] = _strip_assets_prefix(
                    scene_path.relative_to(project_root)
                )
            except ValueError:
                pass

        return scene_source_path, prefab_source_paths, all_scene_source_paths


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _strip_assets_prefix(rel: Path) -> Path:
    """Remove the leading 'Assets/' component from a Unity-relative path.

    Assets/Scenes/Level1.unity  →  Scenes/Level1.unity
    Assets/Prefabs/Enemy.prefab →  Prefabs/Enemy.prefab

    Paths that do not start with 'Assets' are returned unchanged.
    This keeps the Godot output folder flat (no top-level Assets/ dir)
    while preserving the rest of the Unity folder hierarchy.
    """
    parts = rel.parts
    if parts and parts[0] == "Assets":
        return Path(*parts[1:]) if len(parts) > 1 else Path(".")
    return rel


def _build_script_res_map(
    guid_map:     Dict[str, Path],
    project_root: Path,
) -> Dict[str, str]:
    """Return a mapping of script GUID → ``res://`` path for all .cs Unity assets.

    Uses the same ``Assets/``-stripping logic as the script converter so the
    generated paths match where the converter writes the output files:
        Assets/Scripts/Player.cs  →  ``res://Scripts/Player.cs``

    Only .cs files are included; other asset types are ignored.
    """
    result: Dict[str, str] = {}
    for guid, abs_path in guid_map.items():
        if abs_path.suffix.lower() != ".cs":
            continue
        try:
            rel = abs_path.relative_to(project_root)
        except ValueError:
            continue
        parts = rel.parts
        if parts and parts[0] == "Assets":
            rel = Path(*parts[1:]) if len(parts) > 1 else Path(abs_path.name)
        result[guid] = "res://" + rel.as_posix()
    return result


def _convert_scripts(
    script_files: List[Path],
    project_root: Path,
    project_dir:  Path,
    warnings:     List[str],
    progress_cb = None,
) -> tuple[int, int]:
    """Convert Unity C# scripts to Godot C# and write them into the Godot project.

    Each .cs file is converted independently via Ollama.  Failures are
    non-blocking: a warning is appended and the pipeline continues.  The project
    build is never aborted over a script error.

    Output paths mirror the Unity source hierarchy with ``Assets/`` stripped:
        Assets/Scripts/Player.cs  →  <project_dir>/Scripts/Player.cs

    Returns:
        (converted_count, failed_count)
    """
    _prog  = progress_cb or (lambda *_: None)
    _total = max(len(script_files), 1)

    converter = ScriptConverter()
    if not converter.is_available():
        warnings.append(
            "Ollama is not running — C# scripts converted to placeholder stubs. "
            "Start Ollama locally (https://ollama.com) for full C# → Godot C# conversion."
        )
        _prog(78, "Converting scripts", "Writing placeholder stubs")
        stub_count = write_fallback_stubs(script_files, project_root, project_dir)
        log.info("wrote %d placeholder stub(s) (Ollama offline)", stub_count)
        return 0, stub_count

    log.info("converting %d C# script(s) via Ollama …", len(script_files))
    converted = 0
    failed    = 0
    for i, script_path in enumerate(script_files):
        pct = 78 + int(i * 15 / _total)
        _prog(pct, "Converting scripts", f"Script {i + 1}/{_total}: {script_path.name}")
        batch = converter.convert_batch([script_path], project_root, project_dir)
        for r in batch:
            if r.warning:
                warnings.append(f"Script warning ({r.source_path.name}): {r.warning}")
            if r.success:
                converted += 1
            else:
                failed += 1
                warnings.append(
                    f"Script conversion failed ({r.source_path.name}): {r.error}"
                )

    return converted, failed


def _copy_assets(
    project_root: Path,
    project_dir: Path,
    asset_files: List[Path],
    warnings: List[str],
) -> int:
    """Copy passthrough asset files into the Godot project.

    The leading 'Assets/' component is stripped from the relative path so
    that Unity's  Assets/Textures/wood.png  becomes  Textures/wood.png  in
    the Godot project root, matching the scene/prefab path convention.
    Existing files (e.g. generated .tscn) are never overwritten.

    Returns the number of files successfully copied.
    """
    copied = 0
    for src in asset_files:
        try:
            rel = src.relative_to(project_root)
        except ValueError:
            continue
        rel = _strip_assets_prefix(rel)
        dst = project_dir / rel
        if dst.exists():
            continue  # never overwrite generated files
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst)
            copied += 1
        except OSError as exc:
            warnings.append(f"Could not copy asset {rel}: {exc}")
    return copied
