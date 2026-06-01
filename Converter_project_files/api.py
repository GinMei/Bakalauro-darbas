"""api.py

FastAPI backend for the Unity to IR to Godot conversion system.

Endpoints:
    POST /convert/unity                   Upload .zip — starts conversion, returns {job_id} immediately
    GET  /health                          Liveness check
    GET  /versions                        Supported engine/version options
    GET  /job/{job_id}                    Stored IR + report (disk)
    GET  /job/{job_id}/ir                 Scene IR only
    GET  /job/{job_id}/report             Conversion report only
    GET  /job/{job_id}/result             Full conversion result (in-memory or disk)
    GET  /job/{job_id}/progress           SSE stream: {progress, stage, details, done, error}
    GET  /job/{job_id}/download           Download main.tscn
    GET  /job/{job_id}/download_project   Download full project as ZIP

Fix notes (2024):
    All CPU/IO-heavy work (YAML parsing, file scanning, IR building, ZIP
    creation) is offloaded to a thread pool via asyncio.to_thread() so the
    event loop is never blocked.  Without this, uvicorn becomes unresponsive
    during conversion and the shutdown sequence shows CancelledError.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from version_profiles import (
    SUPPORTED_VERSIONS, UnsupportedVersionError, get_profile,
    SOURCE_ENGINES, TARGET_ENGINES,
    SOURCE_ENGINE_OPTIONS, TARGET_ENGINE_OPTIONS,
    parse_engine_option,
)
from unity_to_godot.godot_project_builder import zip_project
from unity_to_godot.project_scanner import (
    extract_zip_to_dir,
    detect_project_engine,
    detect_godot_version,
    read_unity_version,
)
from conversion_pipeline import ConversionController
from godot_to_unity.godot_to_unity_pipeline import GodotToUnityPipeline
from unity_to_godot.unity_to_godot_script_converter import UnityToGodotConverter, U2GBatchResult

# ---------------------------------------------------------------------------
# Logging — configured before anything else so startup messages are captured
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("api")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

# Anchor all paths to the directory that contains this file, not the process
# working directory.  This is the root cause of "server crashes immediately"
# when uvicorn is started from a different directory (e.g. the repo root vs
# a subdirectory): Path("static") would resolve differently in each case,
# causing StaticFiles() to raise RuntimeError("Directory does not exist")
# at import time, which uvicorn surfaces as a silent crash / CancelledError.
_BASE_DIR  = Path(__file__).resolve().parent
JOBS_DIR   = _BASE_DIR / "jobs"
STATIC_DIR = _BASE_DIR / "static"

# Ensure directories exist before the app object is created.
JOBS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Unity to Godot IR Converter", version="0.2.0")

# StaticFiles raises RuntimeError at mount time if the directory is missing.
# We create STATIC_DIR above, so this is safe.  Using the absolute path
# (str(STATIC_DIR)) rather than the relative string "static" makes the mount
# work correctly regardless of the process working directory.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

log.info("api.py loaded — BASE=%s  JOBS=%s  STATIC=%s", _BASE_DIR, JOBS_DIR, STATIC_DIR)


@app.on_event("startup")
async def _startup_sweep() -> None:
    """Delete expired non-consented jobs; re-schedule deletion for surviving ones.

    TTL is counted from conversion completion, not job creation, so jobs
    with no expires_at.txt are either mid-conversion or crashed — delete them.
    Jobs whose TTL hasn't run out yet get a fresh deletion task scheduled for
    the remaining seconds so they are cleaned up even after a server restart.
    """
    now = datetime.now(timezone.utc)
    deleted = rescheduled = 0
    for d in JOBS_DIR.iterdir():
        if not d.is_dir():
            continue
        consented_file = d / "consented.txt"
        expires_file   = d / "expires_at.txt"
        if not consented_file.exists():
            continue
        if consented_file.read_text(encoding="utf-8").strip() != "false":
            continue
        if not expires_file.exists():
            # No expiry written yet — conversion never finished (e.g. server crash).
            shutil.rmtree(d, ignore_errors=True)
            deleted += 1
            continue
        try:
            expires_at = datetime.fromisoformat(expires_file.read_text(encoding="utf-8").strip())
        except ValueError:
            shutil.rmtree(d, ignore_errors=True)
            deleted += 1
            continue
        if now >= expires_at:
            shutil.rmtree(d, ignore_errors=True)
            deleted += 1
        else:
            remaining = max((expires_at - now).total_seconds(), 0)
            asyncio.create_task(_schedule_job_deletion(d.name, delay=remaining))
            rescheduled += 1
    if deleted or rescheduled:
        log.info(
            "startup sweep: deleted=%d expired, rescheduled=%d surviving non-consented job(s)",
            deleted, rescheduled,
        )


# ---------------------------------------------------------------------------
# Per-job progress tracking (in-memory; populated by background conversion tasks)
# ---------------------------------------------------------------------------

# asyncio.Queue per job — SSE endpoint reads from it while worker writes to it.
_job_queues:  Dict[str, asyncio.Queue]  = {}
# Full result dict written by background task; served by GET /job/{job_id}/result.
_job_results: Dict[str, Dict[str, Any]] = {}


class ProgressReporter:
    """Thread-safe callable passed to synchronous pipeline workers.

    Workers run inside asyncio.to_thread() (a thread pool), so they cannot
    await or touch the event loop directly.  This wrapper uses
    loop.call_soon_threadsafe() to enqueue SSE payloads from the worker
    thread onto the asyncio event loop queue.
    """

    def __init__(self, job_id: str, loop: asyncio.AbstractEventLoop) -> None:
        self._job_id = job_id
        self._loop   = loop

    def __call__(self, progress: int, stage: str, details: str = "") -> None:
        q = _job_queues.get(self._job_id)
        if not q:
            return
        payload = json.dumps({
            "progress": int(progress),
            "stage":    stage,
            "details":  details,
            "done":     False,
            "error":    "",
        })
        self._loop.call_soon_threadsafe(q.put_nowait, payload)


# ---------------------------------------------------------------------------
# Project name helpers
# ---------------------------------------------------------------------------

def _clean_name(name: str) -> str:
    """Derive a safe project name from a ZIP filename stem.

    'MyGame'          → 'mygame'
    'Cool Project v2' → 'cool_project_v2'
    'My--Game'        → 'my_game'
    """
    name = name.lower()
    name = re.sub(r'[^a-z0-9]+', '_', name)   # replace any non-alphanumeric run with _
    name = re.sub(r'_+', '_', name)             # collapse multiple underscores
    return name.strip('_') or "project"


# ---------------------------------------------------------------------------
# Pure helpers (synchronous — safe to call from threads)
# ---------------------------------------------------------------------------

def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def _save_project_name(job_id: str, project_name: str) -> None:
    (_job_dir(job_id) / "project_name.txt").write_text(project_name, encoding="utf-8")


def _load_project_name(job_id: str) -> str:
    p = _job_dir(job_id) / "project_name.txt"
    if p.exists():
        return p.read_text(encoding="utf-8").strip() or "project"
    return "project"


def _save_job_type(job_id: str, job_type: str) -> None:
    (_job_dir(job_id) / "job_type.txt").write_text(job_type, encoding="utf-8")


def _load_job_type(job_id: str) -> str:
    p = _job_dir(job_id) / "job_type.txt"
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return "unity_to_godot"


# ---------------------------------------------------------------------------
# Consent & TTL helpers
# ---------------------------------------------------------------------------

_JOB_TTL_MINUTES: int = 15


def _save_consent(job_id: str, consented: bool) -> None:
    (_job_dir(job_id) / "consented.txt").write_text(
        "true" if consented else "false", encoding="utf-8"
    )


def _load_consent(job_id: str) -> Optional[bool]:
    p = _job_dir(job_id) / "consented.txt"
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8").strip() == "true"


def _save_expires_at(job_id: str) -> str:
    """Write expires_at.txt and return the ISO timestamp string."""
    expires = datetime.now(timezone.utc) + timedelta(minutes=_JOB_TTL_MINUTES)
    iso = expires.isoformat()
    (_job_dir(job_id) / "expires_at.txt").write_text(iso, encoding="utf-8")
    return iso


def _delete_job(job_id: str) -> None:
    d = _job_dir(job_id)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    _job_results.pop(job_id, None)
    _job_queues.pop(job_id, None)
    log.info("job=%s  deleted", job_id)


async def _schedule_job_deletion(job_id: str, delay: float = _JOB_TTL_MINUTES * 60) -> None:
    await asyncio.sleep(delay)
    if _load_consent(job_id) is False:
        log.info("job=%s  TTL expired — deleting non-consented job", job_id)
        _delete_job(job_id)


def _write_job(job_id: str, scene_ir: Dict[str, Any], report: Dict[str, Any]) -> None:
    d = _job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "scene_ir.json").write_text(
        json.dumps(scene_ir, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (d / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _build_report(
    job_id: str,
    scene_ir: Dict[str, Any],
    warnings: List[str],
    unity_version: str,
    source_filename: str,
    input_type: str = "unity",
    prefab_count: int = 0,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    total = mapped = partial = unknown = requires_review = 0

    def _walk(nodes: list) -> None:
        nonlocal total, mapped, partial, unknown, requires_review
        for n in nodes:
            total += 1
            status = (n.get("meta") or {}).get("conversion_status", "unknown")
            if status == "mapped":
                mapped += 1
            elif status == "partial":
                partial += 1
            else:
                unknown += 1
            if (n.get("meta") or {}).get("requires_review"):
                requires_review += 1
            _walk(n.get("children") or [])

    _walk(scene_ir.get("nodes") or [])
    score = round(mapped / total, 3) if total else 0.0

    return {
        "job_id":           job_id,
        "created_at":       datetime.now(timezone.utc).isoformat(),
        "unity_version":    unity_version,
        "source_file":      source_filename,
        "input_type":       input_type,
        "prefabs_resolved": prefab_count,
        "error":            error,
        "conversion_score": score,
        "summary": {
            "total_nodes":     total,
            "mapped":          mapped,
            "partial":         partial,
            "unknown":         unknown,
            "requires_review": requires_review,
        },
        "warnings": warnings,
    }


def _build_error_result(
    job_id: str,
    unity_version: str,
    filename: str,
    error_msg: str,
) -> Dict[str, Any]:
    """Build an in-memory error result dict without touching the disk."""
    log.warning("job=%s  error=%s", job_id, error_msg)
    empty_ir = {
        "ir_version": "1.0", "unity_version": unity_version,
        "scene_id": "", "scene_name": "", "source_file": filename,
        "node_count": 0, "nodes": [],
    }
    report = _build_report(job_id, empty_ir, [], unity_version,
                           filename, error=error_msg)
    return {
        "job_id": job_id, "error": error_msg,
        "scene_ir": empty_ir, "report": report,
        "project_available": False,
    }


def _error_response_dict(
    job_id: str,
    unity_version: str,
    filename: str,
    error_msg: str,
) -> Dict[str, Any]:
    """Build and persist an error result dict (no HTTP response object)."""
    result = _build_error_result(job_id, unity_version, filename, error_msg)
    _write_job(job_id, result["scene_ir"], result["report"])
    return result


def _error_response(
    job_id: str,
    unity_version: str,
    filename: str,
    error_msg: str,
) -> JSONResponse:
    content = _error_response_dict(job_id, unity_version, filename, error_msg)
    return JSONResponse(status_code=422, content=content)


# ---------------------------------------------------------------------------
# Synchronous conversion worker
# All blocking I/O and CPU work lives here; called exclusively via
# asyncio.to_thread() so the event loop is never blocked.
# ---------------------------------------------------------------------------

def _convert_zip_worker(
    zip_path: Path,
    extract_dir: Path,
    unity_version: str,
    job_id: str,
    filename: str,
    source_engine: str = "Unity 6000.3.9f1",
    target_engine: str = "Godot 4.5",
    target_version: str = "4.5",
    project_name: str = "project",
    convert_scripts: bool = True,
    progress_cb = None,
) -> Dict[str, Any]:
    """Extract a ZIP then delegate all conversion work to ConversionController.

    This function owns only the concerns that are specific to the HTTP job
    layer: extracting the upload, running the controller, persisting job
    files, and assembling the response dict.  All pipeline logic (scanning,
    parsing, IR building, validation, project generation, asset copying) lives
    in ConversionController.
    """
    _prog = progress_cb or (lambda *_: None)

    _prog(5, "Uploading project", "Extracting archive")
    log.info("job=%s  extracting ZIP  file=%s", job_id, filename)
    project_root = extract_zip_to_dir(zip_path, extract_dir)

    # ── Engine mismatch guard ─────────────────────────────────────────────────
    _prog(10, "Validating archive", "Checking engine compatibility")
    detected = detect_project_engine(project_root)
    if detected == "godot":
        godot_ver = detect_godot_version(project_root)
        ver_info  = f" (Godot {godot_ver})" if godot_ver else ""
        raise ValueError(
            f"Uploaded project appears to be a Godot project{ver_info}, "
            f"but the selected source engine is Unity. "
            f"Please select Godot as the source engine, or upload a Unity project."
        )

    controller = ConversionController(
        source_engine=source_engine,
        target_engine=target_engine,
        unity_version=unity_version,
        target_version=target_version,
    )
    output_dir = _job_dir(job_id) / "godot_project"
    result = controller.convert_zip(
        project_root, job_id, project_name, output_dir,
        convert_scripts=convert_scripts, progress_cb=_prog,
    )

    if not result.primary_ir:
        raise ValueError(result.error or "Conversion produced no IR.")

    _prog(96, "Finalizing output", "Building report")
    # Report building and job persistence are API-layer concerns — they stay here.
    report = _build_report(
        job_id, result.primary_ir, result.warnings, unity_version,
        filename, input_type="zip", prefab_count=result.prefab_count,
    )
    _write_job(job_id, result.primary_ir, report)
    (_job_dir(job_id) / "all_scenes_ir.json").write_text(
        json.dumps(result.all_scenes_ir, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "job_id":            job_id,
        "project_name":      project_name,
        "scene_ir":          result.primary_ir,
        "scenes":            result.scene_names,
        "all_scenes_ir":     result.all_scenes_ir,
        "report":            report,
        "input_type":        "zip",
        "prefabs_found":     result.prefab_count,
        "tscn_available":    result.project_available,
        "project_available": result.project_available,
    }


def _convert_godot_to_unity_worker(
    zip_path:        Path,
    extract_dir:     Path,
    job_id:          str,
    filename:        str,
    project_name:    str  = "project",
    convert_scripts: bool = True,
    progress_cb            = None,
) -> Dict[str, Any]:
    """Extract a Godot project ZIP then convert every .tscn to a Unity project.

    Returns a response dict shaped to match the Unity→Godot response so the
    frontend can handle both conversion directions uniformly.
    """
    _prog = progress_cb or (lambda *_: None)

    _prog(5, "Uploading project", "Extracting archive")
    log.info("job=%s  extracting Godot ZIP  file=%s", job_id, filename)
    project_root = extract_zip_to_dir(zip_path, extract_dir)

    # ── Engine mismatch guard ─────────────────────────────────────────────────
    _prog(10, "Validating archive", "Checking engine compatibility")
    detected = detect_project_engine(project_root)
    if detected == "unity":
        unity_ver = read_unity_version(project_root)
        ver_info  = f" (Unity {unity_ver})" if unity_ver else ""
        if language == "en":
            raise ValueError(
                f"Uploaded project appears to be a Unity project{ver_info}, "
                f"but the selected source engine is Godot. "
                f"Please select Unity as the source engine, or upload a Godot project."
            )
        else:
            raise ValueError(
                f"Įkeltas projektas atrodo kaip Unity projektas{ver_info}, "
                f"tačiau pasirinktas šaltinio variklis yra Godot. "
                f"Pasirinkite Unity kaip šaltinio variklį arba įkelkite Godot projektą."
            )

    pipeline   = GodotToUnityPipeline()
    output_dir = _job_dir(job_id) / "unity_output"
    result     = pipeline.convert(
        godot_root       = project_root,
        output_dir       = output_dir,
        project_name     = project_name,
        convert_scripts  = convert_scripts,
        progress_cb      = _prog,
    )

    if not result.success:
        raise ValueError(result.error or "Godot→Unity conversion produced no scenes.")

    _prog(96, "Finalizing output", "Building report")
    all_scenes_ir: Dict[str, Any] = result.scene_irs
    scene_names  = [Path(p).stem for p in result.scenes_exported]
    exported_set = set(scene_names)
    for k, ir in all_scenes_ir.items():
        if k not in exported_set and ir.get("is_scene", False):
            scene_names.append(k)
            exported_set.add(k)

    primary_ir  = next(iter(all_scenes_ir.values()), {})
    total_nodes = sum(
        ir.get("node_count", len(ir.get("nodes", [])))
        for ir in all_scenes_ir.values()
    )
    num_exported = len(result.scenes_exported)
    num_total    = num_exported + len(result.scenes_failed)
    score        = round(num_exported / num_total, 3) if num_total else 1.0

    report: Dict[str, Any] = {
        "job_id":           job_id,
        "created_at":       datetime.now(timezone.utc).isoformat(),
        "direction":        "godot_to_unity",
        "source_file":      filename,
        "input_type":       "zip",
        "scenes_exported":  [str(p) for p in result.scenes_exported],
        "scenes_failed":    result.scenes_failed,
        "assets_copied":    result.assets_copied,
        "error":            result.error or None,
        "conversion_score": score,
        "summary": {
            "total_nodes":     total_nodes,
            "mapped":          total_nodes,
            "partial":         0,
            "unknown":         0,
            "requires_review": len(result.scenes_failed),
        },
        "warnings": result.warnings,
    }

    _write_job(job_id, primary_ir, report)
    (_job_dir(job_id) / "all_scenes_ir.json").write_text(
        json.dumps(all_scenes_ir, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _save_job_type(job_id, "godot_to_unity")

    return {
        "job_id":           job_id,
        "project_name":     project_name,
        "direction":        "godot_to_unity",
        "scene_ir":         primary_ir,
        "scenes":           scene_names,
        "all_scenes_ir":    all_scenes_ir,
        "report":           report,
        "input_type":       "zip",
        "scenes_exported":  [str(p) for p in result.scenes_exported],
        "scenes_failed":    result.scenes_failed,
        "assets_copied":    result.assets_copied,
        "project_available": True,
        "decisions":        result.decisions,
    }


# ---------------------------------------------------------------------------
# Background conversion coroutines
# ---------------------------------------------------------------------------

async def _run_u2g_conversion(
    job_id:          str,
    zip_path:        Path,
    extract_dir:     Path,
    unity_version:   str,
    filename:        str,
    source_engine:   str,
    target_engine:   str,
    target_version:  str,
    project_name:    str,
    convert_scripts: bool = True,
) -> None:
    """Run Unity→Godot conversion in a thread pool; report progress via SSE queue."""
    loop     = asyncio.get_running_loop()
    reporter = ProgressReporter(job_id, loop)
    error    = ""
    try:
        result = await asyncio.to_thread(
            _convert_zip_worker,
            zip_path, extract_dir, unity_version, job_id, filename,
            source_engine, target_engine, target_version, project_name,
            convert_scripts, reporter,
        )
        _job_results[job_id] = result
    except ValueError as exc:
        error = str(exc)
        _job_results[job_id] = _build_error_result(job_id, unity_version, filename, error)
    except Exception as exc:
        log.exception("job=%s  unexpected error in U2G background task", job_id)
        error = f"Conversion error: {exc}"
        _job_results[job_id] = _build_error_result(job_id, unity_version, filename, error)
    finally:
        consented = _load_consent(job_id)
        if error:
            # Conversion failed — delete the job directory immediately so no
            # partial or invalid project data is stored on disk.
            d = _job_dir(job_id)
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
            log.info("job=%s  conversion error — job directory deleted immediately", job_id)
        elif consented is False:
            # Successful conversion, non-consented — schedule TTL deletion.
            expires_iso = _save_expires_at(job_id)
            asyncio.create_task(_schedule_job_deletion(job_id))
            if job_id in _job_results:
                _job_results[job_id]["expires_at"] = expires_iso
        q = _job_queues.get(job_id)
        if q:
            payload = json.dumps({
                "progress": 100, "stage": "Complete", "details": "",
                "done": True, "error": error,
            })
            q.put_nowait(payload)


async def _run_g2u_conversion(
    job_id:          str,
    zip_path:        Path,
    extract_dir:     Path,
    filename:        str,
    project_name:    str,
    convert_scripts: bool = True,
) -> None:
    """Run Godot→Unity conversion in a thread pool; report progress via SSE queue."""
    loop     = asyncio.get_running_loop()
    reporter = ProgressReporter(job_id, loop)
    error    = ""
    try:
        result = await asyncio.to_thread(
            _convert_godot_to_unity_worker,
            zip_path, extract_dir, job_id, filename, project_name,
            convert_scripts, reporter,
        )
        _job_results[job_id] = result
    except ValueError as exc:
        error = str(exc)
        _job_results[job_id] = _build_error_result(job_id, "godot_4.5", filename, error)
    except Exception as exc:
        log.exception("job=%s  unexpected error in G2U background task", job_id)
        error = f"Conversion error: {exc}"
        _job_results[job_id] = _build_error_result(job_id, "godot_4.5", filename, error)
    finally:
        consented = _load_consent(job_id)
        if error:
            # Conversion failed — delete the job directory immediately so no
            # partial or invalid project data is stored on disk.
            d = _job_dir(job_id)
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
            log.info("job=%s  conversion error — job directory deleted immediately", job_id)
        elif consented is False:
            # Successful conversion, non-consented — schedule TTL deletion.
            expires_iso = _save_expires_at(job_id)
            asyncio.create_task(_schedule_job_deletion(job_id))
            if job_id in _job_results:
                _job_results[job_id]["expires_at"] = expires_iso
        q = _job_queues.get(job_id)
        if q:
            payload = json.dumps({
                "progress": 100, "stage": "Complete", "details": "",
                "done": True, "error": error,
            })
            q.put_nowait(payload)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    """Liveness check — always returns 200 if the server is running."""
    return {"status": "ok", "jobs_dir": str(JOBS_DIR), "static_dir": str(STATIC_DIR)}


@app.get("/")
def root() -> FileResponse:
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(str(index))


@app.get("/versions")
def list_versions() -> Dict[str, Any]:
    return {
        "supported_versions":    SUPPORTED_VERSIONS,
        "source_engines":        SOURCE_ENGINES,
        "target_engines":        TARGET_ENGINES,
        "source_engine_options": SOURCE_ENGINE_OPTIONS,
        "target_engine_options": TARGET_ENGINE_OPTIONS,
    }


@app.post("/convert/unity")
async def convert_unity(
    file: UploadFile = File(..., description="A Unity or Godot project .zip"),
    unity_version:   str  = Form(default="6000.3.9f1",       description="Unity editor version"),
    source_engine:   str  = Form(default="Unity 6000.3.9f1", description="Source engine+version"),
    target_engine:   str  = Form(default="Godot 4.5",        description="Target engine+version"),
    target_version:  str  = Form(default="4.5",              description="Target engine version"),
    convert_scripts: str  = Form(default="true",             description="Whether to convert scripts"),
    consented:       str  = Form(default="false",            description="User consent for training data use"),
    previous_job_id: str  = Form(default="",                 description="Previous job ID to delete if non-consented"),
    language:        str  = Form(default="lt",               description="UI language for error messages (lt/en)"),
) -> JSONResponse:
    """Upload a project .zip and convert scenes between Unity and Godot.

    Routes automatically based on source_engine:
      Unity 6000.3.9f1 → Godot 4.5        (via ConversionController)
      Godot 4.5        → Unity 6000.3.9f1  (via GodotToUnityPipeline)

    Only .zip uploads are accepted.  All heavy work runs in a thread pool.
    """
    orig_filename = file.filename or "upload.zip"
    filename      = orig_filename.lower()
    log.info("convert called  filename=%s  source=%s  target=%s",
             filename, source_engine, target_engine)

    if not filename.endswith(".zip"):
        raise HTTPException(
            status_code=400,
            detail="Only .zip project archives are accepted.",
        )

    do_consent = consented.strip().lower() in ("true", "1", "yes")

    # Delete previous non-consented job when the user starts a new conversion
    prev = previous_job_id.strip()
    if prev and _load_consent(prev) is False:
        log.info("previous_job=%s  deleting on new conversion start", prev)
        _delete_job(prev)

    project_name = _clean_name(Path(orig_filename).stem)
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id       = f"{timestamp}_{project_name}"
    job_path     = _job_dir(job_id)
    # If a job with the same name+second already exists, append a short suffix
    if job_path.exists():
        job_id   = f"{job_id}_{uuid.uuid4().hex[:6]}"
        job_path = _job_dir(job_id)
    job_path.mkdir(parents=True, exist_ok=True)
    log.info("job=%s  created", job_id)

    _save_project_name(job_id, project_name)
    _save_consent(job_id, do_consent)
    log.info("job=%s  project_name=%s  consented=%s", job_id, project_name, do_consent)

    raw_bytes   = await file.read()
    upload_path = job_path / orig_filename
    await asyncio.to_thread(upload_path.write_bytes, raw_bytes)

    src_engine, _ = parse_engine_option(source_engine)
    do_scripts = convert_scripts.strip().lower() not in ("false", "0", "no")

    # Create SSE queue before starting the background task so the client can
    # connect to /job/{job_id}/progress immediately after this response.
    _job_queues[job_id] = asyncio.Queue()

    if src_engine == "godot":
        # ── Godot → Unity ──────────────────────────────────────────────────────
        extract_dir = job_path / "godot_source"
        asyncio.create_task(
            _run_g2u_conversion(job_id, upload_path, extract_dir, orig_filename, project_name,
                                convert_scripts=do_scripts)
        )
    else:
        # ── Unity → Godot ──────────────────────────────────────────────────────
        try:
            get_profile(unity_version)
        except UnsupportedVersionError as exc:
            _delete_job(job_id)
            raise HTTPException(status_code=400, detail=str(exc))

        extract_dir = job_path / "unity_project"
        asyncio.create_task(
            _run_u2g_conversion(
                job_id, upload_path, extract_dir, unity_version, orig_filename,
                source_engine, target_engine, target_version, project_name,
                convert_scripts=do_scripts,
            )
        )

    return JSONResponse(content={"job_id": job_id, "status": "running"})


@app.get("/job/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    d = _job_dir(job_id)
    if not (d / "scene_ir.json").exists():
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    scene_ir = json.loads((d / "scene_ir.json").read_text(encoding="utf-8"))
    report   = json.loads((d / "report.json").read_text(encoding="utf-8")) \
               if (d / "report.json").exists() else {}
    return JSONResponse(content={"job_id": job_id, "scene_ir": scene_ir, "report": report})


@app.get("/job/{job_id}/ir")
def get_job_ir(job_id: str) -> JSONResponse:
    p = _job_dir(job_id) / "scene_ir.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JSONResponse(content=json.loads(p.read_text(encoding="utf-8")))


@app.get("/job/{job_id}/report")
def get_job_report(job_id: str) -> JSONResponse:
    p = _job_dir(job_id) / "report.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JSONResponse(content=json.loads(p.read_text(encoding="utf-8")))


@app.get("/job/{job_id}/result")
def get_job_result(job_id: str) -> JSONResponse:
    """Return the full conversion result once ready.

    Returns 200 + result dict when the job is complete, or 202 + status when
    still running.  Falls back to disk files if the in-memory result was
    evicted (e.g. server restart after a completed job).
    """
    if job_id in _job_results:
        return JSONResponse(content=_job_results[job_id])

    if job_id in _job_queues:
        return JSONResponse(
            content={"job_id": job_id, "status": "running"},
            status_code=202,
        )

    # Disk fallback: reconstruct a minimal result from persisted files.
    d = _job_dir(job_id)
    if not (d / "scene_ir.json").exists():
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    scene_ir = json.loads((d / "scene_ir.json").read_text(encoding="utf-8"))
    report   = (
        json.loads((d / "report.json").read_text(encoding="utf-8"))
        if (d / "report.json").exists() else {}
    )
    all_ir = (
        json.loads((d / "all_scenes_ir.json").read_text(encoding="utf-8"))
        if (d / "all_scenes_ir.json").exists() else {}
    )
    job_type = _load_job_type(job_id)
    project_available = (
        (d / "godot_project").exists() if job_type != "godot_to_unity"
        else (d / "unity_output").exists()
    )
    return JSONResponse(content={
        "job_id":           job_id,
        "scene_ir":         scene_ir,
        "all_scenes_ir":    all_ir,
        "report":           report,
        "project_available": project_available,
        "scenes": [k for k, ir in all_ir.items() if ir.get("is_scene")],
    })


@app.get("/job/{job_id}/progress")
async def progress_stream(job_id: str) -> StreamingResponse:
    """SSE stream that emits structured progress events while the job runs.

    Each event is a JSON object::

        {"progress": 42, "stage": "Converting assets",
         "details": "Parsing scene X", "done": false, "error": ""}

    The final event has ``done: true``.  When ``done`` is true and ``error``
    is non-empty the conversion failed.  The stream closes after the done event.

    Clients that connect after the job is already complete receive a single
    synthetic done event derived from the stored result.
    """
    async def event_gen():
        # If the job completed before the client connected, send one done event.
        if job_id not in _job_queues:
            result = _job_results.get(job_id)
            if result is not None:
                error = result.get("error") or ""
                payload = json.dumps({
                    "progress": 100, "stage": "Complete", "details": "",
                    "done": True, "error": error,
                })
                yield f"data: {payload}\n\n"
            return

        q = _job_queues[job_id]
        try:
            while True:
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                yield f"data: {payload}\n\n"
                if json.loads(payload).get("done"):
                    _job_queues.pop(job_id, None)
                    break
        except (asyncio.CancelledError, Exception):
            pass

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/job/{job_id}/download")
def download_tscn(job_id: str) -> FileResponse:
    p = _job_dir(job_id) / "godot_project" / "main.tscn"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"No .tscn for job '{job_id}'.")
    return FileResponse(path=str(p), media_type="text/plain", filename="main.tscn")


@app.get("/job/{job_id}/download_project")
async def download_project(job_id: str) -> Response:
    """ZIP the converted project folder and return it.

    Serves godot_project/ for Unity→Godot jobs and unity_output/ for
    Godot→Unity jobs.  zip_project() is offloaded to a thread.
    """
    job_type = _load_job_type(job_id)
    if job_type == "godot_to_unity":
        project_dir = _job_dir(job_id) / "unity_output"
        suffix      = "_unity.zip"
    else:
        project_dir = _job_dir(job_id) / "godot_project"
        suffix      = "_godot.zip"

    if not project_dir.exists():
        raise HTTPException(status_code=404, detail=f"No project for job '{job_id}'.")
    zip_bytes    = await asyncio.to_thread(zip_project, project_dir)
    project_name = _load_project_name(job_id)
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition":
                 f'attachment; filename="{project_name}{suffix}"'},
    )


# ---------------------------------------------------------------------------
# Script conversion endpoint — Unity C# → Godot C#
# ---------------------------------------------------------------------------

def _convert_scripts_worker(
    zip_path:    Path,
    extract_dir: Path,
    output_dir:  Path,
    job_id:      str,
) -> Dict[str, Any]:
    """Extract a ZIP of Unity C# scripts and convert them to Godot C#.

    Requires GEMINI_API_KEY in the process environment.
    GEMINI_FALLBACK_API_KEY and OLLAMA_MODEL are optional.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set — "
            "script conversion requires a Gemini API key."
        )
    fallback_key = os.environ.get("GEMINI_FALLBACK_API_KEY", "")
    ollama_model = os.environ.get("OLLAMA_MODEL", "qwen3")

    import zipfile
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    cs_files = sorted(extract_dir.rglob("*.cs"))
    if not cs_files:
        raise ValueError("No .cs files found in the uploaded archive.")

    converter = UnityToGodotConverter(
        gemini_key=gemini_key,
        gemini_fallback_key=fallback_key,
        ollama_model=ollama_model,
    )
    batch = converter.convert_batch(cs_files, output_dir)

    converted_summaries = [
        {
            "source":            r.source_path,
            "output":            r.output_path,
            "stages_completed":  r.stages_completed,
            "todos":             r.todos,
            "warnings":          r.warnings,
            "used_ollama_fallback": r.used_ollama_fallback,
        }
        for r in batch.converted
    ]
    failed_summaries = [
        {
            "source":  r.source_path,
            "error":   r.error,
            "warnings": r.warnings,
        }
        for r in batch.failed
    ]

    return {
        "job_id":         job_id,
        "direction":      "unity_scripts_to_godot",
        "total_files":    batch.total_files,
        "converted":      len(batch.converted),
        "failed":         len(batch.failed),
        "total_todos":    batch.total_todos,
        "total_warnings": batch.total_warnings,
        "files":          converted_summaries,
        "errors":         failed_summaries,
    }


@app.post("/convert/scripts/unity-to-godot")
async def convert_scripts_unity_to_godot(
    file: UploadFile = File(..., description="ZIP archive of Unity C# (.cs) scripts"),
) -> JSONResponse:
    """Convert Unity C# scripts to Godot C# using the five-stage AI pipeline.

    Upload a .zip containing one or more Unity .cs files.
    Requires the server to have GEMINI_API_KEY set in its environment.

    Returns a JSON summary with per-file stage outcomes, TODOs, and warnings.
    """
    orig_filename = file.filename or "scripts.zip"
    if not orig_filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=400,
            detail="Only .zip archives are accepted — zip your .cs files first.",
        )

    job_id   = uuid.uuid4().hex
    job_path = _job_dir(job_id)
    job_path.mkdir(parents=True, exist_ok=True)
    _save_job_type(job_id, "unity_scripts_to_godot")

    raw_bytes   = await file.read()
    upload_path = job_path / orig_filename
    await asyncio.to_thread(upload_path.write_bytes, raw_bytes)

    extract_dir = job_path / "unity_scripts"
    output_dir  = job_path / "godot_scripts"

    try:
        result = await asyncio.to_thread(
            _convert_scripts_worker,
            upload_path, extract_dir, output_dir, job_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        log.exception("job=%s  unexpected error in script conversion", job_id)
        raise HTTPException(
            status_code=500,
            detail=f"Script conversion error: {exc}",
        )

    return JSONResponse(content=result)


@app.get("/job/{job_id}/download_scripts")
async def download_scripts(job_id: str) -> Response:
    """Download the converted Godot C# scripts as a ZIP."""
    scripts_dir = _job_dir(job_id) / "godot_scripts"
    if not scripts_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No converted scripts for job '{job_id}'.",
        )
    zip_bytes    = await asyncio.to_thread(zip_project, scripts_dir)
    project_name = _load_project_name(job_id) or job_id
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition":
                 f'attachment; filename="{project_name}_godot_scripts.zip"'},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
# The if __name__ == "__main__" guard is required on Windows.
# Without it, uvicorn --reload spawns worker processes via multiprocessing.
# On Windows, multiprocessing uses the "spawn" start method: it imports this
# module fresh in each worker, which re-executes all top-level code.
# Without the guard, every worker would try to start its own uvicorn server,
# causing a cascade of bind failures and the CancelledError / KeyboardInterrupt
# storm seen in the terminal.
#
# Usage:
#   python api.py                          # development (uses reload=True)
#   uvicorn api:app                        # production (no reload)
#   uvicorn api:app --reload               # development on Linux/macOS
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,           # safe here because the guard prevents re-entry
        log_level="info",
    )
