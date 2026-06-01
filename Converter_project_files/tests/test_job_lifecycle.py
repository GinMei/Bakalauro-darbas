"""
Tier 4 — api.py job lifecycle helpers.

Tests verify:
  • _save_consent(job_id, True/False)   — writes "true" / "false" to consented.txt
  • _load_consent(job_id)               — reads back bool or None when missing
  • _save_expires_at(job_id)            — writes a future ISO timestamp
  • _delete_job(job_id)                 — removes the directory and clears in-memory dicts
  • Startup sweep (_startup_sweep)      — expired non-consented jobs deleted,
                                          consented jobs preserved,
                                          unexpired non-consented jobs preserved

All tests monkeypatch api.JOBS_DIR to tmp_path so they never touch the real jobs
directory.
"""

import sys
import os
import asyncio
import types
import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Stub out FastAPI and its dependencies before importing api.py.
# api.py uses FastAPI only for the HTTP layer; the lifecycle helpers we test
# are pure synchronous/async functions with no FastAPI dependency at runtime.
# ---------------------------------------------------------------------------

def _passthrough_decorator_factory(*args, **kwargs):
    """Return an identity decorator — used so @app.on_event() doesn't wrap functions."""
    def decorator(func):
        return func
    return decorator


class _MockFastAPIApp:
    """Minimal FastAPI app stub that preserves decorated async functions."""

    def __init__(self, *args, **kwargs):
        pass

    def on_event(self, *args, **kwargs):
        return _passthrough_decorator_factory()

    def mount(self, *args, **kwargs):
        pass

    def include_router(self, *args, **kwargs):
        pass

    def get(self, *args, **kwargs):
        return _passthrough_decorator_factory()

    def post(self, *args, **kwargs):
        return _passthrough_decorator_factory()

    def delete(self, *args, **kwargs):
        return _passthrough_decorator_factory()


def _stub_heavy_imports():
    """Inject minimal stubs so api.py can be imported without FastAPI/yaml/etc."""

    def _FastAPI_constructor(*args, **kwargs):
        return _MockFastAPIApp()

    fa = types.ModuleType("fastapi")
    fa.FastAPI        = _FastAPI_constructor
    fa.File           = MagicMock()
    fa.Form           = MagicMock()
    fa.HTTPException  = MagicMock()
    fa.UploadFile     = MagicMock()
    sys.modules["fastapi"] = fa

    resp = types.ModuleType("fastapi.responses")
    resp.FileResponse      = MagicMock()
    resp.JSONResponse      = MagicMock()
    resp.Response          = MagicMock()
    resp.StreamingResponse = MagicMock()
    sys.modules["fastapi.responses"] = resp

    static = types.ModuleType("fastapi.staticfiles")
    static.StaticFiles = MagicMock(return_value=MagicMock())
    sys.modules["fastapi.staticfiles"] = static

    for mod_name, attrs in [
        ("project_scanner", ["extract_zip_to_dir", "detect_project_engine",
                              "detect_godot_version", "read_unity_version"]),
        ("godot_project_builder",  ["zip_project"]),
        ("conversion_pipeline",    ["ConversionController"]),
        ("unity_to_godot_script_converter", ["UnityToGodotConverter", "U2GBatchResult"]),
    ]:
        if mod_name not in sys.modules:
            mod = types.ModuleType(mod_name)
            for attr in attrs:
                setattr(mod, attr, MagicMock())
            sys.modules[mod_name] = mod


_stub_heavy_imports()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import api as _api_module
from api import (
    _save_consent,
    _load_consent,
    _save_expires_at,
    _delete_job,
    _job_dir,
    _JOB_TTL_MINUTES,
    _job_results,
    _job_queues,
    _startup_sweep,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_job(tmp_path: Path, job_id: str) -> Path:
    """Create a job directory and return its path."""
    d = tmp_path / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _patch_jobs_dir(monkeypatch, tmp_path: Path):
    """Redirect api.JOBS_DIR to tmp_path so helpers use it."""
    monkeypatch.setattr(_api_module, "JOBS_DIR", tmp_path)


# ---------------------------------------------------------------------------
# _save_consent / _load_consent
# ---------------------------------------------------------------------------

class TestSaveLoadConsent:
    def test_save_true_writes_true(self, tmp_path, monkeypatch):
        _patch_jobs_dir(monkeypatch, tmp_path)
        _make_job(tmp_path, "job1")
        _save_consent("job1", True)
        text = (tmp_path / "job1" / "consented.txt").read_text(encoding="utf-8").strip()
        assert text == "true"

    def test_save_false_writes_false(self, tmp_path, monkeypatch):
        _patch_jobs_dir(monkeypatch, tmp_path)
        _make_job(tmp_path, "job2")
        _save_consent("job2", False)
        text = (tmp_path / "job2" / "consented.txt").read_text(encoding="utf-8").strip()
        assert text == "false"

    def test_load_returns_true(self, tmp_path, monkeypatch):
        _patch_jobs_dir(monkeypatch, tmp_path)
        _make_job(tmp_path, "job3")
        _save_consent("job3", True)
        assert _load_consent("job3") is True

    def test_load_returns_false(self, tmp_path, monkeypatch):
        _patch_jobs_dir(monkeypatch, tmp_path)
        _make_job(tmp_path, "job4")
        _save_consent("job4", False)
        assert _load_consent("job4") is False

    def test_load_returns_none_when_file_missing(self, tmp_path, monkeypatch):
        _patch_jobs_dir(monkeypatch, tmp_path)
        _make_job(tmp_path, "job5")
        assert _load_consent("job5") is None

    def test_load_returns_none_when_dir_missing(self, tmp_path, monkeypatch):
        _patch_jobs_dir(monkeypatch, tmp_path)
        assert _load_consent("nonexistent_job") is None

    def test_overwrite_consent(self, tmp_path, monkeypatch):
        _patch_jobs_dir(monkeypatch, tmp_path)
        _make_job(tmp_path, "job6")
        _save_consent("job6", True)
        _save_consent("job6", False)
        assert _load_consent("job6") is False


# ---------------------------------------------------------------------------
# _save_expires_at
# ---------------------------------------------------------------------------

class TestSaveExpiresAt:
    def test_file_created(self, tmp_path, monkeypatch):
        _patch_jobs_dir(monkeypatch, tmp_path)
        _make_job(tmp_path, "exp1")
        _save_expires_at("exp1")
        assert (tmp_path / "exp1" / "expires_at.txt").exists()

    def test_content_is_iso_format(self, tmp_path, monkeypatch):
        _patch_jobs_dir(monkeypatch, tmp_path)
        _make_job(tmp_path, "exp2")
        _save_expires_at("exp2")
        text = (tmp_path / "exp2" / "expires_at.txt").read_text(encoding="utf-8").strip()
        dt = datetime.fromisoformat(text)
        assert dt.tzinfo is not None

    def test_expires_in_future(self, tmp_path, monkeypatch):
        _patch_jobs_dir(monkeypatch, tmp_path)
        _make_job(tmp_path, "exp3")
        before = datetime.now(timezone.utc)
        _save_expires_at("exp3")
        text = (tmp_path / "exp3" / "expires_at.txt").read_text(encoding="utf-8").strip()
        expires_at = datetime.fromisoformat(text)
        assert expires_at > before

    def test_expires_roughly_ttl_minutes_from_now(self, tmp_path, monkeypatch):
        _patch_jobs_dir(monkeypatch, tmp_path)
        _make_job(tmp_path, "exp4")
        _save_expires_at("exp4")
        text = (tmp_path / "exp4" / "expires_at.txt").read_text(encoding="utf-8").strip()
        expires_at = datetime.fromisoformat(text)
        delta = expires_at - datetime.now(timezone.utc)
        expected = timedelta(minutes=_JOB_TTL_MINUTES)
        assert abs(delta.total_seconds() - expected.total_seconds()) < 5


# ---------------------------------------------------------------------------
# _delete_job
# ---------------------------------------------------------------------------

class TestDeleteJob:
    def test_directory_removed(self, tmp_path, monkeypatch):
        _patch_jobs_dir(monkeypatch, tmp_path)
        d = _make_job(tmp_path, "del1")
        _delete_job("del1")
        assert not d.exists()

    def test_no_error_if_dir_missing(self, tmp_path, monkeypatch):
        _patch_jobs_dir(monkeypatch, tmp_path)
        _delete_job("nonexistent")

    def test_in_memory_result_cleared(self, tmp_path, monkeypatch):
        _patch_jobs_dir(monkeypatch, tmp_path)
        _make_job(tmp_path, "del2")
        _job_results["del2"] = {"some": "data"}
        _delete_job("del2")
        assert "del2" not in _job_results

    def test_in_memory_queue_cleared(self, tmp_path, monkeypatch):
        import asyncio
        _patch_jobs_dir(monkeypatch, tmp_path)
        _make_job(tmp_path, "del3")
        _job_queues["del3"] = asyncio.Queue()
        _delete_job("del3")
        assert "del3" not in _job_queues

    def test_files_inside_dir_also_removed(self, tmp_path, monkeypatch):
        _patch_jobs_dir(monkeypatch, tmp_path)
        d = _make_job(tmp_path, "del4")
        (d / "scene_ir.json").write_text("{}", encoding="utf-8")
        (d / "consented.txt").write_text("false", encoding="utf-8")
        _delete_job("del4")
        assert not d.exists()


# ---------------------------------------------------------------------------
# _startup_sweep — async function
# ---------------------------------------------------------------------------

class TestStartupSweep:
    def test_expired_non_consented_job_deleted(self, tmp_path, monkeypatch):
        _patch_jobs_dir(monkeypatch, tmp_path)
        d = _make_job(tmp_path, "sweep1")
        (d / "consented.txt").write_text("false", encoding="utf-8")
        # Write an already-expired timestamp (1 hour in the past)
        expired = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        (d / "expires_at.txt").write_text(expired, encoding="utf-8")

        asyncio.run(_startup_sweep())
        assert not d.exists()

    def test_consented_job_preserved(self, tmp_path, monkeypatch):
        _patch_jobs_dir(monkeypatch, tmp_path)
        d = _make_job(tmp_path, "sweep2")
        (d / "consented.txt").write_text("true", encoding="utf-8")
        expired = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        (d / "expires_at.txt").write_text(expired, encoding="utf-8")

        asyncio.run(_startup_sweep())
        assert d.exists()

    def test_unexpired_non_consented_job_preserved(self, tmp_path, monkeypatch):
        _patch_jobs_dir(monkeypatch, tmp_path)
        d = _make_job(tmp_path, "sweep3")
        (d / "consented.txt").write_text("false", encoding="utf-8")
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        (d / "expires_at.txt").write_text(future, encoding="utf-8")

        asyncio.run(_startup_sweep())
        assert d.exists()

    def test_non_consented_no_expiry_file_deleted(self, tmp_path, monkeypatch):
        _patch_jobs_dir(monkeypatch, tmp_path)
        d = _make_job(tmp_path, "sweep4")
        (d / "consented.txt").write_text("false", encoding="utf-8")
        # No expires_at.txt

        asyncio.run(_startup_sweep())
        assert not d.exists()

    def test_no_consented_file_skipped(self, tmp_path, monkeypatch):
        _patch_jobs_dir(monkeypatch, tmp_path)
        d = _make_job(tmp_path, "sweep5")
        # No consented.txt at all

        asyncio.run(_startup_sweep())
        assert d.exists()

    def test_non_dir_entries_skipped(self, tmp_path, monkeypatch):
        _patch_jobs_dir(monkeypatch, tmp_path)
        (tmp_path / "some_file.txt").write_text("hello", encoding="utf-8")
        asyncio.run(_startup_sweep())
        assert (tmp_path / "some_file.txt").exists()

    def test_invalid_expires_at_causes_deletion(self, tmp_path, monkeypatch):
        _patch_jobs_dir(monkeypatch, tmp_path)
        d = _make_job(tmp_path, "sweep6")
        (d / "consented.txt").write_text("false", encoding="utf-8")
        (d / "expires_at.txt").write_text("not-a-valid-iso-date", encoding="utf-8")

        asyncio.run(_startup_sweep())
        assert not d.exists()
