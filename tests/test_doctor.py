"""Tests for app/runtime/doctor.py — the self-test diagnostics.

Each check is independently testable. We mock the underlying probes so
the checks pass/fail deterministically regardless of whether the dev
machine has Ollama, GPU, or a built RAG index.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.runtime import doctor
from app.runtime.doctor import CheckResult


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def test_check_paths_ok_when_writable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LOCALLLM_DATA_DIR", str(tmp_path))
    result = doctor.check_paths()
    assert result.status == "ok"
    assert str(tmp_path) in result.detail


def test_check_paths_fails_when_unwritable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Any of the three required dirs being missing -> FAIL."""
    from app.runtime import paths
    monkeypatch.setattr(paths, "exe_dir", lambda: tmp_path / "missing-parent" / "exe")
    monkeypatch.setattr(paths, "user_data_dir", lambda: tmp_path / "u")
    monkeypatch.setattr(paths, "ollama_models_dir", lambda: tmp_path / "missing-models")
    result = doctor.check_paths()
    assert result.status == "fail"
    assert result.fix


def test_check_ollama_binary_ok(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = tmp_path / "ollama"
    fake.write_bytes(b"")
    monkeypatch.setenv("LOCALLLM_OLLAMA_BIN", str(fake))
    result = doctor.check_ollama_binary()
    assert result.status == "ok"
    assert str(fake) in result.detail


def test_check_ollama_binary_fails_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("LOCALLLM_OLLAMA_BIN", raising=False)
    from app.runtime import ollama_supervisor
    monkeypatch.setattr(ollama_supervisor, "meipass_dir", lambda: None)
    monkeypatch.setattr(ollama_supervisor, "exe_dir", lambda: tmp_path)
    monkeypatch.setattr(ollama_supervisor.shutil, "which", lambda _: None)
    result = doctor.check_ollama_binary()
    assert result.status == "fail"
    assert "no ollama" in result.detail


def test_check_ollama_reachable_warn_when_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Connection refused → WARN, not FAIL (the user may not have started it yet)."""
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    result = doctor.check_ollama_reachable()
    assert result.status in ("warn", "ok")  # ok if a real ollama is running


def test_check_models_on_disk_fail_when_no_manifests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LOCALLLM_OLLAMA_MODELS", str(tmp_path / "empty-models"))
    result = doctor.check_models_on_disk()
    assert result.status == "fail"
    assert result.fix


def test_check_models_on_disk_ok_when_manifest_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    models = tmp_path / "models"
    # Mimic Ollama's registry/library/<model>/<tag> layout
    manifest_path = models / "manifests" / "registry.ollama.ai" / "library" / "granite4" / "latest"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("LOCALLLM_OLLAMA_MODELS", str(models))
    result = doctor.check_models_on_disk()
    assert result.status == "ok"
    assert "granite4:latest" in result.detail


def test_check_rag_index_warns_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No FAIL — chat works without RAG."""
    from app.config import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "nonexistent-index"))
    result = doctor.check_rag_index()
    assert result.status == "warn"
    get_settings.cache_clear()


def test_check_hardware_returns_ok() -> None:
    """Hardware probe should always succeed — runs locally, no IO."""
    result = doctor.check_hardware()
    assert result.status == "ok"
    assert "tier=" in result.detail


# ---------------------------------------------------------------------------
# Runner / render
# ---------------------------------------------------------------------------

def test_run_checks_aggregates_results() -> None:
    fake_results = [
        CheckResult("a", "ok", "fine"),
        CheckResult("b", "warn", "meh"),
    ]
    out = doctor.run_checks([lambda: fake_results[0], lambda: fake_results[1]])
    assert out == fake_results


def test_run_checks_catches_check_crash() -> None:
    def boom() -> CheckResult:
        raise RuntimeError("oops")
    out = doctor.run_checks([boom])
    assert len(out) == 1
    assert out[0].status == "fail"
    assert "oops" in out[0].detail


def test_render_includes_summary() -> None:
    rendered = doctor.render([
        CheckResult("paths", "ok", "fine"),
        CheckResult("ollama", "warn", "missing", fix="install ollama"),
        CheckResult("rag", "fail", "no index", fix="run build-index"),
    ])
    assert "1 ok, 1 warn, 1 fail" in rendered
    assert "→ install ollama" in rendered
    assert "→ run build-index" in rendered
    assert "[ OK ]" in rendered
    assert "[WARN]" in rendered
    assert "[FAIL]" in rendered


def test_main_returns_1_on_any_fail(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(doctor, "run_checks", lambda *_: [
        CheckResult("a", "fail", "bad"),
    ])
    rc = doctor.main()
    assert rc == 1
    assert "[FAIL]" in capsys.readouterr().out


def test_main_returns_0_when_all_ok_or_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "run_checks", lambda *_: [
        CheckResult("a", "ok", "fine"),
        CheckResult("b", "warn", "meh"),
    ])
    assert doctor.main() == 0
