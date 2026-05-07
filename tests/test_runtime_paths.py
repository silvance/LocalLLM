"""Tests for app/runtime/paths.py — filesystem layout helpers.

The supervisor + cli depend on these for "where does X live" so the
guarantees here matter: ``LOCALLLM_DATA_DIR`` overrides everything,
portable mode wins over user-data when the exe lives somewhere
writable, and ``LOCALLLM_OLLAMA_MODELS`` overrides the models dir.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.runtime import paths  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip env vars the helpers honor — keeps tests independent."""
    monkeypatch.delenv("LOCALLLM_DATA_DIR", raising=False)
    monkeypatch.delenv("LOCALLLM_OLLAMA_MODELS", raising=False)
    monkeypatch.delenv("LOCALLLM_OLLAMA_BIN", raising=False)


# ---------------------------------------------------------------------------
# is_frozen / meipass_dir / exe_dir
# ---------------------------------------------------------------------------

def test_is_frozen_false_in_dev() -> None:
    assert paths.is_frozen() is False


def test_meipass_dir_none_in_dev() -> None:
    assert paths.meipass_dir() is None


def test_exe_dir_falls_back_to_repo_root_in_dev() -> None:
    # In dev, exe_dir should resolve to the repo root (parent of app/).
    assert (paths.exe_dir() / "app").is_dir()


def test_meipass_dir_returns_path_when_frozen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert paths.meipass_dir() == tmp_path


def test_exe_dir_uses_executable_when_frozen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_exe = tmp_path / "LocalLLM.exe"
    fake_exe.write_bytes(b"")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(fake_exe), raising=False)
    assert paths.exe_dir() == tmp_path.resolve()


# ---------------------------------------------------------------------------
# user_data_dir — overrides + portable detection
# ---------------------------------------------------------------------------

def test_user_data_dir_honors_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "custom-data"
    monkeypatch.setenv("LOCALLLM_DATA_DIR", str(target))
    assert paths.user_data_dir() == target.resolve()
    assert target.exists()


def test_user_data_dir_creates_override_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c"
    monkeypatch.setenv("LOCALLLM_DATA_DIR", str(nested))
    paths.user_data_dir()
    assert nested.exists()


def test_user_data_dir_portable_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When exe_dir is writable, portable LocalLLM-Data sibling wins."""
    monkeypatch.setattr(paths, "exe_dir", lambda: tmp_path)
    out = paths.user_data_dir()
    assert out == tmp_path / "LocalLLM-Data"
    assert out.exists()


def test_user_data_dir_falls_back_to_xdg_when_portable_unwritable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Read-only exe_dir → user-home fallback."""
    monkeypatch.setattr(paths, "exe_dir", lambda: tmp_path / "no-write-here")
    monkeypatch.setattr(paths, "_is_writable_or_creatable", lambda _: False)
    fake_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr(sys, "platform", "linux", raising=False)
    out = paths.user_data_dir()
    assert out == fake_home / ".local" / "share" / "LocalLLM"
    assert out.exists()


# ---------------------------------------------------------------------------
# ollama_models_dir
# ---------------------------------------------------------------------------

def test_ollama_models_dir_honors_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "models-elsewhere"
    monkeypatch.setenv("LOCALLLM_OLLAMA_MODELS", str(target))
    assert paths.ollama_models_dir() == target.resolve()
    assert target.exists()


def test_ollama_models_dir_prefers_sibling_when_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(paths, "exe_dir", lambda: tmp_path)
    sibling = tmp_path / "ollama_models"
    sibling.mkdir()
    assert paths.ollama_models_dir() == sibling


def test_ollama_models_dir_falls_back_to_user_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(paths, "exe_dir", lambda: tmp_path)
    # No sibling ollama_models/ exists, so we expect user_data/ollama_models
    user_data = tmp_path / "user-data"
    user_data.mkdir()
    monkeypatch.setattr(paths, "user_data_dir", lambda: user_data)
    out = paths.ollama_models_dir()
    assert out == user_data / "ollama_models"
    assert out.exists()
