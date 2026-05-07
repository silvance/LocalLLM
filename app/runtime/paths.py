"""Filesystem layout for the single-file LocalLLM desktop binary.

PyInstaller onefile extracts the bundle's read-only payload to a temp
``_MEIPASS`` directory on every launch and deletes it on exit. Anything
the user is allowed to mutate (chats, models, the .env file) lives
*outside* that temp dir — alongside the .exe in dev/portable mode, or
under ``%LOCALAPPDATA%\\LocalLLM\\`` when the user installs system-wide.

The functions here are the single source of truth for "where does X
live" so we never sprinkle path-guessing across the codebase.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    """True when running from a PyInstaller-built executable."""
    return getattr(sys, "frozen", False)


def meipass_dir() -> Path | None:
    """Where PyInstaller extracted read-only payload data, or None in dev."""
    if not is_frozen():
        return None
    base = getattr(sys, "_MEIPASS", None)
    return Path(base) if base else None


def exe_dir() -> Path:
    """Directory containing the LocalLLM executable.

    In dev (unfrozen) we fall back to the repo root so callers don't have
    to special-case dev vs frozen.
    """
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent.parent


def user_data_dir() -> Path:
    """Where mutable user data lives (chats, .env, ollama models if not portable).

    Resolution order:
      1. ``LOCALLLM_DATA_DIR`` env var (operator override)
      2. ``<exe_dir>/LocalLLM-Data/`` if writable — portable mode
      3. ``%LOCALAPPDATA%\\LocalLLM\\`` on Windows
      4. ``~/.local/share/LocalLLM/`` on Linux/macOS

    The directory is created if missing. The portable check matters for the
    airgap workflow: if the .exe sits next to its data folder on a USB
    stick, the user can move the whole thing without losing their chats.
    """
    override = os.getenv("LOCALLLM_DATA_DIR")
    if override:
        path = Path(override).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    portable = exe_dir() / "LocalLLM-Data"
    if _is_writable_or_creatable(portable):
        portable.mkdir(parents=True, exist_ok=True)
        return portable

    if sys.platform == "win32":
        base = os.getenv("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        path = Path(base) / "LocalLLM"
    else:
        path = Path.home() / ".local" / "share" / "LocalLLM"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ollama_models_dir() -> Path:
    """Where Ollama should store / find its model blobs.

    Bundled airgap deployments put a pre-populated ``ollama_models/`` next
    to the .exe; we honor that first. Otherwise it lives under user data.
    """
    override = os.getenv("LOCALLLM_OLLAMA_MODELS")
    if override:
        path = Path(override).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    sibling = exe_dir() / "ollama_models"
    if sibling.exists():
        return sibling
    path = user_data_dir() / "ollama_models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def env_file() -> Path:
    """Path to the .env we read/write at runtime (under user data dir)."""
    return user_data_dir() / ".env"


def _is_writable_or_creatable(path: Path) -> bool:
    """True if ``path`` exists and is writable, OR can be created.

    Used to decide whether portable mode is viable. We don't want to
    clobber a system install location that happens to be read-only.
    """
    try:
        if path.exists():
            return os.access(path, os.W_OK)
        # Probe the parent — if we can create the dir, portable is fine.
        return os.access(path.parent, os.W_OK)
    except OSError:
        return False
