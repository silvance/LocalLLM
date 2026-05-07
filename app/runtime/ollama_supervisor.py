"""Lifecycle manager for the embedded Ollama process.

The single-file LocalLLM.exe ships ``ollama.exe`` (Windows) or
``ollama`` (Linux) inside its PyInstaller payload. On launch we extract
it, spawn ``ollama serve`` against a non-default port + a bundle-local
models directory, wait for it to bind, and tear it down on exit.

Why a non-default port: the airgap target may already have an
unrelated Ollama install at ``:11434``. Running ours at ``:11435`` keeps
the two from fighting over the port and prevents us from accidentally
adding our bundled models to the user's existing install.

The supervisor also degrades gracefully:
  * No bundled binary AND no `ollama` on PATH → returns False, caller
    can fall back to "expect the user to have Ollama running already."
  * Bundled binary present → runs it, returns True, terminates on exit.
"""
from __future__ import annotations

import logging
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from .paths import exe_dir, meipass_dir, ollama_models_dir, user_data_dir


log = logging.getLogger(__name__)


# Non-default port — avoids colliding with any host Ollama at :11434.
DEFAULT_BUNDLED_HOST = "127.0.0.1"
DEFAULT_BUNDLED_PORT = 11435


def find_ollama_binary() -> Optional[Path]:
    """Locate the ollama executable, preferring the bundled copy.

    Search order (first match wins):
      1. ``LOCALLLM_OLLAMA_BIN`` env var (operator override)
      2. PyInstaller _MEIPASS/ollama/ (where pyinstaller.spec extracts it)
      3. ``<exe_dir>/ollama/`` next to the .exe (portable / non-frozen layout)
      4. ``ollama`` / ``ollama.exe`` on PATH (user already installed)
    """
    override = os.getenv("LOCALLLM_OLLAMA_BIN")
    if override:
        p = Path(override).expanduser()
        if p.exists():
            return p

    name = "ollama.exe" if sys.platform == "win32" else "ollama"
    for base in (meipass_dir(), exe_dir()):
        if base is None:
            continue
        candidate = base / "ollama" / name
        if candidate.exists():
            return candidate

    on_path = shutil.which("ollama")
    if on_path:
        return Path(on_path)

    return None


def is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """True if a TCP server is accepting connections on host:port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_for_ollama(host: str, port: int, timeout: float = 30.0) -> bool:
    """Poll the Ollama HTTP API until /api/tags responds 200 or timeout."""
    url = f"http://{host}:{port}/api/tags"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if 200 <= resp.status < 300:
                    return True
        except (urllib.error.URLError, OSError, ConnectionError):
            pass
        time.sleep(0.3)
    return False


class OllamaSupervisor:
    """Owns one ``ollama serve`` subprocess.

    Use as a context manager OR call ``start()``/``stop()`` directly.
    Idempotent: ``start()`` on an already-running supervisor is a no-op,
    and ``stop()`` on a never-started one is a no-op.
    """

    def __init__(
        self,
        host: str = DEFAULT_BUNDLED_HOST,
        port: int = DEFAULT_BUNDLED_PORT,
        models_dir: Optional[Path] = None,
        binary: Optional[Path] = None,
        log_path: Optional[Path] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.models_dir = models_dir or ollama_models_dir()
        self.binary = binary or find_ollama_binary()
        self.log_path = log_path or (user_data_dir() / "ollama-localllm.log")
        self._proc: Optional[subprocess.Popen] = None
        self._log_handle = None

    # ----- introspection -------------------------------------------------

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ----- lifecycle -----------------------------------------------------

    def start(self, wait: bool = True, timeout: float = 30.0) -> bool:
        """Start ollama serve. Returns True if Ollama is reachable when we return.

        If something is *already* serving on (host, port) we adopt it
        rather than spawning a duplicate — the user may have started
        Ollama themselves, and we don't want two servers fighting.
        """
        if self.is_running:
            return True

        if is_port_open(self.host, self.port):
            log.info("Ollama already serving on %s:%s — adopting", self.host, self.port)
            return True

        if self.binary is None:
            log.error(
                "No ollama binary found. Set LOCALLLM_OLLAMA_BIN, drop one in "
                "%s/ollama/, or install Ollama on this machine.",
                exe_dir(),
            )
            return False

        env = os.environ.copy()
        env["OLLAMA_HOST"] = f"{self.host}:{self.port}"
        env["OLLAMA_MODELS"] = str(self.models_dir)
        # Belt-and-suspenders: prevent ollama from hijacking telemetry on
        # an airgap network where outbound calls would just hang.
        env.setdefault("OLLAMA_NOHISTORY", "1")

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("ab")

        creationflags = 0
        if sys.platform == "win32":
            # CREATE_NO_WINDOW = 0x08000000 — keeps Ollama from popping
            # up its own console window when we're launched via double-
            # click (we already have one for our own logs).
            creationflags = 0x08000000

        log.info(
            "Spawning ollama: %s serve  (host=%s:%s, models=%s, log=%s)",
            self.binary, self.host, self.port, self.models_dir, self.log_path,
        )
        self._proc = subprocess.Popen(
            [str(self.binary), "serve"],
            env=env,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
            close_fds=(sys.platform != "win32"),
        )

        if not wait:
            return True

        if wait_for_ollama(self.host, self.port, timeout=timeout):
            return True

        log.error(
            "Ollama did not respond within %ss. See log: %s",
            timeout, self.log_path,
        )
        self.stop()
        return False

    def stop(self, timeout: float = 5.0) -> None:
        """Terminate the ollama serve process if we started one."""
        proc = self._proc
        self._proc = None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=timeout)
            except OSError as exc:
                log.warning("Failed to terminate ollama (pid=%s): %s", proc.pid, exc)
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            finally:
                self._log_handle = None

    # ----- context manager ----------------------------------------------

    def __enter__(self) -> "OllamaSupervisor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
