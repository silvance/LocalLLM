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
import threading
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


# Allowlisted env vars passed to the ollama-serve subprocess. Ollama
# reads HTTPS_PROXY / OLLAMA_* / GPU runner vars itself, so we keep
# those — but we do NOT want to forward AWS_*, GH_TOKEN, LOCALLLM_*,
# or the operator's full shell environment to a process that's free
# to phone home with whatever credentials it sees.
_OLLAMA_ENV_ALLOWLIST: frozenset[str] = frozenset({
    # Process-startup
    "PATH", "HOME", "TMPDIR", "TEMP", "TMP",
    "LANG", "LC_ALL", "LC_CTYPE",
    "SYSTEMROOT", "WINDIR", "USERPROFILE", "COMSPEC", "PATHEXT",
    # Ollama-recognised vars the operator may have configured.
    "OLLAMA_HOST", "OLLAMA_MODELS", "OLLAMA_KEEP_ALIVE",
    "OLLAMA_NUM_PARALLEL", "OLLAMA_MAX_LOADED_MODELS",
    "OLLAMA_FLASH_ATTENTION", "OLLAMA_KV_CACHE_TYPE",
    "OLLAMA_DEBUG", "OLLAMA_NOHISTORY", "OLLAMA_RUNNERS_DIR",
    "OLLAMA_TMPDIR", "OLLAMA_LLM_LIBRARY", "OLLAMA_GPU_OVERHEAD",
    # GPU runner discovery (we want Ollama to find the user's CUDA / ROCm).
    "CUDA_VISIBLE_DEVICES", "CUDA_PATH", "CUDA_HOME",
    "ROCM_PATH", "HIP_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION",
    "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH",
})


def _safe_ollama_env() -> dict[str, str]:
    """Filtered process env for the ollama-serve subprocess.
    Allowlist-based so adding a new env var in the parent never
    silently leaks in."""
    src = os.environ
    return {k: src[k] for k in _OLLAMA_ENV_ALLOWLIST if k in src}


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
        self._explicit_binary = binary
        self.models_dir = models_dir or ollama_models_dir()
        self.log_path = log_path or (user_data_dir() / "ollama-localllm.log")
        self._proc: Optional[subprocess.Popen] = None
        self._log_handle = None
        # Guards start/stop against concurrent callers (atexit handler
        # + context-manager exit + explicit stop() can otherwise race
        # to terminate the same Popen).
        self._lifecycle_lock = threading.Lock()

    @property
    def binary(self) -> Optional[Path]:
        """Resolved lazily so callers see a freshly-extracted bundle.

        Returning a cached `__init__`-time value would miss binaries
        materialized after construction (e.g. PyInstaller `_MEIPASS`
        extraction in tests, or operator-set ``LOCALLLM_OLLAMA_BIN``).
        """
        return self._explicit_binary or find_ollama_binary()

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
        with self._lifecycle_lock:
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

            # Allowlisted env — same defense-in-depth shape as the
            # smoke-gate. We don't want to forward AWS keys, GH_TOKEN,
            # arbitrary HTTPS_PROXY etc. to an ollama child that's free
            # to bridge an airgap network on those credentials.
            env = _safe_ollama_env()
            env["OLLAMA_HOST"] = f"{self.host}:{self.port}"
            env["OLLAMA_MODELS"] = str(self.models_dir)
            # Outbound telemetry would just hang on an airgap network.
            env.setdefault("OLLAMA_NOHISTORY", "1")

            self.models_dir.mkdir(parents=True, exist_ok=True)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = self.log_path.open("ab")

            # Suppress Ollama's console window on Windows AND give it
            # its own process group so a Ctrl-C delivered to our shell
            # doesn't kill it before stop() runs. On POSIX,
            # start_new_session does the equivalent so terminate() is
            # scoped to just the child.
            popen_kwargs: dict = {}
            binary = self.binary
            if binary is None:
                log.error("Ollama binary disappeared between start() entry and spawn")
                self._log_handle.close()
                self._log_handle = None
                return False
            if sys.platform == "win32":
                popen_kwargs["creationflags"] = (
                    subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:
                popen_kwargs["start_new_session"] = True
                popen_kwargs["close_fds"] = True

            log.info(
                "Spawning ollama: %s serve  (host=%s:%s, models=%s, log=%s)",
                binary, self.host, self.port, self.models_dir, self.log_path,
            )
            try:
                self._proc = subprocess.Popen(
                    [str(binary), "serve"],
                    env=env,
                    stdout=self._log_handle,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    **popen_kwargs,
                )
            except OSError as exc:
                # Popen failed (binary not executable, missing libs,
                # exec format error) — release the log handle that
                # would otherwise leak until GC since stop() is a no-op
                # when _proc never got assigned.
                log.error("Failed to spawn ollama: %s", exc)
                if self._log_handle is not None:
                    self._log_handle.close()
                    self._log_handle = None
                return False

        # Wait for the daemon to bind OUTSIDE the lifecycle lock — the
        # poll loop can take 30s and we don't want to block stop()
        # callers (e.g. an atexit handler) for that long.
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
        """Terminate the ollama serve process if we started one.

        Lock-guarded against concurrent stop / start callers (e.g. an
        atexit handler firing while the context-manager exit also
        runs) — without this, two threads could each read self._proc
        non-None, both call terminate, and the second would also
        attempt to wait/close the log handle.
        """
        with self._lifecycle_lock:
            proc = self._proc
            self._proc = None
            log_handle = self._log_handle
            self._log_handle = None
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
        if log_handle is not None:
            try:
                log_handle.close()
            except OSError:
                pass

    # ----- context manager ----------------------------------------------

    def __enter__(self) -> "OllamaSupervisor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
