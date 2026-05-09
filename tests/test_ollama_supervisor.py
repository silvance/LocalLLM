"""Tests for app/runtime/ollama_supervisor.py.

Most of the supervisor is wiring around subprocess.Popen and HTTP polling
— hard to unit-test without ACTUALLY spawning ollama. We test the parts
we can isolate: binary discovery (env override, _MEIPASS, exe_dir, PATH),
port-open detection, and the lifecycle no-ops (stop on never-started).
The full subprocess path is exercised end-to-end by the CI smoke check.
"""
from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

import pytest

from app.runtime import ollama_supervisor as os_mod
from app.runtime.ollama_supervisor import (
    OllamaSupervisor,
    find_ollama_binary,
    is_port_open,
    wait_for_ollama,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCALLLM_OLLAMA_BIN", raising=False)


# ---------------------------------------------------------------------------
# find_ollama_binary
# ---------------------------------------------------------------------------

def _binary_name() -> str:
    return "ollama.exe" if sys.platform == "win32" else "ollama"


def test_env_override_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """LOCALLLM_OLLAMA_BIN beats every other candidate."""
    fake = tmp_path / "my-ollama"
    fake.write_bytes(b"")
    monkeypatch.setenv("LOCALLLM_OLLAMA_BIN", str(fake))
    # Even if MEIPASS is set with a different binary, env wins.
    other = tmp_path / "ollama" / _binary_name()
    other.parent.mkdir()
    other.write_bytes(b"")
    monkeypatch.setattr(os_mod, "meipass_dir", lambda: tmp_path)
    assert find_ollama_binary() == fake


def test_meipass_takes_precedence_over_exe_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    meipass = tmp_path / "mei"
    exe_dir = tmp_path / "exe"
    (meipass / "ollama").mkdir(parents=True)
    (exe_dir / "ollama").mkdir(parents=True)
    bin_in_mei = meipass / "ollama" / _binary_name()
    bin_in_exe = exe_dir / "ollama" / _binary_name()
    bin_in_mei.write_bytes(b"")
    bin_in_exe.write_bytes(b"")

    monkeypatch.setattr(os_mod, "meipass_dir", lambda: meipass)
    monkeypatch.setattr(os_mod, "exe_dir", lambda: exe_dir)
    assert find_ollama_binary() == bin_in_mei


def test_falls_back_to_exe_dir_when_no_meipass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bin_in_exe = tmp_path / "ollama" / _binary_name()
    bin_in_exe.parent.mkdir()
    bin_in_exe.write_bytes(b"")

    monkeypatch.setattr(os_mod, "meipass_dir", lambda: None)
    monkeypatch.setattr(os_mod, "exe_dir", lambda: tmp_path)
    assert find_ollama_binary() == bin_in_exe


def test_returns_none_when_nothing_is_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(os_mod, "meipass_dir", lambda: None)
    monkeypatch.setattr(os_mod, "exe_dir", lambda: tmp_path)
    monkeypatch.setattr(os_mod.shutil, "which", lambda _: None)
    assert find_ollama_binary() is None


def test_falls_back_to_path_when_no_bundled_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(os_mod, "meipass_dir", lambda: None)
    monkeypatch.setattr(os_mod, "exe_dir", lambda: tmp_path)
    on_path = tmp_path / "from-path"
    on_path.write_bytes(b"")
    monkeypatch.setattr(os_mod.shutil, "which", lambda _: str(on_path))
    assert find_ollama_binary() == on_path


# ---------------------------------------------------------------------------
# is_port_open
# ---------------------------------------------------------------------------

def test_is_port_open_false_when_nothing_listening() -> None:
    # Pick a random unused port and don't bind anything.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    assert is_port_open("127.0.0.1", port, timeout=0.1) is False


def test_is_port_open_true_when_server_accepts() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    accepted: list[socket.socket] = []

    def _accept_one() -> None:
        try:
            conn, _ = server.accept()
            accepted.append(conn)
        except OSError:
            pass

    t = threading.Thread(target=_accept_one, daemon=True)
    t.start()
    try:
        assert is_port_open("127.0.0.1", port, timeout=2.0) is True
    finally:
        for c in accepted:
            c.close()
        server.close()
        t.join(timeout=1.0)


# ---------------------------------------------------------------------------
# wait_for_ollama — smoke behaviour
# ---------------------------------------------------------------------------

def test_wait_for_ollama_times_out_quickly() -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    start = time.monotonic()
    assert wait_for_ollama("127.0.0.1", port, timeout=0.5) is False
    elapsed = time.monotonic() - start
    # Generous bound — the loop sleeps 0.3s between attempts.
    assert elapsed < 2.5


# ---------------------------------------------------------------------------
# Supervisor lifecycle no-ops
# ---------------------------------------------------------------------------

def test_stop_is_idempotent_on_never_started() -> None:
    sup = OllamaSupervisor(binary=Path("/does/not/exist"))
    sup.stop()  # Should not raise.
    sup.stop()


def test_start_returns_false_when_no_binary_and_port_free(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    sup = OllamaSupervisor(host="127.0.0.1", port=port, binary=None,
                           models_dir=tmp_path / "models",
                           log_path=tmp_path / "log.txt")
    assert sup.start(wait=False) is False
    assert sup.is_running is False


def test_start_adopts_existing_server_on_port() -> None:
    """If something is already listening on (host, port), we adopt it."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    accepted: list[socket.socket] = []

    def _accept_one() -> None:
        try:
            conn, _ = server.accept()
            accepted.append(conn)
        except OSError:
            pass

    t = threading.Thread(target=_accept_one, daemon=True)
    t.start()
    try:
        sup = OllamaSupervisor(
            host="127.0.0.1", port=port,
            binary=Path("/does/not/exist"),  # Should never be invoked.
        )
        assert sup.start(wait=False) is True
        assert sup.is_running is False  # we adopted, didn't spawn
    finally:
        for c in accepted:
            c.close()
        server.close()
        t.join(timeout=1.0)


def test_base_url_format() -> None:
    sup = OllamaSupervisor(host="1.2.3.4", port=9999)
    assert sup.base_url == "http://1.2.3.4:9999"


def test_safe_ollama_env_drops_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Subprocess env must not inherit AWS_*, GH_TOKEN, LOCALLLM_*,
    HTTP_PROXY etc. from the parent shell — the ollama child reads
    HTTPS_PROXY itself and we don't want a stale corp proxy in the
    operator's shell silently routing model traffic."""
    from app.runtime.ollama_supervisor import _safe_ollama_env
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "do-not-leak")
    monkeypatch.setenv("GH_TOKEN", "ghp_do_not_leak")
    monkeypatch.setenv("LOCALLLM_DATA_DIR", "/tmp/sensitive")
    monkeypatch.setenv("HTTPS_PROXY", "http://corp-proxy:8080")
    env = _safe_ollama_env()
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "GH_TOKEN" not in env
    assert "LOCALLLM_DATA_DIR" not in env
    assert "HTTPS_PROXY" not in env


def test_safe_ollama_env_keeps_ollama_runtime_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """OLLAMA_KEEP_ALIVE / OLLAMA_NUM_PARALLEL / CUDA_VISIBLE_DEVICES
    are vars Ollama itself reads — don't strip them."""
    from app.runtime.ollama_supervisor import _safe_ollama_env
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "10m")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    env = _safe_ollama_env()
    assert env.get("OLLAMA_KEEP_ALIVE") == "10m"
    assert env.get("CUDA_VISIBLE_DEVICES") == "0,1"
