"""Tests for app/cli.py — the single-binary subcommand dispatcher.

The dispatcher is glue: it parses one positional, hands the rest to the
named command. We test the dispatch table directly (so we don't have to
spawn a subprocess) and verify each registered subcommand resolves to
the script we expect.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from app import cli


# ---------------------------------------------------------------------------
# Dispatch table — every script we promise in the docs is registered
# ---------------------------------------------------------------------------

EXPECTED_SUBCOMMANDS = {
    "serve",
    "smart-install",
    "recommend-models",
    "build-bundle",
    "build-index",
    "fetch-corpus",
    "model-stats",
    "version",
    "doctor",
}


def test_all_documented_subcommands_registered() -> None:
    assert EXPECTED_SUBCOMMANDS == set(cli.COMMANDS)


# ---------------------------------------------------------------------------
# _parse_top_level — argv slicing
# ---------------------------------------------------------------------------

def test_no_args_defaults_to_serve() -> None:
    assert cli._parse_top_level([]) == ("serve", [])


def test_known_subcommand_passes_remainder() -> None:
    assert cli._parse_top_level(["smart-install", "--dry-run"]) == (
        "smart-install", ["--dry-run"],
    )


def test_help_returns_none_with_empty_remainder() -> None:
    for token in ("-h", "--help", "help"):
        assert cli._parse_top_level([token]) == (None, [])


def test_unknown_subcommand_returns_none_with_error() -> None:
    """Unknown non-flag first arg → None signals 'print help and exit'."""
    cmd, rest = cli._parse_top_level(["bogus-thing"])
    assert cmd is None
    assert rest == []


def test_top_level_flag_routes_to_serve() -> None:
    """`LocalLLM --no-browser` should be treated as `LocalLLM serve --no-browser`."""
    assert cli._parse_top_level(["--no-browser"]) == ("serve", ["--no-browser"])
    assert cli._parse_top_level(["--port", "9000"]) == ("serve", ["--port", "9000"])


# ---------------------------------------------------------------------------
# _proxy — confirms each subcommand actually targets a script's main()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("subcmd,module_name", [
    ("smart-install",    "scripts.smart_install"),
    ("recommend-models", "scripts.recommend_models"),
    ("build-bundle",     "scripts.build_bundle"),
    ("build-index",      "scripts.build_index"),
    ("fetch-corpus",     "scripts.fetch_corpus"),
    ("model-stats",      "scripts.model_stats"),
])
def test_subcommand_proxies_to_correct_module(
    monkeypatch: pytest.MonkeyPatch, subcmd: str, module_name: str
) -> None:
    """Each proxied subcommand should call ``main()`` on the named module
    with sys.argv rewritten to ``[module_name, *forwarded_argv]``.

    We register a stub module in ``sys.modules`` so the proxy's lazy
    ``import_module`` lookup finds it without triggering the real script's
    heavy top-level imports (chromadb, ollama-client, etc.). The minimal
    CI test job intentionally doesn't install those.
    """
    captured: dict[str, object] = {"argv": None}

    def fake_main() -> int:
        captured["argv"] = list(sys.argv)
        return 0

    fake = types.ModuleType(module_name)
    fake.main = fake_main  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, fake)

    rc = cli.COMMANDS[subcmd](["--some-flag", "value"])
    assert rc == 0
    assert captured["argv"] == [module_name, "--some-flag", "value"]


# ---------------------------------------------------------------------------
# version subcommand — runs without crashing, shows the right keys
# ---------------------------------------------------------------------------

def test_version_subcommand_prints_diagnostics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Diagnostics print under a tmp data dir so we don't pollute ~/.local/."""
    monkeypatch.setenv("LOCALLLM_DATA_DIR", str(tmp_path))
    rc = cli.COMMANDS["version"]([])
    assert rc == 0
    out = capsys.readouterr().out
    for marker in ("LocalLLM", "python:", "frozen:", "exe_dir:", "ollama binary:"):
        assert marker in out


# ---------------------------------------------------------------------------
# main() — exit codes
# ---------------------------------------------------------------------------

def test_main_unknown_command_returns_2(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(["bogus"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown subcommand" in err


def test_main_help_returns_0(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(["--help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Subcommands:" in out


# ---------------------------------------------------------------------------
# _ensure_port_free — pre-bind probe with a friendly error message
# ---------------------------------------------------------------------------

def test_ensure_port_free_returns_zero_on_unbound_port() -> None:
    """Picking an obviously-free port — kernel hands one back via
    bind(0) — must produce no error. Regression guard against a
    paranoid implementation that declares everything taken."""
    import socket as _s
    from app.cli import _ensure_port_free

    # Find a port that's free RIGHT NOW. Don't bind — just ask
    # the kernel for one and immediately release it; then the
    # check should still see it as free in the next instant.
    with _s.socket(_s.AF_INET, _s.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]

    rc = _ensure_port_free("127.0.0.1", free_port)
    assert rc == 0


def test_ensure_port_free_reports_conflict_and_suggests_fixes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When the port is already bound, exit non-zero and the error
    message must name the conflicting PID, the kill command, and
    the alternative-port fix. Regression guard for the operator
    experience the user got bitten by — uvicorn's misleading
    "Application startup complete" before bind failure."""
    import socket as _s
    from app.cli import _ensure_port_free

    # Hold the port for real so the probe sees a genuine conflict.
    with _s.socket(_s.AF_INET, _s.SOCK_STREAM) as held:
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        port = held.getsockname()[1]

        rc = _ensure_port_free("127.0.0.1", port)

    assert rc != 0, "expected non-zero exit when port is held"
    err = capsys.readouterr().err
    # Operator-facing message MUST contain the offending port + the
    # essentials of the fix path. Substring checks so wording can
    # drift without breaking the test.
    assert f":{port}" in err, "error must name the conflicting port"
    assert "PID" in err or "another process" in err
    assert "--port" in err, "must suggest the alternative-port flag"
    # Both Windows and POSIX guidance should appear so the message
    # works for the user regardless of platform.
    assert "Get-NetTCPConnection" in err
    assert "lsof" in err


def test_ensure_port_free_handles_missing_psutil(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``_find_pid_on_port`` must degrade gracefully when psutil is
    absent — we still print a usable error, just without a PID. The
    bundled binary path includes psutil, but a developer install or
    a stripped airgap bundle might not, and we don't want the
    helpful-error feature to crash the conflict detection."""
    import sys as _sys
    import socket as _s
    from app.cli import _ensure_port_free

    # Hide psutil so _find_pid_on_port falls into the ImportError
    # branch (returns None → message says "another process").
    monkeypatch.setitem(_sys.modules, "psutil", None)

    with _s.socket(_s.AF_INET, _s.SOCK_STREAM) as held:
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        port = held.getsockname()[1]

        rc = _ensure_port_free("127.0.0.1", port)

    assert rc != 0
    err = capsys.readouterr().err
    # Without psutil we still report the port + the fix paths.
    assert f":{port}" in err
    assert "--port" in err
