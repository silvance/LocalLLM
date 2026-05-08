"""Tests for app/services/review_gates.py — the static-analysis gates
that fence the reviewer model from broken code."""
from __future__ import annotations

import pytest

from app.services.review_gates import (
    DEFAULT_GATES,
    GateResult,
    all_passed,
    format_for_writer,
    gate_imports,
    gate_lint,
    gate_smoke,
    gate_syntax,
    run_gates,
)


# ---------------------------------------------------------------------------
# gate_syntax
# ---------------------------------------------------------------------------

def test_gate_syntax_passes_on_valid_code() -> None:
    r = gate_syntax("def foo():\n    return 1\n")
    assert r.passed is True
    assert r.name == "syntax"


def test_gate_syntax_reports_line_on_failure() -> None:
    r = gate_syntax("def foo(:\n    pass\n")
    assert r.passed is False
    assert r.messages
    assert "line" in r.messages[0]


# ---------------------------------------------------------------------------
# gate_lint (pyflakes)
# ---------------------------------------------------------------------------

def test_gate_lint_passes_on_clean_code() -> None:
    code = "def add(a, b):\n    return a + b\n"
    r = gate_lint(code)
    assert r.passed is True


def test_gate_lint_flags_undefined_name() -> None:
    code = "def use_ghost():\n    return ghost_var\n"
    r = gate_lint(code)
    assert r.passed is False
    assert any("ghost_var" in m for m in r.messages)


# ---------------------------------------------------------------------------
# gate_imports
# ---------------------------------------------------------------------------

def test_gate_imports_passes_on_stdlib_only() -> None:
    code = "import os\nimport sys\nfrom pathlib import Path\n"
    r = gate_imports(code)
    assert r.passed is True


def test_gate_imports_flags_fake_module() -> None:
    code = "import scapy_ng_definitely_does_not_exist\n"
    r = gate_imports(code)
    assert r.passed is False
    assert any("scapy_ng" in m for m in r.messages)


def test_gate_imports_skips_relative_imports() -> None:
    """Relative imports (from . import x) reference siblings of the
    file; we can't resolve them out of context, so the gate skips."""
    code = "from . import sibling\n"
    r = gate_imports(code)
    assert r.passed is True


def test_gate_imports_silent_on_syntax_error() -> None:
    """If the code doesn't parse, gate_imports defers — gate_syntax
    is the one that should report the failure (avoids double-noise)."""
    r = gate_imports("def foo(:\n")
    assert r.passed is True


# ---------------------------------------------------------------------------
# gate_smoke (opt-in)
# ---------------------------------------------------------------------------

def test_gate_smoke_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCALLLM_REVIEW_GATE_SMOKE_ENABLED", raising=False)
    r = gate_smoke("import os\n")
    assert r.passed is True
    assert "disabled" in r.messages[0].lower()


def test_gate_smoke_runs_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCALLLM_REVIEW_GATE_SMOKE_ENABLED", "1")
    r = gate_smoke("x = 1 + 2\n")
    assert r.passed is True


def test_gate_smoke_catches_import_time_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCALLLM_REVIEW_GATE_SMOKE_ENABLED", "1")
    code = "raise RuntimeError('boom at import')\n"
    r = gate_smoke(code, timeout=10.0)
    assert r.passed is False
    assert any("boom" in m or "RuntimeError" in m for m in r.messages)


# ---------------------------------------------------------------------------
# run_gates — short-circuit behaviour
# ---------------------------------------------------------------------------

def test_run_gates_short_circuits_on_first_failure() -> None:
    """If gate 1 fails, gates 2/3/4 should not run."""
    bad = "def foo(:\n"
    results = run_gates(bad)
    assert len(results) == 1
    assert results[0].name == "syntax"
    assert results[0].passed is False


def test_run_gates_runs_all_when_clean() -> None:
    code = "import os\nprint(os.getcwd)\n"
    results = run_gates(code)
    assert len(results) == len(DEFAULT_GATES)
    assert all_passed(results)


def test_run_gates_lint_failure_doesnt_run_imports() -> None:
    code = "def foo():\n    return ghost\n"
    results = run_gates(code)
    assert [r.name for r in results] == ["syntax", "lint"]
    assert results[-1].passed is False


def test_run_gates_continues_past_smoke_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Smoke-disabled means it always passes — full chain runs."""
    monkeypatch.delenv("LOCALLLM_REVIEW_GATE_SMOKE_ENABLED", raising=False)
    results = run_gates("x = 1\n")
    assert all_passed(results)
    assert results[-1].name == "smoke"


def test_run_gates_handles_crashing_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    def crashing(_code: str) -> GateResult:
        raise RuntimeError("oh no")
    crashing.__name__ = "gate_crashing"
    results = run_gates("x = 1\n", gates=[crashing])
    assert len(results) == 1
    assert results[0].passed is False
    assert "oh no" in results[0].messages[0]


# ---------------------------------------------------------------------------
# format_for_writer
# ---------------------------------------------------------------------------

def test_format_for_writer_includes_each_failed_gate() -> None:
    r = format_for_writer([
        GateResult("syntax", passed=False, messages=["line 1: bad"]),
        GateResult("lint", passed=True),
    ])
    assert "syntax" in r
    assert "line 1: bad" in r
    # Passing gates aren't echoed back
    assert "Gate: lint" not in r


def test_format_for_writer_empty_when_all_pass() -> None:
    assert format_for_writer([GateResult("syntax", passed=True)]) == ""


def test_format_for_writer_caps_long_message_lists() -> None:
    msgs = [f"issue {i}" for i in range(60)]
    out = format_for_writer(
        [GateResult("lint", passed=False, messages=msgs)],
        max_msgs_per_gate=10,
    )
    # 10 listed + a "… and N more" hint
    assert out.count("- issue") == 10
    assert "and 50 more" in out
