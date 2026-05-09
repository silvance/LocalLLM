"""Tests for app/services/review_gates.py — the static-analysis gates
that fence the reviewer model from broken code."""
from __future__ import annotations

import pytest

from app.services.review_gates import (
    DEFAULT_GATES,
    KNOWN_THIRD_PARTY_MODULES,
    GateResult,
    all_blocked_only,
    all_passed,
    format_blocked_for_reviewer,
    format_for_writer,
    gate_candidate_count,
    gate_imports,
    gate_lint,
    gate_protocol_interface_mismatch,
    gate_runtime_anti_patterns,
    gate_smoke,
    gate_syntax,
    run_gates,
)


# ---------------------------------------------------------------------------
# gate_candidate_count (Gate 0)
# ---------------------------------------------------------------------------

def test_candidate_count_passes_on_single_python_block() -> None:
    text = (
        "Here you go:\n\n"
        "```python\n"
        "def add(a, b):\n    return a + b\n"
        "```\n"
    )
    r = gate_candidate_count(text)
    assert r.passed is True


def test_candidate_count_fails_when_no_python_block() -> None:
    text = (
        "I think the answer is to use scapy. Run:\n\n"
        "```bash\n"
        "pip install scapy\n"
        "```\n"
    )
    r = gate_candidate_count(text)
    assert r.passed is False
    assert "no_candidate" in r.messages[0]
    # Message must talk about fence syntax, not 'syntax error'.
    assert "```python" in r.messages[0]


def test_candidate_count_fails_on_multiple_python_blocks() -> None:
    text = (
        "Option A:\n```python\ndef a(): pass\n```\n\n"
        "Option B:\n```python\ndef b(): pass\n```\n"
    )
    r = gate_candidate_count(text)
    assert r.passed is False
    assert "multiple_candidates" in r.messages[0]
    assert "2 python code blocks" in r.messages[0]


def test_candidate_count_handles_empty_input() -> None:
    r = gate_candidate_count("")
    assert r.passed is False
    assert "no_candidate" in r.messages[0]


# ---------------------------------------------------------------------------
# gate_no_placeholder_impl
# ---------------------------------------------------------------------------

def test_placeholder_gate_passes_on_real_code() -> None:
    code = (
        "import os\n"
        "def cwd():\n"
        "    return os.getcwd()\n"
    )
    from app.services.review_gates import gate_no_placeholder_impl
    r = gate_no_placeholder_impl(code)
    assert r.passed is True


def test_placeholder_gate_flags_fake_print() -> None:
    """The exact failure mode ChatGPT highlighted — a `print` of a
    string that announces it's simulated."""
    code = (
        "def sniff_ble():\n"
        "    print('simulated BLE packet: 0x42')\n"
    )
    from app.services.review_gates import gate_no_placeholder_impl
    r = gate_no_placeholder_impl(code)
    assert r.passed is False
    assert any("string literal" in m.lower() or "fake" in m.lower() for m in r.messages)


def test_placeholder_gate_flags_todo_implement_comment() -> None:
    code = (
        "def fetch_data():\n"
        "    # TODO: implement actual fetch\n"
        "    return None\n"
    )
    from app.services.review_gates import gate_no_placeholder_impl
    r = gate_no_placeholder_impl(code)
    assert r.passed is False


def test_placeholder_gate_flags_simulated_variable_name() -> None:
    code = (
        "def scan():\n"
        "    simulated_packet = {'rssi': -50}\n"
        "    return simulated_packet\n"
    )
    from app.services.review_gates import gate_no_placeholder_impl
    r = gate_no_placeholder_impl(code)
    assert r.passed is False
    assert any("identifier" in m.lower() for m in r.messages)


def test_placeholder_gate_disabled_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Operator escape hatch — set the env var when reviewing test
    scaffolding (`mock_*` / `fake_*` are legitimate there)."""
    monkeypatch.setenv("LOCALLLM_REVIEW_GATE_PLACEHOLDER_DISABLED", "1")
    code = "def x():\n    print('simulated thing')\n"
    from app.services.review_gates import gate_no_placeholder_impl
    r = gate_no_placeholder_impl(code)
    assert r.passed is True
    assert "disabled" in r.messages[0].lower()


def test_placeholder_gate_in_default_chain() -> None:
    """Regression: gate_no_placeholder_impl must be in DEFAULT_GATES so
    it actually runs without the orchestrator opting in. Sits BEFORE
    gate_smoke because static checks are cheaper than subprocess runs."""
    from app.services.review_gates import (
        DEFAULT_GATES,
        gate_no_placeholder_impl,
        gate_smoke,
    )
    assert gate_no_placeholder_impl in DEFAULT_GATES
    assert DEFAULT_GATES.index(gate_no_placeholder_impl) < DEFAULT_GATES.index(gate_smoke)


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
# gate_imports — known-dep allowlist + taxonomy
# ---------------------------------------------------------------------------

def test_gate_imports_allowlist_lists_real_security_packages() -> None:
    """Sanity-check that the allowlist contains the packages the
    BLE/Wi-Fi sniffer task actually needs — if these regress, the
    writer will start getting bounced for legitimate dependency
    choices."""
    for pkg in ("scapy", "pyshark", "bluetooth", "bleak", "numpy",
                "requests", "fastapi", "pydantic", "cryptography"):
        assert pkg in KNOWN_THIRD_PARTY_MODULES


def test_gate_imports_marks_fake_module_as_likely_fake_import() -> None:
    code = "import made_up_xyz_module_qqq\n"
    r = gate_imports(code)
    assert r.passed is False
    assert r.blocked is False, "fake module is the writer's fault, not blocked"
    assert r.failure_type == "likely_fake_import"
    assert r.is_terminal_failure is True


def test_gate_imports_marks_known_dep_as_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a real third-party package on the allowlist isn't installed
    locally, the gate must mark it BLOCKED rather than failing the
    writer for a hallucination they didn't commit."""
    import importlib.util
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *a, **kw):
        if name == "scapy":
            return None
        return real_find_spec(name, *a, **kw)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    r = gate_imports("import scapy\n")
    assert r.passed is False
    assert r.blocked is True
    assert r.failure_type == "missing_known_dependency"
    assert r.is_terminal_failure is False, "blocked must NOT count as terminal"


def test_gate_imports_mixed_fake_and_blocked_buckets_as_fake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a candidate has BOTH a fake import and a missing-but-real
    dep, the result is `likely_fake_import` (writer's fault dominates).
    The blocked entries are surfaced in the message list too so the
    writer sees the full picture."""
    import importlib.util
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *a, **kw):
        if name in ("scapy", "totally_made_up_qzx"):
            return None
        return real_find_spec(name, *a, **kw)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    code = "import scapy\nimport totally_made_up_qzx\n"
    r = gate_imports(code)
    assert r.passed is False
    assert r.blocked is False
    assert r.failure_type == "likely_fake_import"
    joined = " ".join(r.messages)
    assert "totally_made_up_qzx" in joined
    assert "scapy" in joined  # surfaced too, in the trailing block


def test_run_gates_continues_past_blocked_to_protocol_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gate_imports BLOCKED must NOT short-circuit the chain — the
    static gates after it (protocol_interface_mismatch,
    no_placeholder_impl) still need to run so the reviewer hears
    about all of them at once."""
    import importlib.util
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *a, **kw):
        if name == "scapy":
            return None
        return real_find_spec(name, *a, **kw)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    code = (
        "from scapy.layers.bluetooth4LE import BTLE_ADV\n"
        "from scapy.all import sniff\n"
        "def capture_ble():\n"
        "    return sniff(iface='wlan0mon', count=10), BTLE_ADV\n"
    )
    results = run_gates(code)
    names = [r.name for r in results]
    # Both blocked-import AND protocol_interface_mismatch must appear.
    assert "import" in names
    assert "protocol_interface_mismatch" in names
    import_r = next(r for r in results if r.name == "import")
    mismatch_r = next(r for r in results if r.name == "protocol_interface_mismatch")
    assert import_r.blocked is True
    assert mismatch_r.is_terminal_failure is True


def test_run_gates_skips_smoke_when_known_dep_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Smoke gate would just fail trying to import the missing dep —
    skip it (as blocked) rather than reporting a noisy ImportError."""
    monkeypatch.setenv("LOCALLLM_REVIEW_GATE_SMOKE_ENABLED", "1")
    import importlib.util
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *a, **kw):
        if name == "scapy":
            return None
        return real_find_spec(name, *a, **kw)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    code = "import scapy\nprint(scapy)\n"
    results = run_gates(code)
    smoke_r = next(r for r in results if r.name == "smoke")
    assert smoke_r.blocked is True
    assert smoke_r.passed is False
    assert "skipped" in smoke_r.messages[0].lower()


def test_all_blocked_only_distinguishes_blocked_from_terminal() -> None:
    blocked_only = [
        GateResult("syntax", passed=True),
        GateResult("import", passed=False, blocked=True,
                   failure_type="missing_known_dependency"),
    ]
    assert all_blocked_only(blocked_only) is True

    has_terminal = [
        GateResult("syntax", passed=True),
        GateResult("import", passed=False, blocked=True,
                   failure_type="missing_known_dependency"),
        GateResult("protocol_interface_mismatch", passed=False,
                   failure_type="protocol_interface_mismatch"),
    ]
    assert all_blocked_only(has_terminal) is False

    all_pass = [GateResult("syntax", passed=True), GateResult("lint", passed=True)]
    assert all_blocked_only(all_pass) is False


def test_format_blocked_for_reviewer_includes_known_deps() -> None:
    blocked = [
        GateResult(
            "import", passed=False, blocked=True,
            failure_type="missing_known_dependency",
            messages=["'scapy' (top-level 'scapy') not installed in gate env"],
        ),
    ]
    note = format_blocked_for_reviewer(blocked)
    assert "GATE NOTE" in note
    assert "scapy" in note
    assert "treat them as installed" in note.lower()


def test_format_for_writer_omits_blocked_results() -> None:
    """Writer feedback must not include blocked entries — telling the
    writer to "fix" a legitimate dep choice is the bug we're solving."""
    results = [
        GateResult("import", passed=False, blocked=True,
                   failure_type="missing_known_dependency",
                   messages=["'scapy' not installed locally"]),
    ]
    out = format_for_writer(results)
    assert out == "", "blocked-only results should produce no writer feedback"


# ---------------------------------------------------------------------------
# gate_protocol_interface_mismatch
# ---------------------------------------------------------------------------

def test_protocol_mismatch_passes_on_clean_wifi_only_code() -> None:
    code = (
        "from scapy.all import sniff\n"
        "def cap_wifi():\n"
        "    return sniff(iface='wlan0mon', count=5)\n"
    )
    r = gate_protocol_interface_mismatch(code)
    assert r.passed is True


def test_protocol_mismatch_passes_on_clean_ble_only_code() -> None:
    code = (
        "from scapy.layers.bluetooth4LE import BTLE_ADV\n"
        "import subprocess\n"
        "def cap_ble():\n"
        "    return subprocess.run(['btmon'], capture_output=True), BTLE_ADV\n"
    )
    r = gate_protocol_interface_mismatch(code)
    assert r.passed is True


def test_protocol_mismatch_flags_btle_with_wlan0mon() -> None:
    code = (
        "from scapy.layers.bluetooth4LE import BTLE_ADV\n"
        "from scapy.all import sniff\n"
        "def capture_ble():\n"
        "    return sniff(iface='wlan0mon', count=10), BTLE_ADV\n"
    )
    r = gate_protocol_interface_mismatch(code)
    assert r.passed is False
    assert r.failure_type == "protocol_interface_mismatch"
    assert any("protocol_interface_mismatch" in m for m in r.messages)
    assert any("wlan0mon" in m for m in r.messages)


def test_protocol_mismatch_flags_btle_with_mon0() -> None:
    code = (
        "from scapy.layers.bluetooth import BTLE_ADV\n"
        "from scapy.all import sniff\n"
        "sniff(iface=\"mon0\", count=1)\n"
        "print(BTLE_ADV)\n"
    )
    r = gate_protocol_interface_mismatch(code)
    assert r.passed is False
    assert any("mon0" in m for m in r.messages)


def test_protocol_mismatch_flags_pyshark_hci_capture() -> None:
    code = (
        "import pyshark\n"
        "cap = pyshark.LiveCapture(interface='hci0')\n"
        "for pkt in cap.sniff_continuously():\n"
        "    print(pkt)\n"
    )
    r = gate_protocol_interface_mismatch(code)
    assert r.passed is False
    assert any("unverified_hci_capture_path" in m for m in r.messages)
    assert any("hci0" in m for m in r.messages)


def test_protocol_mismatch_does_not_flag_scapy_hci() -> None:
    """Scapy on hci0 isn't *valid* either, but it's a different bucket
    — this gate is specifically about the BLE-on-Wi-Fi-iface and
    pyshark-on-HCI failure modes the reviewer keeps flagging."""
    code = (
        "from scapy.layers.bluetooth4LE import BTLE_ADV\n"
        "from scapy.all import sniff\n"
        "sniff(iface='hci0', count=1)\n"
        "print(BTLE_ADV)\n"
    )
    r = gate_protocol_interface_mismatch(code)
    assert r.passed is True


def test_protocol_mismatch_in_default_chain() -> None:
    assert gate_protocol_interface_mismatch in DEFAULT_GATES
    # Must run AFTER imports (the BLE indicator strings come from
    # imports) and BEFORE smoke (so the static signal is captured even
    # if smoke is disabled).
    assert (
        DEFAULT_GATES.index(gate_protocol_interface_mismatch)
        > DEFAULT_GATES.index(gate_imports)
    )
    assert (
        DEFAULT_GATES.index(gate_protocol_interface_mismatch)
        < DEFAULT_GATES.index(gate_smoke)
    )


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


def test_gate_smoke_subprocess_env_is_scrubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The smoke subprocess must NOT inherit operator credentials —
    OLLAMA_HOST, AWS keys, GitHub tokens, LOCALLLM_*. Defense in depth
    against an LLM that sneaks `os.environ` exfiltration into a module-
    level call."""
    monkeypatch.setenv("LOCALLLM_REVIEW_GATE_SMOKE_ENABLED", "1")
    monkeypatch.setenv("OLLAMA_HOST", "http://leak.example:9999")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "do-not-leak")
    monkeypatch.setenv("GH_TOKEN", "ghp_dont_leak")
    monkeypatch.setenv("LOCALLLM_DATA_DIR", "/tmp/sensitive")

    # Have the subprocess dump the env vars it sees. If env scrub
    # works, none of the secrets above will appear.
    code = (
        "import os, json, sys\n"
        "sys.stderr.write(json.dumps(dict(os.environ)))\n"
    )
    r = gate_smoke(code, timeout=10.0)
    assert r.passed is True or r.passed is False  # we just care about the env, not pass/fail
    captured = " ".join(r.messages)
    assert "leak.example" not in captured, "OLLAMA_HOST leaked into smoke subprocess"
    assert "do-not-leak" not in captured, "AWS_SECRET_ACCESS_KEY leaked"
    assert "ghp_dont_leak" not in captured, "GH_TOKEN leaked"
    assert "/tmp/sensitive" not in captured, "LOCALLLM_DATA_DIR leaked"


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


# ---------------------------------------------------------------------------
# gate_runtime_anti_patterns
# ---------------------------------------------------------------------------

def test_runtime_anti_patterns_flags_while_true_pass() -> None:
    """The headline anti-pattern from the user's BLE/Wi-Fi sniffer
    screenshot — `while True: pass` to "hold the program open" while
    background captures run. Burns 100% CPU forever; right answer is
    signal.pause() / time.sleep() / Event.wait()."""
    code = (
        "def main():\n"
        "    print('starting')\n"
        "    while True:\n"
        "        pass\n"
    )
    r = gate_runtime_anti_patterns(code)
    assert not r.passed
    assert r.failure_type == "runtime_anti_pattern"
    joined = " ".join(r.messages).lower()
    assert "busy-loop" in joined or "burns 100% cpu" in joined
    # The fix-suggestion must be in the message — the writer needs
    # to know what to do, not just what was wrong.
    assert "signal.pause" in joined or "time.sleep" in joined


def test_runtime_anti_patterns_flags_while_true_continue() -> None:
    """Same anti-pattern, slightly different shape — `while True:
    continue` is also a 100%-CPU spin."""
    code = "while True:\n    continue\n"
    r = gate_runtime_anti_patterns(code)
    assert not r.passed


def test_runtime_anti_patterns_flags_while_one_pass() -> None:
    """`while 1: pass` — older spelling, same bug. The truthy-literal
    check must accept ints, not just `True`."""
    code = "while 1:\n    pass\n"
    r = gate_runtime_anti_patterns(code)
    assert not r.passed


def test_runtime_anti_patterns_accepts_real_event_loops() -> None:
    """`while True: <real work>` is fine — that's how event loops
    work. The gate must NOT false-positive on legitimate patterns
    like the agent's main loop. Body must have something other than
    pass/continue."""
    code = (
        "import time\n"
        "while True:\n"
        "    time.sleep(0.1)\n"
        "    do_work()\n"
    )
    r = gate_runtime_anti_patterns(code)
    assert r.passed


def test_runtime_anti_patterns_accepts_signal_pause() -> None:
    """The right pattern that the gate's message recommends — must
    not fire on its own remediation."""
    code = (
        "import signal\n"
        "def hang():\n"
        "    signal.pause()\n"
    )
    r = gate_runtime_anti_patterns(code)
    assert r.passed


def test_runtime_anti_patterns_flags_conditional_import_referenced_module_wide() -> None:
    """Exact failure from the user's BLE/Wi-Fi sniffer screenshot:
    `import shutil` lives inside `if __name__ == "__main__":` but
    `shutil.which` is called from `check_wifi_prerequisites`. Works
    only when run as `python script.py`; raises NameError as soon as
    the module is imported. Reviewer missed this; the static gate
    has to catch it."""
    code = (
        "def check_wifi_prerequisites():\n"
        "    if not shutil.which('iw'):\n"
        "        return False\n"
        "    return True\n"
        "\n"
        "def main():\n"
        "    check_wifi_prerequisites()\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    import shutil\n"
        "    main()\n"
    )
    r = gate_runtime_anti_patterns(code)
    assert not r.passed
    joined = " ".join(r.messages)
    assert "shutil" in joined
    assert "__main__" in joined
    # Fix-suggestion must be present so the writer knows what to do.
    assert "top of the file" in joined.lower() or "module top" in joined.lower()


def test_runtime_anti_patterns_accepts_main_only_imports_used_only_in_main() -> None:
    """Imports inside `if __name__ == "__main__":` ARE fine when the
    name is only used inside the same block — that's actually a
    reasonable pattern (defer import cost when the file is imported
    as a library). Don't false-positive."""
    code = (
        "def main():\n"
        "    return 42\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    import argparse\n"
        "    parser = argparse.ArgumentParser()\n"
        "    main()\n"
    )
    r = gate_runtime_anti_patterns(code)
    assert r.passed


def test_runtime_anti_patterns_accepts_top_level_import_used_in_function() -> None:
    """Standard pattern — top-level import used by a function. Must
    NOT fire (the gate is specifically about MAIN-block imports)."""
    code = (
        "import shutil\n"
        "\n"
        "def check():\n"
        "    return shutil.which('ls')\n"
    )
    r = gate_runtime_anti_patterns(code)
    assert r.passed


def test_runtime_anti_patterns_handles_attribute_access_chains() -> None:
    """`subprocess.Popen.terminate` — the root Name is `subprocess`.
    The gate's reference-walker must dig through Attribute chains to
    find that root, otherwise it misses cases where the imported
    module is used via `.foo.bar` access."""
    code = (
        "def run():\n"
        "    p = subprocess.Popen(['ls'])\n"
        "    return p\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    import subprocess\n"
        "    run()\n"
    )
    r = gate_runtime_anti_patterns(code)
    assert not r.passed
    assert "subprocess" in " ".join(r.messages)


def test_runtime_anti_patterns_full_screenshot_regression() -> None:
    """End-to-end on the EXACT code from the user's screenshot —
    must catch BOTH the `while True: pass` busy-loop AND the
    `import shutil` inside `if __name__ == "__main__":` while
    `shutil.which` is referenced from check_wifi_prerequisites /
    check_ble_prerequisites. Two findings expected."""
    code = '''
import subprocess

def check_wifi_prerequisites():
    required_tools = ['iw', 'tshark', 'tcpdump']
    for tool in required_tools:
        if not shutil.which(tool):
            return False
    return True

def check_ble_prerequisites():
    required_tools = ['bluetoothctl', 'btmgmt', 'btmon']
    for tool in required_tools:
        if not shutil.which(tool):
            return False
    return True

def main():
    if not check_wifi_prerequisites():
        return
    if not check_ble_prerequisites():
        return
    print("Captures running... Press Ctrl+C to stop.")
    while True:
        pass

if __name__ == "__main__":
    import shutil
    main()
'''
    r = gate_runtime_anti_patterns(code)
    assert not r.passed
    joined = " ".join(r.messages).lower()
    # Both bugs surfaced.
    assert "busy-loop" in joined or "100% cpu" in joined
    assert "shutil" in joined
    assert "__main__" in joined


def test_runtime_anti_patterns_no_main_guard_does_not_false_positive() -> None:
    """Modules without an `if __name__ == "__main__":` block — most
    library files — must pass the conditional-import check trivially.
    Defensive: the gate's traversal must not crash when the guard
    isn't there."""
    code = (
        "import shutil\n"
        "\n"
        "def check():\n"
        "    return shutil.which('ls')\n"
    )
    r = gate_runtime_anti_patterns(code)
    assert r.passed


def test_runtime_anti_patterns_passes_clean_code() -> None:
    """Clean implementation — neither anti-pattern present. Pass."""
    code = (
        "import time\n"
        "\n"
        "def hang():\n"
        "    while not done():\n"
        "        time.sleep(0.5)\n"
        "\n"
        "def done():\n"
        "    return False\n"
    )
    r = gate_runtime_anti_patterns(code)
    assert r.passed


def test_runtime_anti_patterns_short_circuits_on_syntax_error() -> None:
    """If the code doesn't parse, gate_syntax already failed and
    short-circuited the chain. If we somehow get called anyway,
    don't double-report — pass quietly."""
    r = gate_runtime_anti_patterns("def )( bad syntax")
    assert r.passed


def test_runtime_anti_patterns_in_default_chain() -> None:
    """Wiring smoke check — gate must be in DEFAULT_GATES so it runs
    on every review without explicit opt-in."""
    assert gate_runtime_anti_patterns in DEFAULT_GATES


def test_runtime_anti_patterns_runs_after_no_placeholder_impl() -> None:
    """Order matters — runtime-anti-patterns is a quality check that
    only makes sense on code that already passed syntax/lint/imports.
    Must come after gate_no_placeholder_impl (which catches obviously
    fake scaffolding) and before gate_smoke (which actually executes
    the code — no point trying to import code that has busy-loops)."""
    from app.services.review_gates import gate_no_placeholder_impl
    idx_pre = DEFAULT_GATES.index(gate_no_placeholder_impl)
    idx_anti = DEFAULT_GATES.index(gate_runtime_anti_patterns)
    idx_smoke = DEFAULT_GATES.index(gate_smoke)
    assert idx_pre < idx_anti < idx_smoke
