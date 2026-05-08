"""Static-analysis gates for the writer ↔ reviewer review loop.

Each gate is a pure function ``(code: str) -> GateResult``. The runner
runs them in order, short-circuits on the first failure, and the
review thread uses ``all_passed()`` to decide whether to call the
reviewer model or send only the gate errors back to the writer.

Gates ordered by cost / risk:

  1. syntax — ast.parse(). Pure parsing, no execution.
  2. lint   — pyflakes via app.utils.code_linter. Static analysis.
  3. import — importlib.util.find_spec() on each module the code
              imports. Resolves package availability without running
              any model-generated code.
  4. smoke  — actually executes ``import code`` in a fresh subprocess
              with a short timeout. OPT-IN: model-generated code could
              do anything at import time (it's still arbitrary code
              execution), so the operator has to set
              ``LOCALLLM_REVIEW_GATE_SMOKE_ENABLED=1`` to enable it.

Gate 5 (the domain reviewer) and gate 6 (the writer's rebuild) are
the existing reviewer / writer loop in app.web.app — this module is
just gates 1-4.
"""
from __future__ import annotations

import ast
import importlib.util
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable


logger = logging.getLogger("localllm")


@dataclass
class GateResult:
    name: str
    passed: bool
    messages: list[str] = field(default_factory=list)

    def summary_line(self) -> str:
        if self.passed:
            return f"[OK] {self.name}"
        return f"[FAIL] {self.name} ({len(self.messages)} issue{'s' if len(self.messages) != 1 else ''})"


# --- Gates ----------------------------------------------------------------


def gate_syntax(code: str) -> GateResult:
    """Gate 1: code must parse as Python. Caps cascading downstream
    noise — there's no point running pyflakes if the file isn't even
    a valid AST."""
    try:
        ast.parse(code)
    except SyntaxError as exc:
        line = f"line {exc.lineno}" if exc.lineno else "?"
        col = f", col {exc.offset}" if exc.offset else ""
        return GateResult(
            "syntax",
            passed=False,
            messages=[f"{line}{col}: {exc.msg}"],
        )
    return GateResult("syntax", passed=True)


def gate_lint(code: str) -> GateResult:
    """Gate 2: pyflakes — undefined names, unused imports, redefined
    symbols, etc. We reuse code_linter.lint_code so the rules match
    the live in-chat lint hint."""
    from app.utils.code_linter import lint_code
    findings = lint_code(code)
    if not findings:
        return GateResult("lint", passed=True)
    msgs = [f"line {f.line}: {f.message}" for f in findings]
    return GateResult("lint", passed=False, messages=msgs)


_STDLIB_MODULES: frozenset[str] = frozenset(
    getattr(sys, "stdlib_module_names", set())
)


def gate_imports(code: str) -> GateResult:
    """Gate 3: every module the code imports resolves on this Python.

    Static — uses ``importlib.util.find_spec`` on each top-level
    package referenced by ``import`` / ``from ... import``. No code
    is executed; module-level statements stay frozen. Catches the
    common LLM hallucination of inventing plausible-but-fake module
    names (``import scapy_ng``, ``from cryptography.hazmat.primitives.ciphers
    import Salsa42``).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # gate_syntax already reported this — don't double-emit.
        return GateResult("import", passed=True)

    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import
                continue
            if node.module:
                imports.append(node.module)

    failures: list[str] = []
    seen: set[str] = set()
    for fullname in imports:
        top = fullname.split(".", 1)[0]
        if top in seen:
            continue
        seen.add(top)
        if top in _STDLIB_MODULES:
            continue
        try:
            spec = importlib.util.find_spec(top)
        except (ImportError, ValueError) as exc:
            failures.append(f"{fullname!r} not importable ({exc})")
            continue
        if spec is None:
            failures.append(f"{fullname!r} not found")

    return GateResult("import", passed=not failures, messages=failures)


def _smoke_enabled() -> bool:
    """The smoke gate is opt-in because it executes model-generated
    code. Default: off."""
    return os.getenv("LOCALLLM_REVIEW_GATE_SMOKE_ENABLED", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


# Env vars the smoke subprocess is allowed to see. Allowlist (not
# denylist) so a new credential variable in the parent shell can't
# leak in just because we forgot to add it here. Includes only what's
# actually needed for Python to start and import — everything else
# (OLLAMA_HOST, AWS_*, GH_TOKEN, LOCALLLM_*, HTTP_PROXY, …) is dropped.
_SMOKE_ENV_ALLOWLIST: frozenset[str] = frozenset({
    "PATH",          # subprocess on Windows needs it; sys.executable is absolute on POSIX but several stdlib bits expect PATH
    "HOME",          # POSIX libs sometimes blow up without it
    "TMPDIR", "TEMP", "TMP",
    "LANG", "LC_ALL", "LC_CTYPE",
    "PYTHONIOENCODING",
    "SYSTEMROOT", "WINDIR", "USERPROFILE",  # Windows: subprocess can fail to start without these
    "COMSPEC",
    "PATHEXT",
})


def _safe_env() -> dict[str, str]:
    """Filtered process env for the smoke subprocess. Allowlist-based
    so adding a new env var in the parent never silently leaks in."""
    src = os.environ
    return {k: src[k] for k in _SMOKE_ENV_ALLOWLIST if k in src}


def gate_smoke(code: str, *, timeout: float = 5.0) -> GateResult:
    """Gate 4: ``import`` the code in a clean subprocess to surface
    runtime errors that static analysis can't (e.g. ``raise
    NotImplementedError`` at module level, side effects).

    Skipped unless ``LOCALLLM_REVIEW_GATE_SMOKE_ENABLED=1`` because
    arbitrary model code shouldn't execute by default.
    """
    if not _smoke_enabled():
        return GateResult("smoke", passed=True, messages=["(disabled — set LOCALLLM_REVIEW_GATE_SMOKE_ENABLED=1 to opt in)"])

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "review_smoke.py"
        path.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-c", "import review_smoke"],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout,
                # Scrubbed env — drops OLLAMA_HOST, AWS_*, GH_TOKEN,
                # LOCALLLM_*, HTTP_PROXY, etc. so model-generated import-
                # time code can't exfiltrate creds or call out to the
                # operator's configured backends.
                env=_safe_env(),
            )
        except subprocess.TimeoutExpired:
            return GateResult(
                "smoke",
                passed=False,
                messages=[f"import timed out after {timeout}s"],
            )
        except OSError as exc:
            return GateResult(
                "smoke",
                passed=False,
                messages=[f"could not invoke python: {exc}"],
            )
        if proc.returncode == 0:
            return GateResult("smoke", passed=True)
        # Squash the traceback into a few lines so it doesn't
        # dominate the writer's next prompt.
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
        return GateResult("smoke", passed=False, messages=tail or ["import failed"])


# --- Runner ---------------------------------------------------------------


# Default chain. `gate_smoke` is included but no-ops at runtime when
# the env var isn't set, so the chain length doesn't change between
# environments.
DEFAULT_GATES: tuple[Callable[[str], GateResult], ...] = (
    gate_syntax,
    gate_lint,
    gate_imports,
    gate_smoke,
)


def run_gates(
    code: str,
    gates: Iterable[Callable[[str], GateResult]] | None = None,
) -> list[GateResult]:
    """Run gates in order, stopping at the first failure. Returns the
    completed-or-attempted results so the caller can see exactly which
    gate the code died on."""
    out: list[GateResult] = []
    for g in (gates or DEFAULT_GATES):
        try:
            r = g(code)
        except Exception as exc:  # noqa: BLE001 — gates shouldn't crash, but be defensive
            logger.exception("Gate %s raised", getattr(g, "__name__", str(g)))
            r = GateResult(
                getattr(g, "__name__", "gate").removeprefix("gate_"),
                passed=False,
                messages=[f"gate crashed: {exc}"],
            )
        out.append(r)
        if not r.passed:
            break
    return out


def all_passed(results: list[GateResult]) -> bool:
    return all(r.passed for r in results) if results else True


def format_for_writer(results: list[GateResult], max_msgs_per_gate: int = 25) -> str:
    """Format gate failures as a feedback string aimed at the writer
    model. Keeps the message count bounded so a noisy lint doesn't
    explode the prompt."""
    failed = [r for r in results if not r.passed]
    if not failed:
        return ""
    lines = [
        "Your previous code did not pass the static-analysis gates. "
        "Fix every issue listed below and resubmit the full corrected "
        "code in fenced markdown blocks. Do NOT add prose unrelated "
        "to fixing these issues."
    ]
    for r in failed:
        lines.append("")
        lines.append(f"### Gate: {r.name}")
        for msg in r.messages[:max_msgs_per_gate]:
            lines.append(f"- {msg}")
        if len(r.messages) > max_msgs_per_gate:
            lines.append(f"- … and {len(r.messages) - max_msgs_per_gate} more")
    return "\n".join(lines)
