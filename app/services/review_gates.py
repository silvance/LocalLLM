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
import re
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
    # When True, the gate did NOT pass but the failure isn't the
    # writer's fault — typical case: a real third-party dependency
    # we recognize is missing from the local gate environment. The
    # orchestrator runs the reviewer anyway (with a note) and does
    # not bump writer_static_fails.
    blocked: bool = False
    # Taxonomy: free-form short string the orchestrator branches on.
    # Examples:
    #   - "likely_fake_import"          (writer hallucinated a module)
    #   - "missing_known_dependency"    (real dep, not installed locally)
    #   - "protocol_interface_mismatch" (BLE on Wi-Fi iface, etc.)
    failure_type: str = ""
    # Conventional values: info | warning | error | critical. Kept as
    # a free string so new gate categories do not require schema churn.
    severity: str = ""

    def __post_init__(self) -> None:
        if not self.severity:
            if self.passed:
                self.severity = "info"
            elif self.blocked:
                self.severity = "warning"
            else:
                self.severity = "error"

    @property
    def status(self) -> str:
        """New taxonomy while preserving the legacy ``passed`` bool."""
        if self.passed:
            return "pass"
        if self.blocked:
            return "blocked"
        return "fail"

    @property
    def is_terminal_failure(self) -> bool:
        """A failure that should bump writer counters and block the
        reviewer. False if the result is `blocked` (environment / not-
        the-writer's-fault)."""
        return not self.passed and not self.blocked

    def summary_line(self) -> str:
        if self.passed:
            return f"[ OK ] {self.name}"
        if self.blocked:
            return f"[BLOCKED] {self.name} ({len(self.messages)} issue{'s' if len(self.messages) != 1 else ''})"
        return f"[FAIL] {self.name} ({len(self.messages)} issue{'s' if len(self.messages) != 1 else ''})"


# --- Gates ----------------------------------------------------------------


def gate_candidate_count(text: str) -> GateResult:
    """Gate 0: the writer's response must contain exactly one explicit
    fenced ```python``` block.

    - 0 blocks → fail with a message that's specifically about the
      fence syntax, not the code itself ("did you forget the
      ```python tag?"). Helps the writer recover faster than a
      generic syntax-error spew.
    - >1 blocks → fail with "multiple_candidates". The writer often
      gives "here's option A, here's option B" or includes example
      blocks alongside the answer; downstream gates will get
      confused trying to lint a concatenation of two unrelated
      programs.
    - 1 block → pass. Subsequent gates see only the canonical
      candidate.

    Operates on the FULL writer response, not on extracted code, so
    the message can talk about fence syntax. Subsequent gates run on
    the extracted code via the standard run_gates pipeline.
    """
    from app.utils.code_linter import extract_python_blocks
    if _has_unclosed_python_fence(text or ""):
        return GateResult(
            "candidate_count",
            passed=False,
            messages=[
                "invalid_fence: an explicit ```python``` fence was opened "
                "but not closed. Resubmit exactly one complete fenced block.",
            ],
            failure_type="invalid_fence",
            severity="error",
        )
    blocks = extract_python_blocks(text or "")
    n = len(blocks)
    for block in blocks:
        marker = _transcript_marker(block)
        if marker:
            return GateResult(
                "candidate_count",
                passed=False,
                messages=[
                    f"duplicate_output: python block contains transcript / "
                    f"orchestrator text marker {marker!r}. Return only the "
                    f"candidate program, not prior gate feedback or review logs.",
                ],
                failure_type="duplicate_output",
                severity="error",
            )
    if n == 1:
        return GateResult("candidate_count", passed=True)
    if n == 0:
        return GateResult(
            "candidate_count",
            passed=False,
            messages=[
                "no_candidate: no explicit ```python``` fenced code block "
                "found in the response. Resubmit exactly one block "
                "beginning with ```python (lowercase, no language tag is "
                "rejected because untagged blocks were grabbing bash setup "
                "snippets and false-flagging them as Python).",
            ],
            failure_type="no_candidate",
            severity="error",
        )
    return GateResult(
        "candidate_count",
        passed=False,
        messages=[
            f"multiple_candidates: response contains {n} python code "
            f"blocks. Return exactly one complete program. Do not "
            f"include alternative versions, examples, setup commands, "
            f"or 'here's option A / option B' framing. Pick the version "
            f"you actually intend the user to run.",
        ],
        failure_type="multiple_candidates",
        severity="error",
    )


_TRANSCRIPT_MARKERS: tuple[str, ...] = (
    "Static Analysis:",
    "Gate:",
    "Your previous code did not pass",
    "Your previous code failed static-analysis gates",
    "Rewrite:",
    "Switch to",
)


def _transcript_marker(block: str) -> str:
    for marker in _TRANSCRIPT_MARKERS:
        if marker in block:
            return marker
    return ""


def _has_unclosed_python_fence(text: str) -> bool:
    in_python = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("```"):
            continue
        if in_python:
            in_python = False
            continue
        info = stripped[3:].strip().casefold()
        if info in {"python", "py"}:
            in_python = True
    return in_python


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
            failure_type="syntax_error",
            severity="error",
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
    return GateResult(
        "lint",
        passed=False,
        messages=msgs,
        failure_type="lint_error",
        severity="error",
    )


_STDLIB_MODULES: frozenset[str] = frozenset(
    getattr(sys, "stdlib_module_names", set())
)


# Known third-party packages we treat as REAL even when missing locally.
# When a module on this list isn't installed, the failure is an
# environment problem (writer chose a legitimate package, gate env
# doesn't have it), not a hallucination — the orchestrator marks the
# round as "blocked" rather than failing the writer.
#
# Keep this conservative — false-allowlisting a fake module would
# silently let hallucinated imports through. Each entry is a top-
# level module name (what `import x` or `from x import y` evaluates,
# i.e. the first dotted component).
KNOWN_THIRD_PARTY_MODULES: frozenset[str] = frozenset({
    # Networking / packet capture / RF — explicitly listed by the
    # user because the BLE/Wi-Fi sniffer task keeps tripping over them.
    "scapy",
    "pyshark",
    "bluetooth",       # PyBluez
    "bleak",
    "pybluez",
    "bluepy",
    "btlewrap",
    "pcapy",
    "pypcap",
    "dpkt",
    "netifaces",
    "psutil",
    # Common scientific stack — model often reaches for these.
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "sklearn",
    # HTTP / web frameworks.
    "requests",
    "httpx",
    "aiohttp",
    "fastapi",
    "starlette",
    "uvicorn",
    "flask",
    "pydantic",
    # Crypto / security forensics ecosystem.
    "cryptography",
    "nacl",
    "paramiko",
    "pyOpenSSL",
    "OpenSSL",
    # Misc heavy-hitters that show up in security/forensics code.
    "yaml",
    "lxml",
    "bs4",
    "PIL",
    "cv2",
    "serial",
})


def gate_imports(code: str) -> GateResult:
    """Gate 3: every module the code imports resolves on this Python.

    Static — uses ``importlib.util.find_spec`` on each top-level
    package referenced by ``import`` / ``from ... import``. No code
    is executed; module-level statements stay frozen.

    Outcomes:
      - All imports resolve → passed=True.
      - At least one missing import is NOT in the known-third-party
        allowlist → passed=False, failure_type="likely_fake_import".
        The writer is at fault: it invented a module name. Real
        missing-deps shown alongside, but the bucket is still
        "fake" because we have to assume something's hallucinated.
      - All missing imports are in the allowlist → passed=False,
        blocked=True, failure_type="missing_known_dependency". The
        writer picked legitimate packages; the gate env just doesn't
        have them. The orchestrator routes to the reviewer with a
        note instead of bouncing back to the writer.
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

    fake_imports: list[str] = []
    blocked_imports: list[str] = []
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
            # find_spec raised — treat top-level resolution as failed.
            # Bucket the same way as spec-is-None.
            if top in KNOWN_THIRD_PARTY_MODULES:
                blocked_imports.append(
                    f"{fullname!r} (top-level {top!r}) not installed in gate env "
                    f"({exc}); recognized as a real third-party package"
                )
            else:
                fake_imports.append(
                    f"{fullname!r} not importable ({exc}); top-level "
                    f"{top!r} is not on the known-dependency allowlist"
                )
            continue
        if spec is None:
            if top in KNOWN_THIRD_PARTY_MODULES:
                blocked_imports.append(
                    f"{fullname!r} (top-level {top!r}) not installed in gate env; "
                    f"recognized as a real third-party package — install with "
                    f"`pip install {top}` in the LocalLLM venv to enable smoke testing"
                )
            else:
                fake_imports.append(
                    f"{fullname!r} not found; top-level {top!r} is not on "
                    f"the known-dependency allowlist — likely a hallucinated "
                    f"or misspelled module name"
                )

    if not fake_imports and not blocked_imports:
        return GateResult("import", passed=True)

    if fake_imports:
        # Any fake import dominates: the writer made something up.
        # Surface blocked entries too so the writer sees the full
        # picture, but the verdict is "fix your fake imports".
        msgs = list(fake_imports)
        if blocked_imports:
            msgs.append("--- additionally, these real deps are missing locally ---")
            msgs.extend(blocked_imports)
        return GateResult(
            "import",
            passed=False,
            messages=msgs,
            failure_type="likely_fake_import",
            severity="error",
        )

    # Only allowlisted modules missing → environment problem.
    return GateResult(
        "import",
        passed=False,
        messages=blocked_imports,
        blocked=True,
        failure_type="missing_known_dependency",
        severity="warning",
    )


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
        # `errors="surrogatepass"` so lone surrogates (occasionally
        # emitted by quantized models) don't crash the gate before
        # subprocess gets a chance to surface a real error.
        try:
            path.write_text(code, encoding="utf-8", errors="surrogatepass")
        except (OSError, UnicodeError) as exc:
            return GateResult(
                "smoke", passed=False,
                messages=[f"could not write candidate to disk: {exc}"],
                failure_type="smoke_write_error",
                severity="error",
            )
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
                failure_type="smoke_timeout",
                severity="warning",
            )
        except OSError as exc:
            return GateResult(
                "smoke",
                passed=False,
                messages=[f"could not invoke python: {exc}"],
                failure_type="smoke_runtime_error",
                severity="error",
            )
        if proc.returncode == 0:
            return GateResult("smoke", passed=True)
        # Squash the traceback into a few lines so it doesn't
        # dominate the writer's next prompt.
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
        return GateResult(
            "smoke",
            passed=False,
            messages=tail or ["import failed"],
            failure_type="smoke_runtime_error",
            severity="error",
        )


# --- gate_no_placeholder_impl --------------------------------------------

# Patterns that strongly suggest the writer wrote a "looks-finished but
# doesn't actually do the thing" implementation. Each pattern carries
# its own short label so the failure message points at WHY it's
# suspicious, not just WHERE.
#
# Conservative on purpose — false positives waste reviewer time more
# than false negatives, and the reviewer's Class 3 prompt block is the
# secondary check.
_PLACEHOLDER_PATTERNS: tuple[tuple["re.Pattern[str]", str], ...] = (
    # Comments that announce the code is a stand-in. Each `would.* in
    # production`-style alternative is bounded to {0,40} non-newline
    # chars to avoid catastrophic backtracking on a malicious-looking
    # 500-char comment line.
    (re.compile(
        r"#.*\b("
        r"placeholder|stubbed|stub for|for demonstration|for now,? just|"
        r"todo:?\s*(?:implement|actually|real|hook up|wire)|"
        r"fixme:?\s*(?:implement|actually|real|hook up|wire)|"
        r"in (?:a )?real (?:impl|implementation|version)|"
        r"would[^\n]{0,40}? in (?:production|the real thing)|"
        r"simulated\b|simulating\b|simulate\b|"
        r"mocked\b|mocking\b|"
        r"demonstrative purposes?"
        r")\b",
        re.IGNORECASE,
    ), "comment claims fake / simulated / placeholder impl"),
    # Variables / functions whose NAME announces they're fake.
    # We don't try to allow `mock_test_*` / `fake_fixture_*` /
    # similar — too many corner cases for a static regex. If the
    # operator is reviewing test scaffolding, set
    # LOCALLLM_REVIEW_GATE_PLACEHOLDER_DISABLED=1 to skip this gate.
    (re.compile(
        r"\b(?:simulated|fake|mock|dummy|placeholder)_\w+",
        re.IGNORECASE,
    ), "identifier name suggests fake / simulated value"),
    # String literals that label themselves as fake output. Hits e.g.
    # `print("simulated BLE packet: 0x42")` — exactly the failure mode
    # ChatGPT flagged.
    (re.compile(
        r"['\"]\s*(?:simulated|fake|mock(?:ed)?|placeholder|dummy)\b"
        r"[^'\"]{0,80}['\"]",
        re.IGNORECASE,
    ), "string literal labels its content as fake / simulated"),
)


def _placeholder_disabled() -> bool:
    return os.getenv("LOCALLLM_REVIEW_GATE_PLACEHOLDER_DISABLED", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def gate_no_placeholder_impl(code: str) -> GateResult:
    """Static check for "looks complete, doesn't actually do it" code.

    Targets the failure mode where a writer's response parses cleanly,
    lints cleanly, imports cleanly — but the body is `time.sleep(...)`
    + `print("simulated BLE packet")`, or returns a hardcoded fake
    payload. The reviewer's domain prompt catches this too, but a
    static pre-check stops the loop from spending a reviewer round on
    obviously-fake code.

    Operator can opt out with ``LOCALLLM_REVIEW_GATE_PLACEHOLDER_DISABLED=1``
    for legitimate use of `mock_*` / `fake_*` (e.g. test scaffolding
    being reviewed)."""
    if _placeholder_disabled():
        return GateResult(
            "real_implementation", passed=True,
            messages=["(disabled via LOCALLLM_REVIEW_GATE_PLACEHOLDER_DISABLED)"],
        )
    findings: list[str] = []
    for line_no, line in enumerate(code.splitlines(), 1):
        # Cheap whitespace cap — a 5000-char line is almost certainly
        # generated noise, not a real placeholder.
        if len(line) > 500:
            continue
        for pattern, label in _PLACEHOLDER_PATTERNS:
            m = pattern.search(line)
            if m:
                snippet = line.strip()
                if len(snippet) > 100:
                    snippet = snippet[:97] + "..."
                findings.append(f"line {line_no}: {label} — {snippet!r}")
                break  # one finding per line is enough
    if not findings:
        return GateResult("real_implementation", passed=True)
    return GateResult(
        "real_implementation",
        passed=False,
        messages=findings[:15],
        failure_type="fake_implementation",
        severity="critical",
    )


# --- gate_protocol_interface_mismatch ------------------------------------


# BLE Advertising channel constant from Scapy's BTLE module — when the
# code references this, it's clearly building a BLE sniffer. Combine
# with a Wi-Fi-style scapy.sniff() on a non-BLE iface and we have a
# protocol/interface mismatch the reviewer keeps flagging.
_BTLE_INDICATORS: tuple[str, ...] = (
    "BTLE_ADV",
    "BTLE_DATA",
    "BTLE_RF",
    "from scapy.layers.bluetooth",
    "from scapy.layers.bluetooth4LE",
)


# Wi-Fi-style interface names the BLE branch should NEVER be sniffing
# on. Substring match against the inside of an iface-string literal —
# catches "wlan0", "wlan0mon", "mon0", "wlp3s0", "wlx0011…", and "wifi0"
# regardless of any monitor-mode suffix.
_WIFI_IFACE_RE = re.compile(
    r"^(?:wlan\d|mon\d|wlp\d|wlx[0-9a-f]|wifi\d)",
    re.IGNORECASE,
)


# scapy.sniff(iface=...) call — captures the iface argument string so we
# can decide whether the sniff target is Wi-Fi-coded or BLE-coded.
_SCAPY_SNIFF_RE = re.compile(
    r"""(?:scapy\.)?sniff\s*\(\s*[^)]*?iface\s*=\s*(['"][^'"]+['"]|[A-Za-z_]\w*)""",
    re.DOTALL,
)


# pyshark.LiveCapture(interface=...) where interface looks like an HCI
# device. PyShark talks to tshark/Wireshark, which DOES NOT support
# Bluetooth HCI capture out of the box on most platforms — surface this
# as an "unverified path" rather than an outright fail since some
# kernels do expose it via bluetooth-monitor.
_PYSHARK_HCI_RE = re.compile(
    r"""pyshark\.LiveCapture\s*\(\s*[^)]*?interface\s*=\s*['"](hci\d+)['"]""",
    re.DOTALL,
)


def gate_protocol_interface_mismatch(code: str) -> GateResult:
    """Static heuristic that catches two specific repeat-blocker
    patterns the reviewer keeps flagging on BLE/Wi-Fi sniffer attempts:

      1. ``BTLE_*`` / ``scapy.layers.bluetooth*`` referenced AND a
         scapy ``sniff(iface=...)`` call points at a Wi-Fi-coded iface
         (``wlan0``, ``mon0``, ...). Scapy can't pick BLE adverts off
         a Wi-Fi monitor interface — wrong protocol stack entirely.

      2. ``pyshark.LiveCapture(interface="hci0")`` (or any hciN). On
         most stock distros tshark doesn't have a working bluetooth
         capture path; the writer's almost certainly going to produce
         empty captures or a permissions error.

    Conservative on purpose. Doesn't try to validate iface names beyond
    obvious Wi-Fi-pattern strings, doesn't fire on `iface="hci0"` for
    scapy (that's a separate question), and doesn't try to introspect
    function call graphs — just lexical co-occurrence.
    """
    findings: list[str] = []

    has_btle = any(ind in code for ind in _BTLE_INDICATORS)
    if has_btle:
        for m in _SCAPY_SNIFF_RE.finditer(code):
            iface_arg = m.group(1)
            # Only catch literal-string ifaces; variables we can't
            # statically resolve and shouldn't guess.
            if not (iface_arg.startswith("'") or iface_arg.startswith('"')):
                continue
            iface_name = iface_arg.strip("'\"")
            if _WIFI_IFACE_RE.match(iface_name):
                line_no = code.count("\n", 0, m.start()) + 1
                findings.append(
                    f"line {line_no}: protocol_interface_mismatch — code "
                    f"references BLE layers (BTLE_ADV / scapy.layers.bluetooth*) "
                    f"but calls scapy.sniff(iface={iface_arg}) on a Wi-Fi-coded "
                    f"interface. BLE advertisements live on a different radio "
                    f"and don't appear in 802.11 capture; sniff BLE via "
                    f"`btmon` / `bluetoothctl --monitor` or BlueZ HCI sockets, "
                    f"and keep Wi-Fi capture in a separate function on its own "
                    f"monitor-mode interface."
                )

    for m in _PYSHARK_HCI_RE.finditer(code):
        line_no = code.count("\n", 0, m.start()) + 1
        hci = m.group(1)
        findings.append(
            f"line {line_no}: unverified_hci_capture_path — "
            f"pyshark.LiveCapture(interface={hci!r}) requires a tshark "
            f"build with the bluetooth-monitor capture extcap installed, "
            f"which is NOT present on stock Debian/Ubuntu/Raspbian. "
            f"Use BlueZ tooling directly (`btmon`, mgmt-API, HCI raw "
            f"socket via `socket.AF_BLUETOOTH`) instead — don't route BLE "
            f"capture through the pyshark/tshark pipeline."
        )

    if not findings:
        return GateResult("protocol_interface_mismatch", passed=True)
    return GateResult(
        "protocol_interface_mismatch",
        passed=False,
        messages=findings[:10],
        failure_type="protocol_interface_mismatch",
        severity="critical",
    )


# --- Runner ---------------------------------------------------------------


# Default chain. `gate_smoke` is included but no-ops at runtime when
# the env var isn't set, so the chain length doesn't change between
# environments.
DEFAULT_GATES: tuple[Callable[[str], GateResult], ...] = (
    gate_syntax,
    gate_lint,
    gate_imports,
    gate_protocol_interface_mismatch,
    gate_no_placeholder_impl,
    gate_smoke,
)


def run_gates(
    code: str,
    gates: Iterable[Callable[[str], GateResult]] | None = None,
) -> list[GateResult]:
    """Run gates in order. Short-circuit on the first TERMINAL failure
    (writer's fault); keep going past BLOCKED results so we still
    surface static issues like protocol_interface_mismatch alongside
    dependency-environment notes.

    The smoke gate is the one exception — it actually executes the
    candidate, so it's skipped once any blocked-import has been seen
    (running it would just fail trying to import the missing dep)."""
    out: list[GateResult] = []
    seen_blocked = False
    for g in (gates or DEFAULT_GATES):
        # Smoke gate would explode on blocked deps; treat as
        # implicitly-blocked and don't actually run it.
        if seen_blocked and getattr(g, "__name__", "") == "gate_smoke":
            out.append(GateResult(
                "smoke",
                passed=False,
                blocked=True,
                failure_type="missing_known_dependency",
                severity="warning",
                messages=["(skipped — depends on a known third-party package not installed locally)"],
            ))
            continue
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
        if r.blocked:
            seen_blocked = True
            continue
        if not r.passed:
            break
    return out


def all_passed(results: list[GateResult]) -> bool:
    return all(r.passed for r in results) if results else True


def format_for_writer(results: list[GateResult], max_msgs_per_gate: int = 25) -> str:
    """Format gate failures as a feedback string aimed at the writer
    model. Keeps the message count bounded so a noisy lint doesn't
    explode the prompt.

    Blocked results (e.g. missing_known_dependency) are NOT included
    here — the orchestrator handles those on a separate path so the
    writer isn't told to "fix" a legitimate dependency choice.
    """
    failed = [r for r in results if r.is_terminal_failure]
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


def format_blocked_for_reviewer(results: list[GateResult]) -> str:
    """Build a short note for the reviewer prompt explaining which
    real third-party deps the gate env doesn't have. The reviewer
    should treat them as available and review architecture/correctness
    rather than reject the code for "won't run locally"."""
    blocked = [r for r in results if r.blocked]
    if not blocked:
        return ""
    lines = [
        "GATE NOTE — the following imports could not be resolved in the "
        "local gate environment but are recognized as real third-party "
        "packages. Treat them as installed when reviewing; do NOT flag "
        "them as fake or unavailable. Focus your review on architecture, "
        "correctness, hardware/protocol grounding, and whether the code "
        "would actually achieve the user's goal once the deps are present."
    ]
    for r in blocked:
        for msg in r.messages[:25]:
            lines.append(f"- {msg}")
    return "\n".join(lines)


def all_blocked_only(results: list[GateResult]) -> bool:
    """True if at least one gate is blocked AND no gate is a terminal
    failure. Used by the orchestrator to decide whether to take the
    dependency-blocked path (run reviewer with a note, don't bump
    writer counters)."""
    if not results:
        return False
    has_blocked = any(r.blocked for r in results)
    has_terminal = any(r.is_terminal_failure for r in results)
    return has_blocked and not has_terminal
