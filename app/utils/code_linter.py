"""Live linting for streaming writer output on the review page.

Watches the Python code blocks the writer model emits and runs `ast.parse`
+ `pyflakes` on the longest parseable prefix. Findings stream to the
frontend as `lint_state` events keyed by section index; the frontend
displays them in a panel adjacent to each writer section.

What this catches:
    - Undefined names (F821) — your `self.ModuleOption` / forgotten import bug
    - Unused imports (F401)
    - Import shadowing / redefinition
    - Syntax errors (we surface these as severity=error)

What this does NOT catch (semantic-level — would need mypy/pyright):
    - `field()` outside `@dataclass` if `field` is imported
    - Calling a class attribute as a method
    - Type mismatches
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from pyflakes import checker as _flakes_checker


# Python code fenced blocks. Must be `python` or `py` — we used to also
# accept an empty language tag for models that forgot the hint after
# the first block, but that turned out to false-match the closing
# fence of one bash block + prose + opening fence of the next as a
# "Python block." Rather than accept that risk, require explicit
# python/py and anchor to start-of-line so accidental triple-backticks
# inside code (rare) can't confuse us.
_FENCE_PY = re.compile(
    r"^```(?:python|py)\s*\r?\n([\s\S]*?)^```\s*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class LintFinding:
    line: int
    col: int
    severity: str  # "error" | "warning"
    message: str


def extract_python_blocks(text: str) -> list[str]:
    """Extract fenced Python code blocks from arbitrary markdown-ish text."""
    return [m.group(1) for m in _FENCE_PY.finditer(text)]


def parseable_prefix(code: str) -> str:
    """Return the longest line-prefix of `code` that ast.parse can handle.

    Lets us lint partially-emitted code: as the writer streams a function
    body, we lint the previously-completed top-level units even while the
    current one is incomplete.
    """
    if not code.strip():
        return ""
    lines = code.split("\n")
    while lines:
        candidate = "\n".join(lines)
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError as e:
            err_line = e.lineno or len(lines)
            if err_line <= 1:
                return ""
            lines = lines[: err_line - 1]
    return ""


def _to_finding(msg) -> LintFinding:
    text = msg.message
    args = getattr(msg, "message_args", None)
    if args:
        try:
            text = msg.message % args
        except Exception:
            pass
    return LintFinding(
        line=int(getattr(msg, "lineno", 1) or 1),
        col=int(getattr(msg, "col", 0) or 0),
        severity="warning",
        message=text.strip(),
    )


def lint_code(code: str) -> list[LintFinding]:
    """Lint a single Python block. Returns findings (possibly empty)."""
    if not code.strip():
        return []

    prefix = parseable_prefix(code)
    if not prefix:
        # Whole block is unparseable — surface the syntax error if any.
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return [LintFinding(
                line=exc.lineno or 1,
                col=exc.offset or 0,
                severity="error",
                message=f"SyntaxError: {exc.msg}",
            )]
        return []

    try:
        tree = ast.parse(prefix)
    except SyntaxError as exc:
        return [LintFinding(
            line=exc.lineno or 1,
            col=exc.offset or 0,
            severity="error",
            message=f"SyntaxError: {exc.msg}",
        )]

    c = _flakes_checker.Checker(tree, "writer.py")
    return [_to_finding(m) for m in c.messages]


def lint_writer_output(text: str) -> list[dict]:
    """Lint every Python block in the writer's accumulated output. Returns
    a flat list of finding dicts (block, line, col, severity, message)
    ready for SSE serialization."""
    out: list[dict] = []
    for block_idx, block in enumerate(extract_python_blocks(text)):
        for f in lint_code(block):
            out.append({
                "block": block_idx,
                "line": f.line,
                "col": f.col,
                "severity": f.severity,
                "message": f.message,
            })
    return out
