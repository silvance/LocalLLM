"""Tests for the live writer-output linter."""
from __future__ import annotations

from app.utils.code_linter import (
    extract_python_blocks,
    lint_code,
    lint_writer_output,
    parseable_prefix,
)


# ---------------------- extract_python_blocks --------------------------

def test_extracts_explicit_python_blocks() -> None:
    text = "Here you go:\n\n```python\ndef foo():\n    return 1\n```\n\nDone."
    assert extract_python_blocks(text) == ["def foo():\n    return 1\n"]


def test_extracts_unhinted_blocks_too() -> None:
    """Models sometimes drop the `python` hint after the first block."""
    text = "```\nimport os\n```\n```python\nos.path.join('a','b')\n```"
    blocks = extract_python_blocks(text)
    assert "import os\n" in blocks
    assert "os.path.join('a','b')\n" in blocks


def test_no_blocks_returns_empty() -> None:
    assert extract_python_blocks("just prose, no fences") == []


# ---------------------- parseable_prefix -------------------------------

def test_parseable_prefix_returns_full_when_complete() -> None:
    code = "def foo():\n    return 1\n"
    assert parseable_prefix(code) == code.rstrip("\n") or parseable_prefix(code) == code


def test_parseable_prefix_walks_back_on_incomplete_function() -> None:
    """Writer is mid-emission of a second function. We should still get the
    completed first function back."""
    code = "def foo():\n    return 1\n\ndef bar():\n    x ="
    prefix = parseable_prefix(code)
    assert "def foo" in prefix
    assert "def bar" not in prefix or prefix == ""


def test_parseable_prefix_empty_on_garbage() -> None:
    assert parseable_prefix("???\n!!!") == ""


# ---------------------- lint_code (the real test) ----------------------

def test_lint_catches_undefined_name() -> None:
    """The exact bug class the user hit with `self.ModuleOption(...)`."""
    code = "def go():\n    return ModuleOption()\n"
    findings = lint_code(code)
    assert any("ModuleOption" in f.message for f in findings)


def test_lint_catches_unused_import() -> None:
    code = "import os\n\nprint('hello')\n"
    findings = lint_code(code)
    assert any("os" in f.message and "import" in f.message.lower() for f in findings)


def test_lint_clean_code_returns_empty() -> None:
    code = "import os\n\nprint(os.getcwd())\n"
    assert lint_code(code) == []


def test_lint_returns_syntax_error_when_unparseable() -> None:
    code = "def broken(:\n    pass\n"
    findings = lint_code(code)
    assert any(f.severity == "error" and "Syntax" in f.message for f in findings)


def test_lint_partial_code_still_useful() -> None:
    """Writer is mid-stream — first function is complete, second is mid-line.
    We should still get findings for the first function."""
    code = (
        "def first():\n"
        "    return Undefined()\n"
        "\n"
        "def second():\n"
        "    x ="
    )
    findings = lint_code(code)
    assert any("Undefined" in f.message for f in findings)


# ---------------------- lint_writer_output -----------------------------

def test_writer_output_lints_each_block() -> None:
    text = (
        "Here is part one:\n\n"
        "```python\n"
        "import os\n"
        "print('clean')\n"
        "```\n\n"
        "And part two:\n\n"
        "```python\n"
        "def go():\n"
        "    return Undefined()\n"
        "```\n"
    )
    findings = lint_writer_output(text)
    # First block is clean; second block flags Undefined
    assert any(f["message"].startswith("undefined") or "Undefined" in f["message"] for f in findings)
    # Block index is recorded so the UI can show "block 1, line N"
    assert all("block" in f for f in findings)
