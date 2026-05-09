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


def test_unhinted_blocks_are_skipped() -> None:
    """Models sometimes drop the language hint, but treating any
    untagged fence as Python false-matched non-Python content (bash
    setup snippets, the closing-fence-of-X / prose / opening-of-Y
    span). We accept missing the rare untagged Python block in
    exchange for never grabbing bash content."""
    text = "```\nimport os\n```\n```python\nos.path.join('a','b')\n```"
    blocks = extract_python_blocks(text)
    assert blocks == ["os.path.join('a','b')\n"]


def test_does_not_grab_bash_blocks() -> None:
    """Regression for a real failure: a writer's response with one
    Python block + multiple bash setup blocks was producing 4 'matches'
    because the regex's empty-language alternative matched the closing
    fence of one bash block + prose + opening fence of the next."""
    text = (
        "Here:\n\n"
        "```python\n"
        "def add(a, b):\n    return a + b\n"
        "```\n\n"
        "### Setup\n"
        "Run:\n"
        "```bash\n"
        "pip install scapy\n"
        "```\n\n"
        "Then:\n"
        "```bash\n"
        "sudo iw wlan0 set monitor\n"
        "```\n"
    )
    blocks = extract_python_blocks(text)
    assert len(blocks) == 1
    assert "def add" in blocks[0]
    assert "pip install" not in blocks[0]
    assert "iw wlan0" not in blocks[0]


def test_no_blocks_returns_empty() -> None:
    assert extract_python_blocks("just prose, no fences") == []


def test_extracts_block_with_indented_fence() -> None:
    """Regression: deepseek-coder-v2 emits ``` ```python `` (one space
    before the fence) consistently. The original regex anchored the
    fence at column 0 with no allowance, so every response from that
    model failed candidate_count with `no_candidate` and the review
    loop spun until rounds expired. CommonMark allows up to 3 spaces
    of leading indent on a fence; we now match that."""
    text = (
        "Here you go:\n\n"
        " ```python\n"
        "import subprocess\n"
        "def main():\n    pass\n"
        " ```\n"
    )
    blocks = extract_python_blocks(text)
    assert len(blocks) == 1
    assert "import subprocess" in blocks[0]


def test_extracts_block_with_three_space_indent() -> None:
    """Three spaces of leading indent is the CommonMark cutoff;
    must still match. Four+ spaces would be an indented code block
    in markdown (no fence) — out of scope for this gate."""
    text = "Output:\n\n   ```python\nx = 1\n   ```\n"
    blocks = extract_python_blocks(text)
    assert blocks == ["x = 1\n"]


def test_indented_fence_does_not_break_bash_filter() -> None:
    """Allowing leading whitespace on the fence must NOT regress the
    bash-block protection. An indented bash fence must still be
    rejected the same way an unindented one is."""
    text = (
        " ```bash\n"
        "pip install scapy\n"
        " ```\n"
        " ```python\n"
        "def f(): pass\n"
        " ```\n"
    )
    blocks = extract_python_blocks(text)
    assert len(blocks) == 1
    assert "def f" in blocks[0]
    assert "pip install" not in blocks[0]


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
