"""Tests for the shared storage-ID validator. Covers the
Windows-drive-letter trap that the original guards missed."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.utils.storage_ids import safe_storage_path, validate_storage_id


@pytest.mark.parametrize("good", [
    "9acdd50e-c943-4d13-aecb-283e43adc32f",
    "abc123",
    "abcdefABCDEF0123456789",
    "a-b-c",
    "a_b_c",
])
def test_validate_accepts_uuid_like_ids(good: str) -> None:
    validate_storage_id(good, kind="x")


@pytest.mark.parametrize("bad", [
    "",
    " ",
    "../etc/passwd",
    "..",
    ".secret",
    "a/b",
    "a\\b",
    "C:foo",                         # Windows drive letter — the actual bug
    "C:\\Windows\\System32",
    "a:b",                           # any colon
    "foo bar",
    "foo;rm -rf /",
    "a" * 65,                        # over length cap
    "<script>",
    "$(whoami)",
])
def test_validate_rejects_unsafe_ids(bad: str) -> None:
    with pytest.raises(ValueError):
        validate_storage_id(bad, kind="x")


def test_validate_rejects_non_string() -> None:
    with pytest.raises(ValueError):
        validate_storage_id(None, kind="x")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        validate_storage_id(123, kind="x")  # type: ignore[arg-type]


def test_safe_storage_path_returns_resolved_under_base(tmp_path: Path) -> None:
    base = tmp_path / "store"
    base.mkdir()
    p = safe_storage_path(base, "abc123", kind="x")
    assert p.parent == base.resolve()
    assert p.name == "abc123.json"


def test_safe_storage_path_rejects_traversal(tmp_path: Path) -> None:
    base = tmp_path / "store"
    base.mkdir()
    with pytest.raises(ValueError):
        safe_storage_path(base, "../escape", kind="x")
