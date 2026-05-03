"""Zip-slip guard tests for the corpus fetcher."""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pytest


# scripts/ is its own importable package alongside app/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.fetch_corpus import _safe_extract_zip  # noqa: E402


def _zip_with(entries: dict[str, bytes]) -> zipfile.ZipFile:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, payload in entries.items():
            zf.writestr(name, payload)
    buf.seek(0)
    return zipfile.ZipFile(buf, "r")


def test_safe_extract_normal_zip(tmp_path: Path) -> None:
    dest = tmp_path / "out"
    dest.mkdir()
    with _zip_with({"a.txt": b"alpha", "subdir/b.txt": b"bravo"}) as zf:
        _safe_extract_zip(zf, dest)
    assert (dest / "a.txt").read_bytes() == b"alpha"
    assert (dest / "subdir" / "b.txt").read_bytes() == b"bravo"


def test_rejects_parent_traversal(tmp_path: Path) -> None:
    dest = tmp_path / "out"
    dest.mkdir()
    with _zip_with({"../escape.txt": b"x"}) as zf:
        with pytest.raises(ValueError, match="zip-slip"):
            _safe_extract_zip(zf, dest)
    # Nothing leaked outside dest
    assert not (tmp_path / "escape.txt").exists()


def test_rejects_absolute_path(tmp_path: Path) -> None:
    dest = tmp_path / "out"
    dest.mkdir()
    with _zip_with({"/etc/passwd": b"x"}) as zf:
        with pytest.raises(ValueError, match="zip-slip"):
            _safe_extract_zip(zf, dest)


def test_rejects_nested_traversal(tmp_path: Path) -> None:
    dest = tmp_path / "out"
    dest.mkdir()
    with _zip_with({"safe/../../escape.txt": b"x"}) as zf:
        with pytest.raises(ValueError, match="zip-slip"):
            _safe_extract_zip(zf, dest)
