"""Round-trip and tamper-rejection tests for the JSON BM25 serializer."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.utils.bm25_io import load_bm25, save_bm25


def _build_corpus_bm25() -> tuple[BM25Okapi, list[str]]:
    corpus = [
        ["the", "quick", "brown", "fox"],
        ["jumps", "over", "the", "lazy", "dog"],
        ["impacket", "session", "connect", "smb"],
        ["how", "do", "I", "use", "scapy"],
    ]
    chunk_ids = [f"c{i}" for i in range(len(corpus))]
    return BM25Okapi(corpus), chunk_ids


def test_round_trip_preserves_scores() -> None:
    bm25, chunk_ids = _build_corpus_bm25()
    query = ["impacket", "session"]
    expected = list(bm25.get_scores(query))

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "bm25.json"
        save_bm25(bm25, chunk_ids, path)
        loaded = load_bm25(path)
        assert loaded is not None
        bm25_loaded, chunk_ids_loaded = loaded

    assert chunk_ids_loaded == chunk_ids
    actual = list(bm25_loaded.get_scores(query))
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert abs(a - e) < 1e-9


def test_load_returns_none_when_file_missing(tmp_path: Path) -> None:
    assert load_bm25(tmp_path / "missing.json") is None


def test_load_rejects_wrong_type_marker(tmp_path: Path) -> None:
    bad = tmp_path / "bm25.json"
    bad.write_text(json.dumps({"type": "Pickle", "data": "..."}), encoding="utf-8")
    assert load_bm25(bad) is None


def test_load_rejects_unexpected_schema_version(tmp_path: Path) -> None:
    bad = tmp_path / "bm25.json"
    bad.write_text(json.dumps({"type": "BM25Okapi", "schema": 99}), encoding="utf-8")
    assert load_bm25(bad) is None


def test_load_returns_none_on_corrupt_json(tmp_path: Path) -> None:
    bad = tmp_path / "bm25.json"
    bad.write_text("not valid json", encoding="utf-8")
    assert load_bm25(bad) is None


def test_load_returns_none_on_missing_required_fields(tmp_path: Path) -> None:
    bad = tmp_path / "bm25.json"
    bad.write_text(
        json.dumps({"type": "BM25Okapi", "schema": 1}),  # missing everything
        encoding="utf-8",
    )
    assert load_bm25(bad) is None


def test_atomic_write_via_temp_rename(tmp_path: Path) -> None:
    bm25, chunk_ids = _build_corpus_bm25()
    out = tmp_path / "bm25.json"
    save_bm25(bm25, chunk_ids, out)
    # No leftover .tmp file
    leftovers = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []
