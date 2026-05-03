"""JSON serialization for the BM25Okapi index.

Replaces the previous `pickle.dump`/`pickle.load` pair: deserialising a
pickle is arbitrary code execution, and the BM25 artifact is one of the
files we ask the operator to sneakernet into the airgap. JSON is a
recovery from that footgun — only declared fields are read back, no
imported modules or constructors are invoked.

We materialize the BM25Okapi object lazily by constructing an empty
instance via `object.__new__` and populating its attributes from the
deserialised state. This keeps `bm25.get_scores(...)` working unchanged
in the retrieval path.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from rank_bm25 import BM25Okapi


logger = logging.getLogger("localllm")


SCHEMA_VERSION = 1
TYPE_NAME = "BM25Okapi"


def save_bm25(bm25: BM25Okapi, chunk_ids: list[str], path: Path) -> None:
    state = {
        "schema": SCHEMA_VERSION,
        "type": TYPE_NAME,
        "k1": float(bm25.k1),
        "b": float(bm25.b),
        "epsilon": float(getattr(bm25, "epsilon", 0.25)),
        "corpus_size": int(bm25.corpus_size),
        "avgdl": float(bm25.avgdl),
        "average_idf": float(getattr(bm25, "average_idf", 0.0)),
        # Each doc_freqs entry is dict[token -> int]; round-trip through dict()
        # to drop any defaultdict factory.
        "doc_freqs": [dict(d) for d in bm25.doc_freqs],
        "idf": {k: float(v) for k, v in bm25.idf.items()},
        "doc_len": list(bm25.doc_len),
        "chunk_ids": list(chunk_ids),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f)
    tmp.replace(path)


def load_bm25(path: Path) -> tuple[BM25Okapi, list[str]] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception as exc:
        logger.warning("Failed to read BM25 index at %s: %s", path, exc)
        return None

    if not isinstance(state, dict) or state.get("type") != TYPE_NAME:
        logger.warning("BM25 index at %s has wrong type marker; ignoring", path)
        return None

    schema = state.get("schema")
    if schema != SCHEMA_VERSION:
        logger.warning(
            "BM25 index at %s has unexpected schema version %r (expected %d); ignoring",
            path, schema, SCHEMA_VERSION,
        )
        return None

    try:
        # Skip __init__ (which would try to fit on a corpus we don't have)
        # and populate the attributes directly.
        bm25 = object.__new__(BM25Okapi)
        bm25.k1 = float(state["k1"])
        bm25.b = float(state["b"])
        bm25.epsilon = float(state.get("epsilon", 0.25))
        bm25.corpus_size = int(state["corpus_size"])
        bm25.avgdl = float(state["avgdl"])
        bm25.average_idf = float(state.get("average_idf", 0.0))
        bm25.doc_freqs = [dict(d) for d in state["doc_freqs"]]
        bm25.idf = {str(k): float(v) for k, v in state["idf"].items()}
        bm25.doc_len = [int(x) for x in state["doc_len"]]
        bm25.tokenizer = None
        chunk_ids = [str(c) for c in state["chunk_ids"]]
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("BM25 index at %s is malformed: %s", path, exc)
        return None

    return bm25, chunk_ids
