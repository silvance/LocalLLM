"""Tokenizer for BM25 indexing and retrieval.

Must be deterministic — `scripts/build_index.py` and `app.services.rag_service`
both call this and the produced tokens must match for fusion to work.

Behavior:
  - Splits camelCase identifiers: `MyHTTPServer` -> `my http server`
  - Splits snake_case via the non-word splitter (underscore is treated as non-word)
  - Lowercases everything
  - Drops tokens shorter than MIN_TOKEN_LEN
"""
from __future__ import annotations

import re

_CAMEL_LOWER_UPPER = re.compile(r"([a-z\d])([A-Z])")
_CAMEL_ACRONYM = re.compile(r"([A-Z]+)([A-Z][a-z])")
_NONWORD = re.compile(r"[\W_]+")
MIN_TOKEN_LEN = 2


def tokenize(text: str) -> list[str]:
    text = _CAMEL_ACRONYM.sub(r"\1 \2", text)
    text = _CAMEL_LOWER_UPPER.sub(r"\1 \2", text)
    text = text.lower()
    return [t for t in _NONWORD.split(text) if len(t) >= MIN_TOKEN_LEN]
