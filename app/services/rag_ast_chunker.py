"""AST-aware code chunker using Tree-sitter.

Optional dependency — `chunk_file` in `rag_chunker.py` tries this first
and falls back to the recursive splitter if `tree_sitter_language_pack`
isn't installed or if parsing fails.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.services.rag_chunker import (
    MAX_CHUNK_CHARS,
    MIN_CHUNK_CHARS,
    SEPARATORS_BY_LANG,
    split_recursively,
)


# Internal LocalLLM language → tree-sitter grammar name
LANG_TO_TS: dict[str, str] = {
    "python": "python",
    "go": "go",
    "ruby": "ruby",
    "lua": "lua",
}

# Top-level node types treated as standalone chunks (one per definition)
DEF_NODES: dict[str, set[str]] = {
    "python": {
        "function_definition",
        "class_definition",
        "async_function_definition",
        "decorated_definition",
    },
    "go": {
        "function_declaration",
        "method_declaration",
        "type_declaration",
    },
    "ruby": {
        "class",
        "module",
        "method",
        "singleton_method",
    },
    "lua": {
        "function_declaration",
        "function_definition",
        "local_function",
    },
}


@lru_cache(maxsize=None)
def _get_parser(language: str):
    """Returns a tree-sitter Parser for the language, or None on any failure."""
    ts_name = LANG_TO_TS.get(language)
    if not ts_name:
        return None
    # tree-sitter-languages bundles grammars in the wheel (no runtime
    # downloads, important for the airgap target). Note: requires the
    # specific tree-sitter version pinned in requirements.txt.
    try:
        import warnings

        from tree_sitter_languages import get_parser  # noqa: WPS433

        with warnings.catch_warnings():
            # Older tree-sitter API triggers a deprecation warning we can't fix here.
            warnings.simplefilter("ignore")
            return get_parser(ts_name)
    except Exception:
        return None


def _split_if_oversized(text: str, language: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    return split_recursively(text, SEPARATORS_BY_LANG[language], MAX_CHUNK_CHARS)


def ast_chunk_file(path: Path, language: str) -> list[str] | None:
    """Chunk a file using Tree-sitter. Returns None if AST chunking is unavailable
    (caller should fall back). Returns [] for files that are too small to chunk."""
    parser = _get_parser(language)
    if parser is None:
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    if len(text) < MIN_CHUNK_CHARS:
        return []
    try:
        source_bytes = text.encode("utf-8", errors="ignore")
        tree = parser.parse(source_bytes)
    except Exception:
        return None

    root = tree.root_node
    def_types = DEF_NODES.get(language, set())

    chunks: list[str] = []
    preamble_buffer: list = []

    def flush_preamble() -> None:
        if not preamble_buffer:
            return
        first = preamble_buffer[0].start_byte
        last = preamble_buffer[-1].end_byte
        block = source_bytes[first:last].decode("utf-8", errors="ignore").strip()
        preamble_buffer.clear()
        if len(block) >= MIN_CHUNK_CHARS:
            chunks.extend(_split_if_oversized(block, language))

    for child in root.children:
        if child.type in def_types:
            flush_preamble()
            block = source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="ignore").strip()
            if block:
                chunks.extend(_split_if_oversized(block, language))
        else:
            preamble_buffer.append(child)
    flush_preamble()

    if not chunks:
        # Edge case: file with no recognized def nodes (e.g. a Python script).
        # Signal fallback so the recursive splitter can handle it.
        return None
    return chunks
