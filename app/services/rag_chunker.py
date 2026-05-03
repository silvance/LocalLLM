"""Recursive text chunker for RAG indexing. Pure function — no chromadb/ollama deps.

AST-aware chunking lives alongside (will land in a follow-up); this module
is the language-aware fallback / small-file path.
"""
from __future__ import annotations

import logging
from pathlib import Path


logger = logging.getLogger("localllm")

EXT_LANG: dict[str, str] = {
    ".md": "markdown", ".mdown": "markdown", ".markdown": "markdown",
    ".rst": "rst",
    ".py": "python",
    ".go": "go",
    ".rb": "ruby",
    ".ps1": "powershell", ".psm1": "powershell", ".psd1": "powershell",
    ".lua": "lua",
    ".yml": "yaml", ".yaml": "yaml",
    ".txt": "text",
}

MIN_CHUNK_CHARS = 200
MAX_CHUNK_CHARS = 2000

SEPARATORS_BY_LANG: dict[str, list[str]] = {
    "markdown": ["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " "],
    "rst": ["\n=====", "\n-----", "\n~~~~~", "\n\n", "\n", ". ", " "],
    "python": ["\nclass ", "\ndef ", "\nasync def ", "\n\n", "\n", " "],
    "go": ["\nfunc ", "\ntype ", "\npackage ", "\n\n", "\n", " "],
    "ruby": ["\nclass ", "\nmodule ", "\ndef ", "\n\n", "\n", " "],
    "powershell": ["\nfunction ", "\nclass ", "\n\n", "\n", " "],
    "lua": ["\nfunction ", "\nlocal function ", "\n\n", "\n", " "],
    "yaml": ["\n\n", "\n", " "],
    "text": ["\n\n", "\n", ". ", " "],
}


def split_recursively(text: str, separators: list[str], max_size: int) -> list[str]:
    text = text.strip()
    if len(text) <= max_size:
        return [text] if text else []
    if not separators:
        return [text[i:i + max_size] for i in range(0, len(text), max_size)]
    sep = separators[0]
    parts = text.split(sep)
    chunks: list[str] = []
    current = ""
    for i, part in enumerate(parts):
        piece = (sep + part) if i > 0 else part
        candidate = current + piece
        if len(candidate) <= max_size:
            current = candidate
        else:
            if current.strip():
                chunks.append(current.strip())
            if len(piece) > max_size:
                chunks.extend(split_recursively(piece, separators[1:], max_size))
                current = ""
            else:
                current = piece
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if c.strip()]


AST_LANGS = {"python", "go", "ruby", "lua"}


def _recursive_chunk_file(path: Path, language: str) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    if len(text) < MIN_CHUNK_CHARS:
        return []
    separators = SEPARATORS_BY_LANG.get(language, SEPARATORS_BY_LANG["text"])
    return split_recursively(text, separators, MAX_CHUNK_CHARS)


def chunk_file(path: Path, language: str) -> list[str]:
    """Chunk a file. Uses Tree-sitter AST chunking for supported languages
    when available, otherwise falls back to the recursive splitter."""
    if language in AST_LANGS:
        try:
            from app.services.rag_ast_chunker import ast_chunk_file  # lazy
            chunks = ast_chunk_file(path, language)
            if chunks is not None:
                return chunks
            logger.debug(
                "AST chunker returned no chunks for %s (%s); falling back to recursive splitter",
                path.name, language,
            )
        except ImportError:
            logger.debug(
                "tree-sitter not installed; using recursive splitter for %s (%s)",
                path.name, language,
            )
        except Exception as exc:
            logger.warning(
                "AST chunker failed on %s (%s): %s — falling back to recursive splitter",
                path.name, language, exc,
            )
    return _recursive_chunk_file(path, language)
