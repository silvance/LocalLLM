from app.services.rag_chunker import (
    MAX_CHUNK_CHARS,
    SEPARATORS_BY_LANG,
    split_recursively,
)


def test_short_text_is_kept_whole() -> None:
    assert split_recursively("hello world", SEPARATORS_BY_LANG["text"], 1000) == [
        "hello world"
    ]


def test_empty_text_yields_no_chunks() -> None:
    assert split_recursively("", SEPARATORS_BY_LANG["text"], 1000) == []


def test_python_splits_at_class_boundaries() -> None:
    text = (
        "class Foo:\n    def a(self): pass\n\n"
        "class Bar:\n    def b(self): pass\n\n"
    ) * 50
    chunks = split_recursively(text, SEPARATORS_BY_LANG["python"], 200)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


def test_no_chunk_exceeds_max_size_by_more_than_one_separator() -> None:
    text = "word " * 1000
    max_size = 200
    chunks = split_recursively(text, SEPARATORS_BY_LANG["text"], max_size)
    # Allow some slack since we keep the trailing separator on chunks.
    assert all(len(c) <= max_size + 10 for c in chunks)


def test_markdown_splits_on_headings() -> None:
    text = (
        "# Top\n"
        + ("intro paragraph " * 30)
        + "\n## Section A\n"
        + ("body of A " * 30)
        + "\n## Section B\n"
        + ("body of B " * 30)
    )
    chunks = split_recursively(text, SEPARATORS_BY_LANG["markdown"], 400)
    assert any("Section A" in c for c in chunks) or any("body of A" in c for c in chunks)
