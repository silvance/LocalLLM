"""Tests for the pure-function bits of the reranker (parsing, formatting).

Doesn't exercise the Ollama call — that path is mocked elsewhere or
covered by integration smoke testing.
"""
from app.services.rag_reranker import _format_passages, _parse_scores


class _FakeRetrieval:
    def __init__(self, document: str, source_id: str = "src", file_path: str = "f.py", language: str = "python") -> None:
        self.document = document
        self.source_id = source_id
        self.file_path = file_path
        self.language = language


def test_format_passages_numbers_and_truncates() -> None:
    rs = [
        _FakeRetrieval("hello world"),
        _FakeRetrieval("a" * 1000),
    ]
    formatted = _format_passages(rs, max_chars=100)
    assert "[Passage 1]" in formatted
    assert "[Passage 2]" in formatted
    # The 1000-char doc should be truncated with an ellipsis
    assert "…" in formatted
    # First passage fits without ellipsis
    assert formatted.count("…") == 1


def test_parse_scores_happy_path() -> None:
    out = "Result: [3, 7, 0, 9]"
    assert _parse_scores(out, expected_count=4) == [3.0, 7.0, 0.0, 9.0]


def test_parse_scores_with_floats() -> None:
    assert _parse_scores("[7.5, 0.0, 9.2]", expected_count=3) == [7.5, 0.0, 9.2]


def test_parse_scores_returns_none_on_wrong_count() -> None:
    assert _parse_scores("[1, 2, 3]", expected_count=5) is None


def test_parse_scores_returns_none_when_no_array() -> None:
    assert _parse_scores("Sorry, I can't comply.", expected_count=3) is None


def test_parse_scores_picks_first_array_in_response() -> None:
    out = "I think the scores should be [4, 5, 6] which I derived from..."
    assert _parse_scores(out, expected_count=3) == [4.0, 5.0, 6.0]
