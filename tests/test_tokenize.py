from app.services.rag_tokenize import tokenize


def test_camel_case_split() -> None:
    assert tokenize("ImpacketSession") == ["impacket", "session"]


def test_snake_case_split() -> None:
    assert tokenize("connect_smb_server") == ["connect", "smb", "server"]


def test_combined_dot_and_underscore() -> None:
    assert tokenize("ImpacketSession.connect_smb_server") == [
        "impacket", "session", "connect", "smb", "server",
    ]


def test_acronym_splits_cleanly() -> None:
    # MyHTTPServer -> ["my", "http", "server"]
    assert tokenize("MyHTTPServer") == ["my", "http", "server"]


def test_drops_short_tokens() -> None:
    tokens = tokenize("a quick brown fox")
    assert "a" not in tokens
    assert "quick" in tokens


def test_preserves_alphanumeric_runs() -> None:
    assert "sha256" in tokenize("crypto/sha256.NewHash")


def test_lowercase() -> None:
    tokens = tokenize("Some MixedCase Identifier")
    assert all(t == t.lower() for t in tokens)


def test_empty_string() -> None:
    assert tokenize("") == []


def test_query_indexed_text_match() -> None:
    """A query containing a camelCase identifier should produce tokens
    that overlap with the snake_case form in indexed text."""
    query_tokens = set(tokenize("how do I use ImpacketSession"))
    indexed_tokens = set(tokenize("class ImpacketSession(BaseSession):"))
    overlap = query_tokens & indexed_tokens
    assert {"impacket", "session"} <= overlap
