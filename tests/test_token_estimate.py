from app.schemas.chat import ChatMessage
from app.utils.token_estimate import estimate_messages_tokens, estimate_tokens


def test_empty_string_is_zero() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0
    assert estimate_tokens("   ") == 0


def test_short_input_returns_positive() -> None:
    assert estimate_tokens("hello world") > 0


def test_monotonic_with_length() -> None:
    short = estimate_tokens("one two three")
    long = estimate_tokens("one two three four five six seven eight nine ten")
    assert long > short


def test_messages_sum() -> None:
    msgs = [
        ChatMessage(role="user", content="alpha bravo"),
        ChatMessage(role="assistant", content="charlie delta echo foxtrot"),
    ]
    total = estimate_messages_tokens(msgs)
    assert total == estimate_tokens("alpha bravo") + estimate_tokens("charlie delta echo foxtrot")
