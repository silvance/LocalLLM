import pytest

from app.schemas.chat import ChatMessage, ChatRequest
from app.services.model_router import ModelRouter


@pytest.fixture
def router() -> ModelRouter:
    return ModelRouter()


def make_request(text: str) -> ChatRequest:
    return ChatRequest(messages=[ChatMessage(role="user", content=text)])


@pytest.mark.parametrize("prompt,expected", [
    ("hi", "granite"),
    ("Hello there", "granite"),
    ("What is the capital of France?", "granite"),
    ("who is the president", "granite"),
    ("Refactor this Python function to use async/await: def fetch(url): ...", "qwen"),
    ("```python\ndef foo():\n    pass\n```\nWhy does this break in 3.12?", "qwen"),
    ("fix this regex: ^[a-z]+$", "qwen"),
    ("Compare the architectural tradeoffs of monolith vs microservices for a small startup", "gemma"),
    ("Why might a startup prefer a vertical SaaS strategy over a horizontal one in a competitive market with strong network effects?", "gemma"),
])
def test_routes_to_expected_model(router: ModelRouter, prompt: str, expected: str) -> None:
    decision = router.route(make_request(prompt))
    assert decision.selected_model == expected, (
        f"expected {expected}, got {decision.selected_model}; reason: {decision.reason}"
    )


def test_empty_messages_default_to_granite(router: ModelRouter) -> None:
    decision = router.route(ChatRequest(messages=[]))
    assert decision.selected_model == "granite"


def test_word_boundary_rules_out_substring_false_positives(router: ModelRouter) -> None:
    # "hi" must not match inside "this" / "architectural" / "through"
    decision = router.route(make_request(
        "Walk me through this architecture for the reverse proxy"
    ))
    assert "Simple-task keyword" not in decision.reason


def test_decision_carries_score_and_reason(router: ModelRouter) -> None:
    decision = router.route(make_request("Refactor this Python function"))
    assert isinstance(decision.complexity_score, int)
    assert decision.reason
