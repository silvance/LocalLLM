"""Pure-function tests for the agent loop. We mock the chat client so we
don't reach a real Ollama, and mock the tool implementations so we don't
hit the real DuckDuckGo / network."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from app.agent.loop import (
    SYSTEM_PROMPT,
    TOOL_SCHEMAS,
    _coerce_args,
    run_agent,
)


class FakeChatClient:
    """Drop-in for ollama.Client. Returns scripted responses in order."""
    def __init__(self, scripted: list[dict]) -> None:
        self.scripted = list(scripted)
        self.calls: list[dict] = []

    def chat(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        if not self.scripted:
            return {"message": {"content": "(no more scripted responses)"}}
        return {"message": self.scripted.pop(0)}


def test_coerce_args_handles_dict_string_and_garbage() -> None:
    assert _coerce_args({"a": 1}) == {"a": 1}
    assert _coerce_args('{"a": 2}') == {"a": 2}
    assert _coerce_args("not json") == {}
    assert _coerce_args(None) == {}
    assert _coerce_args(["list"]) == {}


def test_run_agent_returns_immediately_on_no_tool_calls() -> None:
    client = FakeChatClient([
        {"role": "assistant", "content": "I already know the answer."},
    ])
    events: list[tuple[str, dict]] = []
    answer = run_agent(
        "what is 2+2?",
        chat_client=client,
        model="granite",
        on_event=lambda n, p: events.append((n, p)),
    )
    assert answer == "I already know the answer."
    # System + user messages were sent
    sent = client.calls[0]
    assert sent["model"] == "granite"
    assert sent["messages"][0]["role"] == "system"
    assert sent["messages"][0]["content"] == SYSTEM_PROMPT
    assert sent["messages"][1]["role"] == "user"
    assert sent["tools"] == TOOL_SCHEMAS
    # Events: step_start, model_text, done
    names = [n for n, _ in events]
    assert names == ["step_start", "model_text", "done"]


def test_run_agent_executes_tool_then_returns_answer() -> None:
    client = FakeChatClient([
        # Round 1: model wants to search
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {"name": "web_search", "arguments": {"query": "X", "max_results": 3}},
            }],
        },
        # Round 2: model gives a final answer using the search result
        {"role": "assistant", "content": "The answer is 42 [1]."},
    ])
    events: list[tuple[str, dict]] = []
    fake_results = {"results": [
        {"title": "T", "url": "https://example.com", "snippet": "S"}
    ]}
    with patch("app.agent.loop._execute_tool", return_value=fake_results):
        answer = run_agent(
            "research X",
            chat_client=client,
            model="granite",
            on_event=lambda n, p: events.append((n, p)),
        )
    assert "42" in answer
    names = [n for n, _ in events]
    # Iteration 0: step_start, tool_call, tool_result. Iteration 1:
    # step_start, model_text, then done.
    assert names == [
        "step_start", "tool_call", "tool_result",
        "step_start", "model_text", "done",
    ]
    # The 'tool' message got appended for the next iteration
    second_call = client.calls[1]
    roles = [m["role"] for m in second_call["messages"]]
    assert "tool" in roles


def test_run_agent_surfaces_tool_errors_and_keeps_going() -> None:
    client = FakeChatClient([
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "web_search", "arguments": {"query": "Z"}}}]},
        {"role": "assistant", "content": "Couldn't find it."},
    ])
    events: list[tuple[str, dict]] = []
    with patch("app.agent.loop._execute_tool", side_effect=RuntimeError("dns fail")):
        answer = run_agent(
            "research Z",
            chat_client=client,
            model="granite",
            on_event=lambda n, p: events.append((n, p)),
        )
    assert answer == "Couldn't find it."
    names = [n for n, _ in events]
    assert "tool_error" in names


def test_run_agent_caps_at_max_iterations() -> None:
    """Model that always wants more tool calls — we should bail."""
    forever = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "web_search", "arguments": {"query": "loop"}}}],
    }
    client = FakeChatClient([dict(forever) for _ in range(20)])
    events: list[tuple[str, dict]] = []
    with patch("app.agent.loop._execute_tool", return_value={"results": []}):
        answer = run_agent(
            "loop forever",
            chat_client=client,
            model="granite",
            max_iterations=3,
            on_event=lambda n, p: events.append((n, p)),
        )
    assert answer == ""
    error_events = [p for n, p in events if n == "error"]
    assert any("3 iterations" in e.get("error", "") for e in error_events)


def test_run_agent_handles_string_args_from_some_models() -> None:
    """Some Ollama models return tool arguments as a JSON-string instead
    of a dict. The loop must accept both."""
    client = FakeChatClient([
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "web_search",
                    "arguments": '{"query": "ok", "max_results": 3}',
                },
            }],
        },
        {"role": "assistant", "content": "Done."},
    ])
    captured: list[dict] = []
    def fake_execute(name: str, args: dict) -> dict:
        captured.append(args)
        return {"results": []}
    with patch("app.agent.loop._execute_tool", side_effect=fake_execute):
        run_agent("x", chat_client=client, model="granite")
    assert captured == [{"query": "ok", "max_results": 3}]
