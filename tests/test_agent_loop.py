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
    _build_system_prompt,
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


def test_run_agent_includes_baseline_in_system_message() -> None:
    """The agent's first system message must include the always-on
    anti-hallucination baseline AND the agent's research-loop prompt.
    Regression guard for the centralized compose() wiring."""
    from app.utils.system_prompt import BASELINE
    client = FakeChatClient([
        {"role": "assistant", "content": "answer."},
    ])
    run_agent("question?", chat_client=client, model="granite")
    sent = client.calls[0]
    sys_content = sent["messages"][0]["content"]
    assert sent["messages"][0]["role"] == "system"
    assert "Calibration" in sys_content  # from BASELINE
    assert "research agent" in sys_content  # from agent SYSTEM_PROMPT
    assert sys_content.index(BASELINE.strip().splitlines()[0]) < sys_content.index("research agent")


def test_system_prompt_forbids_fabrication_on_tool_error() -> None:
    """Regression guard: when both tools fail, the model used to make up
    plausible CVE numbers + URLs from prior knowledge. The system prompt
    must explicitly forbid that and require an error-report final answer."""
    p = SYSTEM_PROMPT.lower()
    # The guidance must mention error results, the no-fabrication rule,
    # and the report-the-error fallback. Keeping these as substring checks
    # so wording can drift without breaking the test.
    assert '"error"' in SYSTEM_PROMPT
    assert "fabricate" in p or "invent" in p
    assert "cve" in p  # specific examples must appear so the model takes them seriously


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
    # Substring rather than exact equality — the always-on baseline
    # prompt is now prepended (see test_run_agent_includes_baseline_in_system_message).
    assert SYSTEM_PROMPT.strip() in sent["messages"][0]["content"]
    assert sent["messages"][1]["role"] == "user"
    assert sent["tools"] == TOOL_SCHEMAS
    # Events: step_start, then done — the FINAL turn's text reaches
    # the UI via done.metadata.answer, not via a separate model_text
    # event. Emitting it as both produced the double-render bug
    # (Reasoning panel + Final Answer panel showing the same paragraph).
    names = [n for n, _ in events]
    assert names == ["step_start", "done"]
    # Done event must carry the answer in metadata so the UI can
    # render it.
    done_events = [p for n, p in events if n == "done"]
    assert done_events[0]["answer"] == "I already know the answer."


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
    # Iteration 0: step_start + tool_call + tool_result.
    # Iteration 1: step_start + (NO model_text — that would double-
    # render against done.metadata.answer) + done.
    assert names == [
        "step_start", "tool_call", "tool_result",
        "step_start", "done",
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


# ---------------------------------------------------------------------------
# Recency discipline — date awareness in the system prompt
# ---------------------------------------------------------------------------

def test_build_system_prompt_injects_today_and_year() -> None:
    """The agent has to know what 'current' means. Pin the date to a
    fixed value and assert it shows up in the prompt — both as the
    full ISO date and as bare year qualifiers in the search-fallback
    instruction."""
    p = _build_system_prompt("2026-05-09")
    assert "Today's date is 2026-05-09" in p
    assert "current calendar year is 2026" in p
    # Year + previous-year qualifiers used by the "search again with
    # a year qualifier" follow-up rule.
    assert "\"2026\"" in p
    assert "\"2025\"" in p


def test_system_prompt_contains_recency_discipline_section() -> None:
    """Catch a regression where someone strips the recency rules
    while editing the prompt for an unrelated reason. The exact
    section heading is the load-bearing token — searches for it in
    docs / changelogs / future code reviews."""
    p = _build_system_prompt("2026-05-09")
    assert "RECENCY DISCIPLINE" in p
    # Hard rules the model must learn — pin the most important
    # phrasings so paraphrase drift gets flagged.
    assert "potentially" in p.lower() and "stale" in p.lower()
    assert "year qualifier" in p.lower()
    assert "publication date" in p.lower() or "published_at" in p


def test_run_agent_renders_today_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """The runner must call _build_system_prompt() each invocation so
    a long-lived server doesn't tell the model an out-of-date today.
    Imports happen once; runs happen continuously."""
    rendered: list[str] = []
    real_build = _build_system_prompt

    def spy(today_iso: str | None = None) -> str:
        out = real_build(today_iso)
        rendered.append(out)
        return out

    monkeypatch.setattr("app.agent.loop._build_system_prompt", spy)
    client = FakeChatClient([{"role": "assistant", "content": "Done."}])
    run_agent("x", chat_client=client, model="granite")
    # _build_system_prompt is called once during run_agent invocation.
    # If a future refactor caches the prompt at import, this fails
    # (rendered would be empty) and the failing test points at the
    # date-drift bug for whoever inherits it.
    assert len(rendered) == 1


def test_system_prompt_documents_published_at_field() -> None:
    """The model needs to know `published_at` exists in tool results.
    Without that, even with date data in hand the model has nothing
    to consult."""
    p = _build_system_prompt("2026-05-09")
    assert "published_at" in p
    # web_search and http_fetch must both be advertised as
    # date-aware so the model uses both signals.
    web_block = p[p.index("web_search"):p.index("http_fetch")]
    fetch_block = p[p.index("http_fetch"):p.index("Strategy:")]
    assert "published_at" in web_block
    assert "published_at" in fetch_block


# ---------------------------------------------------------------------------
# Forced final-synthesis pass at iteration cap
# ---------------------------------------------------------------------------

def test_run_agent_forces_synthesis_when_budget_exhausted() -> None:
    """When the loop runs out of tool-call rounds without a final
    answer, run ONE more turn with no tools to force synthesis. The
    operator gets a real answer instead of a locked UI with empty
    job state — that was the failure mode the user reported when
    the model burned all its iterations on stale-source fetches."""
    # Model wants tools forever for the first 3 iterations, then
    # the synthesis turn (NO tools) returns a real answer.
    forever_tool = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "web_search", "arguments": {"query": "x"}}}],
    }
    synthesis_answer = {
        "role": "assistant",
        "content": "Best answer I can give from the partial data: foo.",
    }
    client = FakeChatClient([
        dict(forever_tool), dict(forever_tool), dict(forever_tool),
        synthesis_answer,
    ])
    events: list[tuple[str, dict]] = []
    with patch("app.agent.loop._execute_tool", return_value={"results": []}):
        answer = run_agent(
            "research X",
            chat_client=client,
            model="granite",
            max_iterations=3,
            on_event=lambda n, p: events.append((n, p)),
        )
    assert "foo" in answer, "synthesis text must become final_answer"

    # The synthesis call must have been made WITHOUT tools — that's
    # what forces termination. If we still passed `tools=`, the
    # model could just emit tool_calls again and the user would
    # be locked out a second time.
    assert len(client.calls) == 4
    synth_call = client.calls[3]
    assert "tools" not in synth_call, (
        f"synthesis call must not carry tools=; got keys: {list(synth_call.keys())}"
    )

    # Must include a user-visible "out of budget" notice so the
    # operator understands why the answer is being labeled.
    text_events = [p["content"] for n, p in events if n == "model_text"]
    assert any("out of tool budget" in t.lower() for t in text_events)

    # `done` event must signal the cap-then-synthesis path so the
    # UI can decorate accordingly.
    done_events = [p for n, p in events if n == "done"]
    assert len(done_events) == 1
    assert done_events[0]["capped"] is True
    assert done_events[0]["answer"] == answer


def test_run_agent_emits_error_when_synthesis_returns_empty() -> None:
    """If even the no-tools synthesis turn comes back empty, surface
    a clear error so the UI can unlock and the operator knows the
    model gave up. Without this, `done` would carry an empty answer
    silently and the operator would think the system hung."""
    forever_tool = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "web_search", "arguments": {"query": "x"}}}],
    }
    # Synthesis turn ALSO returns empty content. The fake client
    # returns scripted responses regardless of whether tools= is
    # passed, so this exercises the "synthesis went sideways" path.
    empty_synthesis = {"role": "assistant", "content": ""}
    client = FakeChatClient([
        dict(forever_tool), dict(forever_tool), empty_synthesis,
    ])
    events: list[tuple[str, dict]] = []
    with patch("app.agent.loop._execute_tool", return_value={"results": []}):
        answer = run_agent(
            "research X",
            chat_client=client,
            model="granite",
            max_iterations=2,
            on_event=lambda n, p: events.append((n, p)),
        )
    assert answer == ""
    error_events = [p for n, p in events if n == "error"]
    assert any(
        "synthesis" in e.get("error", "").lower()
        and "empty" in e.get("error", "").lower()
        for e in error_events
    ), f"expected synthesis-empty error; got events: {error_events}"


def test_run_agent_emits_done_event_after_synthesis_failure() -> None:
    """`done` must always fire so the JobManager closes the job and
    the chat input unlocks. Both for synthesis-success AND synthesis-
    failure paths — the user reported being unable to send a follow-
    up question, and the most common cause is a never-finished job."""
    forever_tool = {
        "role": "assistant", "content": "",
        "tool_calls": [{"function": {"name": "web_search", "arguments": {"query": "x"}}}],
    }
    empty_synthesis = {"role": "assistant", "content": ""}
    client = FakeChatClient([dict(forever_tool), empty_synthesis])
    events: list[tuple[str, dict]] = []
    with patch("app.agent.loop._execute_tool", return_value={"results": []}):
        run_agent(
            "x",
            chat_client=client,
            model="granite",
            max_iterations=1,
            on_event=lambda n, p: events.append((n, p)),
        )
    names = [n for n, _ in events]
    assert "done" in names, (
        f"`done` event missing — UI would stay locked. Events: {names}"
    )


def test_system_prompt_says_skip_stale_dont_fetch() -> None:
    """The original recency rule said treat stale as stale but didn't
    say STOP fetching it. The model burned tool calls on
    confirmed-stale 2020 articles. The rule must be explicit:
    stale `published_at` → skip."""
    p = _build_system_prompt("2026-05-09")
    p_low = p.lower()
    # The skip-stale rule must mention BOTH "stale" / "old" AND a
    # negation ("do not", "skip", "don't"). Substring checks so
    # paraphrase drift doesn't break, but the imperative must
    # remain or this test fails meaningfully.
    assert "do not fetch" in p_low or "don't fetch" in p_low or "skip" in p_low
    # Must explicitly mention the published_at field — the model
    # needs to know which signal to consult.
    assert "published_at" in p
    # Reserve-budget rule: tool calls have a hard cap so we don't
    # exhaust budget on gathering and skip synthesis.
    assert "reserve" in p_low or "budget" in p_low


# ---------------------------------------------------------------------------
# Final-answer single-render — pin against the double-emit regression
# ---------------------------------------------------------------------------

def test_final_answer_text_not_emitted_as_model_text() -> None:
    """Regression: the user's iOS-vulnerabilities run showed the same
    paragraph rendered twice — once as 'Reasoning' (model_text) and
    once as 'Final answer' (done.metadata.answer). The final turn's
    text MUST flow through done only; never through a model_text
    event that the UI would render alongside the final-answer panel."""
    client = FakeChatClient([
        # Single turn, no tools — text becomes final_answer directly.
        {"role": "assistant", "content": "Done — answer is foo."},
    ])
    events: list[tuple[str, dict]] = []
    run_agent(
        "x",
        chat_client=client,
        model="granite",
        on_event=lambda n, p: events.append((n, p)),
    )
    text_events = [p for n, p in events if n == "model_text"]
    assert text_events == [], (
        f"final-iteration text leaked into model_text events: {text_events}"
    )
    # And the answer reached the UI via done as expected.
    done_events = [p for n, p in events if n == "done"]
    assert done_events[0]["answer"] == "Done — answer is foo."


def test_intermediate_reasoning_still_emits_model_text() -> None:
    """Suppressing model_text on the FINAL turn must not also suppress
    it on intermediate tool-calling turns — those are the model's
    legitimate 'thinking out loud' that the user wants to see."""
    client = FakeChatClient([
        # Iter 0: reasoning + tool_call. Text here IS intermediate.
        {
            "role": "assistant",
            "content": "I'll search for that.",
            "tool_calls": [{"function": {"name": "web_search", "arguments": {"query": "x"}}}],
        },
        # Iter 1: final answer.
        {"role": "assistant", "content": "Found it."},
    ])
    events: list[tuple[str, dict]] = []
    with patch("app.agent.loop._execute_tool", return_value={"results": []}):
        run_agent(
            "x",
            chat_client=client,
            model="granite",
            on_event=lambda n, p: events.append((n, p)),
        )
    text_events = [p["content"] for n, p in events if n == "model_text"]
    assert "I'll search for that." in text_events, (
        f"intermediate reasoning was suppressed; got: {text_events}"
    )
    # Final answer text must NOT also appear as model_text.
    assert "Found it." not in text_events


# ---------------------------------------------------------------------------
# Grounding check — strict-mode refusal when the model invents claims
# ---------------------------------------------------------------------------

def test_run_agent_strict_mode_refuses_fabricated_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: a tool-call run where the agent fabricates CVE
    IDs and source URLs that weren't in any tool result. Strict
    mode must REPLACE the final answer with a refusal, and the
    `done` event must carry that refusal — NOT the original
    fabricated text."""
    monkeypatch.setenv("LOCALLLM_AGENT_VERIFY_MODE", "strict")
    client = FakeChatClient([
        # Iter 0: search
        {
            "role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": "web_search",
                                          "arguments": {"query": "ios cve"}}}],
        },
        # Iter 1: model produces a confidently-wrong final answer
        # citing a CVE and URL that DON'T appear in any tool result.
        {
            "role": "assistant",
            "content": (
                "Current iOS vulns include CVE-2099-99999 per "
                "https://totally-fake-cve-site.example.org/page."
            ),
        },
    ])
    fake_search = {"results": [
        {"url": "https://thehackernews.com/real-article",
         "snippet": "real CVE-2024-1 was patched"},
    ]}
    events: list[tuple[str, dict]] = []

    def _exec(name: str, args: dict) -> dict:
        # web_search returns the fake_search; nothing else gets called
        return fake_search if name == "web_search" else {}

    with patch("app.agent.loop._execute_tool", side_effect=_exec):
        answer = run_agent(
            "research ios",
            chat_client=client,
            model="granite",
            on_event=lambda n, p: events.append((n, p)),
        )

    # The fabricated answer MUST NOT reach the user.
    assert "CVE-2099-99999" not in answer or "rejected" in answer.lower(), (
        "strict mode failed to refuse fabricated CVE — answer was: " + answer
    )
    assert "rejected" in answer.lower() or "unverified" in answer.lower(), (
        f"expected refusal-shaped answer; got: {answer[:200]!r}"
    )
    # And the orchestrator must emit a `verification` event so the
    # UI / operator can see what was flagged.
    verification_events = [p for n, p in events if n == "verification"]
    assert verification_events, "no verification event emitted"
    v = verification_events[0]
    assert "CVE-2099-99999" in v["unverified_cves"]
    assert any("totally-fake-cve-site" in u for u in v["unverified_urls"])


def test_run_agent_strict_mode_passes_clean_answer_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When every claim is verifiable, the answer reaches the user
    unchanged AND no `verification` event fires (we only emit on
    unverified-detection). Regression guard against the check
    rewriting clean answers."""
    monkeypatch.setenv("LOCALLLM_AGENT_VERIFY_MODE", "strict")
    client = FakeChatClient([
        {
            "role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": "web_search",
                                          "arguments": {"query": "x"}}}],
        },
        {
            "role": "assistant",
            "content": "Patched in CVE-2024-1 per https://real.com/article.",
        },
    ])
    fake_search = {"results": [
        {"url": "https://real.com/article", "snippet": "CVE-2024-1 patched"},
    ]}
    events: list[tuple[str, dict]] = []
    with patch("app.agent.loop._execute_tool", return_value=fake_search):
        answer = run_agent(
            "x", chat_client=client, model="granite",
            on_event=lambda n, p: events.append((n, p)),
        )
    assert "CVE-2024-1" in answer
    assert "real.com/article" in answer
    assert "rejected" not in answer.lower()
    # No verification event when nothing is unverified.
    assert not [p for n, p in events if n == "verification"]
