"""Agent loop driven by Ollama's native function calling.

Each iteration: model gets the conversation + tool catalog, emits either
a final answer (no tool_calls) or one or more tool calls. We execute the
tools, append results as 'tool' messages, and loop until either a
final answer or the iteration cap.

Designed for a "research and aggregate" workflow — search the web,
fetch a few of the top results, synthesise an answer with citations.
Defaults to ~6 iterations because longer loops tend to wander on
small models.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict
from typing import Any


logger = logging.getLogger("localllm")


TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web with DuckDuckGo. Returns top results with "
                "title, URL, and snippet. Use to discover candidate sources."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query, plain natural language",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (1-20). Default 5.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "http_fetch",
            "description": (
                "Fetch a URL and extract its main text (boilerplate stripped). "
                "Use AFTER web_search to read source content. Capped to 8000 chars."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Absolute URL to fetch"},
                },
                "required": ["url"],
            },
        },
    },
]


_SYSTEM_PROMPT_TEMPLATE = """You are a research agent with access to web tools.

Today's date is {today}. Your training data has a cutoff well before that;
treat the live web as authoritative on anything time-sensitive.

Tools available:
- web_search(query, max_results=5): search the web for candidate sources.
  Each result includes `published_at` when the publication date is
  detectable from the URL or snippet (may be a full YYYY-MM-DD, just
  YYYY-MM, or just YYYY; or null when undetectable).
- http_fetch(url): download a page and extract its main text. Returns
  a `published_at` field when the page advertises one in its metadata.

Strategy:
1. Plan what you need to know.
2. Use web_search with focused queries (not the entire user question verbatim).
3. Pick 2-3 of the most relevant results and http_fetch them.
4. Synthesize a complete, accurate answer that includes inline numeric
   citations like [1], [2] referencing the sources you actually used.
5. End your response with a "Sources:" section listing the URLs **with
   their publication dates** when you have them — e.g. `[1] https://… (2026-03-19)`.

RECENCY DISCIPLINE (mandatory for any "current / latest / known /
recent / today / now" question, or anything mentioning a software
version that ships frequently):
- Today is {today}. The current calendar year is {year}.
- Before citing a source, check its `published_at`. Anything older
  than ~12 months for a recency-sensitive question is potentially
  STALE and must NOT be presented as describing the current state.
- If your top search hits are all stale, run the search AGAIN with a
  year qualifier (e.g. append "{year}" or "{prev_year}") or a more
  specific query. Don't settle for the first plausible-sounding hit.
- If after a follow-up search you still only have stale data, say so
  explicitly in your answer ("I could only find sources from
  YYYY-MM-DD or earlier; the current state may have changed since")
  rather than presenting old facts as current.
- When you DO cite a source, name its date in the prose: "As of
  YYYY-MM-DD, …" or "The most recent source I could find (dated
  YYYY-MM-DD) reports …". Never imply currency you can't verify.
- Concrete examples of stale-source failures to avoid:
  * Citing a 3-year-old article about iOS 16.4 in answer to "what's
    the current iOS version?".
  * Quoting CVE numbers from a 2023 patch round when asked about
    "current" vulnerabilities for software released in 2026.
  * Treating "latest version" claims in old articles as still true.

Constraints:
- At most ~5 tool calls. Do not loop on the same query.
- If a fetch fails, try a different URL — don't keep retrying the same one.
- Don't invent URLs. Only fetch URLs returned by web_search.
- If you already know the answer with confidence AND the question is
  not recency-sensitive, you may skip the tools and answer. For
  anything time-sensitive, search the web — your training data is
  too stale to trust.

CRITICAL — when tools return an error:
- Tool results that look like {{"error": "..."}} mean the tool DID NOT WORK
  and produced NO data. You have nothing to cite from that call.
- Do NOT fabricate concrete facts (CVE IDs, version numbers, dates, names,
  URLs, statistics, quotes) to fill the gap. Inventing specifics is worse
  than admitting you couldn't access the data.
- If every tool call you've made has returned an error, your final answer
  MUST be a short report that states (a) what you tried, (b) the exact
  error message you got, and (c) what the operator should do to fix it
  (usually quoted verbatim from the error). Then stop.
- If only some tool calls failed but at least one returned real data, you
  may answer using that data — but only cite sources you actually fetched.
"""


def _build_system_prompt(today_iso: str | None = None) -> str:
    """Render SYSTEM_PROMPT with today's date filled in. Pulled out as
    a function so tests can pin the date and so the prompt always
    reflects the actual day the agent runs (instead of the day the
    process started — relevant for long-lived servers)."""
    from datetime import date
    today = date.fromisoformat(today_iso) if today_iso else date.today()
    return _SYSTEM_PROMPT_TEMPLATE.format(
        today=today.isoformat(),
        year=today.year,
        prev_year=today.year - 1,
    )


# Back-compat name. Importers that grab `SYSTEM_PROMPT` directly get
# a snapshot rendered at import time; the runner re-renders on each
# call to avoid date drift on long-running processes.
SYSTEM_PROMPT = _build_system_prompt()


def _execute_tool(name: str, args: dict[str, Any]) -> dict:
    """Dispatch to the matching tool implementation. Returns a
    JSON-serializable dict (which gets fed back to the model as the
    'tool' message content)."""
    if name == "web_search":
        from app.agent.tools.web_search import web_search
        results = web_search(
            query=str(args.get("query", "")),
            max_results=int(args.get("max_results", 5)),
        )
        return {"results": [asdict(r) for r in results]}

    if name == "http_fetch":
        from app.agent.tools.http_fetch import http_fetch
        return asdict(http_fetch(url=str(args.get("url", ""))))

    return {"error": f"unknown tool: {name}"}


def _coerce_args(raw: Any) -> dict:
    """Ollama may return tool arguments as a dict OR a JSON string,
    depending on the model. Normalize to dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def run_agent(
    user_prompt: str,
    *,
    chat_client,
    model: str,
    max_iterations: int = 6,
    on_event: Callable[[str, dict], None] | None = None,
) -> str:
    """Run the agent loop synchronously. Returns the final answer.

    `on_event(name, payload)` is called for each step:
        ("step_start",   {"iteration": N})
        ("model_text",   {"content": "..."})         -- assistant prose
        ("tool_call",    {"name": "...", "args": {...}})
        ("tool_result",  {"name": "...", "result": {...}})
        ("tool_error",   {"name": "...", "error": "..."})
        ("done",         {"iterations": N, "answer": "..."})

    Caller is responsible for emitting these events to the UI (we keep
    the loop transport-agnostic).
    """
    def emit(name: str, payload: dict) -> None:
        if on_event is not None:
            try:
                on_event(name, payload)
            except Exception:
                logger.exception("on_event %s callback raised", name)

    # Compose the agent's task-specific prompt with the always-on
    # anti-hallucination baseline. Same composition the chat/review/
    # compare paths use, just with the agent's research-loop guidance
    # as the "task-specific" layer.
    from app.utils.system_prompt import compose as compose_system_prompt
    # Re-render so ``today`` reflects the calendar day the agent
    # actually runs, not the day the server started. The recency-
    # discipline section is date-templated, so an ssh'd box that's
    # been up for weeks would otherwise tell the model the wrong
    # current date.
    system_content = compose_system_prompt(_build_system_prompt())
    messages: list[dict] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_prompt},
    ]

    final_answer = ""

    iteration = 0
    for iteration in range(max_iterations):
        emit("step_start", {"iteration": iteration})
        try:
            resp = chat_client.chat(
                model=model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                stream=False,
            )
        except Exception as exc:
            logger.exception("Agent model call failed")
            emit("error", {"error": str(exc)})
            break

        msg = dict(resp.get("message") or {})
        messages.append(msg)

        text = (msg.get("content") or "").strip()
        if text:
            emit("model_text", {"content": text})

        tool_calls = list(msg.get("tool_calls") or [])
        if not tool_calls:
            final_answer = text
            break

        for tc in tool_calls:
            fn = tc.get("function") or {}
            name = str(fn.get("name") or "")
            args = _coerce_args(fn.get("arguments"))
            emit("tool_call", {"name": name, "args": args})

            try:
                result = _execute_tool(name, args)
                messages.append({
                    "role": "tool",
                    "content": json.dumps(result, ensure_ascii=False),
                })
                emit("tool_result", {"name": name, "result": result})
            except Exception as exc:
                err = str(exc)
                logger.exception("Tool %s failed", name)
                messages.append({
                    "role": "tool",
                    "content": json.dumps({"error": err}, ensure_ascii=False),
                })
                emit("tool_error", {"name": name, "error": err})
    else:
        # Loop hit max_iterations without a final answer
        emit("error", {"error": f"agent stopped after {max_iterations} iterations without a final answer"})

    emit("done", {"iterations": iteration + 1, "answer": final_answer})
    return final_answer
