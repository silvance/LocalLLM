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


SYSTEM_PROMPT = """You are a research agent with access to web tools.

Tools available:
- web_search(query, max_results=5): search the web for candidate sources.
- http_fetch(url): download a page and extract its main text.

Strategy:
1. Plan what you need to know.
2. Use web_search with focused queries (not the entire user question verbatim).
3. Pick 2-3 of the most relevant results and http_fetch them.
4. Synthesize a complete, accurate answer that includes inline numeric
   citations like [1], [2] referencing the sources you actually used.
5. End your response with a "Sources:" section listing the URLs in order.

Constraints:
- At most ~5 tool calls. Do not loop on the same query.
- If a fetch fails, try a different URL — don't keep retrying the same one.
- Don't invent URLs. Only fetch URLs returned by web_search.
- If you already know the answer with confidence, skip the tools and answer.

CRITICAL — when tools return an error:
- Tool results that look like {"error": "..."} mean the tool DID NOT WORK
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
    system_content = compose_system_prompt(SYSTEM_PROMPT)
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
