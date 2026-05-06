"""DuckDuckGo search via the `ddgs` package — no API key required.

DDG rate-limits aggressive scraping. If you hit captchas, throttle the
agent's max iterations or swap to a self-hosted SearXNG instance.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass


logger = logging.getLogger("localllm")


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


def web_search(query: str, max_results: int = 5) -> list[SearchResult]:
    if not query.strip():
        return []
    try:
        from ddgs import DDGS
    except ImportError as exc:
        raise RuntimeError(
            "ddgs is not installed; run `pip install -r requirements-agent.txt`"
        ) from exc

    out: list[SearchResult] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max(1, min(max_results, 20))):
                out.append(SearchResult(
                    title=str(r.get("title") or r.get("name") or "").strip(),
                    url=str(r.get("href") or r.get("link") or r.get("url") or "").strip(),
                    snippet=str(r.get("body") or r.get("snippet") or "").strip(),
                ))
    except Exception as exc:
        logger.exception("DuckDuckGo search failed")
        raise RuntimeError(f"DuckDuckGo search failed: {exc}") from exc
    return out
