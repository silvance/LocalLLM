"""DuckDuckGo search via the `ddgs` package — no API key required.

DDG rate-limits aggressive scraping. If you hit captchas, throttle the
agent's max iterations or swap to a self-hosted SearXNG instance.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass


logger = logging.getLogger("localllm")


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    # Best-effort publication date sniffed from the URL path or
    # snippet ("2026-03-19", "2025-07", or just "2023"). None means
    # we couldn't tell — the agent must treat the source as
    # potentially-stale until http_fetch surfaces a real date.
    published_at: str | None = None


# URL-path date patterns. Order matters — most specific first so we
# don't strip "2026" off "2026/03/19" before matching the day.
_URL_DATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"/(20\d{2})[/-](\d{1,2})[/-](\d{1,2})(?:[/-]|$)"),
    re.compile(r"/(20\d{2})[/-](\d{1,2})(?:[/-]|$)"),
    re.compile(r"/(20\d{2})(?:[/-]|$)"),
)

# Free-text date in snippets — "Mar 19, 2026", "March 2026", "2026".
_SNIPPET_DATE_RE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
    r"(?:\.?\s+\d{1,2},?)?\s+(20\d{2})\b",
    re.IGNORECASE,
)
_BARE_YEAR_RE = re.compile(r"\b(20\d{2})\b")


def _sniff_published_at(url: str, snippet: str) -> str | None:
    """Best-effort publication date for a search hit.

    Returns ISO-ish date strings — full ``YYYY-MM-DD`` when the URL
    path has it, ``YYYY-MM`` for month-only paths, or ``YYYY`` as a
    last resort. The agent uses this for recency triage; precision
    isn't critical, but staleness IS, so we prefer "year only" over
    "no date." Calling code should still treat the value as a hint
    and verify via ``http_fetch`` for authoritative claims."""
    for pat in _URL_DATE_PATTERNS:
        m = pat.search(url)
        if not m:
            continue
        groups = m.groups()
        if len(groups) == 3:
            y, mo, d = groups
            return f"{y}-{int(mo):02d}-{int(d):02d}"
        if len(groups) == 2:
            y, mo = groups
            return f"{y}-{int(mo):02d}"
        return groups[0]

    # Snippet fallback. Prefer "Month YYYY"-shaped dates over a bare
    # year because publication dates more often appear in that
    # format; bare years could be referencing a CVE or version.
    m2 = _SNIPPET_DATE_RE.search(snippet)
    if m2:
        return m2.group(1)
    m3 = _BARE_YEAR_RE.search(snippet[:120])
    if m3:
        return m3.group(1)
    return None


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
                url = str(r.get("href") or r.get("link") or r.get("url") or "").strip()
                snippet = str(r.get("body") or r.get("snippet") or "").strip()
                out.append(SearchResult(
                    title=str(r.get("title") or r.get("name") or "").strip(),
                    url=url,
                    snippet=snippet,
                    published_at=_sniff_published_at(url, snippet),
                ))
    except Exception as exc:
        logger.exception("DuckDuckGo search failed")
        raise RuntimeError(f"DuckDuckGo search failed: {exc}") from exc
    return out
