"""Fetch a URL and extract main-content text. Boilerplate stripped via
trafilatura, which handles SEO chrome / nav / cookie banners better than
naive html2text. Caps response size to keep the model's context manageable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass


logger = logging.getLogger("localllm")


@dataclass
class FetchResult:
    url: str
    title: str
    text: str
    status: int
    chars: int
    truncated: bool
    # Publication date the page advertises in its metadata, when
    # extractable. ``None`` means we couldn't tell — agent should
    # treat the page as undated rather than current. Authoritative
    # for recency triage compared to the URL-path sniff in
    # web_search; trafilatura reads <meta name="article:published_time">,
    # OpenGraph tags, JSON-LD, etc.
    published_at: str | None = None


_DEFAULT_MAX_CHARS = 8_000
_TIMEOUT_SECONDS = 15.0
_USER_AGENT = "LocalLLM-Agent/1.0 (research; +https://github.com/silvance/LocalLLM)"


def http_fetch(url: str, max_chars: int = _DEFAULT_MAX_CHARS) -> FetchResult:
    if not url.strip():
        raise ValueError("empty url")
    try:
        import httpx
        import trafilatura
    except ImportError as exc:
        raise RuntimeError(
            "httpx / trafilatura not installed; run "
            "`pip install -r requirements-agent.txt`"
        ) from exc

    try:
        with httpx.Client(timeout=_TIMEOUT_SECONDS, follow_redirects=True) as client:
            resp = client.get(url, headers={"User-Agent": _USER_AGENT})
    except Exception as exc:
        logger.exception("HTTP fetch failed for %s", url)
        raise RuntimeError(f"HTTP fetch failed: {exc}") from exc

    html = resp.text or ""
    extracted = trafilatura.extract(html, include_comments=False, include_tables=True)
    title = ""
    published_at: str | None = None
    try:
        meta = trafilatura.extract_metadata(html)
        if meta:
            if getattr(meta, "title", None):
                title = str(meta.title).strip()
            # trafilatura returns dates as YYYY-MM-DD strings (or None).
            # Surface this so the agent's recency discipline has something
            # authoritative to consult; previously the agent had no way
            # to know whether a fetched page was 3 years stale.
            d = getattr(meta, "date", None)
            if d:
                published_at = str(d).strip() or None
    except Exception:
        title = ""

    text = (extracted or "").strip()
    truncated = False
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = True

    if not text:
        text = "(no extractable text — page may be JS-rendered, paywalled, or blocked)"

    return FetchResult(
        url=url,
        title=title,
        text=text,
        status=int(resp.status_code),
        chars=len(text),
        truncated=truncated,
        published_at=published_at,
    )
