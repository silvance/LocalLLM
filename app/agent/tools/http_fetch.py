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
    try:
        meta = trafilatura.extract_metadata(html)
        if meta and getattr(meta, "title", None):
            title = str(meta.title).strip()
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
    )
