"""Unit tests for the agent's web tools — date-sniffing logic.

Network-touching paths (DDG search, real http_fetch) are not exercised
here; we test the pure helpers that decide *when something was
published*. Recency-discipline lives in the system prompt, but it can
only work if the tool layer surfaces dates correctly."""
from __future__ import annotations

from app.agent.tools.web_search import (
    SearchResult,
    _sniff_published_at,
)


# ---------------------------------------------------------------------------
# URL-path date extraction
# ---------------------------------------------------------------------------

def test_sniff_full_date_from_url_path() -> None:
    """``/2026/03/19/`` is a very common WordPress / news pattern. We
    must extract the full ISO date so the model can apply month-level
    recency rules, not just year-level."""
    out = _sniff_published_at(
        "https://example.com/2026/03/19/some-article-slug",
        snippet="",
    )
    assert out == "2026-03-19"


def test_sniff_year_month_only_from_url_path() -> None:
    out = _sniff_published_at(
        "https://example.com/2025/07/article-name",
        snippet="",
    )
    assert out == "2025-07"


def test_sniff_year_only_from_url_path() -> None:
    out = _sniff_published_at(
        "https://example.com/blog/2023/some-post",
        snippet="",
    )
    assert out == "2023"


def test_sniff_dashed_date_in_url() -> None:
    """Some sites use ``2026-03-19-slug`` instead of slashes. Both
    forms appear in real DDG result URLs."""
    out = _sniff_published_at(
        "https://example.com/posts/2026-03-19-headline",
        snippet="",
    )
    assert out == "2026-03-19"


def test_sniff_does_not_match_random_4digit_in_path() -> None:
    """``/v1234/`` or ``/article-9999/`` must NOT match — those are
    not dates. The pattern requires a 20XX leading digit pair."""
    out = _sniff_published_at(
        "https://example.com/category/v1234/some-slug",
        snippet="",
    )
    assert out is None


def test_sniff_matches_first_date_only() -> None:
    """``/2024/03/foo/2025/`` — multiple dates in path. We keep the
    first match, which on ordered URL schemes is the publication
    date (later path components are usually slugs / categories)."""
    out = _sniff_published_at(
        "https://example.com/2024/03/19/section/2025/slug",
        snippet="",
    )
    assert out == "2024-03-19"


# ---------------------------------------------------------------------------
# Snippet date fallback
# ---------------------------------------------------------------------------

def test_sniff_full_month_year_from_snippet() -> None:
    """Many news sites omit dates from URLs but include them in the
    DDG-extracted snippet. ``Mar 19, 2026`` is the most common
    machine-readable shape."""
    out = _sniff_published_at(
        "https://example.com/no-date-in-url",
        snippet="Mar 19, 2026 — Apple released iOS …",
    )
    assert out == "2026"


def test_sniff_full_month_word_year_from_snippet() -> None:
    out = _sniff_published_at(
        "https://example.com/no-date-in-url",
        snippet="Posted on March 2026 by …",
    )
    assert out == "2026"


def test_sniff_bare_year_in_short_snippet() -> None:
    """Last-resort fallback — bare year in the first 120 chars of
    the snippet. Less reliable than the month-anchored form because
    "2023" might be a CVE year, but better than no date hint at
    all when we have nothing else."""
    out = _sniff_published_at(
        "https://example.com/no-date",
        snippet="A 2023 retrospective on the patch.",
    )
    assert out == "2023"


def test_sniff_url_takes_precedence_over_snippet() -> None:
    """When both URL and snippet have dates, the URL wins — it's
    almost always the publication-time slug, while snippets often
    quote a date *from the article body* that may be unrelated."""
    out = _sniff_published_at(
        "https://example.com/2026/05/article",
        snippet="In 2019, the team was founded.",
    )
    assert out == "2026-05"


def test_sniff_returns_none_when_no_date_anywhere() -> None:
    out = _sniff_published_at(
        "https://example.com/page",
        snippet="No publication date here.",
    )
    assert out is None


def test_sniff_handles_ios_18_6_arstechnica_url() -> None:
    """Regression — the exact failing URL from the user-reported
    iOS-vulnerabilities run. ``/2025/07/`` must yield ``2025-07``
    so the recency-discipline rules can flag it as ~10 months old."""
    out = _sniff_published_at(
        "https://arstechnica.com/gadgets/2025/07/apple-releases-ios-18-6-macos-15-6-and-other-updates-as-current-gen-winds-down/",
        snippet="",
    )
    assert out == "2025-07"


def test_sniff_handles_appleinsider_short_year_path() -> None:
    """Some sites use 2-digit years in URL paths
    (``/articles/26/03/19/``). Today these are rare enough that we
    don't try to handle them — we'd risk treating ``/articles/99/``
    as 2099. Document the limitation: URL date-sniff returns None
    and we fall through to the snippet fallback."""
    out = _sniff_published_at(
        "https://appleinsider.com/articles/26/03/19/iphone-isnt-safe-on-old-ios-anymore",
        snippet="Older devices need to install iOS 15 or later",
    )
    # No valid date sniffed. The agent's recency-discipline rule says
    # to treat undated sources as potentially stale and search again
    # with a year qualifier — that's the desired fallback behavior.
    assert out is None


# ---------------------------------------------------------------------------
# SearchResult shape
# ---------------------------------------------------------------------------

def test_search_result_default_published_at_is_none() -> None:
    """Older code that constructs SearchResult without a
    published_at must keep working. None means "couldn't tell"."""
    r = SearchResult(title="x", url="https://x.com", snippet="y")
    assert r.published_at is None
