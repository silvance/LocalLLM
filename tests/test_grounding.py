"""Tests for the agent's post-generation grounding check.

The model invents confident-looking specifics (CVE IDs, source URLs)
under synthesis pressure even with three layers of system-prompt
guarding. The grounding check is the deterministic backstop. These
tests pin its behavior so a future "let's relax the regex" change
doesn't quietly let fabrication through.
"""
from __future__ import annotations

import pytest

from app.agent.grounding import (
    ExtractedClaims,
    VerifiedSources,
    add_fetch_result,
    add_search_result,
    apply_verification,
    extract_claims,
    get_verify_mode,
    verify_claims,
)


# ---------------------------------------------------------------------------
# extract_claims
# ---------------------------------------------------------------------------

def test_extract_bare_url() -> None:
    out = extract_claims("See https://example.com/path for details.")
    assert out.urls == ["https://example.com/path"]


def test_extract_url_strips_trailing_punctuation() -> None:
    """Trailing comma / period / paren almost never belongs to the
    URL — they come from the surrounding sentence. Stripping them
    avoids false 'unverified' hits when the operator wrote
    `https://foo.com/bar,` and we already verified `https://foo.com/bar`."""
    text = "See https://foo.com/bar, https://baz.com/qux. and https://x.org/y)."
    urls = extract_claims(text).urls
    assert "https://foo.com/bar" in urls
    assert "https://baz.com/qux" in urls
    assert "https://x.org/y" in urls


def test_extract_url_in_markdown_link() -> None:
    """Markdown link form `[text](url)` is the most common citation
    style in agent answers — must be extracted, not skipped."""
    text = "See [the article](https://example.com/page) for context."
    assert "https://example.com/page" in extract_claims(text).urls


def test_extract_cve_canonical_form() -> None:
    out = extract_claims("Patched in CVE-2024-1234 and cve-2025-99999.")
    # All CVEs uppercased to canonical form for stable matching.
    assert "CVE-2024-1234" in out.cves
    assert "CVE-2025-99999" in out.cves


def test_extract_cve_seven_digit() -> None:
    """2026-era CVEs go to 7 digits — the regex must accept them."""
    out = extract_claims("CVE-2026-1234567 was announced.")
    assert out.cves == ["CVE-2026-1234567"]


def test_extract_cve_rejects_too_few_digits() -> None:
    """3-digit suffix isn't a real CVE; don't match on
    `CVE-2024-123` to avoid false positives on truncated quotes."""
    out = extract_claims("Earlier we discussed CVE-2024-123 (truncated).")
    assert out.cves == []


def test_extract_dedupes_in_first_occurrence_order() -> None:
    text = (
        "First CVE-2024-1 (wait, too short) — sorry, CVE-2024-12345. "
        "Then https://foo.com/a appears, then https://foo.com/a again, "
        "then CVE-2024-12345 again, then CVE-2025-67890."
    )
    out = extract_claims(text)
    assert out.cves == ["CVE-2024-12345", "CVE-2025-67890"]
    assert out.urls == ["https://foo.com/a"]


def test_extract_returns_empty_on_blank() -> None:
    assert extract_claims("").is_empty
    assert extract_claims("   \n").is_empty


# ---------------------------------------------------------------------------
# Verification — URLs
# ---------------------------------------------------------------------------

def test_url_verified_by_exact_search_hit() -> None:
    sources = VerifiedSources()
    add_search_result(sources, {"results": [{"url": "https://foo.com/a", "snippet": ""}]})
    claims = ExtractedClaims(urls=["https://foo.com/a"])
    out = verify_claims(claims, sources)
    assert out.verified_urls == ["https://foo.com/a"]
    assert out.unverified_urls == []


def test_url_verified_by_subpath_of_fetched() -> None:
    """Once we've fetched https://foo.com/article, the model citing
    https://foo.com/article#section or .../article/comments should
    be accepted — same host + base path means the agent isn't
    inventing a deep page on a real site (which would be the next
    failure mode)."""
    sources = VerifiedSources()
    add_fetch_result(sources, {
        "url": "https://foo.com/article",
        "text": "body text",
    })
    claims = ExtractedClaims(urls=[
        "https://foo.com/article#section",
        "https://foo.com/article/comments",
        "https://foo.com/article",
    ])
    out = verify_claims(claims, sources)
    assert len(out.verified_urls) == 3
    assert out.unverified_urls == []


def test_url_unverified_when_only_host_matches() -> None:
    """The model citing https://foo.com/totally/made/up/path when we
    only fetched https://foo.com/something must NOT verify — host-
    only matching would let the model invent deep paths on real
    sites. This is the failure mode beyond the basic 'fake hostname'
    case."""
    sources = VerifiedSources()
    add_fetch_result(sources, {"url": "https://foo.com/something", "text": ""})
    claims = ExtractedClaims(urls=["https://foo.com/totally/made/up"])
    out = verify_claims(claims, sources)
    assert out.unverified_urls == ["https://foo.com/totally/made/up"]


def test_url_match_is_case_insensitive_on_host() -> None:
    """Hostnames are case-insensitive in DNS; the model writing
    https://Foo.COM/page and us having fetched https://foo.com/page
    must match. The verified-list keeps the model's original
    casing (so the operator can correlate with the answer text)
    but the lookup itself is case-insensitive on host. Path case
    preserved (some servers care)."""
    sources = VerifiedSources()
    add_fetch_result(sources, {"url": "https://foo.com/Page", "text": ""})
    claimed = "https://Foo.COM/Page"
    claims = ExtractedClaims(urls=[claimed])
    out = verify_claims(claims, sources)
    assert claimed in out.verified_urls
    assert out.unverified_urls == []


# ---------------------------------------------------------------------------
# Verification — CVEs
# ---------------------------------------------------------------------------

def test_cve_verified_by_appearance_in_corpus() -> None:
    sources = VerifiedSources()
    add_fetch_result(sources, {
        "url": "https://example.com/sec",
        "text": "Apple patched CVE-2023-43010 in iOS 17.2.",
    })
    claims = ExtractedClaims(cves=["CVE-2023-43010"])
    out = verify_claims(claims, sources)
    assert out.verified_cves == ["CVE-2023-43010"]


def test_cve_verified_by_search_snippet() -> None:
    """Snippet-only matches count too — sometimes the search result
    body has the CVE without us having to fetch the page."""
    sources = VerifiedSources()
    add_search_result(sources, {"results": [
        {"url": "https://nvd.nist.gov/vuln/detail/CVE-2024-99",
         "snippet": "CVE-2024-99 is a heap overflow in libfoo."},
    ]})
    claims = ExtractedClaims(cves=["CVE-2024-99"])
    out = verify_claims(claims, sources)
    assert out.verified_cves == ["CVE-2024-99"]


def test_cve_unverified_when_not_in_any_source() -> None:
    """The exact failure from the user-reported iOS run — model
    invents CVE-2025-43529 with no source backing it."""
    sources = VerifiedSources()
    add_fetch_result(sources, {
        "url": "https://example.com/sec",
        "text": "Talks about CVE-2023-43010.",
    })
    claims = ExtractedClaims(cves=["CVE-2025-43529"])
    out = verify_claims(claims, sources)
    assert out.unverified_cves == ["CVE-2025-43529"]


def test_cve_match_is_case_insensitive() -> None:
    """The model often writes CVEs in lowercase even though canonical
    is uppercase. Our extract_claims canonicalizes to upper, but
    the corpus might also have it in lowercase or mixed."""
    sources = VerifiedSources()
    add_fetch_result(sources, {"url": "https://x.com",
                               "text": "see cve-2024-12345 for details"})
    claims = ExtractedClaims(cves=["CVE-2024-12345"])
    out = verify_claims(claims, sources)
    assert out.verified_cves == ["CVE-2024-12345"]


# ---------------------------------------------------------------------------
# apply_verification — the operator-facing rewrite path
# ---------------------------------------------------------------------------

def test_apply_strict_replaces_answer_when_unverified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The headline regression — iOS-vulnerabilities run produced
    CVE-2025-43529 with a fake source URL. Strict mode must
    REPLACE the answer with a refusal so the operator can't trust
    confident-looking fabrication."""
    monkeypatch.setenv("LOCALLLM_AGENT_VERIFY_MODE", "strict")
    sources = VerifiedSources()
    add_search_result(sources, {"results": [
        {"url": "https://thehackernews.com/2026/03/apple.html", "snippet": ""}
    ]})
    add_fetch_result(sources, {
        "url": "https://thehackernews.com/2026/03/apple.html",
        "text": "Patched CVE-2023-43010 in iOS 17.2.",
    })
    # Mix of verified and fabricated claims — mirrors the actual
    # iOS-vulnerabilities answer the user reported (real source +
    # real CVE alongside fabricated ones).
    fake_answer = (
        "Apple patched CVE-2023-43010 in iOS 17.2 per "
        "https://thehackernews.com/2026/03/apple.html. "
        "Current iOS vulnerabilities also include CVE-2025-43529 "
        "and CVE-2025-20253, documented at "
        "https://cvemon.intruder.io/cves/CVE-2025-43529."
    )
    new_answer, result = apply_verification(fake_answer, sources)
    assert new_answer != fake_answer, "strict mode must rewrite"
    assert "rejected" in new_answer.lower() or "unverified" in new_answer.lower()
    # Refusal must NAME the unverified items so the operator can see what failed.
    assert "CVE-2025-43529" in new_answer
    assert "CVE-2025-20253" in new_answer
    assert "cvemon.intruder.io" in new_answer
    # Verified items also surfaced for context — gives the operator
    # a starting point if they want to manually salvage the good parts.
    assert "CVE-2023-43010" in new_answer
    assert "thehackernews.com" in new_answer
    assert result.has_unverified


def test_apply_strict_passes_through_when_all_verified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If every claim is verified, the answer must pass through
    unchanged — no surprise rewrites for clean runs."""
    monkeypatch.setenv("LOCALLLM_AGENT_VERIFY_MODE", "strict")
    sources = VerifiedSources()
    add_fetch_result(sources, {
        "url": "https://example.com/page",
        "text": "CVE-2024-12345 details.",
    })
    answer = "See CVE-2024-12345 documented at https://example.com/page."
    new_answer, result = apply_verification(answer, sources)
    assert new_answer == answer
    assert not result.has_unverified


def test_apply_annotate_tags_unverified_inline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Annotate mode preserves the original prose but tags each
    unverified claim. Operator can then read the answer + decide
    what to trust per-claim, instead of getting a refusal."""
    monkeypatch.setenv("LOCALLLM_AGENT_VERIFY_MODE", "annotate")
    sources = VerifiedSources()
    add_fetch_result(sources, {"url": "https://real.com/page",
                               "text": "CVE-2024-1 is real."})
    answer = (
        "Real CVE-2024-1 at https://real.com/page. "
        "Fake CVE-2099-9999 at https://fake.com/x."
    )
    new_answer, result = apply_verification(answer, sources)
    assert "CVE-2099-9999 [unverified]" in new_answer
    assert "https://fake.com/x [unverified]" in new_answer
    # Verified items NOT tagged.
    assert "CVE-2024-1 [unverified]" not in new_answer
    assert "https://real.com/page [unverified]" not in new_answer
    assert result.has_unverified


def test_apply_off_mode_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """Off mode must be a true no-op — the answer comes back
    unchanged regardless of unverified claims. Used only for
    debugging; never the default."""
    monkeypatch.setenv("LOCALLLM_AGENT_VERIFY_MODE", "off")
    sources = VerifiedSources()  # empty — every claim would be unverified
    answer = "CVE-2099-9999 at https://anywhere.com."
    new_answer, _result = apply_verification(answer, sources)
    assert new_answer == answer


def test_apply_invalid_mode_falls_back_to_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operator typo (LOCALLLM_AGENT_VERIFY_MODE=stict) must NOT
    silently disable the check. Better to over-protect than serve
    fabrication because of a missing letter."""
    monkeypatch.setenv("LOCALLLM_AGENT_VERIFY_MODE", "lenient")
    assert get_verify_mode() == "strict"
    sources = VerifiedSources()
    answer = "Fake CVE-2099-9999 at https://nowhere.com."
    new_answer, _ = apply_verification(answer, sources)
    assert new_answer != answer  # rewrote in strict mode


def test_apply_default_mode_is_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No env var set → strict. Most important behavior: the
    operator who never reads docs gets the safe default."""
    monkeypatch.delenv("LOCALLLM_AGENT_VERIFY_MODE", raising=False)
    assert get_verify_mode() == "strict"


def test_apply_handles_empty_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Edge case: agent's final_answer is empty (synthesis pass
    returned nothing). Verification must not crash — there are no
    claims to check, so the answer passes through unchanged."""
    monkeypatch.setenv("LOCALLLM_AGENT_VERIFY_MODE", "strict")
    sources = VerifiedSources()
    new_answer, result = apply_verification("", sources)
    assert new_answer == ""
    assert not result.has_unverified


# ---------------------------------------------------------------------------
# Sources accumulator wiring
# ---------------------------------------------------------------------------

def test_add_search_result_collects_urls_and_snippets() -> None:
    sources = VerifiedSources()
    add_search_result(sources, {"results": [
        {"url": "https://a.com/x", "snippet": "alpha snippet"},
        {"url": "https://b.com/y", "snippet": "beta snippet"},
    ]})
    assert "https://a.com/x" in sources.search_urls
    assert "https://b.com/y" in sources.search_urls
    assert "alpha snippet" in sources.corpus_text
    assert "beta snippet" in sources.corpus_text


def test_add_fetch_result_collects_url_and_body() -> None:
    sources = VerifiedSources()
    add_fetch_result(sources, {
        "url": "https://x.com/page",
        "text": "page body with CVE-2024-1 mentioned",
    })
    assert "https://x.com/page" in sources.fetched_urls
    # Fetched URLs also count as search-found (the agent might cite
    # them either way).
    assert "https://x.com/page" in sources.search_urls
    assert "CVE-2024-1" in sources.corpus_text


def test_add_handles_missing_fields_gracefully() -> None:
    """Tool results from a future schema change might omit fields
    we expect. Adding empty results must not crash — the corpus
    just doesn't grow."""
    sources = VerifiedSources()
    add_search_result(sources, {})
    add_search_result(sources, {"results": []})
    add_fetch_result(sources, {})
    assert not sources.search_urls
    assert not sources.fetched_urls
    assert not sources.corpus_text.strip()
