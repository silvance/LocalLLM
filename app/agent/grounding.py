"""Post-generation grounding check for the research agent.

The model can — and does — invent confident-looking specifics under
synthesis pressure: fake CVE IDs, fake source URLs, plausible-looking
hostnames that don't exist. Three layers of system prompt didn't stop
this in practice (long-context drift, helpfulness pressure, no
verification step). The fix isn't more prompting — it's a determ-
inistic check that runs AFTER the model has produced its answer
and BEFORE the answer reaches the user.

Scope:
- URLs cited in the answer must equal-or-subpath-of a URL we
  actually fetched, OR appear in some web_search result.
- CVE-YYYY-NNNN strings cited in the answer must appear literally
  in some tool-result body (the actual page text we fetched).

Out of scope (deliberately, for now):
- Free-text product / version names. Too many false positives —
  "Python 3" appears in lots of pages but means different things
  in different contexts.
- Semantic verification ("does this URL actually contain what the
  model claims?"). That requires another model call and is its
  own can of worms.

Modes (operator-configurable via ``LOCALLLM_AGENT_VERIFY_MODE``):
- ``strict`` (default) — replace the answer with a refusal that
  names the unverified claims.
- ``annotate`` — keep the answer but tag each unverified claim
  inline as ``[unverified]``.
- ``off`` — skip the check (debug only; never default).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field


# CVE IDs follow CVE-YYYY-NNNN where N is 4-7 digits. Real-world
# CVEs go to 7 digits in 2026. Capture the canonical form so the
# substring check works against fetched page text.
_CVE_RE = re.compile(r"\bCVE-(?:19|20)\d{2}-\d{4,7}\b", re.IGNORECASE)

# URLs in markdown can wear several costumes:
#   1. Bare:        https://foo.com/bar
#   2. Markdown:    [text](https://foo.com/bar)
#   3. Angle:       <https://foo.com/bar>
#   4. Reference:   [text][1]   (skipped — too rare to bother)
# Strip trailing punctuation that almost never belongs to the URL
# (commas, periods, parens, brackets, semis) — these come from the
# surrounding sentence, not the URL itself. Keep slashes / hashes /
# query strings since those ARE part of URLs.
_URL_RE = re.compile(
    r"https?://[^\s\)\]\>\<,;\"]+",
    re.IGNORECASE,
)


@dataclass
class ExtractedClaims:
    """Concrete factual claims pulled out of the agent's final answer
    for cross-checking against fetched sources."""
    urls: list[str] = field(default_factory=list)
    cves: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.urls and not self.cves


def extract_claims(text: str) -> ExtractedClaims:
    """Pull URLs + CVE IDs out of free-text. Both are normalized:
    URLs lowercased and stripped of trailing punctuation, CVE IDs
    upper-cased to canonical CVE-YYYY-NNNN. Returns deduped lists
    in first-occurrence order so cross-checking is deterministic."""
    if not text:
        return ExtractedClaims()

    urls_seen: dict[str, None] = {}
    for m in _URL_RE.finditer(text):
        url = _clean_url(m.group(0))
        if url and url not in urls_seen:
            urls_seen[url] = None

    cves_seen: dict[str, None] = {}
    for m in _CVE_RE.finditer(text):
        cve = m.group(0).upper()
        if cve not in cves_seen:
            cves_seen[cve] = None

    return ExtractedClaims(urls=list(urls_seen), cves=list(cves_seen))


def _clean_url(url: str) -> str:
    """Strip trailing punctuation that the URL regex didn't already
    catch, lowercase the scheme+host so case-insensitive matching
    works, but preserve path case (paths CAN be case-sensitive on
    some servers)."""
    url = url.rstrip(".,;:!?)\"'>]")
    # Lowercase the scheme://host portion only.
    m = re.match(r"^(https?://[^/]+)(.*)$", url, re.IGNORECASE)
    if not m:
        return url
    return m.group(1).lower() + m.group(2)


@dataclass
class VerifiedSources:
    """Authoritative set of URLs + raw body text the agent actually
    saw during this run. Used as the ground-truth corpus for
    verification."""
    # URLs returned by web_search tool calls. The agent is allowed
    # to cite these even if it didn't fetch them — they're part of
    # the search index, not invented.
    search_urls: set[str] = field(default_factory=set)
    # URLs successfully fetched via http_fetch — strongest signal,
    # since we have the body text behind these.
    fetched_urls: set[str] = field(default_factory=set)
    # Concatenated body text from every successful fetch + every
    # search-result snippet. Used for substring lookups (CVE IDs,
    # exact phrases). Bounded by virtue of httpx max-chars cap on
    # the fetch tool — already trimmed before it gets here.
    corpus_text: str = ""


def add_search_result(sources: VerifiedSources, result: dict) -> None:
    """Fold a web_search tool result into the verified-sources set."""
    for entry in (result.get("results") or []):
        url = _clean_url(str(entry.get("url") or ""))
        if url:
            sources.search_urls.add(url)
        snippet = str(entry.get("snippet") or "")
        if snippet:
            sources.corpus_text += "\n" + snippet


def add_fetch_result(sources: VerifiedSources, result: dict) -> None:
    """Fold an http_fetch tool result into the verified-sources set.
    Trafilatura-extracted text and the URL itself become ground
    truth — these are the strongest verification signal we have."""
    url = _clean_url(str(result.get("url") or ""))
    if url:
        sources.fetched_urls.add(url)
        # Fetched URLs implicitly count as "search-found" too — the
        # agent might have arrived at them via a different search.
        sources.search_urls.add(url)
    text = str(result.get("text") or "")
    if text:
        sources.corpus_text += "\n" + text


def _url_matches(claimed: str, sources: VerifiedSources) -> bool:
    """A claimed URL is verified if it's an exact match with any
    search-or-fetched URL, or if it's a same-page extension (#frag,
    ?query, or a deeper /sub-path) of a fetched URL. Hostname-only
    matching is intentionally NOT enough — the model could invent a
    deep path on a real site, which is a known failure mode."""
    # Normalize claimed URL (lowercase host, strip trailing punct)
    # so case differences and stray punctuation don't false-fail.
    # The verified-source set is already normalized at insert time.
    norm = _clean_url(claimed)
    if norm in sources.search_urls or norm in sources.fetched_urls:
        return True
    for fetched in sources.fetched_urls:
        base = fetched.rstrip("/")
        # Same-page extensions: deeper path (/sub), fragment (#anchor),
        # or query (?param). Anything else with the same host+path
        # prefix would just be a different page.
        if norm == base:
            return True
        if norm.startswith(base):
            tail = norm[len(base):]
            if tail and tail[0] in ("/", "#", "?"):
                return True
    return False


def _cve_matches(cve: str, sources: VerifiedSources) -> bool:
    """A CVE ID is verified if it appears literally in some fetched
    body or search snippet. Case-insensitive (CVE comes in mixed
    case in URLs / titles); the corpus is lowercased on-the-fly
    rather than at ingest time so we don't double the memory."""
    needle = cve.lower()
    return needle in sources.corpus_text.lower()


@dataclass
class VerificationResult:
    """Outcome of cross-checking an extracted set of claims against
    the verified-source corpus."""
    unverified_urls: list[str] = field(default_factory=list)
    unverified_cves: list[str] = field(default_factory=list)
    verified_urls: list[str] = field(default_factory=list)
    verified_cves: list[str] = field(default_factory=list)

    @property
    def has_unverified(self) -> bool:
        return bool(self.unverified_urls or self.unverified_cves)

    @property
    def total_unverified(self) -> int:
        return len(self.unverified_urls) + len(self.unverified_cves)


def verify_claims(claims: ExtractedClaims, sources: VerifiedSources) -> VerificationResult:
    """Cross-check extracted claims against the verified-source set.
    Returns lists of verified + unverified items so callers can
    branch on either set (refuse on unverified, surface verified
    inline, etc.)."""
    out = VerificationResult()
    for url in claims.urls:
        if _url_matches(url, sources):
            out.verified_urls.append(url)
        else:
            out.unverified_urls.append(url)
    for cve in claims.cves:
        if _cve_matches(cve, sources):
            out.verified_cves.append(cve)
        else:
            out.unverified_cves.append(cve)
    return out


# ---------------------------------------------------------------------------
# Mode selection + answer rewriting
# ---------------------------------------------------------------------------


_VALID_MODES = ("strict", "annotate", "off")


def get_verify_mode() -> str:
    """Operator-controlled mode via env var. Default ``strict`` —
    refuse rather than serve confident-looking fabrication.

    Anything unrecognized falls back to strict; we'd rather over-
    protect than silently disable the check on a typo."""
    raw = (os.getenv("LOCALLLM_AGENT_VERIFY_MODE") or "").strip().lower()
    if raw in _VALID_MODES:
        return raw
    return "strict"


def apply_verification(
    answer: str,
    sources: VerifiedSources,
) -> tuple[str, VerificationResult]:
    """Run the grounding check on ``answer`` and return ``(possibly-
    rewritten-answer, verification-result)``. The result is always
    returned so callers can log / emit events even when the answer
    text wasn't modified.

    Behavior by mode:
      - ``off`` — return ``(answer, result)`` unchanged.
      - ``annotate`` — return the answer with each unverified URL /
        CVE marked ``[unverified]``. The original prose is preserved
        so the operator can read what the model produced.
      - ``strict`` — if any unverified claim, REPLACE the answer
        with a refusal message naming the bad claims. Verified
        claims (if any) are listed at the end so the operator can
        decide whether to manually salvage the verified portion.
    """
    mode = get_verify_mode()
    claims = extract_claims(answer)
    result = verify_claims(claims, sources)

    if mode == "off" or not result.has_unverified:
        return answer, result

    if mode == "annotate":
        return _annotate_answer(answer, result), result

    # strict (default) — refuse.
    return _refusal_message(result), result


def _annotate_answer(answer: str, result: VerificationResult) -> str:
    """Mark each unverified URL / CVE with `[unverified]` inline so
    the operator can scan the answer for tags. Replace ALL
    occurrences (the model often cites the same URL multiple times
    in body + sources block)."""
    out = answer
    # Tag CVEs first — URL-replace is more aggressive about boundaries
    # and could swallow part of a CVE ID embedded in a URL.
    for cve in result.unverified_cves:
        # Word-boundary regex so "CVE-2025-1234" doesn't match a
        # longer substring; case-insensitive because CVEs come in
        # mixed case.
        out = re.sub(
            rf"\b{re.escape(cve)}\b",
            f"{cve} [unverified]",
            out,
            flags=re.IGNORECASE,
        )
    for url in result.unverified_urls:
        # Exact-string replace (URLs are case-folded on host but
        # path-case-preserved). Multiple occurrences all get tagged.
        out = out.replace(url, f"{url} [unverified]")
    return out


def _refusal_message(result: VerificationResult) -> str:
    """Replace the answer with an operator-facing refusal that
    names the unverified items and explains what to do next.

    Verified items (if any) are listed too — sometimes the answer
    was 90% sound and the operator can salvage the verified portion
    by re-running with a constrained query."""
    lines = [
        "⚠ **Agent answer rejected: contains citations that could not "
        "be verified against fetched sources.**",
        "",
        "The model produced the following claims that don't appear in "
        "any URL we actually visited or any text we actually fetched. "
        "Treating them as fabricated rather than relaying them.",
        "",
    ]
    if result.unverified_cves:
        lines.append("**Unverified CVE IDs:**")
        for cve in result.unverified_cves[:20]:
            lines.append(f"- `{cve}` — not present in any fetched page text")
        lines.append("")
    if result.unverified_urls:
        lines.append("**Unverified URLs:**")
        for url in result.unverified_urls[:20]:
            lines.append(f"- `{url}` — not in any search result or fetched page")
        lines.append("")
    if result.verified_cves or result.verified_urls:
        lines.append(
            "Some claims WERE verifiable; the agent's answer was "
            "partially grounded. Verified items, for reference:"
        )
        for cve in result.verified_cves[:10]:
            lines.append(f"- CVE: `{cve}`")
        for url in result.verified_urls[:10]:
            lines.append(f"- URL: `{url}`")
        lines.append("")
    lines.append(
        "**To proceed:** re-run the query (the agent may pick "
        "different sources this time), use a more capable agent "
        "model, or set `LOCALLLM_AGENT_VERIFY_MODE=annotate` to "
        "see the original answer with `[unverified]` tags inline."
    )
    return "\n".join(lines)
