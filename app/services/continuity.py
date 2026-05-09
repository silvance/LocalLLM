"""Persistent-continuity primitives for the review loop.

Adapts three patterns from the "Persistent Continuity for Agentic
Systems" paper (May 2026) to LocalLLM's much smaller surface:

1. **Architectural Decision Records (ADRs).** Markdown artifacts under
   ``data/durable/`` with structured YAML frontmatter. Loaded at
   review time so the constrained-kickoff prompt is operator-editable
   instead of a hardcoded Python string.

2. **SITREPs.** End-of-session summaries written to
   ``data/proposed/`` for the operator to triage later. The paper
   calls this "propose, don't promote" — the SITREP is provisional
   until the operator moves it to ``durable/`` (which we also
   support at the path level — no special promotion API yet, just
   `mv`).

3. **Cross-session blocker pattern lookup.** Given a prompt hash,
   walk the outcomes log + saved review sessions to find blockers
   that recurred across prior runs of the same task. Used as opt-in
   "prior lessons" hints; never auto-injected.

Why no graph DB, no vector store: LocalLLM volume is tiny (dozens of
ADRs, hundreds of SITREPs over a project's lifetime). File-scan with
frontmatter is fast enough and stays inspectable — every byte is a
markdown file the operator can read. Matches the paper's section §4.1
non-choice for the same reasons.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import yaml


logger = logging.getLogger("localllm")


# ---------------------------------------------------------------------------
# Markdown frontmatter
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(?P<frontmatter>.*?)\n---\s*\n(?P<body>.*)\Z",
    re.DOTALL,
)


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split a markdown file into ``(frontmatter_dict, body)``.

    Returns ``({}, text)`` if no frontmatter block is present so the
    caller doesn't have to special-case files written without one.
    Raises ``ValueError`` if the frontmatter block is present but
    malformed — silently dropping bad metadata is worse than failing
    loudly when the operator hand-edits an ADR."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    raw = m.group("frontmatter")
    body = m.group("body")
    try:
        parsed = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"malformed frontmatter: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"frontmatter must be a YAML mapping, got {type(parsed).__name__}")
    return parsed, body


def write_with_frontmatter(path: Path, frontmatter: dict, body: str) -> None:
    """Atomically write a markdown file with YAML frontmatter. Used
    for SITREPs (we don't auto-write ADRs — those are operator-
    edited). Atomic via tmp-file + replace so a partially-written
    file can't poison the proposed/ queue if the host loses power
    mid-write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fm = yaml.safe_dump(frontmatter, sort_keys=False, default_flow_style=False).rstrip()
    payload = f"---\n{fm}\n---\n\n{body.rstrip()}\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# ADRs
# ---------------------------------------------------------------------------

# Statuses an ADR can carry. ``durable`` is the only one the writer
# loop loads — the others exist for the operator's audit trail.
_VALID_STATUSES: frozenset[str] = frozenset({
    "provisional", "durable", "superseded", "rejected", "flagged", "archived",
})


@dataclass(frozen=True)
class ADR:
    """An architectural decision record. Bi-temporal — knows both
    when the rule was true in the world (``valid_from`` / ``valid_to``)
    and when we recorded it (``transaction_time``). v1 doesn't query
    by these fields yet, but storing them now means a future operator
    asking "what did we decide about X in March?" has the data to
    reconstruct the timeline without a schema migration."""
    id: str
    title: str
    status: str
    body: str
    valid_from: str | None = None
    valid_to: str | None = None
    transaction_time: str | None = None
    supersedes: tuple[str, ...] = field(default_factory=tuple)
    superseded_by: str | None = None
    applies_to: tuple[str, ...] = field(default_factory=tuple)
    provenance: tuple[str, ...] = field(default_factory=tuple)
    source_path: Path | None = None


def _coerce_str_tuple(v) -> tuple[str, ...]:
    """Accept either a YAML list or a single scalar; normalize to
    tuple-of-str. Lets ADR authors write ``applies_to: protocol_separation``
    or ``applies_to: [protocol_separation, ble]`` and have either work."""
    if v is None:
        return ()
    if isinstance(v, str):
        return (v,) if v else ()
    if isinstance(v, (list, tuple)):
        return tuple(str(x) for x in v if x)
    return (str(v),)


def parse_adr(text: str, source: Path | None = None) -> ADR:
    """Parse an ADR markdown file. Raises ``ValueError`` if any
    required field is missing — better to crash on operator
    hand-editing than to silently skip an ADR that should be
    governing the next review."""
    fm, body = parse_frontmatter(text)
    required = ("id", "title", "status")
    missing = [k for k in required if not fm.get(k)]
    if missing:
        raise ValueError(
            f"ADR{f' at {source}' if source else ''} missing required "
            f"frontmatter fields: {', '.join(missing)}"
        )
    status = str(fm["status"]).strip().lower()
    if status not in _VALID_STATUSES:
        raise ValueError(
            f"ADR{f' at {source}' if source else ''} has invalid status "
            f"{status!r}; expected one of {sorted(_VALID_STATUSES)}"
        )
    return ADR(
        id=str(fm["id"]).strip(),
        title=str(fm["title"]).strip(),
        status=status,
        body=body.strip(),
        valid_from=_str_or_none(fm.get("valid_from")),
        valid_to=_str_or_none(fm.get("valid_to")),
        transaction_time=_str_or_none(fm.get("transaction_time")),
        supersedes=_coerce_str_tuple(fm.get("supersedes")),
        superseded_by=_str_or_none(fm.get("superseded_by")),
        applies_to=_coerce_str_tuple(fm.get("applies_to")),
        provenance=_coerce_str_tuple(fm.get("provenance")),
        source_path=source,
    )


def _str_or_none(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def load_durable_adrs(durable_dir: Path) -> list[ADR]:
    """Load every ``status: durable`` ADR in ``durable_dir``. Skips
    files that fail to parse — and logs them — so a single broken
    ADR doesn't block the whole review loop. The operator sees the
    warning in the uvicorn log and can fix the bad file without
    needing to restart immediately.

    Returns sorted by ``id`` so output is deterministic across hosts.
    Non-recursive: ``durable_dir`` is flat, one ADR per file. Sub-
    directories are reserved for future bank-isolation (per the
    paper §6, deferred to v2)."""
    if not durable_dir.exists():
        return []
    out: list[ADR] = []
    for path in sorted(durable_dir.glob("*.md")):
        try:
            adr = parse_adr(path.read_text(encoding="utf-8"), source=path)
        except (ValueError, OSError) as exc:
            logger.warning("Skipping malformed ADR %s: %s", path, exc)
            continue
        if adr.status == "durable":
            out.append(adr)
    return out


def render_adrs_for_prompt(adrs: Iterable[ADR]) -> str:
    """Format ADRs as plain-text rules for embedding in a model
    prompt. Drops frontmatter (the model doesn't care about
    transaction_time); keeps title + body so the rule is
    self-contained.

    Returns ``""`` if the iterable is empty so the caller can
    concatenate without a guard."""
    chunks: list[str] = []
    for adr in adrs:
        chunks.append(f"### {adr.id}: {adr.title}\n\n{adr.body}")
    return "\n\n".join(chunks)


# ---------------------------------------------------------------------------
# SITREPs
# ---------------------------------------------------------------------------

# Keys the SITREP frontmatter MUST have so the operator can query the
# proposed/ queue without opening every file. Anything beyond these is
# free-form.
_SITREP_REQUIRED_FIELDS: tuple[str, ...] = (
    "id", "review_id", "prompt_hash", "writer_models",
    "reviewer_model", "outcome", "rounds", "status",
    "transaction_time",
)


def write_sitrep(
    *,
    proposed_dir: Path,
    review_id: str,
    prompt: str,
    prompt_hash: str,
    writer_models: list[str],
    reviewer_model: str,
    outcome: str,                          # "done" | "stopped" | "error" | "abort"
    rounds_completed: int,
    constrained_kickoff_used: bool,
    repeated_blockers: list[str],
    abort_reason: str | None,
    fallback_count: int,
) -> Path:
    """Write a single end-of-session summary to ``proposed_dir``.

    The SITREP is a *proposal*. Nothing in the live review loop reads
    from ``proposed_dir`` — the operator is supposed to triage the
    queue manually (move to ``durable_dir``, edit, or rm). This is the
    paper's "propose, don't promote" boundary: we capture, we never
    auto-promote.

    Filename: ``sitrep-<UTC-date>-<short-review-id>.md``. UTC so
    SITREPs sort the same across operator timezones.
    """
    proposed_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    short_rid = (review_id or "unknown").split("-", 1)[0][:12]
    sitrep_id = f"sitrep-{date_str}-{short_rid}"
    path = proposed_dir / f"{sitrep_id}.md"

    # Truncate the prompt for the body — full hash is in frontmatter
    # if the operator wants to grep for the same task across SITREPs.
    prompt_preview = prompt.strip()
    if len(prompt_preview) > 600:
        prompt_preview = prompt_preview[:597] + "..."

    frontmatter = {
        "id": sitrep_id,
        "review_id": review_id,
        "prompt_hash": prompt_hash,
        "writer_models": list(writer_models),
        "reviewer_model": reviewer_model,
        "outcome": outcome,
        "rounds": int(rounds_completed),
        "constrained_kickoff_used": bool(constrained_kickoff_used),
        "fallback_count": int(fallback_count),
        "status": "provisional",
        "transaction_time": now.isoformat(),
    }

    body_lines = [f"# {sitrep_id}", ""]
    body_lines.append(f"**Outcome:** `{outcome}` after {rounds_completed} round(s).")
    body_lines.append("")
    body_lines.append(f"**Writer chain:** {' → '.join(writer_models) or '(none)'}")
    body_lines.append(f"**Reviewer:** {reviewer_model}")
    body_lines.append(f"**Constrained-kickoff fired:** {'yes' if constrained_kickoff_used else 'no'}")
    body_lines.append(f"**Fallback swaps:** {fallback_count}")
    if abort_reason:
        body_lines.append("")
        body_lines.append(f"**Abort reason:** {abort_reason}")
    body_lines.append("")
    body_lines.append("## Prompt")
    body_lines.append("")
    body_lines.append("```")
    body_lines.append(prompt_preview)
    body_lines.append("```")
    if repeated_blockers:
        body_lines.append("")
        body_lines.append("## Repeated reviewer blockers")
        body_lines.append("")
        body_lines.append(
            "These blockers recurred past `MAX_SAME_BLOCKER_REPEATS` in this run. "
            "If they keep showing up across runs of the same task, consider "
            "promoting them into a durable ADR (move/extend the relevant file "
            "in `data/durable/`)."
        )
        body_lines.append("")
        for b in repeated_blockers[:20]:
            body_lines.append(f"- {b}")
    body_lines.append("")
    body_lines.append("## Operator action")
    body_lines.append("")
    body_lines.append(
        "This SITREP is **provisional**. To act on it: "
        "edit, move to `data/durable/` to promote, or `rm` to discard."
    )

    write_with_frontmatter(path, frontmatter, "\n".join(body_lines))
    return path


# ---------------------------------------------------------------------------
# Cross-session blocker lookup
# ---------------------------------------------------------------------------


def find_prior_blockers(
    *,
    outcomes_log_path: Path,
    prompt_hash: str,
    current_review_id: str | None = None,
    min_recurrences: int = 2,
    max_age_days: int | None = None,
    max_results: int = 10,
) -> list[str]:
    """Scan the outcomes log for prior runs of the same prompt hash
    and return blockers that appeared in ``min_recurrences`` or more
    PRIOR sessions.

    Reads only the outcomes log — does not load review sessions.
    That's a deliberate scope cut: blocker text must be present in
    the outcome log entry (we just extended ``OutcomeLog.append`` to
    accept ``blockers=``). Older log entries from before that change
    won't have blocker text and are silently skipped.

    Filters:
      - Excludes ``current_review_id`` so an in-flight session
        doesn't see its own emitted blockers.
      - Optional ``max_age_days``: don't surface lessons from a stale
        codebase the operator may have already addressed.
      - ``max_results``: hard cap so we never bloat a writer prompt
        with 50+ "previously seen" entries.
    """
    if not outcomes_log_path.exists():
        return []
    cutoff_iso: str | None = None
    if max_age_days is not None and max_age_days > 0:
        cutoff_dt = datetime.now(timezone.utc).timestamp() - max_age_days * 86400
        cutoff_iso = datetime.fromtimestamp(cutoff_dt, tz=timezone.utc).isoformat()

    counts: dict[str, int] = {}
    seen_per_session: dict[str, set[str]] = {}
    try:
        with outcomes_log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("prompt_hash") != prompt_hash:
                    continue
                if row.get("event") != "review_block":
                    continue
                rid = str(row.get("review_id") or "")
                if not rid or rid == current_review_id:
                    continue
                if cutoff_iso is not None:
                    ts = str(row.get("ts") or "")
                    if ts and ts < cutoff_iso:
                        continue
                blockers = row.get("blockers") or []
                if not isinstance(blockers, list):
                    continue
                # De-dupe within a single session — we want to count
                # number of SESSIONS that produced a blocker, not
                # how many rounds within one session reproduced it.
                seen = seen_per_session.setdefault(rid, set())
                for b in blockers:
                    text = _normalize_blocker(str(b))
                    if not text or text in seen:
                        continue
                    seen.add(text)
                    counts[text] = counts.get(text, 0) + 1
    except OSError as exc:
        logger.warning("find_prior_blockers: could not read %s: %s", outcomes_log_path, exc)
        return []

    candidates = [
        (text, n) for text, n in counts.items()
        if n >= min_recurrences
    ]
    candidates.sort(key=lambda kv: (-kv[1], kv[0]))
    return [text for text, _ in candidates[:max_results]]


def _normalize_blocker(text: str) -> str:
    """Mirrors app.web.app._normalize_blocker — kept duplicated rather
    than imported because this module is loaded by both the request
    path and the SITREP-writer path, and we don't want a circular
    import via app.web.app."""
    return " ".join(text.casefold().split())[:200]


def render_prior_lessons_for_writer(blockers: list[str]) -> str:
    """Format prior-session blockers as a self-contained hint block
    for the writer's first prompt. **Always framed as advisory** —
    the paper's whole point is that one session's wrong inference
    must not become the next session's premise. The operator opted
    in by setting ``apply_prior_lessons``, but the writer is told
    explicitly these are observations, not facts."""
    if not blockers:
        return ""
    lines = [
        "PRIOR-SESSION OBSERVATIONS — these reviewer blockers came up in "
        "earlier runs of the same task. They are HINTS, not established "
        "facts. The operator opted in to surfacing them; treat each as a "
        "thing to consider, and if your design avoids the underlying "
        "problem you do NOT need to acknowledge them in your output.",
        "",
    ]
    for b in blockers:
        lines.append(f"- {b}")
    return "\n".join(lines)
