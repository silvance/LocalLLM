"""Tests for app/services/continuity.py — ADR loading, SITREP
writing, and cross-session blocker lookup.

The continuity layer is the architectural answer (paper §2-§4) to
context poisoning + cross-session amnesia in the writer ↔ reviewer
loop. Each test pins one behavior the rest of the orchestrator
relies on so the next refactor doesn't silently regress it."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.continuity import (
    ADR,
    find_prior_blockers,
    load_durable_adrs,
    parse_adr,
    parse_frontmatter,
    render_adrs_for_prompt,
    render_prior_lessons_for_writer,
    write_sitrep,
    write_with_frontmatter,
)


# ---------------------------------------------------------------------------
# Frontmatter parser
# ---------------------------------------------------------------------------

def test_parse_frontmatter_no_block_returns_empty_dict() -> None:
    """Files without a frontmatter block must round-trip as
    ``({}, original_text)`` — callers should not have to special-case
    legacy markdown that predates the ADR convention."""
    fm, body = parse_frontmatter("# Just a heading\n\nNo frontmatter here.\n")
    assert fm == {}
    assert "# Just a heading" in body


def test_parse_frontmatter_extracts_yaml_mapping() -> None:
    text = (
        "---\n"
        "id: adr-0007\n"
        "title: Cache invalidation strategy\n"
        "status: durable\n"
        "applies_to:\n"
        "  - caching\n"
        "  - rate_limit\n"
        "---\n"
        "\n"
        "Body of the ADR.\n"
    )
    fm, body = parse_frontmatter(text)
    assert fm["id"] == "adr-0007"
    assert fm["status"] == "durable"
    assert fm["applies_to"] == ["caching", "rate_limit"]
    assert body.strip() == "Body of the ADR."


def test_parse_frontmatter_raises_on_malformed_yaml() -> None:
    """Operator-edited frontmatter that's broken must fail loudly —
    silently dropping bad metadata would let an ADR governing the
    next review go missing without anyone noticing."""
    text = "---\nid: oops\n  badly: indented\nbroken\n---\n\nbody\n"
    with pytest.raises(ValueError, match="frontmatter"):
        parse_frontmatter(text)


def test_parse_frontmatter_rejects_non_mapping() -> None:
    """A frontmatter block that's a YAML scalar or list (e.g. just
    ``- one\n- two``) is not a valid ADR header — reject explicitly."""
    text = "---\n- not\n- a mapping\n---\n\nbody\n"
    with pytest.raises(ValueError, match="must be a YAML mapping"):
        parse_frontmatter(text)


def test_write_with_frontmatter_round_trips(tmp_path: Path) -> None:
    """write → parse must round-trip frontmatter values cleanly so a
    SITREP written today can be read tomorrow without surprises."""
    path = tmp_path / "x.md"
    write_with_frontmatter(
        path,
        {"id": "sitrep-1", "outcome": "done", "rounds": 3},
        "Some body content",
    )
    text = path.read_text(encoding="utf-8")
    fm, body = parse_frontmatter(text)
    assert fm["id"] == "sitrep-1"
    assert fm["outcome"] == "done"
    assert fm["rounds"] == 3
    assert "Some body content" in body


def test_write_with_frontmatter_is_atomic(tmp_path: Path) -> None:
    """Tmp-file + rename so a partial write can't poison the queue.
    We can't easily simulate a power-loss, but we can confirm no
    .tmp file is left behind on success."""
    path = tmp_path / "y.md"
    write_with_frontmatter(path, {"id": "x"}, "body")
    assert path.exists()
    leftover = list(tmp_path.glob("*.tmp"))
    assert leftover == [], f"leftover tmp files: {leftover}"


# ---------------------------------------------------------------------------
# ADR parsing + loading
# ---------------------------------------------------------------------------

_VALID_ADR = """---
id: adr-0001
title: Wi-Fi and BLE belong on separate backends
status: durable
applies_to:
  - protocol_separation
  - ble
provenance:
  - PR-38 review-loop hardening
---

BLE goes through BlueZ. Wi-Fi goes through scapy. Don't mix.
"""


def test_parse_adr_extracts_required_fields() -> None:
    adr = parse_adr(_VALID_ADR)
    assert isinstance(adr, ADR)
    assert adr.id == "adr-0001"
    assert adr.title == "Wi-Fi and BLE belong on separate backends"
    assert adr.status == "durable"
    assert "BlueZ" in adr.body
    assert adr.applies_to == ("protocol_separation", "ble")
    assert adr.provenance == ("PR-38 review-loop hardening",)


def test_parse_adr_rejects_missing_required_field() -> None:
    text = "---\nid: only-id\n---\n\nbody\n"
    with pytest.raises(ValueError, match="missing required"):
        parse_adr(text)


def test_parse_adr_rejects_unknown_status() -> None:
    text = (
        "---\nid: adr-x\ntitle: Test\nstatus: pending\n---\n\nbody\n"
    )
    with pytest.raises(ValueError, match="invalid status"):
        parse_adr(text)


def test_parse_adr_normalizes_status_case() -> None:
    """Operator-typed status with stray uppercase must still load —
    we normalize rather than reject. Avoids a class of trivial
    operator errors blocking a review."""
    text = "---\nid: adr-x\ntitle: Test\nstatus: DURABLE\n---\n\nbody\n"
    adr = parse_adr(text)
    assert adr.status == "durable"


def test_parse_adr_accepts_scalar_for_list_field() -> None:
    """``applies_to: caching`` as a scalar must be accepted as a
    one-item list. Treats ergonomic operator shortcuts as equivalent
    to the explicit YAML list form."""
    text = (
        "---\nid: x\ntitle: Test\nstatus: durable\n"
        "applies_to: caching\n---\n\nbody\n"
    )
    adr = parse_adr(text)
    assert adr.applies_to == ("caching",)


def test_load_durable_adrs_returns_only_durable(tmp_path: Path) -> None:
    """Provisional / superseded ADRs must NOT influence the next
    review — that's the entire point of the operator-confirmation
    boundary (paper §3)."""
    (tmp_path / "a.md").write_text(_VALID_ADR, encoding="utf-8")
    (tmp_path / "b.md").write_text(
        _VALID_ADR.replace("status: durable", "status: provisional"),
        encoding="utf-8",
    )
    (tmp_path / "c.md").write_text(
        _VALID_ADR.replace("status: durable", "status: superseded")
                  .replace("adr-0001", "adr-0002"),
        encoding="utf-8",
    )
    adrs = load_durable_adrs(tmp_path)
    assert len(adrs) == 1
    assert adrs[0].status == "durable"


def test_load_durable_adrs_skips_malformed_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """One bad ADR must not block the rest. Operator sees a warning
    in the uvicorn log and can fix the file without restarting.

    Captured via capsys (stderr) rather than caplog because
    ``setup_logger`` sets ``propagate=False`` on the localllm logger
    when the app is imported during another test, breaking caplog's
    default root-level capture."""
    import logging
    (tmp_path / "good.md").write_text(_VALID_ADR, encoding="utf-8")
    (tmp_path / "broken.md").write_text(
        "---\n[: not yaml\n---\nbody\n", encoding="utf-8",
    )
    # Attach a stderr handler to the localllm logger directly so the
    # warning is captured regardless of propagate / pre-existing
    # handler config.
    log = logging.getLogger("localllm")
    handler = logging.StreamHandler()
    log.addHandler(handler)
    log.setLevel(logging.WARNING)
    try:
        adrs = load_durable_adrs(tmp_path)
    finally:
        log.removeHandler(handler)
    captured = capsys.readouterr()
    assert len(adrs) == 1
    assert "broken.md" in captured.err


def test_load_durable_adrs_returns_sorted_by_id(tmp_path: Path) -> None:
    """Determinism: same on-disk state must produce the same prompt
    text so reviews of the same prompt repeat."""
    (tmp_path / "z.md").write_text(
        _VALID_ADR.replace("adr-0001", "adr-0099"), encoding="utf-8",
    )
    (tmp_path / "a.md").write_text(
        _VALID_ADR.replace("adr-0001", "adr-0001"), encoding="utf-8",
    )
    adrs = load_durable_adrs(tmp_path)
    assert [a.id for a in adrs] == ["adr-0001", "adr-0099"]


def test_load_durable_adrs_empty_dir_returns_empty_list(tmp_path: Path) -> None:
    assert load_durable_adrs(tmp_path) == []
    assert load_durable_adrs(tmp_path / "nonexistent") == []


def test_render_adrs_for_prompt_includes_title_and_body() -> None:
    adr = parse_adr(_VALID_ADR)
    out = render_adrs_for_prompt([adr])
    assert "adr-0001" in out
    assert "Wi-Fi and BLE belong on separate backends" in out
    assert "BlueZ" in out


def test_render_adrs_for_prompt_empty_list_returns_empty_string() -> None:
    assert render_adrs_for_prompt([]) == ""


# ---------------------------------------------------------------------------
# SITREP writer
# ---------------------------------------------------------------------------

def test_write_sitrep_creates_file_with_frontmatter(tmp_path: Path) -> None:
    path = write_sitrep(
        proposed_dir=tmp_path,
        review_id="abc-123-def",
        prompt="Build a BLE/Wi-Fi sniffer",
        prompt_hash="sha256:deadbeef00",
        writer_models=["qwen", "granite"],
        reviewer_model="gemma",
        outcome="done",
        rounds_completed=4,
        constrained_kickoff_used=True,
        repeated_blockers=["BLE cannot be sniffed from wlan0"],
        abort_reason=None,
        fallback_count=1,
    )
    assert path.exists()
    fm, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    assert fm["review_id"] == "abc-123-def"
    assert fm["status"] == "provisional"
    assert fm["outcome"] == "done"
    assert fm["rounds"] == 4
    assert fm["constrained_kickoff_used"] is True
    assert fm["writer_models"] == ["qwen", "granite"]
    # Body must contain the human-readable summary AND the repeated
    # blockers section.
    assert "Outcome" in body
    assert "BLE cannot be sniffed from wlan0" in body


def test_write_sitrep_filename_starts_with_date_and_review_prefix(
    tmp_path: Path,
) -> None:
    """SITREP filenames must sort chronologically AND let the
    operator grep for "all SITREPs from review X" without opening
    files. Format: ``sitrep-<date>-<short-review-id>.md``."""
    path = write_sitrep(
        proposed_dir=tmp_path,
        review_id="ffeeddccbbaa1234",
        prompt="x", prompt_hash="sha256:00",
        writer_models=["qwen"], reviewer_model="gemma",
        outcome="done", rounds_completed=1,
        constrained_kickoff_used=False, repeated_blockers=[],
        abort_reason=None, fallback_count=0,
    )
    assert path.name.startswith("sitrep-20")  # YYYY-MM-DD prefix
    assert path.name.endswith(".md")
    # Short review ID must appear so the operator can correlate
    # SITREP → review session.
    assert "ffeeddccbbaa" in path.name


def test_write_sitrep_truncates_long_prompts(tmp_path: Path) -> None:
    """A 50KB prompt should NOT bloat the SITREP file. Truncate the
    body but keep the full hash in frontmatter for grep-by-task."""
    long_prompt = "x" * 50_000
    path = write_sitrep(
        proposed_dir=tmp_path,
        review_id="r1",
        prompt=long_prompt, prompt_hash="sha256:abc",
        writer_models=["qwen"], reviewer_model="gemma",
        outcome="done", rounds_completed=1,
        constrained_kickoff_used=False, repeated_blockers=[],
        abort_reason=None, fallback_count=0,
    )
    text = path.read_text(encoding="utf-8")
    # Truncated to ~600 chars + ellipsis. NOT 50_000.
    assert len(text) < 5_000
    assert "..." in text  # truncation marker


def test_write_sitrep_records_abort_reason(tmp_path: Path) -> None:
    path = write_sitrep(
        proposed_dir=tmp_path,
        review_id="r1", prompt="p", prompt_hash="h",
        writer_models=["qwen"], reviewer_model="gemma",
        outcome="done", rounds_completed=8,
        constrained_kickoff_used=True,
        repeated_blockers=["BLE on wlan0"],
        abort_reason="Same blocker survived a constrained-architecture rebuild",
        fallback_count=0,
    )
    text = path.read_text(encoding="utf-8")
    assert "constrained-architecture rebuild" in text


# ---------------------------------------------------------------------------
# Cross-session blocker lookup
# ---------------------------------------------------------------------------

def _write_log(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_find_prior_blockers_returns_recurring_across_sessions(
    tmp_path: Path,
) -> None:
    """Same blocker in two different prior sessions of the same
    prompt hash → returned. Same blocker in one session → not
    returned (need ``min_recurrences`` distinct sessions)."""
    log = tmp_path / "outcomes.jsonl"
    _write_log(log, [
        {"event": "review_block", "review_id": "r1", "writer": "qwen",
         "prompt_hash": "h-abc",
         "blockers": ["BLE on wlan0", "Module not pinned"]},
        {"event": "review_block", "review_id": "r2", "writer": "granite",
         "prompt_hash": "h-abc",
         "blockers": ["BLE on wlan0", "Some other thing"]},
        {"event": "review_block", "review_id": "r3", "writer": "qwen",
         "prompt_hash": "h-different",
         "blockers": ["BLE on wlan0"]},
    ])
    out = find_prior_blockers(
        outcomes_log_path=log, prompt_hash="h-abc", min_recurrences=2,
    )
    # "BLE on wlan0" appeared in r1 + r2 (same hash). "Module not pinned"
    # only in r1. "Some other thing" only in r2. Different prompt
    # hashes are excluded.
    assert any("ble on wlan0" in b for b in out)
    assert not any("module not pinned" in b for b in out)
    assert not any("some other thing" in b for b in out)


def test_find_prior_blockers_excludes_current_session(tmp_path: Path) -> None:
    """In-flight session must not see its own blockers — would create
    a feedback loop where each round inherits the previous round's
    failures."""
    log = tmp_path / "outcomes.jsonl"
    _write_log(log, [
        {"event": "review_block", "review_id": "current", "writer": "qwen",
         "prompt_hash": "h", "blockers": ["X"]},
        {"event": "review_block", "review_id": "current", "writer": "qwen",
         "prompt_hash": "h", "blockers": ["X"]},
    ])
    out = find_prior_blockers(
        outcomes_log_path=log, prompt_hash="h",
        current_review_id="current", min_recurrences=2,
    )
    assert out == []


def test_find_prior_blockers_dedupes_within_session(tmp_path: Path) -> None:
    """Same blocker in 5 rounds of one session = 1 session-vote, not
    5. Counting rounds would over-weight a single bad run."""
    log = tmp_path / "outcomes.jsonl"
    _write_log(log, [
        {"event": "review_block", "review_id": "r1", "writer": "qwen",
         "prompt_hash": "h", "blockers": ["X"]},
        {"event": "review_block", "review_id": "r1", "writer": "qwen",
         "prompt_hash": "h", "blockers": ["X"]},
        {"event": "review_block", "review_id": "r1", "writer": "qwen",
         "prompt_hash": "h", "blockers": ["X"]},
    ])
    out = find_prior_blockers(
        outcomes_log_path=log, prompt_hash="h", min_recurrences=2,
    )
    # X appeared in 1 session only → not returned (need 2+).
    assert out == []


def test_find_prior_blockers_skips_non_review_block_events(tmp_path: Path) -> None:
    log = tmp_path / "outcomes.jsonl"
    _write_log(log, [
        {"event": "gate_pass", "review_id": "r1", "writer": "qwen",
         "prompt_hash": "h"},
        {"event": "review_pass", "review_id": "r2", "writer": "qwen",
         "prompt_hash": "h"},
        {"event": "abort", "review_id": "r3", "writer": "qwen",
         "prompt_hash": "h"},
    ])
    out = find_prior_blockers(
        outcomes_log_path=log, prompt_hash="h", min_recurrences=1,
    )
    assert out == []


def test_find_prior_blockers_handles_missing_log_file(tmp_path: Path) -> None:
    """Brand-new install with no log yet must return [] without
    raising — the writer just gets no prior-lessons hint."""
    out = find_prior_blockers(
        outcomes_log_path=tmp_path / "does-not-exist.jsonl",
        prompt_hash="h",
    )
    assert out == []


def test_find_prior_blockers_skips_legacy_rows_without_blockers(
    tmp_path: Path,
) -> None:
    """Outcome log entries from before the blockers field was added
    don't have blocker text. Must skip silently — the field is
    optional, not malformed."""
    log = tmp_path / "outcomes.jsonl"
    _write_log(log, [
        {"event": "review_block", "review_id": "r1", "writer": "qwen",
         "prompt_hash": "h", "blocker_count": 3},  # old schema
    ])
    out = find_prior_blockers(
        outcomes_log_path=log, prompt_hash="h", min_recurrences=1,
    )
    assert out == []


def test_find_prior_blockers_caps_results(tmp_path: Path) -> None:
    """``max_results`` keeps the writer prompt bounded — a 50-blocker
    history should not become a 50-bullet hint block."""
    log = tmp_path / "outcomes.jsonl"
    rows = []
    for i in range(20):
        rows.append({
            "event": "review_block",
            "review_id": f"session-a-{i}",
            "writer": "qwen", "prompt_hash": "h",
            "blockers": [f"unique-blocker-{i}"],
        })
        rows.append({
            "event": "review_block",
            "review_id": f"session-b-{i}",
            "writer": "qwen", "prompt_hash": "h",
            "blockers": [f"unique-blocker-{i}"],
        })
    _write_log(log, rows)
    out = find_prior_blockers(
        outcomes_log_path=log, prompt_hash="h",
        min_recurrences=2, max_results=5,
    )
    assert len(out) == 5


def test_render_prior_lessons_for_writer_emits_advisory_framing() -> None:
    """The output must explicitly tell the writer these are HINTS,
    not facts — paper §5: never auto-promote inferences across
    sessions."""
    out = render_prior_lessons_for_writer(["BLE on wlan0", "iface name typo"])
    assert "HINTS" in out or "hints" in out.lower()
    assert "BLE on wlan0" in out
    # "Advisory" / "consider" framing — NOT "fix" or "you must".
    assert "you must" not in out.lower()


def test_render_prior_lessons_for_writer_empty_returns_empty_string() -> None:
    assert render_prior_lessons_for_writer([]) == ""
