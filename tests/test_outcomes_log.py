"""Tests for app/utils/outcomes_log.py — append-only JSONL log of
review outcomes and the per-writer aggregator that the model-stats
CLI consumes."""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pytest

from app.utils.outcomes_log import (
    EVENT_TYPES,
    ModelStats,
    OutcomeLog,
    aggregate_by_writer,
    hash_prompt,
)


# ---------------------------------------------------------------------------
# OutcomeLog.append
# ---------------------------------------------------------------------------

def test_append_writes_one_line_per_event(tmp_path: Path) -> None:
    log = OutcomeLog(tmp_path / "out.jsonl")
    log.append(review_id="r1", writer="qwen3-coder:30b",
               event="gate_pass", prompt_hash="sha256:abc")
    log.append(review_id="r1", writer="qwen3-coder:30b",
               event="review_pass", prompt_hash="sha256:abc")
    contents = (tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(contents) == 2
    rows = [json.loads(line) for line in contents]
    assert rows[0]["event"] == "gate_pass"
    assert rows[1]["event"] == "review_pass"
    # ts is a sortable ISO timestamp
    for r in rows:
        _dt.datetime.fromisoformat(r["ts"])


def test_append_drops_unknown_events(tmp_path: Path) -> None:
    log = OutcomeLog(tmp_path / "out.jsonl")
    log.append(review_id="r1", writer="x", event="bogus_event",
               prompt_hash="sha256:abc")
    assert not (tmp_path / "out.jsonl").exists() or \
           (tmp_path / "out.jsonl").read_text(encoding="utf-8") == ""


def test_append_truncates_long_detail(tmp_path: Path) -> None:
    log = OutcomeLog(tmp_path / "out.jsonl")
    log.append(review_id="r1", writer="x", event="abort",
               prompt_hash="sha256:abc",
               detail="!" * 10_000)
    rows = list(log.read_all())
    assert len(rows[0]["detail"]) <= 200


def test_append_creates_parent_dir(tmp_path: Path) -> None:
    deep = tmp_path / "a" / "b" / "c" / "out.jsonl"
    log = OutcomeLog(deep)
    log.append(review_id="r1", writer="x", event="gate_pass",
               prompt_hash="sha256:abc")
    assert deep.exists()


def test_event_types_match_documented_enum() -> None:
    assert EVENT_TYPES == {
        "gate_pass", "gate_fail", "gate_blocked",
        "review_pass", "review_block", "abort",
    }


# ---------------------------------------------------------------------------
# read_all + aggregate_by_writer
# ---------------------------------------------------------------------------

def test_read_all_returns_empty_when_file_absent(tmp_path: Path) -> None:
    log = OutcomeLog(tmp_path / "missing.jsonl")
    assert list(log.read_all()) == []


def test_read_all_skips_malformed_lines(tmp_path: Path) -> None:
    p = tmp_path / "out.jsonl"
    p.write_text(
        '{"writer": "qwen", "event": "gate_pass"}\n'
        'not valid json\n'
        '{"writer": "granite", "event": "gate_fail", "gate": "lint"}\n',
        encoding="utf-8",
    )
    rows = list(OutcomeLog(p).read_all())
    assert [r["event"] for r in rows] == ["gate_pass", "gate_fail"]


def test_aggregate_counts_each_event_per_writer(tmp_path: Path) -> None:
    log = OutcomeLog(tmp_path / "out.jsonl")
    for ev, gate in [
        ("gate_pass", None),
        ("gate_fail", "lint"),
        ("gate_fail", "lint"),
        ("gate_fail", "imports"),
        ("review_pass", None),
        ("review_block", None),
    ]:
        log.append(review_id="r1", writer="qwen",
                   event=ev, gate=gate, prompt_hash="sha256:abc")
    log.append(review_id="r1", writer="granite",
               event="gate_pass", prompt_hash="sha256:abc")

    stats = aggregate_by_writer(log.read_all())
    assert "qwen" in stats and "granite" in stats
    qwen = stats["qwen"]
    assert qwen.gate_passes == 1
    assert qwen.gate_fails == 3
    assert qwen.review_passes == 1
    assert qwen.review_blocks == 1
    assert qwen.gate_failure_counts == {"lint": 2, "imports": 1}
    assert qwen.top_gate_failure == "lint (2)"


def test_aggregate_pass_rates() -> None:
    s = ModelStats(writer="x")
    s.gate_passes = 7
    s.gate_fails = 3
    s.review_passes = 4
    s.review_blocks = 1
    assert s.gate_pass_rate == pytest.approx(0.7)
    assert s.review_pass_rate == pytest.approx(0.8)


def test_aggregate_pass_rate_zero_when_no_data() -> None:
    s = ModelStats(writer="x")
    assert s.gate_pass_rate == 0.0
    assert s.review_pass_rate == 0.0


def test_aggregate_runs_count_unique_review_ids(tmp_path: Path) -> None:
    """A single review produces multiple events — ModelStats.runs
    should count the review_ids, not the events."""
    log = OutcomeLog(tmp_path / "out.jsonl")
    for ev in ("gate_pass", "review_pass"):
        log.append(review_id="r1", writer="qwen", event=ev, prompt_hash="h1")
    for ev in ("gate_fail", "gate_pass"):
        log.append(review_id="r2", writer="qwen",
                   event=ev, gate="lint", prompt_hash="h2")
    stats = aggregate_by_writer(log.read_all())
    assert stats["qwen"].runs == 2


# ---------------------------------------------------------------------------
# hash_prompt
# ---------------------------------------------------------------------------

def test_hash_prompt_stable() -> None:
    assert hash_prompt("hello") == hash_prompt("hello")
    assert hash_prompt("hello") != hash_prompt("world")


def test_hash_prompt_short_form_for_log_compactness() -> None:
    h = hash_prompt("anything")
    assert h.startswith("sha256:")
    # 16 hex chars is enough for grouping without the full 64.
    assert len(h.split(":", 1)[1]) == 16


def test_concurrent_appends_dont_interleave_lines(tmp_path: Path) -> None:
    """Two threads writing simultaneously to the same OutcomeLog must
    produce a clean JSONL — never partial lines that read_all would
    silently drop as malformed."""
    import threading

    log = OutcomeLog(tmp_path / "out.jsonl")
    n_per_thread = 200
    barrier = threading.Barrier(2)

    def _writer(writer_id: str) -> None:
        barrier.wait()
        for i in range(n_per_thread):
            log.append(
                review_id=f"r-{writer_id}-{i}",
                writer=writer_id,
                event="gate_pass",
                prompt_hash=f"sha256:{i:016x}",
            )

    t1 = threading.Thread(target=_writer, args=("threadA",))
    t2 = threading.Thread(target=_writer, args=("threadB",))
    t1.start(); t2.start()
    t1.join(); t2.join()

    rows = list(log.read_all())
    # All 400 rows must parse — no half-lines lost to interleave.
    assert len(rows) == n_per_thread * 2
    by_writer: dict[str, int] = {}
    for r in rows:
        by_writer[r["writer"]] = by_writer.get(r["writer"], 0) + 1
    assert by_writer == {"threadA": n_per_thread, "threadB": n_per_thread}
