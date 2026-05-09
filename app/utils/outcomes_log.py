"""Append-only JSONL log of review-loop outcomes.

The review pipeline already labels failures by gate name and verdict
shape; this module persists those labels per writer-model so the
operator can answer questions like:

  - "qwen3-coder fails imports 40% of the time, granite4 only 10%"
  - "gemma is the best reviewer for Python, qwen3-coder is best for
     research-style tasks"

Without this log, comparison stays at the vibes level. With it, the
operator gets actual model-eval data accumulated across real runs.

File layout: one JSON object per line, written to
``<user_data>/review_outcomes.jsonl``. Append-only, never compacted —
the file is small (a few hundred bytes per outcome) and small models
on a security workstation produce single-digit-MB-per-year logs.

Reading is via ``read_all()``; aggregation is in
``aggregate_by_writer()`` so different consumers (CLI, future web
dashboard) share the same shape.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


logger = logging.getLogger("localllm")


# Closed set so consumers can switch over it cleanly. New event types
# need a deliberate add; "unknown" rows from older log files are
# skipped during aggregation.
EVENT_TYPES: frozenset[str] = frozenset({
    "gate_pass",       # all static gates passed for a writer round
    "gate_fail",       # at least one static gate failed (writer's fault)
    "gate_blocked",    # only-blocked failure — real third-party dep not
                       # installed in gate env; NOT writer's fault, the
                       # reviewer still ran. Counted separately so the
                       # writer's gate_pass_rate isn't depressed by
                       # operator environment quirks.
    "review_pass",     # reviewer returned no blockers
    "review_block",    # reviewer returned ≥1 blocker
    "abort",           # orchestrator stopped the run (blocker repeat etc.)
})


@dataclass
class ModelStats:
    writer: str
    runs: int = 0
    gate_passes: int = 0
    gate_fails: int = 0
    gate_blocked: int = 0
    review_passes: int = 0
    review_blocks: int = 0
    aborts: int = 0
    # gate name → count, populated for gate_fail events
    gate_failure_counts: dict[str, int] = field(default_factory=dict)

    @property
    def gate_pass_rate(self) -> float:
        total = self.gate_passes + self.gate_fails
        return self.gate_passes / total if total else 0.0

    @property
    def review_pass_rate(self) -> float:
        total = self.review_passes + self.review_blocks
        return self.review_passes / total if total else 0.0

    @property
    def top_gate_failure(self) -> str:
        if not self.gate_failure_counts:
            return ""
        name, count = max(self.gate_failure_counts.items(), key=lambda kv: kv[1])
        return f"{name} ({count})"


def hash_prompt(prompt: str) -> str:
    """Short stable hash of the prompt — lets the operator group runs
    of the same task across writer changes without storing the prompt
    text (which could be sensitive on a forensics workstation)."""
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return f"sha256:{digest[:16]}"


class OutcomeLog:
    """Thin append-only wrapper. Single instance per server is fine —
    JSONL appends are atomic up to the OS pagesize for our line lengths.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Concurrent _runner threads (two reviews running in parallel)
        # both call append() — POSIX guarantees atomic appends only up
        # to PIPE_BUF (~4KB), and Windows doesn't even guarantee that.
        # Serialize through this lock so JSONL stays one-row-per-line
        # and a power-loss mid-write doesn't lose buffered events from
        # other threads.
        self._lock = threading.Lock()

    def append(
        self,
        *,
        review_id: str,
        writer: str,
        event: str,
        prompt_hash: str,
        gate: str | None = None,
        blocker_count: int | None = None,
        detail: str | None = None,
    ) -> None:
        if event not in EVENT_TYPES:
            logger.warning("OutcomeLog: unknown event %r — dropping", event)
            return
        row = {
            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds"),
            "review_id": review_id,
            "writer": writer,
            "event": event,
            "prompt_hash": prompt_hash,
        }
        if gate is not None:
            row["gate"] = gate
        if blocker_count is not None:
            row["blocker_count"] = blocker_count
        if detail is not None:
            row["detail"] = detail[:200]
        line = json.dumps(row, ensure_ascii=False)
        try:
            with self._lock, self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    # fsync may not be supported on all filesystems
                    # (e.g. some network mounts) — line is at least
                    # in the OS buffer cache.
                    pass
        except OSError as exc:
            # Logging failure must never break a review — swallow + warn.
            logger.warning("OutcomeLog write failed: %s", exc)

    def read_all(self) -> Iterator[dict]:
        if not self.path.exists():
            return iter([])
        def _gen() -> Iterator[dict]:
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        return _gen()


def aggregate_by_writer(entries: Iterator[dict]) -> dict[str, ModelStats]:
    """Roll up outcomes into per-writer ModelStats. Unknown event types
    are silently dropped — the log is forward-compatible by design."""
    out: dict[str, ModelStats] = {}
    seen_runs: dict[str, set[str]] = {}  # writer → review_ids seen
    for row in entries:
        writer = str(row.get("writer") or "")
        event = str(row.get("event") or "")
        if not writer or event not in EVENT_TYPES:
            continue
        s = out.setdefault(writer, ModelStats(writer=writer))
        rid = str(row.get("review_id") or "")
        if rid:
            seen = seen_runs.setdefault(writer, set())
            if rid not in seen:
                seen.add(rid)
                s.runs += 1
        if event == "gate_pass":
            s.gate_passes += 1
        elif event == "gate_fail":
            s.gate_fails += 1
            gate = str(row.get("gate") or "unknown")
            s.gate_failure_counts[gate] = s.gate_failure_counts.get(gate, 0) + 1
        elif event == "gate_blocked":
            s.gate_blocked += 1
        elif event == "review_pass":
            s.review_passes += 1
        elif event == "review_block":
            s.review_blocks += 1
        elif event == "abort":
            s.aborts += 1
    return out


def default_log_path() -> Path:
    """Where the runtime writes outcomes. Override for tests via
    ``LOCALLLM_OUTCOMES_LOG``; otherwise lands under user_data_dir."""
    override = os.getenv("LOCALLLM_OUTCOMES_LOG")
    if override:
        return Path(override).expanduser().resolve()
    from app.runtime.paths import user_data_dir
    return user_data_dir() / "review_outcomes.jsonl"
