"""Print per-model failure statistics from the review-outcome log.

Reads ``review_outcomes.jsonl`` (default location: user_data_dir, or
override via ``LOCALLLM_OUTCOMES_LOG``) and emits a fixed-width table
ordered by total runs descending. Top failure column highlights the
gate name a writer has hit most often — the model-eval data the
operator was missing when comparing Qwen / Granite / Gemma / etc.

Usage:
    LocalLLM model-stats
    LocalLLM model-stats --since 7d
    LocalLLM model-stats --log /path/to/review_outcomes.jsonl
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path


def _parse_since(spec: str) -> _dt.datetime:
    """Accept a duration suffix (7d / 24h / 30m) or an ISO timestamp."""
    if not spec:
        return _dt.datetime.min.replace(tzinfo=_dt.timezone.utc)
    try:
        return _dt.datetime.fromisoformat(spec)
    except ValueError:
        pass
    if spec[-1] in "smhd" and spec[:-1].isdigit():
        n = int(spec[:-1])
        unit = spec[-1]
        delta = {
            "s": _dt.timedelta(seconds=n),
            "m": _dt.timedelta(minutes=n),
            "h": _dt.timedelta(hours=n),
            "d": _dt.timedelta(days=n),
        }[unit]
        return _dt.datetime.now(_dt.timezone.utc) - delta
    raise argparse.ArgumentTypeError(f"--since: expected ISO timestamp or N[smhd], got {spec!r}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate review outcomes per writer model.",
    )
    parser.add_argument("--log", type=Path, default=None,
                        help="Override outcomes-log path (default: user_data_dir).")
    parser.add_argument("--since", type=_parse_since, default=None,
                        help="Only count entries newer than this (ISO ts or N[smhd]).")
    args = parser.parse_args()

    # Local imports so this script doesn't pay the import cost when
    # someone just wants `--help`.
    from app.utils.outcomes_log import (
        OutcomeLog,
        aggregate_by_writer,
        default_log_path,
    )

    path = args.log or default_log_path()
    log = OutcomeLog(path)
    if not path.exists():
        print(f"No outcomes log at {path}.", file=sys.stderr)
        print("Run a few reviews first; this file is appended at every "
              "gate / reviewer decision point.", file=sys.stderr)
        return 0

    def _gen():
        cutoff = args.since
        for row in log.read_all():
            if cutoff is not None:
                ts = row.get("ts")
                try:
                    rowtime = _dt.datetime.fromisoformat(ts) if ts else None
                except ValueError:
                    rowtime = None
                if rowtime is not None and rowtime < cutoff:
                    continue
            yield row

    stats = aggregate_by_writer(_gen())
    if not stats:
        print(f"{path}: no matching entries.")
        return 0

    rows = sorted(stats.values(), key=lambda s: -s.runs)
    name_w = max(8, max(len(s.writer) for s in rows))
    fmt = "{:<%d}  {:>5}  {:>11}  {:>13}  {:<25}" % name_w
    print(fmt.format("Writer", "Runs", "Gate-pass %", "Review-pass %", "Top gate failure"))
    print(fmt.format("-" * name_w, "-" * 5, "-" * 11, "-" * 13, "-" * 25))
    for s in rows:
        print(fmt.format(
            s.writer,
            s.runs,
            f"{s.gate_pass_rate * 100:.1f}",
            f"{s.review_pass_rate * 100:.1f}",
            s.top_gate_failure or "—",
        ))
    print()
    print(f"Source: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
