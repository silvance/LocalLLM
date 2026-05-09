"""Machine-readable reviewer verdict.

Reviewer rounds used to be free-form prose, which means the
orchestrator could only *count* failures (gates failed N times → swap
writer). With a structured verdict we can branch on whether the bug
is repairable code (rebuild same writer), conceptual / hardware-level
(rebuild different writer), or unsalvageable (abort the loop).

The reviewer is asked to append a JSON object to its prose, wrapped
in a ```json fenced block. We extract + parse it; if the model fails
to produce valid JSON we fall back to treating the whole response as
a "blocker prose" with a conservative recommendation. That keeps the
loop functional even on smaller models that drift on structured
output, while the orchestrator gets the cleaner signal whenever the
reviewer cooperates.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Literal


logger = logging.getLogger("localllm")


# Closed set so the orchestrator can switch over it cleanly. Adding
# a new action means the orchestrator gets a deliberate update too.
NextAction = Literal[
    "rebuild_same_writer",       # default — apply the blockers and try again
    "rebuild_different_writer",  # writer-class problem; engage the fallback
    "abort",                     # unsalvageable — stop the review loop
    "approve",                   # nothing to fix; finish the review
]


VALID_ACTIONS: frozenset[str] = frozenset(
    ("rebuild_same_writer", "rebuild_different_writer", "abort", "approve")
)


@dataclass
class ReviewerVerdict:
    passed: bool
    blockers: list[str] = field(default_factory=list)
    safe_to_rebuild: bool = True
    next_action: NextAction = "rebuild_same_writer"
    raw_text: str = ""  # the full reviewer response, for the UI

    def display_text(self) -> str:
        """What we render in the review-page section. Strips the JSON
        block (the prose carries the actual content for the human;
        the JSON is for the orchestrator)."""
        return _strip_json_block(self.raw_text).strip() or self.raw_text.strip()


# Schema instructions appended to REVIEWER_INSTRUCTION so the model
# knows the orchestrator parses its output. Kept concise — long
# specs make small models drift.
SCHEMA_INSTRUCTION = """\
After your prose review, append a SINGLE JSON object inside a fenced
```json``` block. Schema:

```json
{
  "pass": false,
  "blockers": ["short summary of each issue, one per array entry"],
  "safe_to_rebuild": true,
  "recommended_next_action": "rebuild_same_writer"
}
```

Field rules:
- `pass`: true only when there are NO blockers; false otherwise.
- `blockers`: at most 8 short strings. Empty array if pass is true.
- `safe_to_rebuild`: false only if the task is fundamentally
  impossible on the stated hardware/OS, otherwise true.
- `recommended_next_action` MUST be one of:
  - "rebuild_same_writer" — normal repairable bugs (typos, missing
    imports, control-flow issues). The same writer can fix these.
  - "rebuild_different_writer" — the writer keeps making the same
    domain / hardware-grounding mistake (e.g. claims BLE works on
    wlan0). A different model is more likely to succeed.
  - "abort" — the task can't be done as specified. Examples: hardware
    not present, fundamentally impossible operation. Stop the loop.
  - "approve" — code is correct; reviewer has no blockers. Use only
    when `pass` is true.

Output ONLY one such JSON block. Do not nest or repeat."""


_FENCED_JSON_RE = re.compile(
    r"```\s*json\s*\n(?P<body>\{.*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)


def _extract_json_block(text: str) -> str | None:
    """Pull a fenced ```json {...} ``` block from the reviewer's
    response. We match the LAST one so the reviewer can include
    examples earlier without confusing us."""
    matches = list(_FENCED_JSON_RE.finditer(text))
    if not matches:
        return None
    return matches[-1].group("body")


def _strip_json_block(text: str) -> str:
    """Inverse of _extract_json_block — used for display so the
    rendered review section doesn't have raw JSON dangling at the
    bottom."""
    return _FENCED_JSON_RE.sub("", text)


def parse(text: str) -> ReviewerVerdict:
    """Parse a reviewer's response into a structured verdict.

    On any parse failure (no JSON block, malformed JSON, missing
    fields), we return a "couldn't parse, treat as repairable" verdict
    so the orchestrator keeps making forward progress. The raw_text is
    always populated so the UI can fall back to showing the prose.
    """
    if not text or not text.strip():
        return ReviewerVerdict(
            passed=True,  # vacuously
            next_action="approve",
            raw_text=text,
        )

    block = _extract_json_block(text)
    if block is None:
        # No JSON at all — treat the prose as a list of one blocker.
        return ReviewerVerdict(
            passed=False,
            blockers=[text.strip()[:500]],
            safe_to_rebuild=True,
            next_action="rebuild_same_writer",
            raw_text=text,
        )

    try:
        data = json.loads(block)
    except json.JSONDecodeError as exc:
        logger.warning("Reviewer JSON unparseable: %s", exc)
        return ReviewerVerdict(
            passed=False,
            blockers=[f"reviewer emitted unparseable JSON: {exc}"],
            safe_to_rebuild=True,
            next_action="rebuild_same_writer",
            raw_text=text,
        )

    if not isinstance(data, dict):
        return ReviewerVerdict(
            passed=False,
            blockers=["reviewer JSON wasn't an object"],
            raw_text=text,
        )

    passed = bool(data.get("pass", False))
    raw_blockers = data.get("blockers") or []
    blockers = [str(b) for b in raw_blockers if isinstance(b, str)]
    safe = bool(data.get("safe_to_rebuild", True))
    action_raw = str(data.get("recommended_next_action") or "").strip()
    if action_raw not in VALID_ACTIONS:
        # Unknown action → fall back to safest interpretation.
        action_raw = "approve" if passed else "rebuild_same_writer"

    # Force consistency: if the model says pass=true the orchestrator
    # MUST short-circuit to approve, even if the model also wrote
    # `recommended_next_action: rebuild_same_writer` (a contradictory
    # combination some local models emit). Without this, the review
    # loop runs another writer round on already-approved code.
    if passed and action_raw != "approve":
        action_raw = "approve"
    # Inverse — pass=false but action="approve" — bumps to the safest
    # repairable interpretation.
    if not passed and action_raw == "approve":
        action_raw = "rebuild_same_writer"

    return ReviewerVerdict(
        passed=passed,
        blockers=blockers,
        safe_to_rebuild=safe,
        next_action=action_raw,  # type: ignore[arg-type]
        raw_text=text,
    )
