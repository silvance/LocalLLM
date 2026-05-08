"""Centralized system-prompt assembly.

A hidden `BASELINE` layer is prepended to every model interaction
(chat, review, compare, agent) so anti-hallucination behavior is
consistent regardless of which page or which user-supplied system
prompt is in play. Local quantized models drift toward fabrication
on long answers, especially for things like CVE IDs and API
signatures — the baseline pulls them back without the user having
to remember to re-state the rules every time.

Layered ordering (top-to-bottom in the final system message):
  1. BASELINE      — always-on anti-hallucination guardrails
  2. task-specific — caller-supplied (e.g. agent's research playbook,
                     review's writer/reviewer instructions)
  3. user prompt   — whatever the user typed in the sidebar textarea

Each later layer can refine the earlier ones but should not contradict
them. If a model takes the user's "answer confidently" instruction at
face value and overrides the baseline's "say I don't know," that's a
small-model alignment gap, not a layering bug.

Disable for debugging by setting `LOCALLLM_DISABLE_BASELINE_PROMPT=1`.
Tests rely on `compose()` honoring that env var so we can isolate the
behavior under test from the always-on guardrail.
"""
from __future__ import annotations

import os


BASELINE = """\
Calibration:
- If you don't know something, say so plainly. Do not invent facts to fill gaps.
- Specific details — version numbers, CVE IDs, dates, exact API or module names, \
line numbers, command flags, file paths, URLs, statistics — must come from the \
conversation, retrieved context, or your confident knowledge. If you are guessing, \
mark it explicitly ("I think", "likely", "verify this").
- For code: only use functions, classes, and modules you can name with confidence. \
If unsure of an exact import path or signature, write a placeholder with an inline \
comment ("# verify this import") instead of a fabricated call.
- Citing a source you actually used is better than vague hand-waving; vague \
hand-waving is better than confident-sounding fabrication.
- A partial answer with honest gaps is better than a complete-looking answer with \
made-up specifics."""


def baseline_enabled() -> bool:
    """The operator can opt out via env var. Default: on."""
    raw = os.getenv("LOCALLLM_DISABLE_BASELINE_PROMPT", "").strip().lower()
    return raw not in ("1", "true", "yes", "on")


def compose(*layers: str) -> str:
    """Stack non-empty layers with a blank line between, prepending the
    baseline guardrail (unless disabled). Trims whitespace per layer
    so callers don't have to.

    Returns an empty string if every layer is empty AND the baseline
    is disabled — caller should treat that as "don't add a system
    message at all."
    """
    parts: list[str] = []
    if baseline_enabled():
        parts.append(BASELINE.strip())
    for layer in layers:
        if not layer:
            continue
        cleaned = layer.strip()
        if cleaned:
            parts.append(cleaned)
    return "\n\n".join(parts)
