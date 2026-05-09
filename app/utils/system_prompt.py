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


# Named runtime-quality anti-patterns the writer keeps producing in
# real review runs even though every prompt-level "write good code"
# guidance is in place. Each entry comes from an observed failure
# in the user's review sessions — the BLE/Wi-Fi sniffer screenshot,
# specifically. These are STRUCTURAL guidance ("don't write X, write
# Y instead") rather than aesthetic ("style your code well") because
# small models route on concrete patterns much better than on abstract
# guidance. The static gate `gate_runtime_anti_patterns` catches these
# AFTER they're produced; this layer is the prevention pass.
REVIEW_WRITER_GUIDANCE = """\
Code-quality requirements (these are real bugs that have shipped from \
prior runs — do not produce any of them):

- Never write `while True: pass` or `while True: continue` to keep a \
process alive. That burns 100% CPU forever. Use `signal.pause()`, \
`time.sleep(...)` in the loop body alongside real work, or a \
`threading.Event().wait()` blocked on the actual termination condition.
- Every `import` must live at the top of the file. NEVER put an `import` \
inside `if __name__ == "__main__":` if the imported name is referenced \
from a module-level function. That works as a script but breaks the \
moment anyone imports the file as a module — the function fires \
NameError. Stdlib imports especially (`shutil`, `os`, `subprocess`) \
belong at module top.
- When you spawn a subprocess and later call `.terminate()` or `.kill()`, \
also call `.wait()` (with a timeout). Otherwise the parent exits before \
the child is reaped and you leave zombies / stale pcap files.
- Verify command-line flags exist before citing them. `btmon --output` \
is NOT a real flag (the real flag is `-w`). If you don't remember a \
specific flag, say so and use a placeholder rather than inventing one \
that looks plausible. Same rule for `iw`, `tshark`, `nmcli`, etc.
- Don't catch `Exception` to print and continue when the failure means \
the next step can't possibly work — either let it propagate or actually \
recover. Empty `except: pass` blocks hide the bug from the operator."""


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


def compose_for_writer(*layers: str) -> str:
    """Same as ``compose()`` but stacks ``REVIEW_WRITER_GUIDANCE``
    after the baseline. The /review writer in particular keeps
    producing the named anti-patterns (busy-loops, conditional
    imports, fabricated CLI flags) that the static gate then has to
    catch — this layer is the prevention pass that names them
    before the model writes the first character. Sits ABOVE the
    user-supplied system prompt so a user who specifically asks for
    "produce a placeholder" can still override.

    Honors the same baseline-disabled env var as ``compose()``."""
    return compose(REVIEW_WRITER_GUIDANCE, *layers)
