"""Cheap token-count estimate for context-window guard rails.

Not a tokenizer — just a ratio that's close enough to surface "this prompt
is going to overflow num_ctx" before we send it. Counts whitespace-split
words and applies a 1.3× multiplier (empirical for nomic/llama-family
tokenizers on English prose; code skews higher).
"""
from __future__ import annotations


def estimate_tokens(text: str | None) -> int:
    if not text:
        return 0
    words = text.split()
    if not words:
        return 0
    # +1 covers the BOS/EOS overhead of a single message; better to slightly
    # over-estimate than to miss-fit.
    return int(len(words) * 1.3) + 1


def estimate_messages_tokens(messages) -> int:
    """Sum of estimates over a list of objects with a `.content` attribute."""
    return sum(estimate_tokens(getattr(m, "content", "")) for m in messages)
