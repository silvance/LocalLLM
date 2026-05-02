"""LLM-based re-ranker. Single batched prompt: ask the LLM to score each
(query, passage) pair on 0-10 in a JSON list, then re-sort.

Optional — disabled by default via RAG_RERANK_ENABLED. Adds one extra LLM
call per retrieval, but uses a small fast model (default granite4:tiny-h)
so the overhead is bounded (~1-3 sec on the airgap 3070).
"""
from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from ollama import Client


if TYPE_CHECKING:
    from app.services.rag_service import Retrieval

logger = logging.getLogger("localllm")

PROMPT_TEMPLATE = """\
You are a relevance scorer. Rate each passage's relevance to the query on a
scale of 0-10 (0=irrelevant, 10=directly answers the query).

Output ONLY a JSON array of {n} integers, in the same order as the passages.
No prose, no explanation, no markdown.

Query: {query}

{passages}

Output (JSON array of {n} integers):"""

# Be conservative about how much of each chunk the reranker sees;
# more text = slower scoring and more chance the small model loses focus.
MAX_PASSAGE_CHARS = 600


def _format_passages(retrievals: list, max_chars: int = MAX_PASSAGE_CHARS) -> str:
    blocks: list[str] = []
    for i, r in enumerate(retrievals, start=1):
        snippet = r.document[:max_chars]
        if len(r.document) > max_chars:
            snippet += "…"
        blocks.append(
            f"[Passage {i}] ({r.source_id}:{r.file_path}, {r.language or 'n/a'})\n{snippet}"
        )
    return "\n\n".join(blocks)


def _parse_scores(response: str, expected_count: int) -> list[float] | None:
    """Find a JSON array of numbers and return floats, or None on any malformation."""
    match = re.search(r"\[\s*\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?)*\s*\]", response)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except Exception:
        return None
    if not isinstance(parsed, list) or len(parsed) != expected_count:
        return None
    if not all(isinstance(x, (int, float)) for x in parsed):
        return None
    return [float(x) for x in parsed]


class LLMReranker:
    def __init__(self, ollama_host: str, model: str) -> None:
        self.client = Client(host=ollama_host)
        self.model = model

    def rerank(self, query: str, retrievals: list, top_k: int) -> list:
        if not retrievals:
            return []
        if len(retrievals) <= 1:
            return retrievals[:top_k]

        prompt = PROMPT_TEMPLATE.format(
            n=len(retrievals),
            query=query.strip(),
            passages=_format_passages(retrievals),
        )

        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                options={"temperature": 0.0, "num_predict": 256},
            )
            text = response["message"]["content"]
        except Exception:
            logger.exception("LLM reranker call failed; returning fusion order")
            return retrievals[:top_k]

        scores = _parse_scores(text, len(retrievals))
        if scores is None:
            logger.warning(
                "Reranker output unparseable; returning fusion order. raw=%r",
                text[:200],
            )
            return retrievals[:top_k]

        # Re-sort by reranker score, replacing the score field with the
        # normalized 0-1 reranker score so the UI surfaces it clearly.
        from app.services.rag_service import Retrieval as _R

        scored = sorted(zip(retrievals, scores), key=lambda x: x[1], reverse=True)
        return [
            _R(
                document=r.document,
                source_id=r.source_id,
                file_path=r.file_path,
                category=r.category,
                language=r.language,
                score=s / 10.0,
            )
            for r, s in scored[:top_k]
        ]
