"""Reciprocal Rank Fusion. Pure function — kept here so tests don't pull chromadb."""
from __future__ import annotations

RRF_K = 60


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Fuse multiple ranked id lists into a single score per id.

    score(id) = sum over rankings of 1 / (k + rank_in_that_ranking + 1).
    Higher is better.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores
