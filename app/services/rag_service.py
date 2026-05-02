"""Query-time RAG retrieval: vector search + BM25, fused with RRF."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from ollama import Client

from app.config import get_settings
from app.services.rag_tokenize import tokenize


RRF_K = 60


@dataclass
class Retrieval:
    document: str
    source_id: str
    file_path: str
    category: str
    language: str
    score: float


def _rrf(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


class RAGService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.index_dir = Path(self.settings.rag_index_dir)
        self._chroma = None
        self._collection = None
        self._ollama = Client(host=self.settings.ollama_host)
        self._bm25: Any = None
        self._bm25_chunk_ids: list[str] | None = None
        self._bm25_loaded = False

    def _ensure_collection(self):
        if self._collection is not None:
            return self._collection
        if not self.index_dir.exists():
            raise RuntimeError(
                f"Index dir {self.index_dir} does not exist; run scripts/build_index.py"
            )
        self._chroma = chromadb.PersistentClient(path=str(self.index_dir))
        self._collection = self._chroma.get_or_create_collection(
            self.settings.rag_collection,
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    def _ensure_bm25(self) -> tuple[Any, list[str] | None]:
        if self._bm25_loaded:
            return self._bm25, self._bm25_chunk_ids
        self._bm25_loaded = True
        bm25_path = self.index_dir / "bm25.pkl"
        if not bm25_path.exists():
            return None, None
        try:
            with bm25_path.open("rb") as f:
                data = pickle.load(f)
            self._bm25 = data.get("bm25")
            self._bm25_chunk_ids = list(data.get("chunk_ids") or [])
        except Exception:
            self._bm25 = None
            self._bm25_chunk_ids = None
        return self._bm25, self._bm25_chunk_ids

    def is_ready(self) -> bool:
        try:
            return self._ensure_collection().count() > 0
        except Exception:
            return False

    def count(self) -> int:
        try:
            return self._ensure_collection().count()
        except Exception:
            return 0

    def _vector_candidates(self, query: str, candidate_k: int) -> tuple[list[str], dict[str, float]]:
        try:
            collection = self._ensure_collection()
            embedding = self._ollama.embeddings(
                model=self.settings.rag_embedding_model,
                prompt=query,
            )["embedding"]
            results = collection.query(
                query_embeddings=[embedding],
                n_results=candidate_k,
            )
        except Exception:
            return [], {}
        ids = (results.get("ids") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        return ids, dict(zip(ids, distances))

    def _bm25_candidates(self, query: str, candidate_k: int) -> list[str]:
        bm25, chunk_ids = self._ensure_bm25()
        if bm25 is None or not chunk_ids:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        try:
            scores = bm25.get_scores(query_tokens)
        except Exception:
            return []
        top_idx = sorted(
            range(len(scores)),
            key=lambda i: float(scores[i]),
            reverse=True,
        )[:candidate_k]
        return [chunk_ids[i] for i in top_idx if float(scores[i]) > 0.0]

    def retrieve(self, query: str, k: int | None = None) -> list[Retrieval]:
        if not query.strip():
            return []
        k = k or self.settings.rag_retrieval_k
        candidate_k = max(k * 4, 20)

        vec_ids, vec_distances = self._vector_candidates(query, candidate_k)
        bm25_ids = self._bm25_candidates(query, candidate_k)

        if vec_ids and bm25_ids:
            fused = _rrf([vec_ids, bm25_ids])
            top_ids = sorted(fused, key=lambda x: fused[x], reverse=True)[:k]
            fused_used = True
        elif vec_ids:
            top_ids = vec_ids[:k]
            fused = {}
            fused_used = False
        elif bm25_ids:
            top_ids = bm25_ids[:k]
            fused = {}
            fused_used = False
        else:
            return []

        try:
            collection = self._ensure_collection()
            hydrated = collection.get(ids=top_ids)
        except Exception:
            return []

        by_id: dict[str, tuple[str, dict]] = {}
        h_ids = hydrated.get("ids") or []
        h_docs = hydrated.get("documents") or []
        h_metas = hydrated.get("metadatas") or [{}] * len(h_ids)
        for cid, doc, meta in zip(h_ids, h_docs, h_metas):
            by_id[cid] = (doc, meta or {})

        retrievals: list[Retrieval] = []
        for cid in top_ids:
            if cid not in by_id:
                continue
            doc, meta = by_id[cid]
            if cid in vec_distances:
                score = 1.0 - float(vec_distances[cid])
            elif fused_used:
                score = fused.get(cid, 0.0)
            else:
                score = 0.0
            retrievals.append(Retrieval(
                document=doc,
                source_id=meta.get("source_id", ""),
                file_path=meta.get("file_path", ""),
                category=meta.get("category", ""),
                language=meta.get("source_language") or meta.get("language", ""),
                score=score,
            ))
        return retrievals

    def format_context(self, retrievals: list[Retrieval], max_chars: int | None = None) -> str:
        if not retrievals:
            return ""
        cap = max_chars or self.settings.rag_max_context_chars
        parts: list[str] = []
        used = 0
        for r in retrievals:
            header = f"--- {r.source_id}:{r.file_path} ({r.language}) ---"
            block = f"{header}\n{r.document}\n"
            if used + len(block) > cap:
                break
            parts.append(block)
            used += len(block)
        return "\n".join(parts)
