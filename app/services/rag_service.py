"""Query-time RAG retrieval: vector + BM25 + RRF, with optional LLM rerank."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from ollama import Client

from app.config import get_settings
from app.services.rag_fusion import reciprocal_rank_fusion
from app.services.rag_tokenize import tokenize
from app.utils.bm25_io import load_bm25


logger = logging.getLogger("localllm")


@dataclass
class Retrieval:
    document: str
    source_id: str
    file_path: str
    category: str
    language: str
    score: float


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
        self._reranker = None
        self.last_error: str | None = None

    # ------------------------------------------------------------------
    # Lazy resource accessors
    # ------------------------------------------------------------------

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
        bm25_path = self.index_dir / "bm25.json"
        legacy = self.index_dir / "bm25.pkl"
        if legacy.exists() and not bm25_path.exists():
            logger.warning(
                "Found legacy bm25.pkl at %s — refusing to load (pickle = RCE). "
                "Re-run scripts/build_index.py to produce bm25.json.",
                legacy,
            )
            return None, None
        loaded = load_bm25(bm25_path)
        if loaded is None:
            return None, None
        self._bm25, self._bm25_chunk_ids = loaded
        return self._bm25, self._bm25_chunk_ids

    def _ensure_reranker(self):
        if not self.settings.rag_rerank_enabled:
            return None
        if self._reranker is None:
            from app.services.rag_reranker import LLMReranker  # lazy import
            self._reranker = LLMReranker(
                ollama_host=self.settings.ollama_host,
                model=self.settings.rag_rerank_model,
            )
        return self._reranker

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        return self.count() > 0

    def count(self) -> int:
        try:
            return self._ensure_collection().count()
        except Exception:
            return 0

    def retrieve(self, query: str, k: int | None = None) -> list[Retrieval]:
        self.last_error = None
        stripped = query.strip()
        if not stripped or len(stripped) < self.settings.rag_min_query_len:
            return []
        k = k or self.settings.rag_retrieval_k

        reranker = self._ensure_reranker()
        pool_size = self.settings.rag_rerank_pool if reranker else k
        candidate_k = max(pool_size * 4, 20)

        vec_ids, vec_distances = self._vector_candidates(query, candidate_k)
        bm25_ids = self._bm25_candidates(query, candidate_k)
        top_ids, fused = self._fuse(vec_ids, bm25_ids, pool_size)
        if not top_ids:
            return []

        retrievals = self._hydrate(top_ids, vec_distances, fused)
        if not retrievals:
            return []

        if reranker is not None and len(retrievals) > 1:
            try:
                return reranker.rerank(query, retrievals, top_k=k)
            except Exception as exc:
                logger.exception("Reranker failed; falling back to fusion order")
                self.last_error = f"rerank: {exc}"
        return retrievals[:k]

    def format_context(
        self,
        retrievals: list[Retrieval],
        max_chars: int | None = None,
    ) -> str:
        if not retrievals:
            return ""
        cap = max_chars or self.settings.rag_max_context_chars
        parts: list[str] = []
        used = 0
        for r in retrievals:
            block = f"--- {r.source_id}:{r.file_path} ({r.language}) ---\n{r.document}\n"
            if used + len(block) > cap:
                break
            parts.append(block)
            used += len(block)
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Retrieval internals
    # ------------------------------------------------------------------

    def _vector_candidates(
        self,
        query: str,
        candidate_k: int,
    ) -> tuple[list[str], dict[str, float]]:
        try:
            collection = self._ensure_collection()
            embedding = self._ollama.embeddings(
                model=self.settings.rag_embedding_model,
                prompt=query,
            )["embedding"]
            results = collection.query(query_embeddings=[embedding], n_results=candidate_k)
        except Exception as exc:
            logger.exception("Vector retrieval failed")
            self.last_error = f"vector: {exc}"
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
        except Exception as exc:
            logger.exception("BM25 scoring failed")
            self.last_error = f"bm25: {exc}"
            return []
        ranked = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        return [chunk_ids[i] for i in ranked[:candidate_k] if float(scores[i]) > 0.0]

    def _fuse(
        self,
        vec_ids: list[str],
        bm25_ids: list[str],
        pool_size: int,
    ) -> tuple[list[str], dict[str, float] | None]:
        """Fuse the two ranked id lists. Returns (top_ids, fused_scores).
        fused_scores is None if only one retriever produced results."""
        if vec_ids and bm25_ids:
            scores = reciprocal_rank_fusion([vec_ids, bm25_ids])
            top = sorted(scores, key=lambda i: scores[i], reverse=True)[:pool_size]
            return top, scores
        if vec_ids:
            return vec_ids[:pool_size], None
        if bm25_ids:
            return bm25_ids[:pool_size], None
        return [], None

    def _hydrate(
        self,
        top_ids: list[str],
        vec_distances: dict[str, float],
        fused: dict[str, float] | None,
    ) -> list[Retrieval]:
        try:
            collection = self._ensure_collection()
            data = collection.get(ids=top_ids)
        except Exception as exc:
            logger.exception("Failed to hydrate candidates from Chroma")
            self.last_error = f"hydrate: {exc}"
            return []

        by_id: dict[str, tuple[str, dict]] = {}
        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or [{}] * len(ids)
        for cid, doc, meta in zip(ids, docs, metas):
            by_id[cid] = (doc, meta or {})

        out: list[Retrieval] = []
        for cid in top_ids:
            if cid not in by_id:
                continue
            doc, meta = by_id[cid]
            if cid in vec_distances:
                score = 1.0 - float(vec_distances[cid])
            elif fused is not None:
                score = fused.get(cid, 0.0)
            else:
                score = 0.0
            out.append(Retrieval(
                document=doc,
                source_id=meta.get("source_id", ""),
                file_path=meta.get("file_path", ""),
                category=meta.get("category", ""),
                language=meta.get("source_language") or meta.get("language", ""),
                score=score,
            ))
        return out
