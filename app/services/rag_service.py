"""Query-time RAG retrieval service. Backed by Chroma + Ollama embeddings."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from ollama import Client

from app.config import get_settings


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

    def retrieve(self, query: str, k: int | None = None) -> list[Retrieval]:
        if not query.strip():
            return []
        k = k or self.settings.rag_retrieval_k
        try:
            collection = self._ensure_collection()
            embedding = self._ollama.embeddings(
                model=self.settings.rag_embedding_model,
                prompt=query,
            )["embedding"]
            results = collection.query(
                query_embeddings=[embedding],
                n_results=k,
            )
        except Exception:
            return []

        retrievals: list[Retrieval] = []
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            meta = meta or {}
            retrievals.append(Retrieval(
                document=doc,
                source_id=meta.get("source_id", ""),
                file_path=meta.get("file_path", ""),
                category=meta.get("category", ""),
                language=meta.get("source_language") or meta.get("language", ""),
                score=1.0 - float(dist),
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
