"""Chunk fetched corpus, embed via Ollama, and store in Chroma.

Idempotent: existing chunks (matched by deterministic id) are upserted.
Use --reset to drop the collection and rebuild from scratch.
"""
from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
from pathlib import Path
from typing import Iterator

import chromadb
import yaml
from ollama import Client
from rank_bm25 import BM25Okapi


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.services.rag_chunker import EXT_LANG, chunk_file  # noqa: E402
from app.services.rag_tokenize import tokenize  # noqa: E402

SKIP_DIRS = {".git", "node_modules", "vendor", "_build", "build", "dist", "__pycache__"}
MAX_FILE_SIZE = 500_000


def iter_corpus_files(data_dir: Path) -> Iterator[tuple[Path, Path]]:
    for source_dir in sorted(data_dir.iterdir()):
        if not source_dir.is_dir():
            continue
        for path in source_dir.rglob("*"):
            if not path.is_file():
                continue
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if path.suffix.lower() not in EXT_LANG:
                continue
            try:
                if path.stat().st_size > MAX_FILE_SIZE:
                    continue
            except OSError:
                continue
            yield source_dir, path


def chunk_id(source_id: str, relpath: str, chunk_idx: int) -> str:
    h = hashlib.sha1(f"{source_id}|{relpath}|{chunk_idx}".encode()).hexdigest()[:16]
    return f"{source_id}-{h}"


def load_source_metadata(manifest: dict) -> dict[str, dict]:
    return {
        s["id"]: {
            "category": s.get("category", ""),
            "language": s.get("language") or "",
            "tier": s.get("tier", 0),
        }
        for s in manifest.get("sources", [])
    }


def existing_ids_for_file(collection, source_id: str, relpath: str) -> set[str]:
    try:
        existing = collection.get(where={"$and": [
            {"source_id": source_id},
            {"file_path": relpath},
        ]})
        return set(existing.get("ids", []) or [])
    except Exception:
        return set()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build RAG index")
    parser.add_argument("--manifest", default="corpus.yaml")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument("--collection", default="corpus")
    parser.add_argument("--limit", type=int, default=None, help="Cap files indexed (debug)")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate collection")
    args = parser.parse_args()

    manifest_path = REPO_ROOT / args.manifest
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    source_meta = load_source_metadata(manifest)
    settings = manifest.get("settings", {})
    data_dir = REPO_ROOT / settings.get("data_dir", "data/corpus")
    index_dir = REPO_ROOT / settings.get("index_dir", "data/index")
    index_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"Corpus dir {data_dir} does not exist. Run scripts/fetch_corpus.py first.")
        return 1

    print(f"Index dir:   {index_dir}")
    print(f"Corpus dir:  {data_dir}")
    print(f"Embed model: {args.embed_model}")

    chroma = chromadb.PersistentClient(path=str(index_dir))
    if args.reset:
        try:
            chroma.delete_collection(args.collection)
            print("Reset existing collection.")
        except Exception:
            pass
    collection = chroma.get_or_create_collection(
        args.collection,
        metadata={"hnsw:space": "cosine"},
    )

    ollama = Client(host=args.ollama_host)
    try:
        ollama.embeddings(model=args.embed_model, prompt="health check")
    except Exception as exc:
        print(f"Failed to call Ollama for embeddings: {exc}")
        print(f"Make sure `ollama pull {args.embed_model}` has been run.")
        return 2

    files_seen = 0
    chunks_indexed = 0
    chunks_skipped = 0

    for source_dir, path in iter_corpus_files(data_dir):
        if args.limit is not None and files_seen >= args.limit:
            break
        source_id = source_dir.name
        meta = source_meta.get(source_id, {})
        relpath = str(path.relative_to(source_dir)).replace("\\", "/")
        language = EXT_LANG.get(path.suffix.lower(), "text")

        chunks = chunk_file(path, language)
        files_seen += 1
        if not chunks:
            continue

        already = existing_ids_for_file(collection, source_id, relpath)

        ids: list[str] = []
        documents: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict] = []

        for idx, chunk in enumerate(chunks):
            cid = chunk_id(source_id, relpath, idx)
            if cid in already:
                chunks_skipped += 1
                continue
            try:
                embed = ollama.embeddings(model=args.embed_model, prompt=chunk)["embedding"]
            except Exception as exc:
                print(f"  embed failed for {source_id}:{relpath}#{idx}: {exc}")
                continue
            ids.append(cid)
            documents.append(chunk)
            embeddings.append(embed)
            metadatas.append({
                "source_id": source_id,
                "file_path": relpath,
                "chunk_index": idx,
                "language": language,
                "source_language": meta.get("language") or "",
                "category": meta.get("category", ""),
                "tier": meta.get("tier", 0),
            })

        if ids:
            collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            chunks_indexed += len(ids)

        if files_seen % 50 == 0:
            print(f"  files={files_seen}  indexed={chunks_indexed}  skipped={chunks_skipped}")

    print(f"\nDone. files={files_seen}  indexed={chunks_indexed}  skipped={chunks_skipped}")
    print(f"Collection size: {collection.count()}")

    rebuild_bm25(collection, index_dir)
    return 0


def rebuild_bm25(collection, index_dir: Path) -> None:
    """Fit BM25 over the full Chroma collection and pickle to disk.

    Done at the end of every build so IDF stats reflect the global corpus.
    """
    print("\nRebuilding BM25 index over the full collection...")
    all_data = collection.get(include=["documents"])
    chunk_ids = list(all_data.get("ids") or [])
    documents = list(all_data.get("documents") or [])
    if not documents:
        print("  collection empty; skipping BM25")
        return
    print(f"  tokenizing {len(documents)} chunks...")
    tokenized = [tokenize(d) for d in documents]
    print("  fitting BM25 (Okapi)...")
    bm25 = BM25Okapi(tokenized)
    out = index_dir / "bm25.pkl"
    with out.open("wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids}, f)
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"  saved {out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
