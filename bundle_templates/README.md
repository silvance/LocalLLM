# LocalLLM Airgap Bundle

A self-contained, offline-deployable bundle of:
- **`LocalLLM.exe`** — single-file binary embedding the FastAPI app, ChromaDB,
  the BM25 + reranker pipeline, the pyflakes live linter, and the Ollama
  server itself. Nothing else needs to be installed on the target machine.
- **`ollama_models/`** — pre-pulled Granite 4, Gemma 4, Qwen3-Coder-30B, and
  nomic-embed-text (sit next to the .exe; can't fit inside it at ~25 GB).
- **`SHA256SUMS.txt`** + `verify.ps1` — integrity check for sneakernet transfers.
- **`install.bat`** — one-shot smart-install that detects hardware and writes
  a sensible `DEFAULT_MODEL` into the user's .env. Re-runnable.
- **`start.bat`** — thin wrapper around `LocalLLM.exe` (or just double-click
  the .exe directly).

See `INSTALL.md` for the full runbook.
