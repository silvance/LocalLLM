# LocalLLM — single-exe install runbook

## What's in the bundle

```
LocalLLM-Bundle/
├── LocalLLM.exe                 ← The whole app: FastAPI + Chroma + BM25 + embedded Ollama
├── ollama_models/               Pre-pulled LLM and embedding models (sit next to the .exe)
│   ├── manifests/
│   └── blobs/
├── bundle_stamp.json            Build metadata (date, target, model list)
├── SHA256SUMS.txt               Integrity manifest for verify.ps1
├── install.bat                  Runs smart-install once (idempotent)
├── start.bat                    Thin wrapper around LocalLLM.exe
├── verify.ps1
├── INSTALL.md (this file)
└── README.md
```

The .exe is **fully self-contained** — it embeds Python, FastAPI, ChromaDB,
the pyflakes linter, the RAG pipeline, AND the Ollama server itself.
Nothing else needs to be installed on the target machine.

The only thing that can't fit inside the .exe is the LLM model blobs
(~25 GB across all four). They live in `ollama_models/` next to the
binary, where the embedded Ollama looks for them.

## On the airgapped machine

1. Copy the bundle directory somewhere stable, e.g. `C:\LocalLLM-Bundle\`.
2. **Verify integrity BEFORE first launch** (so you're checking the
   exact files that were transferred, not anything written at runtime):
   ```powershell
   powershell -ExecutionPolicy Bypass -File verify.ps1
   ```
   Mismatches = corrupted transfer or tampering. Re-transfer.
3. Run `install.bat` once. It calls `LocalLLM.exe smart-install`, which
   detects this machine's CPU/RAM/GPU and writes a sensible `DEFAULT_MODEL`
   into the user's `.env` so a slow box doesn't open with `qwen3-coder:30b`
   pre-selected. The bundle still ships ALL models — this only steers
   the UI defaults. Re-run anytime via `LocalLLM.exe smart-install` if
   hardware changes.
4. **Double-click `LocalLLM.exe`** (or `start.bat`) to launch.
   - First launch unpacks the embedded payload to `%TEMP%\_MEIxxxxxx\`
     (~5–10 s on a cold disk). Subsequent launches reuse what's still
     in temp where possible.
   - LocalLLM.exe spawns its embedded Ollama on `127.0.0.1:11435`
     against `ollama_models/`, starts the FastAPI app on a free port
     (8765+), and opens your default browser.

## Bundle-local Ollama instance

LocalLLM.exe runs its OWN Ollama on a non-default port (`11435`) so it
can coexist with any other Ollama install you have at `:11434`. Models
are read from `ollama_models/` next to the .exe — the bundle never
touches `%USERPROFILE%\.ollama\` or the default Ollama service port.

Tearing the deployment down is `del /S /Q LocalLLM-Bundle\` —
nothing else on the machine sees these models.

If you actually want LocalLLM to talk to your existing Ollama instead
of spawning its own, launch with `--no-ollama` and set the host:

```
set OLLAMA_HOST=http://127.0.0.1:11434
LocalLLM.exe serve --no-ollama
```

## Subcommands

LocalLLM.exe is also a CLI dispatcher. From a terminal:

```
LocalLLM.exe                       Run the desktop app (default).
LocalLLM.exe serve --no-browser    Run the server only, no browser.
LocalLLM.exe smart-install         Reconfigure DEFAULT_MODEL for current hardware.
LocalLLM.exe recommend-models      Print recommended models for this machine.
LocalLLM.exe build-bundle          Build a new bundle (home-machine workflow).
LocalLLM.exe build-index           Index the corpus (Chroma + BM25).
LocalLLM.exe fetch-corpus          Pull source repos defined in corpus.yaml.
LocalLLM.exe version               Print version + paths + diagnostics.
LocalLLM.exe <subcommand> --help   See options for any subcommand.
```

## Updating the deployment

When the corpus, models, or app code changes on the home machine:

1. On the home machine, build a fresh `LocalLLM.exe` (CI does this on
   every push to `main`; you can also `pyinstaller pyinstaller.spec`
   locally).
2. Re-run `python scripts/build_bundle.py` (or `LocalLLM.exe build-bundle`)
   to assemble the new bundle directory.
3. Sneakernet to the airgap target. Robocopy diff is fine — only the
   changed model blobs need to move.

## Performance expectations on the RTX 3070 (8 GB VRAM)

| Model | Size | VRAM use | Speed |
|---|---|---|---|
| `granite4:latest` (3B Micro) | 2.1 GB | full GPU | very fast |
| `granite4:tiny-h` (7B / 1B active MoE) | 4.2 GB | full GPU | very fast |
| `gemma4` E4B (4B effective) | 9.6 GB | partial offload, 1–2 layers on CPU | fast |
| `qwen3-coder:30b` (30B / 3.3B active MoE) | 18 GB | partial offload, most on CPU/RAM | 4–8 tok/s |
| `nomic-embed-text` | 270 MB | full GPU when active | embeddings are fast |

The 128 GB system RAM has plenty of headroom for partial-offload of qwen3-coder.

## Troubleshooting

- **`LocalLLM.exe` does nothing on double-click** — open `cmd`, `cd` to the
  bundle, run `LocalLLM.exe version`. That prints the discovered paths
  and the location of the embedded Ollama binary; missing entries point
  at the issue.
- **`Ollama did not respond within 30s`** — check `ollama-localllm.log`
  in the user data directory printed by `LocalLLM.exe version`. Could
  be a port conflict on `:11435` or a corrupt model blob.
- **No retrievals when "Use RAG" is on** — sidebar surfaces a warning
  if retrieval errored. Verify `data/index/chroma.sqlite3` exists in the
  bundle root (or wherever `corpus_index_dir` in the .env points).
- **`verify.ps1` reports MISMATCH** — bundle was modified or corrupted in
  transit. Re-transfer.
- **First launch is slow** — onefile extraction. The .exe unpacks ~500 MB
  to `%TEMP%\_MEIxxxxxx\` on first run. Subsequent launches are faster.
