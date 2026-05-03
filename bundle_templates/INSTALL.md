# LocalLLM — airgap install runbook

## Before transferring the bundle

The build script does not auto-download these because URLs change and you may
need a specific version. Drop into `installers/` on the home machine before
sneakernet:

| File | Source |
|---|---|
| `OllamaSetup.exe` | <https://ollama.com/download/OllamaSetup.exe> |
| `python-3.12.x-amd64.exe` | <https://www.python.org/downloads/> |

`installers/PUT_INSTALLERS_HERE.txt` is the same reminder.

## On the airgapped machine

1. Copy the entire bundle directory somewhere stable, e.g. `C:\LocalLLM-Bundle\`.
2. **Verify integrity BEFORE running install.bat** (running install first creates `LocalLLM\.venv\` and `*.pyc` files which would then mismatch the manifest):
   ```powershell
   powershell -ExecutionPolicy Bypass -File verify.ps1
   ```
   Checks every file in the bundle against `SHA256SUMS.txt`. Mismatches = corrupted transfer or tampering. Re-transfer if anything fails.
3. Open an **Administrator** PowerShell or cmd, `cd` to the bundle, run `install.bat`.
   It will:
   - Show `bundle_stamp.json` (build date, target platform, model list, installer hashes)
   - Run `verify-installers.ps1` to confirm each installer in `installers/` matches the SHA-256 recorded at bundle build time. The operator can pin specific Ollama / Python releases by dropping the binaries in *before* running `build_bundle.py` — the build script hashes them into the stamp, install.bat refuses to run if they don't match later.
   - Install Python 3.12 silently and add it to PATH
   - Install Ollama silently
   - Create `LocalLLM\.venv\` and `pip install` from the bundled wheels (offline)
   - Copy `.env.example` → `.env` if there's no existing config
4. **Open a fresh shell** (so the new PATH applies), then `start.bat`.

> Note: after `install.bat` runs, `verify.ps1` will report mismatches in `LocalLLM/.venv/` and `LocalLLM/__pycache__/` directories — that's expected (those didn't exist at build time). To re-verify the originally-shipped files, run `verify.ps1` only on a fresh extraction of the bundle.

Streamlit opens at <http://127.0.0.1:8501>.

## Bundle-local Ollama instance

`start.bat` does **not** add the bundled models to your existing Ollama install,
and does **not** depend on whatever Ollama service Windows installed at the
default port. Specifically:

- `OLLAMA_MODELS` is set to `<bundle>\ollama_models\` (relative to the bundle).
- `OLLAMA_HOST` is set to `127.0.0.1:11435` (not the default `11434`).
- `start.bat` launches a fresh `ollama serve` against those settings, polls
  `/api/tags` until it's up, then runs Streamlit.

This means the bundle is fully isolated: tearing it down is `del /S` of the
bundle directory; nothing else on the machine sees these models. If you also
have an unrelated Ollama install running at `:11434`, the two coexist.

## What's in the bundle

```
LocalLLM-Bundle/
├── installers/                  ← drop OllamaSetup.exe + Python installer here
├── ollama_models/               Pre-pulled LLM and embedding models (bundle-local)
│   ├── manifests/
│   └── blobs/
├── wheels/                      Python deps as .whl files (offline pip)
├── LocalLLM/                    The app
│   ├── app/
│   ├── scripts/
│   ├── data/index/              Pre-built Chroma RAG index + bm25.pkl
│   ├── corpus.yaml
│   ├── requirements.txt
│   └── .env.example             (operator copies to .env on first install)
├── bundle_stamp.json            Build metadata (date, target, model list)
├── SHA256SUMS.txt               Integrity manifest for verify.ps1
├── install.bat
├── start.bat
├── verify.ps1
├── INSTALL.md (this file)
└── README.md
```

## Updating the deployment

When the corpus, models, or app code changes on the home machine:

1. Re-run `python scripts/build_bundle.py`.
2. Transfer the updated bundle. (Robocopy diff is fine — only the changed blobs need to move.)
3. On the airgap box, re-run `install.bat` (idempotent). For an index-only
   change you can just replace `LocalLLM\data\index\`.

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

- **`'python' is not recognized`** — open a new shell after `install.bat`; PATH only refreshes for new processes.
- **`'streamlit' is not recognized`** — venv didn't activate. Make sure `install.bat` and `start.bat` ran from the same bundle directory.
- **`Ollama did not come up within 20 seconds`** — check `ollama-localllm.log` in the bundle root. Could be a port conflict on `:11435` or a missing/corrupt model blob.
- **No retrievals when "Use RAG" is on** — the sidebar will surface a warning if retrieval errored. Check that `LocalLLM\data\index\chroma.sqlite3` exists and is non-empty.
- **`pip install` fails offline** — one or more wheels are missing for the target platform. Re-run `build_bundle.py` on the home machine; one of the deps probably doesn't have a `win_amd64` binary wheel and needs source build (rare).
- **`verify.ps1` reports MISMATCH** — bundle was modified or corrupted in transit. Re-transfer.
