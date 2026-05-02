# LocalLLM — Airgap install

## Before transferring the bundle

The build script does **not** auto-download the Ollama and Python installers
(URLs change, you may want a specific version). Drop these into `installers/`
on the home machine before sneakernetting:

| File | Source |
|---|---|
| `OllamaSetup.exe` | <https://ollama.com/download/OllamaSetup.exe> |
| `python-3.12.x-amd64.exe` | <https://www.python.org/downloads/> (any 3.12 release) |

The bundle's `installers/PUT_INSTALLERS_HERE.txt` has the same reminder.

## On the airgapped machine

1. Copy the entire bundle directory somewhere stable, e.g. `C:\LocalLLM-Bundle\`.
2. Open an **Administrator** PowerShell or cmd, `cd` to that directory.
3. Run `install.bat`. It will:
   - Install Python 3.12 silently and add it to PATH
   - Install Ollama silently
   - Copy the bundled models into `%USERPROFILE%\.ollama\models\`
   - Create a Python venv inside `LocalLLM\.venv\`
   - Install all Python deps from the bundled wheels (no network needed)
4. **Open a fresh shell** (so the new PATH takes effect), then run `start.bat`.

The Streamlit UI opens at <http://localhost:8501>.

## What's in the bundle

```
LocalLLM-Bundle/
├── installers/                  ← drop OllamaSetup.exe + Python installer here
├── ollama_models/               Pre-pulled LLM and embedding models
│   ├── manifests/
│   └── blobs/
├── wheels/                      Python deps as .whl files (offline pip)
├── LocalLLM/                    The app
│   ├── app/
│   ├── scripts/
│   ├── data/index/              Pre-built Chroma RAG index
│   ├── corpus.yaml
│   ├── requirements.txt
│   └── .env
├── install.bat
├── start.bat
├── INSTALL.md  (this file)
└── README.md
```

## Updating the airgap deployment

When the corpus, models, or app code changes on the home machine:

1. Re-run `python scripts/build_bundle.py` on the home machine.
2. Transfer the updated bundle (only the changed files actually need to move —
   diffing manually with rsync/robocopy works).
3. Re-run `install.bat` on the airgap box. It overwrites in place and is
   safe to re-run.

If you only changed the RAG index, you don't need to re-run install.bat —
just replace `LocalLLM\data\index\` with the new directory.

## Performance expectations on the RTX 3070 (8 GB VRAM)

| Model | Size | VRAM use | Speed |
|---|---|---|---|
| `granite4:latest` (3 B) | 2.1 GB | full GPU | very fast |
| `granite4:tiny-h` (7 B / 1 B active) | 4.2 GB | full GPU | very fast |
| `gemma4:e4b` (≈4 B) | 9.6 GB | partial offload, 1-2 layers on CPU | fast |
| `qwen3-coder:30b` (30 B / 3.3 B active MoE) | 18 GB | partial offload, most on CPU/RAM | 4-8 tok/s |
| `nomic-embed-text` | 270 MB | full GPU when active | embeddings are fast |

The 128 GB system RAM has plenty of headroom for partial-offload of qwen3-coder.

## Troubleshooting

- **`'python' is not recognized`** — open a new shell after `install.bat`; PATH only refreshes for new processes.
- **`'streamlit' is not recognized`** — the venv didn't activate. Make sure you ran `install.bat` and `start.bat` from the same bundle directory.
- **Streamlit can't reach Ollama** — Ollama may not have auto-started. Run `ollama serve` in a separate terminal.
- **No retrievals shown when "Use RAG" is on** — Confirm `LocalLLM\data\index\chroma.sqlite3` exists and is non-empty. If not, the bundle was built without an index.
- **`pip install` fails offline** — One or more wheels are missing. Re-run the build on the home machine; one of the deps probably doesn't have a `win_amd64` binary wheel and needs source build (rare).
- **Model fails to load** — Check `ollama list` to confirm the model is registered. If a model is missing, the manifest was bundled but blobs were not — re-run the build.
