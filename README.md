# LocalLLM

A Streamlit chat app over local Ollama models with retrieval-augmented generation, designed to be packaged and deployed to an airgapped Windows machine.

## Stack

- **Models** (via Ollama):
  - `granite4` — IBM Granite 4.0 Micro, fast small generalist
  - `gemma4` — Google Gemma 4 E4B, multimodal generalist
  - `qwen3-coder:30b` — Alibaba Qwen3-Coder 30B (3.3B active, MoE), coding specialist
  - `nomic-embed-text` — embedding model for RAG
- **Routing** — keyword-based router sends each prompt to the best-fit model (`auto` mode); manual override via sidebar
- **RAG** — Chroma vector store + BM25 + Reciprocal Rank Fusion over a configurable corpus
- **UI** — Streamlit app with a side-by-side `Compare Models` page
- **Airgap deploy** — `scripts/build_bundle.py` produces a self-contained directory for sneakernet to a target Windows machine

## Quickstart (home machine, from source)

```powershell
# 1. Install Ollama and pull models
ollama pull granite4 gemma4 qwen3-coder:30b nomic-embed-text

# 2. Create the venv + install ALL Python deps in one shot.
#    This is the "from source" setup — distinct from the bundle's
#    install.bat, which targets the prebuilt LocalLLM.exe and has
#    its deps baked in.
powershell -ExecutionPolicy Bypass -File scripts\dev-setup.ps1
# Linux/macOS: ./scripts/dev-setup.sh

# 3. Configure
Copy-Item .env.example .env

# 4. (Optional) Build the RAG corpus
python scripts/fetch_corpus.py --tiers 1
python scripts/build_index.py

# 5. Run
.\.venv\Scripts\Activate.ps1
python scripts/run_web.py        # FastAPI UI on http://127.0.0.1:8000
```

`scripts/dev-setup.ps1` installs `requirements.txt` + `requirements-agent.txt`
+ `requirements-extras.txt`. Pass `-NoAgent` to skip the online `/agent`
deps if you don't plan to use that page; pass `-NoExtras` to skip the
tree-sitter packages used for AST-aware code chunking. Re-run after every
`git pull` to pick up new dependencies.

## Building the airgap bundle

```powershell
# After indexing the corpus on the home machine:
python scripts/build_bundle.py
# Drop OllamaSetup.exe and python-3.12.x-amd64.exe into dist/LocalLLM-Bundle/installers/
# Transfer the bundle directory to the airgapped machine
# Run install.bat (Administrator), then start.bat
```

See `bundle_templates/INSTALL.md` for the airgap install runbook.

## Project layout

```
app/                    Streamlit app + Ollama adapter + chat/RAG services
scripts/                Corpus fetch, index build, bundle build
bundle_templates/       Files copied into the airgap bundle
corpus.yaml             Manifest of RAG sources (4 tiers, 17 sources)
requirements.txt        Python deps (snapshotted by `pip download` at bundle time)
tests/                  Unit tests (pytest)
```

## Hardware targets

- **Home (build) machine** — AMD 7900 XT (20 GB VRAM), 96 GB RAM, 12700K
- **Airgap (deploy) machine** — RTX 3070 (8 GB VRAM), 128 GB RAM, Xeon Silver 4314

`qwen3-coder:30b` Q4_K_M is ~17 GB; on the 8 GB 3070 it partial-offloads to system RAM. With the model's 3.3B active params (MoE) this still runs at ~4–8 tok/s on the airgapped box.
