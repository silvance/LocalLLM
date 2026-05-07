# PyInstaller spec for the single-file LocalLLM desktop binary.
#
# Build (locally on the target platform — PyInstaller can't cross-compile):
#     pip install -r requirements.txt pyinstaller
#     pyinstaller pyinstaller.spec --clean --noconfirm
# Output:
#     dist/LocalLLM.exe   (Windows)
#     dist/LocalLLM       (Linux/macOS)
#
# Onefile vs onedir: this spec uses onefile so the user gets a single
# .exe to double-click. PyInstaller extracts it to %TEMP%/_MEIxxxxxx on
# every launch (~5–10s the first time on cold disks). The chromadb
# hnswlib DLL-search-path issue that previously discouraged onefile is
# handled by build_support/pyi_rthook_chromadb.py.
#
# Embedded sidecars: if vendor/ollama/<binary> exists at build time, the
# Ollama executable is embedded so the .exe is fully self-contained
# (Ollama lifecycle managed by app/runtime/ollama_supervisor.py at
# launch). The CI workflow downloads it before invoking pyinstaller.
# Locally, you can drop ollama.exe into vendor/ollama/ to test the same
# path.

# ruff: noqa  (this is a config file, not Python source)
import sys
from pathlib import Path

block_cipher = None
HERE = Path(SPECPATH).resolve()


# --- Data files -----------------------------------------------------------
# Anything our app reads at runtime that isn't a Python module.

datas = [
    # Templates + static for the FastAPI UI.
    (str(HERE / "app" / "web" / "templates"), "app/web/templates"),
    (str(HERE / "app" / "web" / "static"), "app/web/static"),
    # corpus.yaml — read by build_index.py; harmless to ship for runtime too.
    (str(HERE / "corpus.yaml"), "."),
    # Default env template — copied to the user's data dir on first run.
    (str(HERE / ".env.example"), "."),
]

# scripts/ is collected as a package so the CLI dispatcher can do
# `import scripts.smart_install` etc. when frozen.
scripts_dir = HERE / "scripts"
if scripts_dir.exists():
    datas.append((str(scripts_dir), "scripts"))

# Bundle templates — needed when running `LocalLLM build-bundle` from
# a frozen .exe on the home machine.
bundle_templates = HERE / "bundle_templates"
if bundle_templates.exists():
    datas.append((str(bundle_templates), "bundle_templates"))

# Pre-built RAG index — optional. CI doesn't include it (no corpus
# available); the home-machine build picks it up automatically when
# data/index/ exists.
rag_index = HERE / "data" / "index"
if rag_index.exists():
    datas.append((str(rag_index), "data/index"))

# Online-agent UI assets — only present in non-airgap builds.
agent_templates = HERE / "app" / "agent" / "templates"
agent_static = HERE / "app" / "agent" / "static"
if agent_templates.exists():
    datas.append((str(agent_templates), "app/agent/templates"))
if agent_static.exists():
    datas.append((str(agent_static), "app/agent/static"))


# --- Embedded binaries (Ollama) ------------------------------------------
# vendor/ollama/ may contain ollama.exe (Windows) or ollama (Linux).
# CI downloads it before invoking pyinstaller. If absent, the .exe still
# works — supervisor falls back to PATH or prints a friendly error.

binaries = []
vendor_ollama = HERE / "vendor" / "ollama"
if vendor_ollama.exists():
    # Recursively include ollama.exe + its runners (CUDA/CPU subfolder).
    # Destination "ollama" matches what supervisor.find_ollama_binary()
    # looks for under _MEIPASS / exe_dir.
    for path in vendor_ollama.rglob("*"):
        if path.is_file():
            rel = path.relative_to(vendor_ollama)
            dest_dir = "ollama"
            if rel.parent != Path("."):
                dest_dir = f"ollama/{rel.parent.as_posix()}"
            binaries.append((str(path), dest_dir))


# --- Hidden imports -------------------------------------------------------
# chromadb's plugin loader uses runtime imports PyInstaller can't detect
# statically. uvicorn similarly loads its protocols dynamically.

from PyInstaller.utils.hooks import collect_all

datas_extra = []
binaries_extra = []
hiddenimports = []

for pkg in ("chromadb", "uvicorn", "fastapi", "ollama", "starlette",
            "rank_bm25", "pyflakes"):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(pkg)
    datas_extra += pkg_datas
    binaries_extra += pkg_binaries
    hiddenimports += pkg_hiddenimports

# Modules that PyInstaller's static analysis misses on chromadb's plugin tree.
hiddenimports += [
    "chromadb.telemetry.product.posthog",
    "chromadb.api.fastapi",
    "chromadb.api.segment",
    "chromadb.db.impl.sqlite",
    "chromadb.segment.impl.manager.local",
    "chromadb.segment.impl.metadata.sqlite",
    "chromadb.segment.impl.vector.local_persistent_hnsw",
    # Subcommand targets — proxied dynamically by app/cli.py via importlib.
    # Listing them here forces PyInstaller's static analysis to include them.
    "scripts.smart_install",
    "scripts.recommend_models",
    "scripts.build_bundle",
    "scripts.build_index",
    "scripts.fetch_corpus",
]

datas += datas_extra
binaries += binaries_extra


# --- Analysis -------------------------------------------------------------

a = Analysis(
    [str(HERE / "app" / "cli.py")],
    pathex=[str(HERE)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[str(HERE / "build_support" / "pyi_rthook_chromadb.py")],
    excludes=[
        # We don't ship Streamlit through the desktop binary — the legacy
        # UI is dev-only.
        "streamlit", "altair", "pandas", "pyarrow",
        # Heavy modules pulled in by chromadb's optional clients we don't use.
        "tensorflow", "torch", "transformers",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)


# --- Onefile EXE ----------------------------------------------------------
# All binaries / data / pyz collapsed into a single self-extracting .exe.

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="LocalLLM",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=True,         # Keep the console — closing it stops the server.
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon=str(HERE / "icon.ico"),  # add when we have one
)
