# PyInstaller spec for LocalLLM desktop binary.
#
# Build (locally on the target platform — PyInstaller CAN'T cross-compile):
#     pip install -r requirements.txt pyinstaller
#     pyinstaller pyinstaller.spec --clean --noconfirm
# Output:
#     dist/LocalLLM/LocalLLM        (Linux/macOS — onefile=False, see below)
#     dist/LocalLLM/LocalLLM.exe    (Windows)
#
# Why onefile=False:
#     onefile=True produces a single .exe that unpacks to a temp dir on
#     each launch. That's fine for most apps but chromadb's hnswlib has
#     Windows DLL search-path quirks that get worse from a temp dir.
#     The "onedir" output is still a single folder you can ship — just
#     bigger and contains a few more files alongside the .exe.

# ruff: noqa  (this is a config file, not Python source)
import sys
from pathlib import Path

block_cipher = None
HERE = Path(SPECPATH).resolve()


# --- Data files -----------------------------------------------------------
# Anything our app reads at runtime that isn't a Python module.

datas = [
    # Templates + static for the FastAPI UI
    (str(HERE / "app" / "web" / "templates"), "app/web/templates"),
    (str(HERE / "app" / "web" / "static"), "app/web/static"),
    # corpus.yaml is referenced by scripts/build_index.py and read by the
    # app via its config path
    (str(HERE / "corpus.yaml"), "."),
    # Default env template — the app uses ".env" if present, fallback to
    # config defaults otherwise. Ship the example so users can copy it.
    (str(HERE / ".env.example"), "."),
]

# If the agent module is present (online-connected variant), bundle its
# templates + static too. Excluded from the airgap-target build by removing
# the directory before running pyinstaller.
agent_templates = HERE / "app" / "agent" / "templates"
agent_static = HERE / "app" / "agent" / "static"
if agent_templates.exists():
    datas.append((str(agent_templates), "app/agent/templates"))
if agent_static.exists():
    datas.append((str(agent_static), "app/agent/static"))


# --- Hidden imports -------------------------------------------------------
# chromadb's plugin loader uses runtime imports PyInstaller can't detect
# statically. uvicorn similarly loads its protocols dynamically. Use
# collect_all to grab the full transitive surface for the modules that
# matter, plus explicit hidden imports for things known to bite.

from PyInstaller.utils.hooks import collect_all, collect_submodules

datas_extra = []
binaries_extra = []
hiddenimports = []

for pkg in ("chromadb", "uvicorn", "fastapi", "ollama", "starlette",
            "rank_bm25", "pyflakes"):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(pkg)
    datas_extra += pkg_datas
    binaries_extra += pkg_binaries
    hiddenimports += pkg_hiddenimports

# Modules that PyInstaller's static analysis misses on chromadb's plugin tree
hiddenimports += [
    "chromadb.telemetry.product.posthog",
    "chromadb.api.fastapi",
    "chromadb.api.segment",
    "chromadb.db.impl.sqlite",
    "chromadb.segment.impl.manager.local",
    "chromadb.segment.impl.metadata.sqlite",
    "chromadb.segment.impl.vector.local_persistent_hnsw",
]

datas += datas_extra


# --- Analysis -------------------------------------------------------------

a = Analysis(
    [str(HERE / "scripts" / "run_desktop.py")],
    pathex=[str(HERE)],
    binaries=binaries_extra,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # We don't ship Streamlit through the desktop binary — the legacy
        # UI is dev-only.
        "streamlit", "altair", "pandas", "pyarrow",
        # Common heavy modules pulled in by chromadb's optional clients
        # that we don't actually use:
        "tensorflow", "torch", "transformers",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LocalLLM",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,         # Keep the small console window so users can close it to stop
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon=str(HERE / "icon.ico"),  # add when we have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="LocalLLM",
)
