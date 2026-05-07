"""PyInstaller runtime hook: fix up DLL search paths for chromadb on
Windows under onefile mode.

In onefile mode PyInstaller extracts the bundle into a fresh ``_MEIPASS``
temp dir on every launch. chromadb's hnswlib (compiled extension) loads
sibling DLLs via the standard Windows DLL search order, which doesn't
automatically include the temp extraction dir on Python 3.8+.

We add ``_MEIPASS`` (and ``_MEIPASS\\chromadb`` if present) to the DLL
search path so the import survives. No-op on non-Windows or in dev mode.
"""
from __future__ import annotations

import os
import sys


def _add() -> None:
    if sys.platform != "win32":
        return
    if not getattr(sys, "frozen", False):
        return
    base = getattr(sys, "_MEIPASS", None)
    if not base:
        return
    candidates = [base]
    chroma_dir = os.path.join(base, "chromadb")
    if os.path.isdir(chroma_dir):
        candidates.append(chroma_dir)
    add = getattr(os, "add_dll_directory", None)
    if add is None:
        return
    for path in candidates:
        try:
            add(path)
        except OSError:
            pass


_add()
