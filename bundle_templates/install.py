#!/usr/bin/env python3
"""Cross-platform installer for LocalLLM bundles.

This is the Linux/macOS/Windows alternative to install.bat. Runs the
bundled LocalLLM binary's `smart-install` subcommand to configure the
DEFAULT_MODEL based on detected hardware. Re-runnable.

Usage:
    python3 install.py        # Linux / macOS
    python install.py         # Windows (or just double-click install.bat)
"""
from __future__ import annotations

import os
import platform
import stat
import subprocess
import sys
from pathlib import Path


def find_binary(here: Path) -> Path | None:
    """Locate the bundled LocalLLM binary alongside this script."""
    name = "LocalLLM.exe" if platform.system() == "Windows" else "LocalLLM"
    candidate = here / name
    return candidate if candidate.is_file() else None


def ensure_executable(binary: Path) -> None:
    """Set the +x bit on POSIX. PyInstaller's tar/zip archive doesn't
    always preserve permissions through every transfer (USB, samba,
    Windows hosts), so we re-stamp it here."""
    if platform.system() == "Windows":
        return
    mode = binary.stat().st_mode
    binary.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def print_bundle_stamp(here: Path) -> None:
    stamp = here / "bundle_stamp.json"
    if not stamp.exists():
        return
    print("Bundle stamp:")
    try:
        print(stamp.read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"  (could not read bundle_stamp.json: {exc})")
    print()


def main() -> int:
    here = Path(__file__).resolve().parent
    os.chdir(here)

    binary = find_binary(here)
    if binary is None:
        name = "LocalLLM.exe" if platform.system() == "Windows" else "LocalLLM"
        print(f"ERROR: {name} not found in {here}.", file=sys.stderr)
        print("       Make sure you copied the entire bundle, not just install.py.",
              file=sys.stderr)
        return 1

    print("=== LocalLLM Installer ===")
    print_bundle_stamp(here)

    ensure_executable(binary)

    print("[1/1] Detecting hardware and configuring defaults...")
    try:
        result = subprocess.run([str(binary), "smart-install"], check=False)
    except OSError as exc:
        print(f"ERROR: failed to launch {binary}: {exc}", file=sys.stderr)
        return 1
    if result.returncode != 0:
        print("WARNING: smart-install failed; LocalLLM will start with conservative defaults.")

    print()
    if platform.system() == "Windows":
        print("Install complete. Double-click LocalLLM.exe (or run start.bat) to launch.")
    else:
        print(f"Install complete. Run: ./{binary.name}  (or python3 start.py)")
    print(f"Re-run this anytime hardware changes: python3 {Path(__file__).name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
