#!/usr/bin/env python3
"""Cross-platform launcher for LocalLLM bundles.

The Linux/macOS/Windows alternative to start.bat — exec()'s the bundled
LocalLLM binary with whatever args you passed. Closing the terminal
stops the server.

Usage:
    python3 start.py                  # default: serve + open browser
    python3 start.py serve --no-browser
    python3 start.py doctor
"""
from __future__ import annotations

import os
import platform
import stat
import sys
from pathlib import Path


def main() -> int:
    here = Path(__file__).resolve().parent
    os.chdir(here)

    name = "LocalLLM.exe" if platform.system() == "Windows" else "LocalLLM"
    binary = here / name
    if not binary.is_file():
        print(f"ERROR: {name} not found in {here}.", file=sys.stderr)
        print("       Make sure you copied the entire bundle.", file=sys.stderr)
        return 1

    if platform.system() != "Windows":
        # Re-stamp +x in case the bundle was transferred via a fs that
        # didn't preserve POSIX permissions (Windows host, samba, etc.).
        mode = binary.stat().st_mode
        binary.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    args = [str(binary), *sys.argv[1:]]
    if platform.system() == "Windows":
        # exec on Windows can't fully replace the console process; spawn
        # via subprocess.call instead so Ctrl-C reaches the child cleanly.
        import subprocess
        return subprocess.call(args)
    # POSIX: replace this process so signals + exit codes pass through.
    os.execv(str(binary), args)
    return 0  # unreachable


if __name__ == "__main__":
    sys.exit(main())
