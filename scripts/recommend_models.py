"""Detect local hardware and print a recommended model lineup.

Usage:
    python scripts/recommend_models.py            # human-readable summary
    python scripts/recommend_models.py --json     # machine-readable

The recommendation includes copy-paste `ollama pull` commands for any
recommended models that aren't already installed locally.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.utils.hardware_info import (  # noqa: E402
    detect_hardware,
    recommend_models,
    to_dict,
)


def _installed_models() -> list[str]:
    """Best-effort list of model names from `ollama list`. Empty on any
    failure (no Ollama installed, not on PATH, etc.)."""
    if not shutil.which("ollama"):
        return []
    try:
        out = subprocess.check_output(["ollama", "list"], text=True, timeout=5)
    except Exception:
        return []
    names: list[str] = []
    for line in out.strip().splitlines()[1:]:  # skip header row
        parts = line.split()
        if parts:
            names.append(parts[0])
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description="Recommend Ollama models for this machine")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    args = parser.parse_args()

    hw = detect_hardware()
    rec = recommend_models(hw)
    installed = _installed_models()
    payload = to_dict(hw, rec, installed=installed)

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    # Human-readable summary
    print("=" * 60)
    print(f"Detected hardware on {hw.os_name} {hw.os_release}")
    print("=" * 60)
    print(f"  CPU:    {hw.cpu_model}  ({hw.cpu_cores} cores)")
    print(f"  RAM:    {hw.ram_gb} GB")
    if hw.gpus:
        for g in hw.gpus:
            vram = f"{g.vram_gb} GB" if g.vram_gb is not None else "VRAM unknown"
            print(f"  GPU:    {g.vendor}: {g.name}  ({vram})")
    else:
        print("  GPU:    none detected")

    print()
    print(f"Tier: {rec.tier} — {rec.label}")
    print(f"Why:  {rec.rationale}")
    print()
    print("Recommended models:")
    for m in rec.models:
        status = payload["recommendation"]["model_status"][m]
        marker = "✓ installed" if status == "installed" else "✗ missing"
        print(f"  {marker:14} {m}")

    missing = payload["recommendation"]["missing"]
    if missing:
        print()
        print("Pull the missing ones with:")
        for m in missing:
            print(f"  ollama pull {m}")
    else:
        print()
        print("All recommended models are installed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
