"""First-install hardware-aware configurator.

Runs once at install time (called by bundle_templates/install.bat after
the Python venv is set up). Detects local hardware, picks a sensible
default model + sidebar selection for the tier, and writes those into
.env so the examiner gets a usable setup without having to learn the
trade-offs.

The bundle itself still ships ALL models — they're all on disk and
all loadable. We just steer the UI defaults so e.g. a CPU-only box
doesn't open with `auto` routing pointing at qwen3-coder:30b (which
would take minutes to load).

Re-runnable any time hardware changes (RAM upgrade, GPU swap):
    python scripts/smart_install.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.utils.hardware_info import (  # noqa: E402
    HardwareInfo,
    Recommendation,
    detect_hardware,
    recommend_models,
    to_dict,
)


# Default sidebar selection per hardware tier. `auto` routing always
# considers all models; this just picks what the dropdown shows on
# first launch so a slow box doesn't open with qwen pre-selected.
TIER_DEFAULT_MODEL: dict[str, str] = {
    "high":      "auto",
    "upper-mid": "auto",
    "mid":       "auto",       # 3070-class — qwen partial-offload OK
    "low-gpu":   "gemma",      # qwen too slow; gemma fits
    "tiny-gpu":  "granite",
    "cpu-high":  "granite",    # CPU-only — start fast, user can promote
    "cpu-low":   "granite",
}


def pick_default_model(rec: Recommendation) -> str:
    return TIER_DEFAULT_MODEL.get(rec.tier, "granite")


def update_env_settings(lines: list[str], updates: dict[str, str]) -> list[str]:
    """Apply key=value updates to a .env content (as a list of lines).

    Replaces the existing key= line if present, otherwise appends. Comment
    lines and blank lines are preserved in place. Returns the new lines.
    """
    seen: set[str] = set()
    result: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            result.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in updates:
            result.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            result.append(line)
    for key, value in updates.items():
        if key not in seen:
            result.append(f"{key}={value}")
    return result


def write_env(env_path: Path, updates: dict[str, str]) -> None:
    """Update env_path with the given updates. If env_path doesn't exist,
    seed from .env.example in the same directory; if THAT doesn't exist
    either, start from an empty file."""
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    else:
        example = env_path.with_name(".env.example")
        if example.exists():
            lines = example.read_text(encoding="utf-8").splitlines()
        else:
            lines = []
    new_lines = update_env_settings(lines, updates)
    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def _print_summary(hw: HardwareInfo, rec: Recommendation, default_model: str) -> None:
    print("=" * 64)
    print("LocalLLM smart install — detected hardware")
    print("=" * 64)
    print(f"  CPU: {hw.cpu_model} ({hw.cpu_cores} cores)")
    print(f"  RAM: {hw.ram_gb} GB")
    if hw.gpus:
        for g in hw.gpus:
            vram = f"{g.vram_gb} GB" if g.vram_gb is not None else "VRAM unknown"
            print(f"  GPU: {g.vendor}: {g.name} ({vram})")
    else:
        print("  GPU: none — CPU-only inference")
    print()
    print(f"Tier: {rec.tier} — {rec.label}")
    print(f"Reasoning: {rec.rationale}")
    print()
    print(f"Recommended models: {', '.join(rec.models)}")
    print(f"  (all four are bundled and on disk; this is which auto-routing prefers)")
    print()
    print(f"Default sidebar selection: {default_model}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Configure LocalLLM defaults based on detected hardware.",
    )
    parser.add_argument(
        "--env", default=str(REPO_ROOT / ".env"),
        help="Path to the .env file to write (default: repo root)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress the human-readable summary; just write the file.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would change but don't touch any files.",
    )
    args = parser.parse_args()

    hw = detect_hardware()
    rec = recommend_models(hw)
    default_model = pick_default_model(rec)

    if not args.quiet:
        _print_summary(hw, rec, default_model)

    updates = {"DEFAULT_MODEL": default_model}

    if args.dry_run:
        print(f"[dry-run] Would write to {args.env}: {updates}")
        return 0

    env_path = Path(args.env)
    env_path.parent.mkdir(parents=True, exist_ok=True)
    write_env(env_path, updates)

    if not args.quiet:
        print(f"Wrote {env_path}")
        print("Done. Re-run this script if hardware changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
