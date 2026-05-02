"""Build a portable Windows bundle for the airgapped LocalLLM deployment.

Run this on the home machine after:
  - models pulled via `ollama pull granite4 gemma4 qwen3-coder:30b nomic-embed-text`
  - corpus indexed via `python scripts/build_index.py`

Output goes to dist/LocalLLM-Bundle/ by default. Transfer that whole directory
to the airgapped machine (USB/sneakernet) and follow bundle/INSTALL.md there.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODELS = [
    "granite4:latest",
    "gemma4:latest",
    "qwen3-coder:30b",
    "nomic-embed-text:latest",
]

APP_INCLUDES = [
    "app",
    "scripts",
    "corpus.yaml",
    "requirements.txt",
    ".env",
]

APP_IGNORE = shutil.ignore_patterns(
    ".git", ".gitignore", "__pycache__", "*.pyc",
    ".venv", "venv", ".pytest_cache",
    ".vscode", ".idea",
    "data", "logs", "dist", "node_modules",
)


def find_ollama_models_dir() -> Path:
    env_dir = os.environ.get("OLLAMA_MODELS")
    if env_dir:
        return Path(env_dir)
    return Path.home() / ".ollama" / "models"


def parse_model_manifest(models_dir: Path, model: str) -> tuple[Path, list[str]]:
    name, _, tag = model.partition(":")
    if not tag:
        tag = "latest"
    manifest_path = models_dir / "manifests" / "registry.ollama.ai" / "library" / name / tag
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found for {model} (expected at {manifest_path}). "
            f"Run `ollama pull {model}` first."
        )
    with manifest_path.open() as f:
        manifest = json.load(f)
    digests: list[str] = [layer["digest"] for layer in manifest.get("layers", [])]
    config = manifest.get("config")
    if config and "digest" in config:
        digests.append(config["digest"])
    return manifest_path, digests


def digest_blob_path(models_dir: Path, digest: str) -> Path:
    return models_dir / "blobs" / digest.replace(":", "-")


def copy_app(dst_root: Path) -> None:
    dst = dst_root / "LocalLLM"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for entry in APP_INCLUDES:
        src = REPO_ROOT / entry
        if not src.exists():
            print(f"  skipping missing {entry}")
            continue
        target = dst / entry
        if src.is_dir():
            shutil.copytree(src, target, ignore=APP_IGNORE)
        else:
            shutil.copy2(src, target)
        print(f"  copied {entry}")


def copy_index(dst_root: Path) -> bool:
    src = REPO_ROOT / "data" / "index"
    if not src.exists() or not any(src.iterdir()):
        print("  no Chroma index at data/index — skipping")
        print("  (run scripts/build_index.py to populate)")
        return False
    dst = dst_root / "LocalLLM" / "data" / "index"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    size_mb = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file()) / 1024 / 1024
    print(f"  copied Chroma index ({size_mb:.1f} MB)")
    return True


def copy_ollama_models(models: list[str], models_dir: Path, dst_root: Path) -> None:
    if not models_dir.exists():
        print(f"  Ollama models dir not found at {models_dir}")
        print("  set OLLAMA_MODELS env var or skip with --skip-models")
        sys.exit(2)

    dst = dst_root / "ollama_models"
    manifests_dst = dst / "manifests"
    blobs_dst = dst / "blobs"
    manifests_dst.mkdir(parents=True, exist_ok=True)
    blobs_dst.mkdir(parents=True, exist_ok=True)

    blobs_to_copy: set[str] = set()
    for model in models:
        manifest_src, digests = parse_model_manifest(models_dir, model)
        manifest_rel = manifest_src.relative_to(models_dir / "manifests")
        manifest_dst = manifests_dst / manifest_rel
        manifest_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_src, manifest_dst)
        blobs_to_copy.update(digests)
        print(f"  {model}: 1 manifest + {len(digests)} blobs")

    total_bytes = 0
    skipped = 0
    for digest in sorted(blobs_to_copy):
        src = digest_blob_path(models_dir, digest)
        if not src.exists():
            print(f"  WARN: missing blob {digest} (referenced by manifest)")
            continue
        dst_blob = blobs_dst / src.name
        if dst_blob.exists() and dst_blob.stat().st_size == src.stat().st_size:
            skipped += 1
            continue
        shutil.copy2(src, dst_blob)
        total_bytes += src.stat().st_size

    total_gb = total_bytes / 1024 / 1024 / 1024
    print(f"  {len(blobs_to_copy)} unique blobs, {skipped} unchanged, {total_gb:.2f} GB copied")


def download_python_wheels(dst_root: Path, platform: str, py_version: str) -> None:
    dst = dst_root / "wheels"
    dst.mkdir(parents=True, exist_ok=True)
    requirements = REPO_ROOT / "requirements.txt"
    cmd = [
        sys.executable, "-m", "pip", "download",
        "-r", str(requirements),
        "-d", str(dst),
        "--platform", platform,
        "--python-version", py_version,
        "--only-binary=:all:",
    ]
    print(f"  $ {' '.join(cmd)}")
    subprocess.check_call(cmd)
    n_wheels = len(list(dst.glob("*.whl")))
    print(f"  downloaded {n_wheels} wheel(s) for {platform}")


def copy_templates(dst_root: Path) -> None:
    src = REPO_ROOT / "bundle_templates"
    if not src.exists():
        print("  WARN: bundle_templates/ not found — bundle will lack install/start scripts")
        return
    for name in ["install.bat", "start.bat", "INSTALL.md", "README.md"]:
        s = src / name
        if s.exists():
            shutil.copy2(s, dst_root / name)
            print(f"  copied {name}")
    installers = dst_root / "installers"
    installers.mkdir(exist_ok=True)
    placeholder = installers / "PUT_INSTALLERS_HERE.txt"
    if not placeholder.exists():
        placeholder.write_text(
            "Place these files in this directory before transferring the bundle:\n"
            "  - OllamaSetup.exe         (https://ollama.com/download/OllamaSetup.exe)\n"
            "  - python-3.12.x-amd64.exe (https://www.python.org/downloads/)\n",
            encoding="utf-8",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build airgap bundle")
    parser.add_argument("--out", default="dist/LocalLLM-Bundle", help="Output bundle root")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Ollama models to bundle")
    parser.add_argument("--platform", default="win_amd64", help="pip wheel platform tag")
    parser.add_argument("--python-version", default="3.12", help="Target Python version for wheels")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-wheels", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    args = parser.parse_args()

    out = (REPO_ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"Building bundle in {out}\n")

    print("[1/5] App code")
    copy_app(out)

    print("\n[2/5] Chroma index")
    if args.skip_index:
        print("  skipped")
    else:
        copy_index(out)

    print("\n[3/5] Python wheels")
    if args.skip_wheels:
        print("  skipped")
    else:
        download_python_wheels(out, args.platform, args.python_version)

    print("\n[4/5] Ollama models")
    if args.skip_models:
        print("  skipped")
    else:
        copy_ollama_models(args.models, find_ollama_models_dir(), out)

    print("\n[5/5] Bundle templates")
    copy_templates(out)

    print(f"\nBundle ready at: {out}")
    print("Before transferring: place OllamaSetup.exe and python-3.12.x-amd64.exe in installers/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
