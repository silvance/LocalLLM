"""Build a portable Windows bundle for the airgapped LocalLLM deployment.

Run on the home machine after:
  - models pulled via `ollama pull granite4 gemma4 qwen3-coder:30b nomic-embed-text`
  - corpus indexed via `python scripts/build_index.py`

Output goes to dist/LocalLLM-Bundle/ by default. Transfer that whole directory
to the airgapped machine (USB/sneakernet) and follow INSTALL.md there.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
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
    ".env.example",
]

APP_IGNORE = shutil.ignore_patterns(
    ".git", ".gitignore", "__pycache__", "*.pyc",
    ".venv", "venv", ".pytest_cache",
    ".vscode", ".idea",
    "data", "logs", "dist", "node_modules",
    ".env",  # never bundle the local .env (operator copies .env.example)
)

# Files (relative to bundle root) that exist for verification meta — skip
# them when computing/verifying the manifest.
MANIFEST_SELF = "SHA256SUMS.txt"


def find_ollama_models_dir() -> Path:
    env_dir = os.environ.get("OLLAMA_MODELS")
    if env_dir:
        return Path(env_dir)
    return Path.home() / ".ollama" / "models"


def parse_model_manifest(models_dir: Path, model: str) -> tuple[Path, list[str]]:
    """Look up an Ollama model's manifest. Tries the standard
    registry.ollama.ai/library/<name>/<tag> path first, then falls back to
    searching any namespace under manifests/."""
    name, _, tag = model.partition(":")
    if not tag:
        tag = "latest"

    primary = models_dir / "manifests" / "registry.ollama.ai" / "library" / name / tag
    if primary.exists():
        return _read_manifest(primary)

    # Fallback: search any namespace, e.g. manifests/<host>/<user>/<name>/<tag>
    for candidate in (models_dir / "manifests").rglob(tag):
        if candidate.is_file() and candidate.parent.name == name:
            return _read_manifest(candidate)

    raise FileNotFoundError(
        f"Manifest not found for {model} under {models_dir / 'manifests'}. "
        f"Run `ollama pull {model}` first, or check that OLLAMA_MODELS points "
        f"to the right directory."
    )


def _read_manifest(manifest_path: Path) -> tuple[Path, list[str]]:
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


def copy_ollama_models(models: list[str], models_dir: Path, dst_root: Path) -> dict:
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
    copied = 0
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
        copied += 1

    total_gb = total_bytes / 1024 / 1024 / 1024
    print(f"  {len(blobs_to_copy)} unique blobs, {skipped} unchanged, "
          f"{copied} new ({total_gb:.2f} GB)")
    return {
        "total_blobs": len(blobs_to_copy),
        "total_bytes": sum(p.stat().st_size for p in blobs_dst.iterdir() if p.is_file()),
    }


def download_python_wheels(dst_root: Path, platform_tag: str, py_version: str) -> int:
    dst = dst_root / "wheels"
    dst.mkdir(parents=True, exist_ok=True)
    requirements = REPO_ROOT / "requirements.txt"
    cmd = [
        sys.executable, "-m", "pip", "download",
        "-r", str(requirements),
        "-d", str(dst),
        "--platform", platform_tag,
        "--python-version", py_version,
        "--only-binary=:all:",
    ]
    print(f"  $ {' '.join(cmd)}")
    subprocess.check_call(cmd)
    n_wheels = len(list(dst.glob("*.whl")))
    print(f"  downloaded {n_wheels} wheel(s) for {platform_tag}")
    return n_wheels


def copy_templates(dst_root: Path) -> None:
    src = REPO_ROOT / "bundle_templates"
    if not src.exists():
        print("  WARN: bundle_templates/ not found — bundle will lack install/start scripts")
        return
    for name in [
        "install.bat",
        "start.bat",
        "verify.ps1",
        "verify-installers.ps1",
        "INSTALL.md",
        "README.md",
    ]:
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


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _scan_installers(dst_root: Path) -> list[dict]:
    """Hash any installer the operator dropped into installers/ before
    bundling. We can't pin Ollama's "latest" URL upstream, but we can pin
    *the binary that's about to ship* by recording its SHA-256 in the
    stamp — install.bat verifies this before running anything.
    """
    installers_dir = dst_root / "installers"
    if not installers_dir.exists():
        return []
    out: list[dict] = []
    for path in sorted(installers_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name == "PUT_INSTALLERS_HERE.txt":
            continue
        out.append({
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _hash_file(path),
        })
    return out


def write_bundle_stamp(
    dst_root: Path,
    models: list[str],
    target_platform: str,
    target_py_version: str,
    wheels_count: int,
    models_info: dict,
) -> None:
    stamp = {
        "schema_version": 2,
        "build_date": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "build_host": {
            "os": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
        },
        "target": {
            "platform": target_platform,
            "python_version": target_py_version,
        },
        "models": list(models),
        "models_total_bytes": models_info.get("total_bytes", 0) if models_info else 0,
        "wheels_count": wheels_count,
        # Empty list if the operator hasn't dropped installers in yet — they
        # can re-run --skip-models --skip-wheels --skip-index to refresh the
        # stamp after adding them. install.bat warns when this is empty.
        "installers": _scan_installers(dst_root),
    }
    out = dst_root / "bundle_stamp.json"
    out.write_text(json.dumps(stamp, indent=2), encoding="utf-8")
    print(f"  wrote {out.name}")


def write_sha256_manifest(dst_root: Path) -> None:
    """Walk the bundle and produce SHA256SUMS.txt at the root.

    Format is sha256sum-compatible: '<hex>  <relpath>' with forward-slash paths
    so it round-trips through both Linux sha256sum -c and Windows verify.ps1.
    """
    print("  hashing bundle (this can take a few minutes for large bundles)...")
    rows: list[tuple[str, str]] = []
    for path in sorted(dst_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(dst_root).as_posix()
        if rel == MANIFEST_SELF:
            continue
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        rows.append((h.hexdigest(), rel))

    out = dst_root / MANIFEST_SELF
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for digest, rel in rows:
            f.write(f"{digest}  {rel}\n")
    print(f"  wrote {MANIFEST_SELF} ({len(rows)} entries)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build airgap bundle")
    parser.add_argument("--out", default="dist/LocalLLM-Bundle", help="Output bundle root")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Ollama models to bundle")
    parser.add_argument("--platform", default="win_amd64", help="pip wheel platform tag")
    parser.add_argument("--python-version", default="3.12", help="Target Python version for wheels")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-wheels", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--skip-manifest", action="store_true",
                        help="Skip the SHA256 manifest (faster for iterative dev)")
    args = parser.parse_args()

    out = (REPO_ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"Building bundle in {out}\n")

    print("[1/7] App code")
    copy_app(out)

    print("\n[2/7] Chroma index")
    if args.skip_index:
        print("  skipped")
    else:
        copy_index(out)

    print("\n[3/7] Python wheels")
    wheels_count = 0
    if args.skip_wheels:
        print("  skipped")
    else:
        wheels_count = download_python_wheels(out, args.platform, args.python_version)

    print("\n[4/7] Ollama models")
    models_info: dict = {}
    if args.skip_models:
        print("  skipped")
    else:
        models_info = copy_ollama_models(args.models, find_ollama_models_dir(), out)

    print("\n[5/7] Bundle templates")
    copy_templates(out)

    print("\n[6/7] Bundle stamp")
    write_bundle_stamp(
        out,
        models=args.models if not args.skip_models else [],
        target_platform=args.platform,
        target_py_version=args.python_version,
        wheels_count=wheels_count,
        models_info=models_info,
    )

    print("\n[7/7] SHA256 manifest")
    if args.skip_manifest:
        print("  skipped")
    else:
        write_sha256_manifest(out)

    print(f"\nBundle ready at: {out}")
    print("Before transferring: place OllamaSetup.exe and python-3.12.x-amd64.exe in installers/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
