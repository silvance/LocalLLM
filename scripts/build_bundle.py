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
    # Online-only agent variant: never goes onto the airgap target.
    # Excluding the directory keeps internet-tool code paths off-disk
    # entirely on the airgap deploy — no inert attack surface.
    "agent",
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


def copy_localllm_exe(dst_root: Path) -> bool:
    """Copy the PyInstaller-built single-file binary into the bundle root.

    The new bundle templates (install.bat / start.bat) drive everything
    through this exe — Python, FastAPI, ChromaDB, and Ollama are all
    embedded in it. Caller is responsible for having run pyinstaller
    first; we just locate `dist/LocalLLM(.exe)` and copy it.

    Returns True on success, False if no exe was found (still recoverable —
    the legacy wheels-based install path can take over).
    """
    candidates = [
        REPO_ROOT / "dist" / "LocalLLM.exe",
        REPO_ROOT / "dist" / "LocalLLM",
    ]
    for src in candidates:
        if src.exists() and src.is_file():
            dst = dst_root / src.name
            shutil.copy2(src, dst)
            size_mb = dst.stat().st_size / 1024 / 1024
            print(f"  copied {src.name} ({size_mb:.1f} MB)")
            return True
    print("  no LocalLLM(.exe) at dist/ — run `pyinstaller pyinstaller.spec` first")
    print("  (the new install.bat / start.bat templates depend on it)")
    return False


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
    """Download wheels for the bundle's target platform.

    `pip download --platform X` doesn't evaluate environment markers in the
    target platform's context — it uses the host's environment for marker
    evaluation. That breaks transitive deps like uvloop, which has marker
    `sys_platform != "win32" and extra == "standard"`: chromadb requires
    `uvicorn[standard]`, which on a Linux host evaluates the marker as TRUE
    even when targeting Windows, and the resolver fails because uvloop has
    no Windows wheel.

    Workaround: walk the dep tree ourselves with `pip download --no-deps`,
    parse each wheel's METADATA, and evaluate `Requires-Dist` markers with
    `sys_platform="win32"` (or whatever the target is) so transitive deps
    that genuinely don't apply to the target platform are skipped.
    """
    dst = dst_root / "wheels"
    dst.mkdir(parents=True, exist_ok=True)
    requirements = REPO_ROOT / "requirements.txt"

    target_env = _target_marker_env(platform_tag, py_version)

    seen: set[str] = set()
    queue: list[tuple[str, frozenset[str]]] = []

    # Seed the queue from requirements.txt (top-level specs, with their extras).
    for raw_line in requirements.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        spec, extras = _split_extras(line)
        queue.append((spec, extras))

    print(f"  resolving wheels for {platform_tag} (env-marker aware)...")
    while queue:
        spec, requested_extras = queue.pop(0)
        base_name = _normalize_name(_strip_specifier(spec))
        if not base_name or base_name in seen:
            continue
        seen.add(base_name)

        # Download this package only (no deps). pip download still does
        # platform-tag matching, so a package with no win_amd64 wheel will
        # legitimately fail here — and that's the right behaviour.
        cmd = [
            sys.executable, "-m", "pip", "download", spec,
            "--no-deps",
            "-d", str(dst),
            "--platform", platform_tag,
            "--python-version", py_version,
            "--only-binary=:all:",
        ]
        try:
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:
            # If the failure is "no compatible wheel" AND no extras requested
            # us specifically, treat it as a legitimate skip (e.g. uvloop on Windows).
            print(f"    skipping {spec}: no compatible wheel for {platform_tag}")
            continue

        # Crack the just-downloaded wheel's METADATA to find its Requires-Dist
        # entries. Filter them by target-platform markers and requested extras.
        for req_spec, req_extras in _wheel_requirements(dst, base_name, requested_extras, target_env):
            queue.append((req_spec, req_extras))

    n_wheels = len(list(dst.glob("*.whl")))
    print(f"  downloaded {n_wheels} wheel(s) for {platform_tag}")
    return n_wheels


# --- wheel resolver helpers ------------------------------------------------

import re as _re
import zipfile as _zipfile


_SPECIFIER_RE = _re.compile(r"^([A-Za-z0-9_.\-]+)")
_EXTRAS_RE = _re.compile(r"\[([^\]]+)\]")


def _normalize_name(name: str) -> str:
    return _re.sub(r"[-_.]+", "-", name.strip()).lower()


def _strip_specifier(spec: str) -> str:
    """'uvicorn[standard]>=0.18.3' -> 'uvicorn'."""
    spec = spec.strip()
    spec = _re.sub(r";.*$", "", spec).strip()  # drop env marker
    spec = _EXTRAS_RE.sub("", spec).strip()    # drop extras
    m = _SPECIFIER_RE.match(spec)
    return m.group(1) if m else ""


def _split_extras(spec: str) -> tuple[str, frozenset[str]]:
    """Split a requirement spec into (cleaned_spec, extras_set)."""
    spec = spec.strip()
    extras_match = _EXTRAS_RE.search(spec)
    if not extras_match:
        return spec, frozenset()
    extras = frozenset(e.strip() for e in extras_match.group(1).split(",") if e.strip())
    cleaned = _EXTRAS_RE.sub("", spec, count=1)
    return cleaned, extras


def _target_marker_env(platform_tag: str, py_version: str) -> dict:
    """Build a marker-evaluation environment as if we were the target."""
    sys_platform = "win32" if platform_tag.startswith("win") else "linux"
    platform_system = "Windows" if sys_platform == "win32" else "Linux"
    return {
        "sys_platform": sys_platform,
        "platform_system": platform_system,
        "platform_python_implementation": "CPython",
        "implementation_name": "cpython",
        "python_version": py_version,
        "python_full_version": py_version + ".0",
        "os_name": "nt" if sys_platform == "win32" else "posix",
    }


def _wheel_requirements(
    wheel_dir: Path,
    base_name: str,
    requested_extras: frozenset[str],
    target_env: dict,
) -> list[tuple[str, frozenset[str]]]:
    """Parse the wheel's METADATA Requires-Dist lines and return the deps
    that apply for our target platform + requested extras."""
    candidates = sorted(wheel_dir.glob(f"{base_name.replace('-', '_')}*.whl"))
    if not candidates:
        candidates = sorted(wheel_dir.glob(f"{base_name}*.whl"))
    if not candidates:
        return []

    try:
        with _zipfile.ZipFile(candidates[-1]) as z:
            meta_name = next(
                (n for n in z.namelist() if n.endswith(".dist-info/METADATA")), None
            )
            if meta_name is None:
                return []
            meta = z.read(meta_name).decode("utf-8", errors="ignore")
    except Exception:
        return []

    try:
        from packaging.markers import Marker, InvalidMarker
        from packaging.requirements import Requirement, InvalidRequirement
    except ImportError:
        # packaging is in pip's deps and almost always present, but if not
        # we conservatively include all deps (better to fail loudly later).
        return []

    out: list[tuple[str, frozenset[str]]] = []
    for line in meta.splitlines():
        if not line.lower().startswith("requires-dist:"):
            continue
        spec = line.split(":", 1)[1].strip()
        try:
            req = Requirement(spec)
        except InvalidRequirement:
            continue
        # Evaluate the marker with each requested-extra. If the dep applies
        # for the empty extra OR for any of our requested extras, queue it.
        if req.marker is None:
            applies = True
        else:
            extras_to_try = list(requested_extras) + [""]
            applies = False
            for x in extras_to_try:
                env = dict(target_env, extra=x)
                try:
                    if req.marker.evaluate(env):
                        applies = True
                        break
                except Exception:
                    applies = True
                    break
        if not applies:
            continue
        # Re-emit with the dep's own extras preserved (so e.g. if chromadb
        # asks for uvicorn[standard], we propagate {"standard"}).
        out.append((str(req), frozenset(req.extras or [])))
    return out


def copy_templates(dst_root: Path) -> None:
    src = REPO_ROOT / "bundle_templates"
    if not src.exists():
        print("  WARN: bundle_templates/ not found — bundle will lack install/start scripts")
        return
    for name in [
        "install.bat",
        "install.py",
        "start.bat",
        "start.py",
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
    parser.add_argument("--skip-exe", action="store_true",
                        help="Skip copying dist/LocalLLM(.exe) into the bundle.")
    parser.add_argument("--skip-manifest", action="store_true",
                        help="Skip the SHA256 manifest (faster for iterative dev)")
    args = parser.parse_args()

    out = (REPO_ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"Building bundle in {out}\n")

    print("[1/8] LocalLLM single-file binary")
    if args.skip_exe:
        print("  skipped")
    else:
        copy_localllm_exe(out)

    print("\n[2/8] App code")
    copy_app(out)

    print("\n[3/8] Chroma index")
    if args.skip_index:
        print("  skipped")
    else:
        copy_index(out)

    print("\n[4/8] Python wheels")
    wheels_count = 0
    if args.skip_wheels:
        print("  skipped")
    else:
        wheels_count = download_python_wheels(out, args.platform, args.python_version)

    print("\n[5/8] Ollama models")
    models_info: dict = {}
    if args.skip_models:
        print("  skipped")
    else:
        models_info = copy_ollama_models(args.models, find_ollama_models_dir(), out)

    print("\n[6/8] Bundle templates")
    copy_templates(out)

    print("\n[7/8] Bundle stamp")
    write_bundle_stamp(
        out,
        models=args.models if not args.skip_models else [],
        target_platform=args.platform,
        target_py_version=args.python_version,
        wheels_count=wheels_count,
        models_info=models_info,
    )

    print("\n[8/8] SHA256 manifest")
    if args.skip_manifest:
        print("  skipped")
    else:
        write_sha256_manifest(out)

    print(f"\nBundle ready at: {out}")
    print("If you used --skip-exe, run `pyinstaller pyinstaller.spec --clean --noconfirm` first")
    print("and rebuild — install.bat / start.bat depend on LocalLLM.exe being present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
