"""Fetch corpus sources defined in corpus.yaml.

Idempotent: skips sources whose target dir already exists unless --update is set.
Uses git sparse-checkout to keep clones small for repos with path_filter.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Source:
    id: str
    tier: int
    kind: str
    url: str
    path_filter: list[str]
    description: str
    license: str
    category: str
    language: str | None


def load_manifest(manifest_path: Path) -> tuple[dict, list[Source]]:
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    sources = [
        Source(
            id=s["id"],
            tier=int(s.get("tier", 0)),
            kind=s["kind"],
            url=s["url"],
            path_filter=list(s.get("path_filter", [])),
            description=s.get("description", ""),
            license=s.get("license", ""),
            category=s.get("category", ""),
            language=s.get("language"),
        )
        for s in manifest.get("sources", [])
    ]
    return manifest, sources


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"  $ {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def fetch_git_repo(source: Source, dest: Path, depth: int) -> None:
    if dest.exists():
        print(f"[{source.id}] target exists; skipping (use --update to re-fetch)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)

    if source.path_filter:
        run([
            "git", "clone",
            "--depth", str(depth),
            "--filter=blob:none",
            "--no-checkout",
            source.url,
            str(dest),
        ])
        run(["git", "sparse-checkout", "init", "--cone"], cwd=dest)
        run(["git", "sparse-checkout", "set", *source.path_filter], cwd=dest)
        run(["git", "checkout"], cwd=dest)
    else:
        run([
            "git", "clone",
            "--depth", str(depth),
            source.url,
            str(dest),
        ])


def fetch_zip(source: Source, dest: Path) -> None:
    if dest.exists():
        print(f"[{source.id}] target exists; skipping")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".zip.tmp")
    print(f"  downloading {source.url}", flush=True)
    urllib.request.urlretrieve(source.url, tmp)
    print(f"  extracting to {dest}", flush=True)
    with zipfile.ZipFile(tmp) as zf:
        zf.extractall(dest)
    tmp.unlink(missing_ok=True)


def fetch_raw(source: Source, dest: Path) -> None:
    if dest.exists():
        print(f"[{source.id}] target exists; skipping")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {source.url}", flush=True)
    urllib.request.urlretrieve(source.url, dest)


def fetch_source(source: Source, data_dir: Path, depth: int, update: bool) -> bool:
    dest = data_dir / source.id
    if update and dest.exists():
        print(f"[{source.id}] removing existing dir for --update")
        shutil.rmtree(dest)

    print(f"\n[{source.id}] tier={source.tier} kind={source.kind}  {source.description}")
    try:
        if source.kind == "git_repo":
            fetch_git_repo(source, dest, depth)
        elif source.kind == "zip_url":
            fetch_zip(source, dest)
        elif source.kind == "raw_url":
            fetch_raw(source, dest)
        else:
            print(f"[{source.id}] unknown kind {source.kind!r}; skipping")
            return False
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[{source.id}] FAILED: {exc}")
        return False
    except Exception as exc:
        print(f"[{source.id}] FAILED: {exc!r}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch corpus sources")
    parser.add_argument("--manifest", default="corpus.yaml", help="Path to manifest")
    parser.add_argument("--tiers", default="", help="Comma-separated tiers to fetch (default: all)")
    parser.add_argument("--ids", default="", help="Comma-separated source ids to fetch (default: all)")
    parser.add_argument("--update", action="store_true", help="Re-fetch existing sources")
    args = parser.parse_args()

    manifest_path = REPO_ROOT / args.manifest
    manifest, sources = load_manifest(manifest_path)

    settings = manifest.get("settings", {})
    data_dir = REPO_ROOT / settings.get("data_dir", "data/corpus")
    depth = int(settings.get("default_clone_depth", 1))

    selected_tiers = {int(t) for t in args.tiers.split(",") if t.strip()} if args.tiers else None
    selected_ids = {s.strip() for s in args.ids.split(",") if s.strip()} if args.ids else None

    targets = [
        s for s in sources
        if (selected_tiers is None or s.tier in selected_tiers)
        and (selected_ids is None or s.id in selected_ids)
    ]
    if not targets:
        print("No sources matched the filters.")
        return 1

    print(f"Fetching {len(targets)} source(s) into {data_dir}")
    succeeded = 0
    failed = 0
    for source in targets:
        if fetch_source(source, data_dir, depth, args.update):
            succeeded += 1
        else:
            failed += 1

    print(f"\nDone. {succeeded} succeeded, {failed} failed.")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
