"""Self-test diagnostics for the LocalLLM single-file binary.

Runs a series of quick checks — paths writable, Ollama discoverable,
RAG index loadable, hardware detected — and prints a labeled status
line for each. Designed to be the first thing an operator runs when
something is wrong: ``LocalLLM.exe doctor`` should usually identify
what to fix.

Each check returns a ``CheckResult``; ``run_all_checks()`` prints them
in order and returns a non-zero exit code if any check FAILed (warns
are exit-code 0 — they're advisories).
"""
from __future__ import annotations

import os
import shutil
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal


Status = Literal["ok", "warn", "fail"]


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    detail: str
    fix: str = ""


# Map status → emoji marker so output is scannable on a console without
# colors (operators may run via plain cmd.exe). Avoids raw "OK"/"FAIL"
# stringly checks at call sites — callers compare on `.status` instead.
_BADGE: dict[Status, str] = {"ok": "[ OK ]", "warn": "[WARN]", "fail": "[FAIL]"}


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_paths() -> CheckResult:
    """exe_dir / user_data_dir / ollama_models_dir resolved and writable."""
    from app.runtime.paths import exe_dir, ollama_models_dir, user_data_dir
    issues: list[str] = []
    for label, path in (
        ("exe_dir", exe_dir()),
        ("user_data", user_data_dir()),
        ("ollama_models", ollama_models_dir()),
    ):
        if not path.exists():
            issues.append(f"{label}={path} (missing)")
        elif not os.access(path, os.W_OK):
            issues.append(f"{label}={path} (not writable)")
    if issues:
        return CheckResult(
            "filesystem layout",
            "fail",
            "; ".join(issues),
            fix="Set LOCALLLM_DATA_DIR to a writable directory and retry.",
        )
    return CheckResult("filesystem layout", "ok",
                       f"user_data={user_data_dir()}")


def check_ollama_binary() -> CheckResult:
    """Embedded or PATH-installed Ollama executable is discoverable."""
    from app.runtime.ollama_supervisor import find_ollama_binary
    binary = find_ollama_binary()
    if binary is None:
        return CheckResult(
            "ollama binary",
            "fail",
            "no ollama executable found",
            fix="Install Ollama, set LOCALLLM_OLLAMA_BIN, or ensure the bundle "
                "ships ollama/ alongside the .exe.",
        )
    return CheckResult("ollama binary", "ok", str(binary))


def check_ollama_reachable() -> CheckResult:
    """OLLAMA_HOST or the supervisor's default port responds to /api/tags.

    Uses urllib only (no ollama-python) so this works even before any
    Ollama integration imports happen. WARN (not FAIL) when unreachable —
    the user may intend to start Ollama themselves later.
    """
    import urllib.error
    import urllib.request
    from app.runtime.ollama_supervisor import DEFAULT_BUNDLED_HOST, DEFAULT_BUNDLED_PORT
    host_env = os.getenv("OLLAMA_HOST")
    if host_env:
        url = host_env.rstrip("/") + "/api/tags"
    else:
        url = f"http://{DEFAULT_BUNDLED_HOST}:{DEFAULT_BUNDLED_PORT}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=2.0) as resp:
            if 200 <= resp.status < 300:
                return CheckResult("ollama reachable", "ok", url)
            return CheckResult("ollama reachable", "warn",
                               f"{url} returned HTTP {resp.status}")
    except (urllib.error.URLError, OSError, ConnectionError) as exc:
        return CheckResult(
            "ollama reachable",
            "warn",
            f"{url}: {exc}",
            fix="Start LocalLLM (the serve command spawns Ollama for you), or "
                "run `ollama serve` separately.",
        )


def check_models_on_disk() -> CheckResult:
    """At least one model manifest exists under OLLAMA_MODELS."""
    from app.runtime.paths import ollama_models_dir
    models_dir = ollama_models_dir()
    manifests = models_dir / "manifests"
    if not manifests.exists():
        return CheckResult(
            "models on disk",
            "fail",
            f"no manifests/ under {models_dir}",
            fix="Copy ollama_models/ from the bundle, or run "
                "`ollama pull granite4` to fetch a starter model.",
        )
    found: list[str] = []
    for tag_file in manifests.rglob("*"):
        if tag_file.is_file():
            # Manifest path encodes registry/library/<model>/<tag>
            rel = tag_file.relative_to(manifests).as_posix().split("/")
            if len(rel) >= 4:
                found.append(f"{rel[-2]}:{rel[-1]}")
    if not found:
        return CheckResult(
            "models on disk",
            "fail",
            f"manifests/ exists but contains no models",
            fix="Run `ollama pull granite4` or copy models from the bundle.",
        )
    return CheckResult("models on disk", "ok",
                       f"{len(found)} model(s): {', '.join(sorted(set(found))[:5])}")


def check_rag_index() -> CheckResult:
    """Chroma index exists and has documents indexed.

    WARN when missing — the chat path works without RAG, the user just
    won't get retrieval-augmented answers.
    """
    try:
        from app.config import get_settings
    except Exception as exc:
        return CheckResult("rag index", "warn", f"settings unreadable: {exc}")
    index_dir = Path(get_settings().rag_index_dir)
    if not index_dir.exists():
        return CheckResult(
            "rag index",
            "warn",
            f"{index_dir} missing — RAG queries will return no context",
            fix="Run `LocalLLM build-index` after fetching the corpus.",
        )
    sqlite = index_dir / "chroma.sqlite3"
    if not sqlite.exists():
        return CheckResult(
            "rag index",
            "warn",
            f"{sqlite} missing",
            fix="Re-run `LocalLLM build-index`.",
        )
    bm25 = index_dir / "bm25.json"
    extras = "with bm25" if bm25.exists() else "no bm25 (vector-only)"
    return CheckResult("rag index", "ok", f"{index_dir} ({extras})")


def check_hardware() -> CheckResult:
    """Hardware detection completes and produces a tier."""
    try:
        from app.utils.hardware_info import detect_hardware, recommend_models
    except Exception as exc:
        return CheckResult("hardware probe", "fail", f"import failed: {exc}")
    try:
        hw = detect_hardware()
        rec = recommend_models(hw)
    except Exception as exc:
        return CheckResult("hardware probe", "fail", str(exc))
    gpu_summary = (
        f"{hw.gpus[0].name} ({hw.gpus[0].vram_gb} GB)"
        if hw.gpus else "no GPU"
    )
    return CheckResult(
        "hardware probe", "ok",
        f"{hw.cpu_cores}c / {hw.ram_gb} GB RAM / {gpu_summary} → tier={rec.tier}",
    )


def check_agent_deps() -> CheckResult:
    """Optional packages for the online /agent page (ddgs / trafilatura /
    httpx). WARN-level — the airgap deploy doesn't ship the agent module
    and shouldn't have these installed."""
    # No agent module on disk → check is irrelevant (airgap build).
    try:
        from app.agent.routes import _missing_agent_deps  # type: ignore
    except ImportError:
        return CheckResult("agent deps", "ok", "agent module not present (airgap build)")
    missing = _missing_agent_deps()
    if not missing:
        return CheckResult("agent deps", "ok", "all importable")
    return CheckResult(
        "agent deps",
        "warn",
        f"missing: {', '.join(missing)} — /agent will refuse to run",
        fix="Run `pip install -r requirements-agent.txt` in the venv that runs uvicorn.",
    )


def check_port_free() -> CheckResult:
    """The default app port (8000) is free, or another LocalLLM is running on it.

    INFO-level: the dispatcher will pick a free port automatically — this
    just helps the user understand if there's a stale process around.
    """
    from app.runtime.ollama_supervisor import is_port_open
    if is_port_open("127.0.0.1", 8000):
        return CheckResult(
            "default port (8000)",
            "warn",
            "something is listening — serve will fall back to a free port",
        )
    return CheckResult("default port (8000)", "ok", "available")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

DEFAULT_CHECKS: tuple[Callable[[], CheckResult], ...] = (
    check_paths,
    check_hardware,
    check_ollama_binary,
    check_ollama_reachable,
    check_models_on_disk,
    check_rag_index,
    check_agent_deps,
    check_port_free,
)


def run_checks(
    checks: Iterable[Callable[[], CheckResult]] = DEFAULT_CHECKS,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    for fn in checks:
        try:
            results.append(fn())
        except Exception as exc:  # noqa: BLE001 — unexpected check crash
            results.append(CheckResult(fn.__name__, "fail", f"check raised: {exc}"))
    return results


def render(results: list[CheckResult]) -> str:
    lines = ["LocalLLM doctor", "=" * 60]
    width = max(len(r.name) for r in results)
    for r in results:
        lines.append(f"{_BADGE[r.status]}  {r.name.ljust(width)}  {r.detail}")
        if r.fix and r.status != "ok":
            lines.append(f"        → {r.fix}")
    fails = sum(1 for r in results if r.status == "fail")
    warns = sum(1 for r in results if r.status == "warn")
    oks = sum(1 for r in results if r.status == "ok")
    lines.append("-" * 60)
    lines.append(f"Summary: {oks} ok, {warns} warn, {fails} fail")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entrypoint for `LocalLLM doctor`. Exit code: 1 if any FAIL, else 0."""
    results = run_checks()
    print(render(results))
    return 1 if any(r.status == "fail" for r in results) else 0
