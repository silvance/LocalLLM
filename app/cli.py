"""LocalLLM single-binary entry point.

This is the file PyInstaller bundles into ``LocalLLM.exe``. Double-
clicking the .exe (no args) runs ``serve``: spawn the embedded Ollama,
start the FastAPI app on a free port, open the user's browser. Power
users can pass a subcommand to access dev/build tooling without
needing a separate Python install.

Subcommands:
    serve              Run the desktop app (default)
    smart-install      Detect hardware, write DEFAULT_MODEL into .env
    recommend-models   Print recommended models for this machine
    build-bundle       Build an airgap bundle (home-machine workflow)
    build-index        Index the corpus (Chroma + BM25)
    fetch-corpus       Pull source repos defined in corpus.yaml
    version            Print version + frozen-binary diagnostics

Each subcommand forwards remaining argv to the underlying script's
own argparse, so ``LocalLLM.exe build-index --reset`` is identical to
``python scripts/build_index.py --reset``.
"""
from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Callable, Optional


# When frozen by PyInstaller, scripts/ isn't on a normal Python path —
# it's collected into _MEIPASS as a regular package. Make sure both dev
# and frozen modes can `import scripts.foo`.
def _ensure_paths() -> None:
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", "."))
    else:
        base = Path(__file__).resolve().parent.parent
    p = str(base)
    if p not in sys.path:
        sys.path.insert(0, p)


_ensure_paths()


# ---------------------------------------------------------------------------
# serve — the double-click path
# ---------------------------------------------------------------------------

def _find_free_port(start: int = 8765) -> int:
    for port in range(start, start + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"no free TCP port in {start}-{start + 100}")


def _wait_for_http(host: str, port: int, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def cmd_serve(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="LocalLLM serve",
        description="Run the LocalLLM desktop app (default action).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("LOCALLLM_PORT", "0")),
        help="Port to bind (0 = pick a free one in 8765+)",
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Don't auto-open the browser. Useful for headless / inspection workflows.",
    )
    parser.add_argument(
        "--no-ollama", action="store_true",
        help="Don't start the embedded Ollama. Use this when you already have "
             "Ollama running and want LocalLLM to talk to it via OLLAMA_HOST.",
    )
    parser.add_argument(
        "--ollama-port", type=int, default=11435,
        help="Port for the embedded Ollama (default 11435 — non-default to avoid "
             "colliding with any host Ollama at :11434).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Spin up Ollama BEFORE importing the FastAPI app so the app's
    # config sees the right OLLAMA_HOST when modules read os.environ.
    supervisor = None
    if not args.no_ollama:
        from app.runtime.ollama_supervisor import OllamaSupervisor

        supervisor = OllamaSupervisor(host="127.0.0.1", port=args.ollama_port)
        if supervisor.start(wait=True, timeout=30.0):
            os.environ["OLLAMA_HOST"] = supervisor.base_url
            print(f"Ollama serving at {supervisor.base_url}", flush=True)
        else:
            print(
                "WARNING: embedded Ollama failed to start. The app will run but "
                "model calls will fail until an Ollama instance is reachable at "
                f"OLLAMA_HOST={os.getenv('OLLAMA_HOST', 'http://localhost:11434')}",
                file=sys.stderr,
                flush=True,
            )
            supervisor = None  # Don't try to stop something that never started.

    try:
        import uvicorn  # noqa: F401
        import app.web.app  # noqa: F401
    except ImportError as exc:
        print(f"ERROR: missing runtime dependency — {exc}", file=sys.stderr)
        if supervisor:
            supervisor.stop()
        return 1

    port = args.port or _find_free_port()
    url = f"http://{args.host}:{port}"
    print(f"Starting LocalLLM on {url} ...", flush=True)

    def _serve() -> None:
        import uvicorn
        uvicorn.run(
            "app.web.app:app",
            host=args.host,
            port=port,
            log_level="warning",
            access_log=False,
        )

    server_thread = threading.Thread(target=_serve, daemon=True, name="localllm-uvicorn")
    server_thread.start()

    if not _wait_for_http(args.host, port, timeout=30.0):
        print(f"ERROR: server didn't bind on {args.host}:{port} within 30s", file=sys.stderr)
        if supervisor:
            supervisor.stop()
        return 1

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception as exc:  # noqa: BLE001
            print(f"Could not auto-open browser ({exc}). Open manually: {url}", flush=True)

    print(f"LocalLLM is running at {url}.", flush=True)
    print("Close this window to stop.", flush=True)

    try:
        server_thread.join()
    except KeyboardInterrupt:
        print("\nStopping LocalLLM ...", flush=True)
    finally:
        if supervisor:
            supervisor.stop()
    return 0


# ---------------------------------------------------------------------------
# Subcommand dispatch — proxy each script's own main()
# ---------------------------------------------------------------------------

def _proxy(import_path: str) -> Callable[[list[str]], int]:
    """Return a thunk that imports ``import_path`` and calls its ``main()``.

    We re-route ``sys.argv`` so the inner argparse sees ``[prog, *argv]``.
    Imports are lazy so e.g. invoking ``serve`` doesn't pay the cost of
    pulling chromadb in for ``recommend-models``.
    """
    def run(argv: list[str]) -> int:
        module_name, _, func_name = import_path.partition(":")
        func_name = func_name or "main"
        from importlib import import_module
        mod = import_module(module_name)
        fn = getattr(mod, func_name)
        prev_argv = sys.argv
        try:
            sys.argv = [module_name, *argv]
            result = fn()
        finally:
            sys.argv = prev_argv
        return int(result or 0)
    return run


def cmd_version(argv: list[str]) -> int:
    """Print version + diagnostics. Useful first-stop when something's wrong."""
    parser = argparse.ArgumentParser(prog="LocalLLM version")
    parser.parse_args(argv)
    from app.runtime.paths import (
        env_file, exe_dir, is_frozen, meipass_dir, ollama_models_dir,
        user_data_dir,
    )
    from app.runtime.ollama_supervisor import find_ollama_binary

    print("LocalLLM")
    print(f"  python:        {sys.version.split()[0]}")
    print(f"  frozen:        {is_frozen()}")
    print(f"  exe:           {sys.executable}")
    print(f"  exe_dir:       {exe_dir()}")
    if is_frozen():
        print(f"  _MEIPASS:      {meipass_dir()}")
    print(f"  user_data:     {user_data_dir()}")
    print(f"  env file:      {env_file()}")
    print(f"  ollama_models: {ollama_models_dir()}")
    binary = find_ollama_binary()
    print(f"  ollama binary: {binary if binary else '(not found)'}")
    return 0


COMMANDS: dict[str, Callable[[list[str]], int]] = {
    "serve":            cmd_serve,
    "smart-install":    _proxy("scripts.smart_install"),
    "recommend-models": _proxy("scripts.recommend_models"),
    "build-bundle":     _proxy("scripts.build_bundle"),
    "build-index":      _proxy("scripts.build_index"),
    "fetch-corpus":     _proxy("scripts.fetch_corpus"),
    "version":          cmd_version,
}


def _print_help() -> None:
    print("LocalLLM — local LLM coding assistant with RAG")
    print()
    print("Usage:")
    print("  LocalLLM                     Run the desktop app (default).")
    print("  LocalLLM <subcommand> [args] Run a specific tool.")
    print()
    print("Subcommands:")
    for name in COMMANDS:
        print(f"  {name}")
    print()
    print("Pass --help to any subcommand for its options, e.g.:")
    print("  LocalLLM serve --help")


def _parse_top_level(argv: list[str]) -> tuple[Optional[str], list[str]]:
    """Pull the subcommand off the front; everything else goes to it.

    No subcommand → ``serve`` is the implicit default (the double-click
    path). ``--help`` / ``-h`` at the top level → built-in help and exit.
    """
    if not argv:
        return "serve", []
    head = argv[0]
    if head in ("-h", "--help", "help"):
        return None, []
    if head in COMMANDS:
        return head, argv[1:]
    # Unknown first arg — assume the user is passing flags to `serve`
    # (e.g. `LocalLLM --no-browser`).
    if head.startswith("-"):
        return "serve", argv
    print(f"ERROR: unknown subcommand: {head!r}", file=sys.stderr)
    print(file=sys.stderr)
    return None, []


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd, rest = _parse_top_level(argv)
    if cmd is None:
        _print_help()
        return 0 if argv and argv[0] in ("-h", "--help", "help") else 2
    return COMMANDS[cmd](rest)


if __name__ == "__main__":
    sys.exit(main())
