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
    doctor             Run self-test checks (paths, ollama, models, RAG)

Each subcommand forwards remaining argv to the underlying script's
own argparse, so ``LocalLLM.exe build-index --reset`` is identical to
``python scripts/build_index.py --reset``.
"""
from __future__ import annotations

import argparse
import contextlib
import logging
import os
import socket
import sys
import threading
import webbrowser
from importlib import import_module
from typing import Callable, Optional

from app.runtime.paths import exe_dir, meipass_dir


# When frozen by PyInstaller, scripts/ isn't on a normal Python path —
# it's collected into _MEIPASS as a regular package. Make sure both dev
# and frozen modes can `import scripts.foo`.
_BASE = meipass_dir() or exe_dir()
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))


HELP_TOKENS = ("-h", "--help", "help")
OLLAMA_HOST_ENV = "OLLAMA_HOST"


# ---------------------------------------------------------------------------
# serve — the double-click path
# ---------------------------------------------------------------------------

def _pick_free_port() -> int:
    """Bind to port 0 and let the kernel hand back a free one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_http(host: str, port: int, timeout: float = 30.0) -> bool:
    """Poll until host:port accepts TCP, or give up after `timeout` seconds."""
    from app.runtime.ollama_supervisor import is_port_open
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_port_open(host, port, timeout=0.5):
            return True
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
        help="Port to bind (0 = pick a free one)",
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
    log = logging.getLogger("localllm.cli")

    # Spin up Ollama BEFORE importing the FastAPI app so the app's
    # config sees the right OLLAMA_HOST when modules read os.environ.
    with contextlib.ExitStack() as stack:
        if not args.no_ollama:
            from app.runtime.ollama_supervisor import OllamaSupervisor
            supervisor = OllamaSupervisor(host="127.0.0.1", port=args.ollama_port)
            if supervisor.start(wait=True, timeout=30.0):
                stack.callback(supervisor.stop)
                os.environ[OLLAMA_HOST_ENV] = supervisor.base_url
                print(f"Ollama serving at {supervisor.base_url}", flush=True)
            else:
                log.warning(
                    "embedded Ollama failed to start; model calls will fail until "
                    "an Ollama instance is reachable at OLLAMA_HOST=%s",
                    os.getenv(OLLAMA_HOST_ENV, "http://localhost:11434"),
                )

        try:
            import app.web.app  # noqa: F401
        except ImportError as exc:
            log.error("missing runtime dependency — %s", exc)
            return 1

        port = args.port or _pick_free_port()
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
            log.error("server didn't bind on %s:%s within 30s", args.host, port)
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
        env_file, is_frozen, ollama_models_dir, user_data_dir,
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


def cmd_doctor(argv: list[str]) -> int:
    """Run self-test checks (paths, ollama, models, RAG)."""
    parser = argparse.ArgumentParser(
        prog="LocalLLM doctor",
        description="Run diagnostic checks. Exits 1 if any check FAILs.",
    )
    parser.parse_args(argv)
    from app.runtime.doctor import main as doctor_main
    return doctor_main()


COMMANDS: dict[str, Callable[[list[str]], int]] = {
    "serve":            cmd_serve,
    "smart-install":    _proxy("scripts.smart_install"),
    "recommend-models": _proxy("scripts.recommend_models"),
    "build-bundle":     _proxy("scripts.build_bundle"),
    "build-index":      _proxy("scripts.build_index"),
    "fetch-corpus":     _proxy("scripts.fetch_corpus"),
    "version":          cmd_version,
    "doctor":           cmd_doctor,
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
    if head in HELP_TOKENS:
        return None, []
    if head in COMMANDS:
        return head, argv[1:]
    if head.startswith("-"):
        # Unknown leading flag — assume the user is passing flags to `serve`.
        return "serve", argv
    print(f"ERROR: unknown subcommand: {head!r}", file=sys.stderr)
    print(file=sys.stderr)
    return None, []


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd, rest = _parse_top_level(argv)
    if cmd is None:
        _print_help()
        return 0 if argv and argv[0] in HELP_TOKENS else 2
    return COMMANDS[cmd](rest)


if __name__ == "__main__":
    sys.exit(main())
