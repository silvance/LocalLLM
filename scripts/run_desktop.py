"""Desktop entry point — what the bundled .exe (or Linux binary) runs.

Starts the FastAPI app on a free local port, then opens the user's default
browser to that URL. Stays in the foreground so closing the window stops
the server (the only "Quit" mechanism for now).

We keep this distinct from `scripts/run_web.py` because:
  - run_web.py is the dev-time entrypoint (returns immediately if uvicorn
    is missing, no browser open).
  - run_desktop.py is the end-user entrypoint (auto-opens browser, friendly
    failure messaging, can be wrapped by PyInstaller).
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path


# When frozen by PyInstaller, sys.frozen is set and the app code lives
# alongside the executable in the temp _MEIPASS directory.
def _is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def _bundle_root() -> Path:
    """Where the bundled app data lives — _MEIPASS when frozen, repo root otherwise."""
    if _is_frozen():
        return Path(getattr(sys, "_MEIPASS", "."))  # type: ignore[arg-type]
    return Path(__file__).resolve().parent.parent


def _ensure_path() -> None:
    repo_root = _bundle_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _find_free_port(start: int = 8765) -> int:
    for port in range(start, start + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError("no free port in 8765-8864 range")


def _wait_for_server(host: str, port: int, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                s.connect((host, port))
                return True
        except OSError:
            time.sleep(0.2)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="LocalLLM desktop launcher")
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("LOCALLLM_PORT", "0")),
        help="Port to bind (0 = pick a free one)",
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Don't auto-open the browser (useful if you want to inspect via VM/other browser)",
    )
    args = parser.parse_args()

    _ensure_path()

    try:
        import uvicorn  # noqa: F401  (just verifying it's importable)
        import app.web.app  # noqa: F401
    except ImportError as exc:
        print(f"ERROR: missing runtime dependency — {exc}", file=sys.stderr)
        print("Install with: pip install -r requirements.txt", file=sys.stderr)
        return 1

    host = "127.0.0.1"
    port = args.port or _find_free_port()
    url = f"http://{host}:{port}"

    print(f"Starting LocalLLM on {url} ...", flush=True)

    def _serve() -> None:
        import uvicorn
        uvicorn.run(
            "app.web.app:app",
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
        )

    server_thread = threading.Thread(target=_serve, daemon=True, name="localllm-uvicorn")
    server_thread.start()

    if not _wait_for_server(host, port, timeout=30.0):
        print(
            f"ERROR: server didn't bind on {host}:{port} within 30s. Check ollama-localllm.log if "
            f"you launched via start.bat.",
            file=sys.stderr,
        )
        return 1

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception as exc:  # noqa: BLE001
            print(f"Could not auto-open the browser ({exc}). Open this URL manually: {url}",
                  flush=True)

    print(f"LocalLLM is running at {url}.", flush=True)
    print("Close this window to stop the server.", flush=True)

    try:
        # Block on the server thread (which itself blocks forever inside uvicorn.run)
        server_thread.join()
    except KeyboardInterrupt:
        print("\nStopping LocalLLM ...", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
