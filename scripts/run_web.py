"""Launch the FastAPI web UI.

Defaults: 127.0.0.1:8000, no auto-reload. Override with env vars
LOCALLLM_HOST, LOCALLLM_PORT or pass --host/--port via uvicorn directly.

Hardened against silent failures: pre-bind probe before uvicorn announces
startup, post-return diagnostic, and exception capture. Without these,
``uvicorn.run()`` exiting unexpectedly (signal, swallowed exception,
port collision after the bind) leaves no evidence in the parent shell
beyond the prompt returning, and the operator has no way to tell what
went wrong.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


_INSTALL_HINT = """
ERROR: missing dependency `{package}`.

Install the runtime requirements first (with your venv active):

    pip install -r requirements.txt

If you also want the online `/agent` page:

    pip install -r requirements-agent.txt

Then re-run this script.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the LocalLLM web UI")
    parser.add_argument("--host", default=os.getenv("LOCALLLM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("LOCALLLM_PORT", "8000")))
    parser.add_argument("--reload", action="store_true", help="autoreload on file change (dev)")
    args = parser.parse_args()

    # Friendly hints when the most common missing deps haven't been installed.
    # Without this, users see a bare `ModuleNotFoundError: No module named X`
    # which doesn't tell them they forgot the requirements step.
    for required in ("uvicorn", "fastapi", "ollama"):
        try:
            __import__(required)
        except ImportError:
            print(_INSTALL_HINT.format(package=required), file=sys.stderr)
            return 1

    # Pre-bind probe — same logic the cli.py path uses. If port is
    # already taken, exit BEFORE uvicorn announces "Application
    # startup complete," which it prints before its own bind attempt
    # (so a port-conflict failure looks like a successful start +
    # mysterious crash to the operator).
    from app.cli import _ensure_port_free
    rc = _ensure_port_free(args.host, args.port)
    if rc != 0:
        return rc

    print(
        f"[run_web] starting uvicorn on http://{args.host}:{args.port} "
        f"(reload={args.reload})",
        flush=True,
    )

    import uvicorn

    # Wrap uvicorn.run() so any exception it raises gets logged with a
    # full traceback, and a post-return diagnostic prints regardless.
    # uvicorn.run() blocks until shutdown — if it returns at all,
    # something has stopped the server, and we want the operator to
    # see WHAT, not just a returned shell prompt.
    exc_info: BaseException | None = None
    try:
        uvicorn.run(
            "app.web.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
    except KeyboardInterrupt:
        # Operator hit Ctrl+C — clean stop, not a crash. Print a
        # confirmation so they can tell the difference from the
        # silent-exit case below.
        print("\n[run_web] received KeyboardInterrupt — stopped cleanly", flush=True)
        return 0
    except SystemExit as exc:
        # uvicorn or one of its handlers called sys.exit(). Surface
        # the exit code so the script's parent (update-and-run.ps1)
        # can react, instead of swallowing it.
        print(
            f"[run_web] uvicorn called sys.exit({exc.code!r})",
            file=sys.stderr, flush=True,
        )
        return int(exc.code) if isinstance(exc.code, int) else 1
    except BaseException as exc:  # noqa: BLE001 — diagnostic catch-all
        exc_info = exc
        print(
            f"[run_web] uvicorn.run() raised {type(exc).__name__}: {exc}",
            file=sys.stderr, flush=True,
        )
        traceback.print_exc()
        return 1

    # uvicorn.run() returned WITHOUT raising. This is unusual on a
    # long-running server — it shouldn't happen unless something
    # called the lifespan shutdown explicitly or a signal was
    # handled. Print a clear diagnostic so the operator can tell
    # this case apart from "still running" or "process killed."
    print(
        "[run_web] uvicorn.run() returned without raising. "
        "If you didn't press Ctrl+C, the server was likely stopped "
        "by a signal (Windows: console close, antivirus, or job-"
        "object termination). Check Event Viewer / antivirus logs "
        "if this keeps happening.",
        file=sys.stderr, flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
