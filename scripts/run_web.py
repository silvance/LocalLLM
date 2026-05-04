"""Launch the FastAPI web UI.

Defaults: 127.0.0.1:8000, no auto-reload. Override with env vars
LOCALLLM_HOST, LOCALLLM_PORT or pass --host/--port via uvicorn directly.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the LocalLLM web UI")
    parser.add_argument("--host", default=os.getenv("LOCALLLM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("LOCALLLM_PORT", "8000")))
    parser.add_argument("--reload", action="store_true", help="autoreload on file change (dev)")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        "app.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
