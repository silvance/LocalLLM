"""FastAPI router for the /agent page. Mounted from app/web/app.py via
try-import so the airgap deploy (which excludes app/agent/) silently
skips it."""
from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.agent.loop import run_agent
from app.config import get_settings


logger = logging.getLogger("localllm")
settings = get_settings()

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_STATIC_DIR = Path(__file__).resolve().parent / "static"

agent_templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
agent_static = StaticFiles(directory=str(_STATIC_DIR))

router = APIRouter()


def _missing_agent_deps() -> list[str]:
    """Return the list of agent-tool packages that aren't importable.

    Surfaces install issues at /agent page load instead of waiting for
    a tool to fail mid-loop with a buried stack trace. The packages
    here mirror requirements-agent.txt; the names are the *import* names
    (which sometimes differ from pip names — none do here, but check
    `import` not `pip show`).
    """
    missing: list[str] = []
    for module_name in ("ddgs", "trafilatura", "httpx"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    return missing


@router.get("/agent", response_class=HTMLResponse)
async def agent_page(request: Request):
    return agent_templates.TemplateResponse(
        request,
        "agent.html",
        {
            "settings": settings,
            "model_options": ["granite", "gemma", "qwen"],
            "missing_deps": _missing_agent_deps(),
        },
    )


@router.post("/api/agent")
async def start_agent(request: Request) -> JSONResponse:
    # Late binding to avoid an import cycle with app.web.app
    from app.web.app import chat_service, job_manager

    missing = _missing_agent_deps()
    if missing:
        # Return 503 rather than starting a doomed loop — the model would
        # otherwise see only tool errors and (per past observation) make
        # up plausible-looking facts to compensate.
        raise HTTPException(
            503,
            f"agent tools unavailable: missing {', '.join(missing)}. "
            f"Run `pip install -r requirements-agent.txt` and reload.",
        )

    body = await request.json()
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(400, "empty prompt")

    model_key = body.get("model") or "granite"
    if model_key not in chat_service.adapters:
        raise HTTPException(400, f"unknown model: {model_key}")

    adapter = chat_service.adapters[model_key]
    job = job_manager.create(
        chat_id=f"agent-{int(asyncio.get_running_loop().time())}",
        request_data={
            "kind": "agent",
            "model": model_key,
            "prompt_chars": len(prompt),
        },
    )
    loop = asyncio.get_running_loop()

    def _runner() -> None:
        def on_event(name: str, payload: dict) -> None:
            job_manager.emit_event(job.id, name, payload, loop)

        try:
            answer = run_agent(
                user_prompt=prompt,
                chat_client=adapter.client,
                model=adapter.model_name,
                on_event=on_event,
            )
            job_manager.finish(
                job.id,
                "done",
                loop,
                metadata={"kind": "agent", "model": model_key, "answer": answer},
            )
        except Exception as exc:
            logger.exception("Agent job %s failed", job.id)
            job_manager.finish(
                job.id,
                "error",
                loop,
                error=str(exc),
                metadata={"kind": "agent"},
            )

    threading.Thread(target=_runner, daemon=True, name=f"agent-{job.id[:8]}").start()
    return JSONResponse({"job_id": job.id})
