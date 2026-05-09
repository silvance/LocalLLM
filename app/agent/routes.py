"""FastAPI router for the /agent page. Mounted from app/web/app.py via
try-import so the airgap deploy (which excludes app/agent/) silently
skips it."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
import logging
import threading
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.agent.loop import run_agent
from app.config import get_settings
from app.utils.agent_storage import (
    AgentSession,
    AgentStorage,
    new_session as new_agent_session,
)
from app.web.app import NoCacheStaticFiles


logger = logging.getLogger("localllm")
settings = get_settings()

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_STATIC_DIR = Path(__file__).resolve().parent / "static"
_AGENTS_DIR = Path("data/agents")

agent_templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
agent_static = NoCacheStaticFiles(directory=str(_STATIC_DIR))
agent_storage = AgentStorage(_AGENTS_DIR)

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


def _job_to_payload(job) -> dict:
    return {
        "job_id": job.id,
        "agent_id": str(job.request_data.get("agent_id") or ""),
        "model": str(job.request_data.get("model") or ""),
        "prompt": str(job.request_data.get("prompt") or ""),
        "status": job.status,
    }


def _find_agent_job(agent_id: str) -> Optional[dict]:
    """Return the most recent in-memory job for one saved agent session.

    Agent sessions are now durable, but active SSE replay still depends on
    the in-memory JobManager while the server process is alive.
    """
    from app.web.app import job_manager
    candidates = list(job_manager.list_with_chat_prefix(f"agent:{agent_id}"))
    if not candidates:
        return None
    job = max(candidates, key=lambda j: j.started_at)
    return _job_to_payload(job)


def _find_latest_agent_job() -> Optional[dict]:
    """Most-recent agent job, regardless of status.

    This preserves the old /agent reload behavior while the new sidebar
    provides durable access to older sessions.
    """
    from app.web.app import job_manager
    candidates = list(job_manager.list_with_chat_prefix("agent:"))
    if not candidates:
        return None
    job = max(candidates, key=lambda j: j.started_at)
    return _job_to_payload(job)


def _agent_summaries() -> list[dict]:
    return [asdict(s) for s in agent_storage.list_summaries()]


def _serialize_session(session: AgentSession | None) -> dict | None:
    return asdict(session) if session is not None else None


def _is_running_job(payload: dict | None) -> bool:
    return bool(payload and payload.get("status") in {"pending", "streaming"})


async def _render_agent(request: Request, loaded: AgentSession | None = None) -> HTMLResponse:
    # Match the chat/review/compare dropdown behaviour: list every
    # installed Ollama model, not just the three preset slots.
    from app.web.app import _model_choices, get_ollama_status

    active_job = _find_agent_job(loaded.id) if loaded else _find_latest_agent_job()
    if loaded and not _is_running_job(active_job):
        active_job = None
    return agent_templates.TemplateResponse(
        request,
        "agent.html",
        {
            "settings": settings,
            "model_options": _model_choices(include_auto=False),
            "missing_deps": _missing_agent_deps(),
            "active_job": active_job,
            "loaded_agent": _serialize_session(loaded),
            "summaries": _agent_summaries(),
            "ollama_status": get_ollama_status(),
        },
    )


@router.get("/agent", response_class=HTMLResponse)
async def agent_page(request: Request):
    return await _render_agent(request)


@router.get("/agent/{agent_id}", response_class=HTMLResponse)
async def agent_load_page(agent_id: str, request: Request):
    loaded = agent_storage.load(agent_id)
    if loaded is None:
        raise HTTPException(404, f"agent session {agent_id} not found")
    return await _render_agent(request, loaded)


@router.delete("/agent/{agent_id}")
async def delete_agent(agent_id: str) -> JSONResponse:
    agent_storage.delete(agent_id)
    return JSONResponse({"ok": True})


@router.post("/api/agent")
async def start_agent(request: Request) -> JSONResponse:
    body = await request.json()
    prompt = (body.get("prompt") or "").strip()
    model_key = body.get("model") or "granite"
    _validate_agent_request(prompt, model_key)

    session = new_agent_session(prompt, model_key)
    payload = _start_agent_job(
        session=session,
        prompt=prompt,
        run_prompt=prompt,
        model_key=model_key,
        loop=asyncio.get_running_loop(),
    )
    return JSONResponse(payload)


@router.post("/api/agent/{agent_id}/messages")
async def continue_agent(agent_id: str, request: Request) -> JSONResponse:
    session = agent_storage.load(agent_id)
    if session is None:
        raise HTTPException(404, f"agent session {agent_id} not found")
    if _is_running_job(_find_agent_job(agent_id)):
        raise HTTPException(409, "agent session is already running")

    body = await request.json()
    prompt = (body.get("prompt") or "").strip()
    model_key = body.get("model") or session.model or "granite"
    _validate_agent_request(prompt, model_key)

    payload = _start_agent_job(
        session=session,
        prompt=prompt,
        run_prompt=_followup_prompt(session, prompt),
        model_key=model_key,
        loop=asyncio.get_running_loop(),
    )
    return JSONResponse(payload)


def _validate_agent_request(prompt: str, model_key: str) -> None:
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

    if not prompt:
        raise HTTPException(400, "empty prompt")

    # Accept either a preset slot name or a raw Ollama model name —
    # ChatService.get_adapter handles both, so the agent dropdown can
    # surface every installed model the same way chat/review/compare do.
    from app.web.app import _model_choices
    if model_key not in set(_model_choices(include_auto=False)):
        raise HTTPException(400, f"unknown model: {model_key}")


def _followup_prompt(session: AgentSession, prompt: str) -> str:
    turns: list[str] = []
    seen_initial = False
    for event in session.events:
        if event.kind == "user_prompt":
            text = str(event.payload.get("prompt") or "").strip()
            if text:
                seen_initial = True
                turns.append(f"User: {text}")
        elif event.kind == "final":
            text = str(event.payload.get("answer") or "").strip()
            if text:
                turns.append(f"Agent: {text}")
    if not seen_initial and session.prompt.strip():
        turns.insert(0, f"User: {session.prompt.strip()}")
    if session.answer.strip() and not any(t.startswith("Agent:") for t in turns):
        turns.append(f"Agent: {session.answer.strip()}")

    transcript = "\n\n".join(turns[-8:]) or "(no prior transcript saved)"
    return (
        "Continue this saved online-agent conversation.\n\n"
        "Use the prior transcript for context, but answer the new user "
        "message directly. Use web tools again when the follow-up asks for "
        "current facts, sources, or verification. Do not cite sources you "
        "did not fetch in this new run unless you explicitly say they came "
        "from the saved prior transcript.\n\n"
        f"Prior transcript:\n{transcript}\n\n"
        f"New user message:\n{prompt}"
    )


def _start_agent_job(
    *,
    session: AgentSession,
    prompt: str,
    run_prompt: str,
    model_key: str,
    loop: asyncio.AbstractEventLoop,
) -> dict:
    # Late binding to avoid an import cycle with app.web.app
    from app.web.app import chat_service, job_manager

    adapter = chat_service.get_adapter(model_key)
    session.status = "running"
    session.model = model_key
    session.error = None
    agent_storage.save(session)
    job = job_manager.create(
        chat_id=f"agent:{session.id}",
        request_data={
            "kind": "agent",
            "agent_id": session.id,
            "model": model_key,
            "prompt": prompt,
            "prompt_chars": len(prompt),
        },
    )
    session.job_id = job.id
    agent_storage.save(session)
    agent_storage.append_event(session.id, "user_prompt", {"prompt": prompt})
    job_manager.emit_event(job.id, "user_prompt", {"prompt": prompt}, loop)

    def _runner() -> None:
        def on_event(name: str, payload: dict) -> None:
            # run_agent emits its own "done" bookkeeping event before
            # returning. The JobManager also uses "done" as the terminal
            # SSE event, so forwarding the loop's internal version would
            # close the stream before finish() can send status/metadata.
            if name == "done":
                return
            agent_storage.append_event(session.id, name, payload)
            job_manager.emit_event(job.id, name, payload, loop)

        try:
            answer = run_agent(
                user_prompt=run_prompt,
                chat_client=adapter.client,
                model=adapter.model_name,
                on_event=on_event,
            )
            agent_storage.append_event(
                session.id,
                "final",
                {"answer": answer, "model": model_key, "status": "done"},
            )
            saved = agent_storage.load(session.id)
            if saved is not None:
                saved.status = "done"
                saved.answer = answer
                saved.error = None
                saved.job_id = job.id
                agent_storage.save(saved)
            job_manager.finish(
                job.id,
                "done",
                loop,
                metadata={
                    "kind": "agent",
                    "agent_id": session.id,
                    "model": model_key,
                    "answer": answer,
                },
            )
        except Exception as exc:
            logger.exception("Agent job %s failed", job.id)
            agent_storage.append_event(session.id, "error", {"error": str(exc)})
            saved = agent_storage.load(session.id)
            if saved is not None:
                saved.status = "error"
                saved.error = str(exc)
                saved.job_id = job.id
                agent_storage.save(saved)
            job_manager.finish(
                job.id,
                "error",
                loop,
                error=str(exc),
                metadata={"kind": "agent", "agent_id": session.id, "model": model_key},
            )

    threading.Thread(target=_runner, daemon=True, name=f"agent-{job.id[:8]}").start()
    return {"job_id": job.id, "agent_id": session.id, "title": session.title}
