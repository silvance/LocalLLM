"""FastAPI web UI for LocalLLM.

Replaces the Streamlit app with a server-side architecture: generation
jobs run as background threads coordinated by JobManager, the browser
subscribes via Server-Sent Events. Tab-switch / refresh / disconnect no
longer kills the generation — clients just resubscribe.

Mounted routes:

  GET  /                                  redirect to most recent chat (or new)
  GET  /chats/{chat_id}                   chat page (HTML)
  POST /chats                             create a new chat, redirect into it
  DELETE /chats/{chat_id}                 delete a chat
  GET  /api/chats                         list of chat summaries (JSON, sidebar)
  POST /api/chats/{chat_id}/messages      send a user message, kick off a job
  GET  /api/jobs/{job_id}/stream          SSE token stream for a job
  POST /api/jobs/{job_id}/stop            request stop (cooperative)
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import threading
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import get_settings
from app.schemas.chat import ChatMessage, ChatRequest
from app.services.chat_service import ChatService
from app.utils.chat_storage import (
    ChatSession,
    ChatStorage,
    derive_title,
    new_session,
)
from app.utils.logger import setup_logger
from app.web.job_manager import Job, JobManager


setup_logger()
logger = logging.getLogger("localllm")
settings = get_settings()


# ---------------------------------------------------------------------------
# Single-process singletons
# ---------------------------------------------------------------------------

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_STATIC_DIR = Path(__file__).resolve().parent / "static"
_CHATS_DIR = Path("data/chats")

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
chat_service = ChatService()
chat_storage = ChatStorage(_CHATS_DIR)
job_manager = JobManager()


app = FastAPI(title="LocalLLM")
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_or_create_active_chat() -> ChatSession:
    summaries = chat_storage.list_summaries()
    if summaries:
        existing = chat_storage.load(summaries[0].id)
        if existing is not None:
            return existing
    s = new_session()
    chat_storage.save(s)
    return s


def _serialize_routing(decision) -> dict | None:
    if decision is None:
        return None
    return {
        "selected_model": decision.selected_model,
        "complexity_score": decision.complexity_score,
        "reason": decision.reason,
    }


def _serialize_retrievals(retrievals) -> list[dict]:
    return [
        {
            "source_id": r.source_id,
            "file_path": r.file_path,
            "language": r.language,
            "category": r.category,
            "score": r.score,
            "snippet": (r.document[:600] + "…") if len(r.document) > 600 else r.document,
        }
        for r in retrievals
    ]


def _start_generation_thread(
    job: Job,
    request_obj: ChatRequest,
    selection: str,
    use_rag: bool,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Run the synchronous Ollama stream in a background thread, pushing
    chunks back into the JobManager (which broadcasts to subscribers)."""

    def _runner() -> None:
        try:
            execution = chat_service.stream_chat(
                request=request_obj,
                selection=selection,
                use_rag=use_rag,
            )
            for chunk in execution.stream:
                if job_manager.is_stop_requested(job.id):
                    metadata = {
                        "selected_model": execution.selected_model,
                        "routing": _serialize_routing(execution.routing_decision),
                        "retrievals": _serialize_retrievals(execution.retrievals),
                        "stopped": True,
                    }
                    job_manager.finish(job.id, "stopped", loop, metadata=metadata)
                    _persist_assistant_message(job)
                    return
                if chunk.content:
                    job_manager.append_chunk(job.id, chunk.content, loop)

            metadata = {
                "selected_model": execution.selected_model,
                "routing": _serialize_routing(execution.routing_decision),
                "retrievals": _serialize_retrievals(execution.retrievals),
                "rag_error": chat_service.rag.last_error if use_rag else None,
            }
            job_manager.finish(job.id, "done", loop, metadata=metadata)
            _persist_assistant_message(job)
        except Exception as exc:
            logger.exception("Generation failed for job %s", job.id)
            job_manager.finish(job.id, "error", loop, error=str(exc))

    threading.Thread(target=_runner, daemon=True, name=f"gen-{job.id[:8]}").start()


def _persist_assistant_message(job: Job) -> None:
    """When a job finishes (done or stopped), append the accumulated text
    to the chat session and save."""
    session = chat_storage.load(job.chat_id)
    if session is None:
        return
    text = job.text.strip()
    if not text and job.status == "stopped":
        text = "_(stopped before any output)_"
    if not text:
        return
    if job.status == "stopped":
        text = text + "\n\n_[stopped]_"
    session.messages.append(ChatMessage(role="assistant", content=text))
    chat_storage.save(session)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root() -> RedirectResponse:
    s = _get_or_create_active_chat()
    return RedirectResponse(url=f"/chats/{s.id}", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/chats/{chat_id}", response_class=HTMLResponse)
async def chat_page(chat_id: str, request: Request):
    session = chat_storage.load(chat_id)
    if session is None:
        raise HTTPException(404, f"chat {chat_id} not found")

    summaries = chat_storage.list_summaries()
    active_jobs = job_manager.jobs_active_for_chat(chat_id)
    active_job_id = active_jobs[0].id if active_jobs else None

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "chat": session,
            "summaries": summaries,
            "settings": settings,
            "model_options": ["auto", "granite", "gemma", "qwen"],
            "active_job_id": active_job_id,
        },
    )


@app.post("/chats")
async def create_chat() -> RedirectResponse:
    s = new_session()
    chat_storage.save(s)
    return RedirectResponse(url=f"/chats/{s.id}", status_code=status.HTTP_303_SEE_OTHER)


@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str) -> JSONResponse:
    chat_storage.delete(chat_id)
    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------

@app.get("/api/chats")
async def list_chats() -> JSONResponse:
    summaries = chat_storage.list_summaries()
    return JSONResponse(
        [dataclasses.asdict(s) for s in summaries]
    )


@app.post("/api/chats/{chat_id}/messages")
async def send_message(chat_id: str, request: Request) -> JSONResponse:
    """User submits a message. We persist the user message immediately,
    create a Job, kick off background generation, return the job_id so
    the client can subscribe to /api/jobs/{job_id}/stream."""
    body = await _parse_message_body(request)
    user_text = (body.get("content") or "").strip()
    if not user_text:
        raise HTTPException(400, "empty message")

    session = chat_storage.load(chat_id)
    if session is None:
        raise HTTPException(404, f"chat {chat_id} not found")

    # 1. Persist the user message right away so the next page render shows it.
    session.messages.append(ChatMessage(role="user", content=user_text))
    chat_storage.save(session)

    # 2. Build the chat request: optional system prompt + full message history.
    selection: str = body.get("model_selection") or settings.default_model or "auto"
    use_rag: bool = bool(body.get("use_rag", False))
    system_prompt: str = (body.get("system_prompt") or "").strip()
    temperature = body.get("temperature")
    max_tokens = body.get("max_tokens")
    num_ctx = body.get("num_ctx")

    request_messages: list[ChatMessage] = []
    if system_prompt:
        request_messages.append(ChatMessage(role="system", content=system_prompt))
    request_messages.extend(session.messages)

    chat_request = ChatRequest(
        messages=request_messages,
        stream=True,
        temperature=float(temperature) if temperature is not None else None,
        max_tokens=int(max_tokens) if max_tokens is not None else None,
        num_ctx=int(num_ctx) if num_ctx is not None else None,
    )

    # 3. Create job + spawn background producer
    job = job_manager.create(
        chat_id=chat_id,
        request_data={
            "selection": selection,
            "use_rag": use_rag,
            "user_message_chars": len(user_text),
        },
    )
    loop = asyncio.get_running_loop()
    _start_generation_thread(job, chat_request, selection, use_rag, loop)

    return JSONResponse({"job_id": job.id, "user_message": user_text})


async def _parse_message_body(request: Request) -> dict:
    """Accepts either application/json or form-encoded so HTMX works either way."""
    ctype = request.headers.get("content-type", "")
    if "application/json" in ctype:
        return await request.json()
    form = await request.form()
    out = dict(form)
    # HTMX sends checkboxes as "on"/missing; coerce booleans.
    if "use_rag" in out:
        out["use_rag"] = out["use_rag"] in ("on", "true", "1", "yes")
    return out


# ---------------------------------------------------------------------------
# Streaming + stop
# ---------------------------------------------------------------------------

@app.get("/api/jobs/{job_id}/stream")
async def stream_job(job_id: str) -> StreamingResponse:
    """Server-Sent Events stream of a job's tokens.

    On connect, we replay the current buffer immediately, then live-stream
    tokens as the producer emits them. If the job is already in a terminal
    state we emit one 'done' event and close.
    """
    job, queue = job_manager.subscribe(job_id)
    if job is None or queue is None:
        raise HTTPException(404, f"job {job_id} not found")

    async def gen() -> AsyncIterator[bytes]:
        try:
            # Replay current buffer
            if job.text:
                yield _sse("token", {"chunk": job.text})
            if job.status in ("done", "error", "stopped"):
                yield _sse(
                    "done",
                    {
                        "status": job.status,
                        "error": job.error,
                        "metadata": job.metadata,
                        "text": job.text,
                    },
                )
                return

            while True:
                event, payload = await queue.get()
                yield _sse(event, payload)
                if event == "done":
                    return
        finally:
            job_manager.unsubscribe(job_id, queue)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx-style buffering if anyone proxies
        },
    )


@app.post("/api/jobs/{job_id}/stop")
async def stop_job(job_id: str) -> JSONResponse:
    if not job_manager.request_stop(job_id):
        raise HTTPException(404, f"job {job_id} not found")
    return JSONResponse({"ok": True})


def _sse(event: str, data: dict) -> bytes:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


# ---------------------------------------------------------------------------
# Review page (writer ↔ reviewer adversarial loop)
# ---------------------------------------------------------------------------

REVIEWER_INSTRUCTION = (
    "You are a senior code reviewer. Review the following code for problems "
    "that would prevent it from running cleanly: missing imports, undefined "
    "names, dataclass misuse, attribute references that don't exist on the "
    "class, broken control flow, security footguns, and any other concrete "
    "bugs. List issues briefly and specifically — line refs or quoted "
    "snippets where helpful. Do not rewrite the code yourself; just call out "
    "what's wrong."
)
REVISE_INSTRUCTION = (
    "You wrote the following code. A reviewer identified the issues below. "
    "Produce a revised version that addresses every point. Output the full "
    "revised code in fenced markdown blocks — do not skip unchanged sections."
)


@app.get("/review", response_class=HTMLResponse)
async def review_page(request: Request):
    return templates.TemplateResponse(
        request,
        "review.html",
        {
            "settings": settings,
            "model_options": ["auto", "granite", "gemma", "qwen"],
        },
    )


@app.post("/api/review")
async def start_review(request: Request) -> JSONResponse:
    """Kick off a writer ↔ reviewer ↔ writer loop as a single Job.

    The runner thread streams each role's output through the same Job buffer,
    and emits 'section_start' SSE events between roles so the frontend can
    label each block.
    """
    body = await _parse_message_body(request)
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(400, "empty prompt")

    writer_model = body.get("writer_model") or "qwen"
    reviewer_model = body.get("reviewer_model") or "gemma"
    rounds = int(body.get("rounds") or 3)
    if rounds < 2:
        raise HTTPException(400, "rounds must be at least 2 (write + review)")
    if rounds > 6:
        raise HTTPException(400, "rounds capped at 6")

    system_prompt = (body.get("system_prompt") or "").strip() or settings.default_system_prompt
    temperature = body.get("temperature")
    max_tokens = body.get("max_tokens")
    num_ctx = body.get("num_ctx")

    job = job_manager.create(
        chat_id=f"review-{int(asyncio.get_running_loop().time())}",
        request_data={
            "kind": "review",
            "writer_model": writer_model,
            "reviewer_model": reviewer_model,
            "rounds": rounds,
            "prompt_chars": len(prompt),
        },
    )
    loop = asyncio.get_running_loop()
    _start_review_thread(
        job=job,
        prompt=prompt,
        writer_model=writer_model,
        reviewer_model=reviewer_model,
        rounds=rounds,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        num_ctx=num_ctx,
        loop=loop,
    )
    return JSONResponse({"job_id": job.id})


def _start_review_thread(
    *,
    job: Job,
    prompt: str,
    writer_model: str,
    reviewer_model: str,
    rounds: int,
    system_prompt: str,
    temperature: float | None,
    max_tokens: int | None,
    num_ctx: int | None,
    loop: asyncio.AbstractEventLoop,
) -> None:
    def _runner() -> None:
        sections: list[dict] = []

        def run_section(*, role: str, model: str, messages: list[ChatMessage]) -> str:
            """Stream one role's response and capture its text. Returns the
            collected text. Honors stop requests cooperatively."""
            section_idx = len(sections)
            job_manager.emit_event(
                job.id,
                "section_start",
                {"index": section_idx, "role": role, "model": model},
                loop,
            )

            req = ChatRequest(
                messages=messages,
                stream=True,
                temperature=float(temperature) if temperature is not None else None,
                max_tokens=int(max_tokens) if max_tokens is not None else None,
                num_ctx=int(num_ctx) if num_ctx is not None else None,
            )
            execution = chat_service.stream_chat(request=req, selection=model, use_rag=False)
            collected: list[str] = []
            for chunk in execution.stream:
                if job_manager.is_stop_requested(job.id):
                    raise _StopRequested(execution.selected_model)
                if chunk.content:
                    collected.append(chunk.content)
                    job_manager.append_chunk(job.id, chunk.content, loop)
            return "".join(collected)

        def base_messages() -> list[ChatMessage]:
            msgs: list[ChatMessage] = []
            if system_prompt:
                msgs.append(ChatMessage(role="system", content=system_prompt))
            return msgs

        try:
            # Round 0 — writer produces initial code
            writer_msgs = base_messages() + [ChatMessage(role="user", content=prompt)]
            writer_text = run_section(role="writer", model=writer_model, messages=writer_msgs)
            sections.append({"role": "writer", "model": writer_model, "text": writer_text})

            # Round 1 — reviewer critiques
            review_user = (
                f"{REVIEWER_INSTRUCTION}\n\n---\n\nOriginal task:\n{prompt}\n\n"
                f"---\n\nCode under review:\n\n{writer_text}"
            )
            reviewer_msgs = base_messages() + [ChatMessage(role="user", content=review_user)]
            review_text = run_section(role="reviewer", model=reviewer_model, messages=reviewer_msgs)
            sections.append({"role": "reviewer", "model": reviewer_model, "text": review_text})

            # Subsequent rounds alternate writer/reviewer up to `rounds` total
            for i in range(2, rounds):
                if i % 2 == 0:
                    # Writer revises
                    revise_user = (
                        f"{REVISE_INSTRUCTION}\n\n---\n\nOriginal task:\n{prompt}\n\n"
                        f"---\n\nYour previous code:\n\n{sections[-2]['text']}\n\n"
                        f"---\n\nReviewer's notes:\n\n{sections[-1]['text']}"
                    )
                    msgs = base_messages() + [ChatMessage(role="user", content=revise_user)]
                    text = run_section(role="writer", model=writer_model, messages=msgs)
                    sections.append({"role": "writer", "model": writer_model, "text": text})
                else:
                    # Reviewer rounds 2 onwards: review the latest revision
                    review_user = (
                        f"{REVIEWER_INSTRUCTION}\n\n---\n\nOriginal task:\n{prompt}\n\n"
                        f"---\n\nLatest code:\n\n{sections[-1]['text']}"
                    )
                    msgs = base_messages() + [ChatMessage(role="user", content=review_user)]
                    text = run_section(role="reviewer", model=reviewer_model, messages=msgs)
                    sections.append({"role": "reviewer", "model": reviewer_model, "text": text})

            job_manager.finish(
                job.id,
                "done",
                loop,
                metadata={"kind": "review", "sections": sections},
            )
        except _StopRequested as stop:
            job_manager.finish(
                job.id,
                "stopped",
                loop,
                metadata={
                    "kind": "review",
                    "sections": sections,
                    "stopped_during": stop.model,
                },
            )
        except Exception as exc:
            logger.exception("Review job %s failed", job.id)
            job_manager.finish(
                job.id,
                "error",
                loop,
                error=str(exc),
                metadata={"kind": "review", "sections": sections},
            )

    threading.Thread(target=_runner, daemon=True, name=f"review-{job.id[:8]}").start()


class _StopRequested(Exception):
    def __init__(self, model: str) -> None:
        self.model = model
