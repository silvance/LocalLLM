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
from app.utils.code_linter import lint_writer_output
from app.utils.hardware_info import (
    detect_hardware,
    recommend_models,
    to_dict as hw_to_dict,
)
from app.utils.logger import setup_logger
from app.utils.comparison_storage import (
    ComparisonRun,
    ComparisonStorage,
    ModelOutput,
    new_run as new_comparison_run,
)
from app.utils.review_storage import (
    ReviewSection,
    ReviewStorage,
    new_session as new_review_session,
)
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
_REVIEWS_DIR = Path("data/reviews")
_COMPARISONS_DIR = Path("data/comparisons")

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
chat_service = ChatService()
chat_storage = ChatStorage(_CHATS_DIR)
review_storage = ReviewStorage(_REVIEWS_DIR)
comparison_storage = ComparisonStorage(_COMPARISONS_DIR)
job_manager = JobManager()


app = FastAPI(title="LocalLLM")
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# Online agent variant — only present when app/agent/ is on disk. The
# airgap bundle excludes that directory, so this try-import quietly
# becomes a no-op there. Locally, it adds the /agent page + /api/agent.
_agent_enabled = False
try:
    from app.agent.routes import agent_static, router as agent_router
    app.include_router(agent_router)
    app.mount("/agent-static", agent_static, name="agent-static")
    _agent_enabled = True
except ImportError:
    pass


def is_agent_enabled() -> bool:
    return _agent_enabled


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
            "agent_enabled": _agent_enabled,
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

    session.messages.append(ChatMessage(role="user", content=user_text))
    chat_storage.save(session)
    job_id = await _kick_off_generation(session, body)
    return JSONResponse({"job_id": job_id, "user_message": user_text})


@app.post("/api/chats/{chat_id}/messages/{index}/regenerate")
async def regenerate_message(
    chat_id: str, index: int, request: Request,
) -> JSONResponse:
    """Discard messages from `index` onward and re-run inference using the
    user message that preceded `index`. Use case: assistant gave a bad
    answer; the user wants another roll of the dice with the same prompt.
    """
    body = await _parse_message_body(request)
    session = chat_storage.load(chat_id)
    if session is None:
        raise HTTPException(404, f"chat {chat_id} not found")
    if not 0 < index <= len(session.messages):
        raise HTTPException(400, "invalid message index")
    if session.messages[index - 1].role != "user":
        # Regenerating an assistant turn that wasn't preceded by a user
        # message would re-run with no fresh prompt — not meaningful.
        raise HTTPException(400, "no preceding user message to regenerate from")
    session.messages = session.messages[:index]
    chat_storage.save(session)
    job_id = await _kick_off_generation(session, body)
    return JSONResponse({"job_id": job_id})


@app.post("/api/chats/{chat_id}/messages/{index}/edit")
async def edit_message(
    chat_id: str, index: int, request: Request,
) -> JSONResponse:
    """Replace the user message at `index` with new content and re-run
    everything after it. Server-side truncation prevents stale assistant
    turns from leaking into the next request's context."""
    body = await _parse_message_body(request)
    new_text = (body.get("content") or "").strip()
    if not new_text:
        raise HTTPException(400, "empty message")
    session = chat_storage.load(chat_id)
    if session is None:
        raise HTTPException(404, f"chat {chat_id} not found")
    if not 0 <= index < len(session.messages):
        raise HTTPException(400, "invalid message index")
    if session.messages[index].role != "user":
        raise HTTPException(400, "can only edit user messages")
    session.messages = session.messages[:index]
    session.messages.append(ChatMessage(role="user", content=new_text))
    chat_storage.save(session)
    job_id = await _kick_off_generation(session, body)
    return JSONResponse({"job_id": job_id, "user_message": new_text})


async def _kick_off_generation(session: ChatSession, body: dict) -> str:
    """Build the ChatRequest from body settings, create a job, start the
    background producer, return the job_id. Shared by send/regenerate/edit
    so they can't drift on routing/RAG/system-prompt handling."""
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

    last = session.messages[-1] if session.messages else None
    job = job_manager.create(
        chat_id=session.id,
        request_data={
            "selection": selection,
            "use_rag": use_rag,
            "user_message_chars": len(last.content) if last and last.role == "user" else 0,
        },
    )
    loop = asyncio.get_running_loop()
    _start_generation_thread(job, chat_request, selection, use_rag, loop)
    return job.id


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
            # Replay buffered state. If the job has checkpoints (e.g. review
            # page section_starts), interleave them with the corresponding
            # text slices so late subscribers see the structure too. Without
            # this, a client that subscribed after a section_start fires
            # would see the tokens for that section but no marker telling
            # the UI which role/model they belong to — i.e. silent void.
            if job.checkpoints:
                last_pos = 0
                for event, payload, pos in job.checkpoints:
                    if pos > last_pos:
                        yield _sse("token", {"chunk": job.text[last_pos:pos]})
                    yield _sse(event, payload)
                    last_pos = pos
                if last_pos < len(job.text):
                    yield _sse("token", {"chunk": job.text[last_pos:]})
            elif job.text:
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
    return _render_review_page(request, loaded=None)


@app.get("/review/{review_id}", response_class=HTMLResponse)
async def review_load_page(review_id: str, request: Request):
    session = review_storage.load(review_id)
    if session is None:
        raise HTTPException(404, f"review {review_id} not found")
    return _render_review_page(request, loaded=session)


def _render_review_page(request: Request, loaded) -> HTMLResponse:
    summaries = review_storage.list_summaries()
    return templates.TemplateResponse(
        request,
        "review.html",
        {
            "settings": settings,
            "model_options": ["auto", "granite", "gemma", "qwen"],
            "summaries": summaries,
            "loaded": loaded,
            "loaded_dict": (
                {
                    "id": loaded.id,
                    "title": loaded.title,
                    "prompt": loaded.prompt,
                    "writer_model": loaded.writer_model,
                    "reviewer_model": loaded.reviewer_model,
                    "rounds": loaded.rounds,
                    "status": loaded.status,
                    "sections": [
                        {"role": s.role, "model": s.model, "text": s.text}
                        for s in loaded.sections
                    ],
                }
                if loaded is not None
                else None
            ),
        },
    )


@app.delete("/review/{review_id}")
async def delete_review(review_id: str) -> JSONResponse:
    review_storage.delete(review_id)
    return JSONResponse({"ok": True})


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

    # Create the persisted review session up front so it shows up in the
    # sidebar immediately. The runner thread updates it as sections complete.
    review_session = new_review_session(prompt, writer_model, reviewer_model, rounds)
    review_storage.save(review_session)

    job = job_manager.create(
        chat_id=f"review-{review_session.id}",
        request_data={
            "kind": "review",
            "review_id": review_session.id,
            "writer_model": writer_model,
            "reviewer_model": reviewer_model,
            "rounds": rounds,
            "prompt_chars": len(prompt),
        },
    )
    loop = asyncio.get_running_loop()
    _start_review_thread(
        job=job,
        review_id=review_session.id,
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
    return JSONResponse({"job_id": job.id, "review_id": review_session.id})


def _start_review_thread(
    *,
    job: Job,
    review_id: str,
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
    def _persist(status: str) -> None:
        """Save the current sections list to disk under the review session
        we created upfront. Idempotent — overwrites the file each time."""
        try:
            session = review_storage.load(review_id)
            if session is None:
                return
            session.status = status
            session.sections = [
                ReviewSection(role=s["role"], model=s["model"], text=s["text"])
                for s in sections
            ]
            review_storage.save(session)
        except Exception:
            logger.exception("Failed to persist review %s", review_id)

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
            chunks_since_lint = 0
            # Lint the writer's output every ~30 chunks (≈30 tokens). Cheap
            # (pyflakes is milliseconds) and doesn't block streaming since we
            # already run in a producer thread.
            LINT_INTERVAL_CHUNKS = 30

            for chunk in execution.stream:
                if job_manager.is_stop_requested(job.id):
                    raise _StopRequested(execution.selected_model)
                if chunk.content:
                    collected.append(chunk.content)
                    job_manager.append_chunk(job.id, chunk.content, loop)

                    if role == "writer":
                        chunks_since_lint += 1
                        if chunks_since_lint >= LINT_INTERVAL_CHUNKS:
                            chunks_since_lint = 0
                            _emit_lint(section_idx, "".join(collected))

            # One last lint pass after the section completes so the panel
            # reflects the final state (and not an interval-old snapshot).
            if role == "writer":
                _emit_lint(section_idx, "".join(collected))
            return "".join(collected)

        def _emit_lint(section_index: int, text: str) -> None:
            try:
                findings = lint_writer_output(text)
            except Exception:
                logger.exception("Lint pass failed (skipping)")
                return
            job_manager.emit_event(
                job.id,
                "lint_state",
                {"section_index": section_index, "findings": findings},
                loop,
            )

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

            _persist("done")
            job_manager.finish(
                job.id,
                "done",
                loop,
                metadata={"kind": "review", "sections": sections, "review_id": review_id},
            )
        except _StopRequested as stop:
            _persist("stopped")
            job_manager.finish(
                job.id,
                "stopped",
                loop,
                metadata={
                    "kind": "review",
                    "sections": sections,
                    "review_id": review_id,
                    "stopped_during": stop.model,
                },
            )
        except Exception as exc:
            logger.exception("Review job %s failed", job.id)
            _persist("error")
            job_manager.finish(
                job.id,
                "error",
                loop,
                error=str(exc),
                metadata={"kind": "review", "sections": sections, "review_id": review_id},
            )

    threading.Thread(target=_runner, daemon=True, name=f"review-{job.id[:8]}").start()


class _StopRequested(Exception):
    def __init__(self, model: str) -> None:
        self.model = model


# ---------------------------------------------------------------------------
# Hardware detection + model recommendation
# ---------------------------------------------------------------------------

def _ollama_installed_models() -> list[str]:
    """Names of models present in the user's Ollama. Reuses the chat_service
    client so we go through the same configured OLLAMA_HOST."""
    try:
        resp = chat_service.adapters["granite"].client.list()
    except Exception:
        return []
    names: list[str] = []
    for entry in resp.get("models", []):
        n = entry.get("name") or entry.get("model")
        if n:
            names.append(n)
    return names


# ---------------------------------------------------------------------------
# Compare page (one prompt → multiple models, side-by-side)
# ---------------------------------------------------------------------------

def _start_compare_thread(
    job: Job,
    request_obj: ChatRequest,
    model_key: str,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """One job per model. Streams content; on finish, attaches per-model
    metrics (token counts, elapsed, tok/s) so the UI can show metadata."""
    import time as _time
    from contextlib import closing

    def _runner() -> None:
        try:
            adapter = chat_service.adapters[model_key]
            wall_started = _time.perf_counter()
            prompt_tokens: int | None = None
            completion_tokens: int | None = None
            eval_duration_ns: int | None = None
            total_duration_ns: int | None = None

            with closing(adapter.stream_chat(request_obj)) as stream:
                for chunk in stream:
                    if job_manager.is_stop_requested(job.id):
                        job_manager.finish(job.id, "stopped", loop, metadata={
                            "model_key": model_key,
                            "model_name": adapter.model_name,
                        })
                        return
                    if chunk.content:
                        job_manager.append_chunk(job.id, chunk.content, loop)
                    if chunk.done:
                        prompt_tokens = chunk.prompt_tokens
                        completion_tokens = chunk.completion_tokens
                        eval_duration_ns = chunk.eval_duration_ns
                        total_duration_ns = chunk.total_duration_ns

            wall_elapsed = _time.perf_counter() - wall_started
            elapsed_s = (
                total_duration_ns / 1_000_000_000
                if total_duration_ns else wall_elapsed
            )
            tps = (
                completion_tokens / (eval_duration_ns / 1_000_000_000)
                if completion_tokens and eval_duration_ns else None
            )
            job_manager.finish(job.id, "done", loop, metadata={
                "model_key": model_key,
                "model_name": adapter.model_name,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "elapsed_s": elapsed_s,
                "tokens_per_sec": tps,
            })
        except Exception as exc:
            logger.exception("Compare run failed for model %s", model_key)
            job_manager.finish(job.id, "error", loop, error=str(exc), metadata={
                "model_key": model_key,
            })

    threading.Thread(
        target=_runner, daemon=True, name=f"compare-{model_key}-{job.id[:8]}",
    ).start()


def _compare_summaries() -> list[dict]:
    """Compact list rendering for the sidebar — preview + winner + timestamp."""
    out = []
    for r in comparison_storage.list_runs():
        preview = r.prompt[:60] + ("…" if len(r.prompt) > 60 else "")
        out.append({
            "id": r.id,
            "timestamp": r.timestamp,
            "preview": preview,
            "winner": r.winner,
            "model_count": len(r.outputs),
        })
    return out


@app.get("/compare", response_class=HTMLResponse)
async def compare_page(request: Request):
    return templates.TemplateResponse(
        request,
        "compare.html",
        {
            "model_keys": list(chat_service.adapters.keys()),
            "summaries": _compare_summaries(),
            "tally": comparison_storage.winner_tally(),
            "loaded": None,
            "agent_enabled": _agent_enabled,
        },
    )


@app.get("/compare/{run_id}", response_class=HTMLResponse)
async def compare_load_page(run_id: str, request: Request):
    loaded = comparison_storage.load(run_id)
    if loaded is None:
        raise HTTPException(404, f"comparison {run_id} not found")
    return templates.TemplateResponse(
        request,
        "compare.html",
        {
            "model_keys": list(chat_service.adapters.keys()),
            "summaries": _compare_summaries(),
            "tally": comparison_storage.winner_tally(),
            "loaded": dataclasses.asdict(loaded),
            "agent_enabled": _agent_enabled,
        },
    )


@app.delete("/compare/{run_id}")
async def delete_compare(run_id: str) -> JSONResponse:
    comparison_storage.delete(run_id)
    return JSONResponse({"ok": True})


@app.post("/api/compare")
async def start_compare(request: Request) -> JSONResponse:
    """Spawn one job per requested model. The client subscribes to each
    job's existing /api/jobs/{job_id}/stream endpoint to render columns
    in parallel. Run id is server-generated so save round-trips can use it."""
    body = await request.json()
    prompt = (body.get("prompt") or "").strip()
    system_prompt = (body.get("system_prompt") or "").strip()
    requested = body.get("models") or []
    if not prompt:
        raise HTTPException(400, "empty prompt")
    if not requested:
        raise HTTPException(400, "no models selected")
    available = chat_service.adapters
    unknown = [m for m in requested if m not in available]
    if unknown:
        raise HTTPException(400, f"unknown model(s): {', '.join(unknown)}")

    messages: list[ChatMessage] = []
    if system_prompt:
        messages.append(ChatMessage(role="system", content=system_prompt))
    messages.append(ChatMessage(role="user", content=prompt))
    chat_request = ChatRequest(messages=messages, stream=True)

    loop = asyncio.get_running_loop()
    run_id = new_comparison_run(prompt=prompt, system_prompt=system_prompt).id
    jobs: list[dict] = []
    for model_key in requested:
        job = job_manager.create(
            chat_id=f"compare:{run_id}",
            request_data={"selection": model_key, "model_key": model_key},
        )
        _start_compare_thread(job, chat_request, model_key, loop)
        jobs.append({"model_key": model_key, "job_id": job.id})

    return JSONResponse({"run_id": run_id, "jobs": jobs})


@app.post("/api/compare/save")
async def save_compare(request: Request) -> JSONResponse:
    """Persist a completed run. Client supplies the full payload (text +
    metrics from each job's done event) — server doesn't aggregate from
    JobManager because jobs may have been GC'd by now."""
    body = await request.json()
    run_id = body.get("run_id") or ""
    prompt = (body.get("prompt") or "").strip()
    system_prompt = (body.get("system_prompt") or "").strip()
    raw_outputs = body.get("outputs") or []
    winner = body.get("winner") or None
    if not run_id or not prompt or not raw_outputs:
        raise HTTPException(400, "missing run_id, prompt, or outputs")

    run = ComparisonRun(
        id=run_id,
        timestamp=new_comparison_run("").timestamp,
        prompt=prompt,
        system_prompt=system_prompt,
    )
    for o in raw_outputs:
        run.outputs.append(ModelOutput(
            model_key=str(o.get("model_key") or ""),
            model_name=str(o.get("model_name") or ""),
            text=str(o.get("text") or ""),
            prompt_tokens=o.get("prompt_tokens"),
            completion_tokens=o.get("completion_tokens"),
            elapsed_s=o.get("elapsed_s"),
            tokens_per_sec=o.get("tokens_per_sec"),
            error=o.get("error"),
        ))
    run.winner = winner if winner in {o.model_key for o in run.outputs} else None
    comparison_storage.save(run)
    return JSONResponse({"ok": True, "id": run.id})


@app.get("/hardware", response_class=HTMLResponse)
async def hardware_page(request: Request):
    hw = detect_hardware()
    rec = recommend_models(hw)
    payload = hw_to_dict(hw, rec, installed=_ollama_installed_models())
    return templates.TemplateResponse(
        request,
        "hardware.html",
        {
            "hw": payload["hardware"],
            "rec": payload["recommendation"],
        },
    )


@app.get("/api/hardware")
async def hardware_api() -> JSONResponse:
    hw = detect_hardware()
    rec = recommend_models(hw)
    return JSONResponse(hw_to_dict(hw, rec, installed=_ollama_installed_models()))
