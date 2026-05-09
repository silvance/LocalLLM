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
import os
import threading
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


class NoCacheStaticFiles(StaticFiles):
    """StaticFiles with `Cache-Control: no-cache` on every response.

    Default StaticFiles relies on browser heuristic caching (may keep
    a CSS file for hours), which means an updated stylesheet won't
    show up until the user hard-refreshes — pretty hostile during
    iterative UI work. `no-cache` tells the browser to revalidate on
    every request; combined with the parent's ETag/Last-Modified
    handling, unchanged files still get a cheap 304.
    """
    async def get_response(self, path, scope):  # type: ignore[override]
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response

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
from app.services.review_gates import (
    all_passed as gates_all_passed,
    format_for_writer as gates_format_for_writer,
    gate_candidate_count,
    run_gates,
)
from app.utils.outcomes_log import OutcomeLog, default_log_path, hash_prompt
from app.services.reviewer_verdict import (
    SCHEMA_INSTRUCTION as REVIEWER_VERDICT_SCHEMA,
    ReviewerVerdict,
    parse as parse_reviewer_verdict,
)
from app.utils.code_linter import extract_python_blocks
from app.utils.review_storage import (
    ReviewSection,
    ReviewStorage,
    new_session as new_review_session,
)
from app.utils.system_prompt import compose as compose_system_prompt
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
outcomes_log = OutcomeLog(default_log_path())


app = FastAPI(title="LocalLLM")


# ---------------------------------------------------------------------------
# Origin guard (CSRF defense for the local-only threat model)
# ---------------------------------------------------------------------------
# LocalLLM's threat model assumes the operator only ever uses it from a
# browser tab pointed at 127.0.0.1. Without an Origin check, ANY web
# page the operator visits during a forensics session can issue
# DELETE /chats/<id>, POST /api/ollama/pull (queue arbitrary downloads),
# POST /api/review (burn GPU cycles), etc. Reject non-GET/HEAD requests
# whose Origin or Referer host isn't localhost.
#
# Disable for testing / proxied setups via LOCALLLM_DISABLE_ORIGIN_GUARD=1.

_SAFE_ORIGIN_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})


def _origin_guard_disabled() -> bool:
    return os.environ.get("LOCALLLM_DISABLE_ORIGIN_GUARD", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


@app.exception_handler(ValueError)
async def _value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """ValueError at the API boundary maps to a 400. Storage-layer
    ID validators raise ValueError on invalid chat / review / run IDs
    (e.g. ``C:foo`` from a CSRF attempt or a fuzzer); without this
    handler they bubble up as 500 and leak the stack trace."""
    return JSONResponse({"detail": str(exc)}, status_code=400)


@app.middleware("http")
async def _origin_guard(request: Request, call_next):
    method = request.method.upper()
    if method in ("GET", "HEAD", "OPTIONS"):
        return await call_next(request)
    if _origin_guard_disabled():
        return await call_next(request)

    # Same-origin browser fetches always carry one of these headers.
    # CLI / curl requests don't, but they don't send Origin either, so
    # they pass — that's the local-only operator using the app
    # programmatically. Cross-origin browser requests DO send Origin
    # (for non-simple methods / non-simple bodies) and that's what we
    # reject.
    origin = request.headers.get("origin") or request.headers.get("referer")
    if origin is None:
        return await call_next(request)

    from urllib.parse import urlparse
    try:
        host = urlparse(origin).hostname or ""
    except Exception:
        host = ""
    if host in _SAFE_ORIGIN_HOSTS:
        return await call_next(request)

    logger.warning("Rejecting %s %s from origin %r", method, request.url.path, origin)
    return JSONResponse(
        status_code=403,
        content={"detail": f"cross-origin request from {host!r} blocked"},
    )


app.mount("/static", NoCacheStaticFiles(directory=str(_STATIC_DIR)), name="static")

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
            "model_options": _model_choices(),
            "active_job_id": active_job_id,
            "agent_enabled": _agent_enabled,
            "ollama_status": get_ollama_status(),
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
    composed_system = compose_system_prompt(system_prompt)
    if composed_system:
        request_messages.append(ChatMessage(role="system", content=composed_system))
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
    job, queue, snapshot = job_manager.subscribe_with_snapshot(job_id)
    if job is None or queue is None or snapshot is None:
        raise HTTPException(404, f"job {job_id} not found")

    async def gen() -> AsyncIterator[bytes]:
        try:
            # Replay from the FROZEN snapshot taken under the lock at
            # subscribe time, not from live job.* fields — otherwise the
            # producer thread can append a checkpoint between snapshot
            # reads and we end up duplicating chunks (or pointing at a
            # `pos` that doesn't exist in the text we sliced). The
            # remainder of the live stream comes through `queue` below.
            text = snapshot["text"]
            checkpoints = snapshot["checkpoints"]
            status = snapshot["status"]
            if checkpoints:
                last_pos = 0
                for event, payload, pos in checkpoints:
                    if pos > last_pos:
                        yield _sse("token", {"chunk": text[last_pos:pos]})
                    yield _sse(event, payload)
                    last_pos = pos
                if last_pos < len(text):
                    yield _sse("token", {"chunk": text[last_pos:]})
            elif text:
                yield _sse("token", {"chunk": text})

            if status in ("done", "error", "stopped"):
                yield _sse(
                    "done",
                    {
                        "status": status,
                        "error": snapshot["error"],
                        "metadata": snapshot["metadata"],
                        "text": text,
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
    "You are a senior code reviewer for security/forensics tooling. "
    "Review the code under review for two classes of problem.\n\n"

    "## Class 1: code-correctness bugs\n"
    "Missing imports, undefined names, dataclass misuse, attribute "
    "references that don't exist on the class, broken control flow, "
    "wrong exception types, security footguns (subprocess shell=True, "
    "yaml.load without SafeLoader, eval, pickle on untrusted input), "
    "concurrency hazards, etc.\n\n"

    "## Class 3: real vs simulated implementation\n"
    "Models will sometimes return code that LOOKS finished but doesn't "
    "actually do the thing. Flag any of these:\n\n"
    "- Comments like `# simulated`, `# placeholder`, `# for "
    "demonstration`, `# would do X in production`, `# TODO: actually "
    "implement` next to code that pretends to be the real thing.\n"
    "- Variable / function names like `simulated_packet`, "
    "`fake_device`, `mock_response`, `dummy_data` in code that's "
    "supposed to be production logic (not a test fixture).\n"
    "- String literals like `print(\"simulated BLE packet: 0x42\")` — "
    "the model is announcing that the output is fabricated.\n"
    "- Function bodies that are just `time.sleep(...)` + a `print` "
    "loop when the function name implies actual I/O (`sniff_*`, "
    "`fetch_*`, `scan_*`, `parse_*`).\n"
    "- Hardcoded \"realistic-looking\" return values: "
    "`return [{\"name\": \"AC:DE:48:00:11:22\", \"rssi\": -50}]` "
    "from a function that should be doing live discovery.\n\n"
    "If you see this, the right verdict is `pass: false`, "
    "`recommended_next_action: \"rebuild_different_writer\"` — the "
    "writer is taking a shortcut and a different model is more likely "
    "to attempt the real implementation.\n\n"

    "## Class 2: domain / hardware / protocol grounding\n"
    "Verify the implementation is actually possible on the stated "
    "hardware and OS — this is where local code-gen models hallucinate "
    "the loudest. Specifically flag any of these:\n\n"
    "- BLE / Bluetooth packets being sniffed, decoded, or transmitted "
    "from a Wi-Fi interface (wlan0, mon0, wlp*, en*, anything in monitor "
    "mode for 802.11). BLE goes through HCI / btmon / BlueZ on Linux, "
    "Core Bluetooth on macOS, the Microsoft Bluetooth stack on Windows. "
    "It cannot be sniffed from a Wi-Fi card.\n"
    "- 802.11 monitor-mode operations (airodump-ng, scapy.sniff on a "
    "wlan iface, deauth frames) being driven via Bluetooth / HCI APIs.\n"
    "- SDR work (RTL-SDR, HackRF, BladeRF) being attempted via socket / "
    "raw network APIs instead of the SDR's actual driver (pyrtlsdr, "
    "soapy, GNU Radio, etc.).\n"
    "- USB peripheral access via /dev/ttyUSB* or pyserial when the "
    "device speaks HID, libusb, or a vendor protocol.\n"
    "- Linux-only paths (/proc, /sys, iw, hcitool, ip-link) used "
    "without a Windows or macOS branch when the original task didn't "
    "pin an OS.\n"
    "- Privilege assumptions (root, CAP_NET_RAW, CAP_NET_ADMIN) used "
    "without acknowledging them.\n"
    "- Network-namespace / monitor-mode / promiscuous-mode commands run "
    "without a clear teardown so the user's interface stays in a "
    "broken state.\n"
    "- Protocols / standards confused with each other (Zigbee vs "
    "Thread vs Z-Wave; LoRa vs LoRaWAN; UART vs SPI vs I²C).\n\n"

    "List every issue briefly and specifically — line refs or quoted "
    "snippets where helpful. Do not rewrite the code yourself; just "
    "call out what's wrong."
)
REVISE_INSTRUCTION = (
    "You wrote the following code. A reviewer identified the issues below. "
    "Produce a revised version that addresses every point. Output the full "
    "revised code in fenced markdown blocks — do not skip unchanged sections."
)


@app.get("/review", response_class=HTMLResponse)
async def review_page(request: Request):
    # If a review is currently running, jump straight to its detail page
    # so the user lands on the live stream instead of a blank form. The
    # /review/{id} route's `loaded` state + active_job_id combo handles
    # resume from there.
    in_flight_id = _find_active_review_id()
    if in_flight_id:
        return RedirectResponse(
            url=f"/review/{in_flight_id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )
    return _render_review_page(request, loaded=None, active_job_id=None)


@app.get("/review/{review_id}", response_class=HTMLResponse)
async def review_load_page(review_id: str, request: Request):
    session = review_storage.load(review_id)
    if session is None:
        raise HTTPException(404, f"review {review_id} not found")
    active = job_manager.jobs_active_for_chat(f"review-{review_id}")
    return _render_review_page(
        request,
        loaded=session,
        active_job_id=active[0].id if active else None,
    )


def _find_active_review_id() -> Optional[str]:
    """Most-recent review whose job is still streaming. Returns the
    review_id (which doubles as a route key) so /review can redirect."""
    candidates = [
        j for j in job_manager.list_with_chat_prefix("review-")
        if j.status in ("pending", "streaming")
    ]
    if not candidates:
        return None
    job = max(candidates, key=lambda j: j.started_at)
    rid = str(job.request_data.get("review_id") or "")
    return rid or None


def _render_review_page(request: Request, loaded, active_job_id=None) -> HTMLResponse:
    summaries = review_storage.list_summaries()
    return templates.TemplateResponse(
        request,
        "review.html",
        {
            "settings": settings,
            "model_options": _model_choices(),
            "summaries": summaries,
            "loaded": loaded,
            "active_job_id": active_job_id,
            "ollama_status": get_ollama_status(),
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
    fallback_writer_model = (body.get("fallback_writer_model") or "").strip() or None
    rounds = int(body.get("rounds") or DEFAULT_MAX_TOTAL_ROUNDS)
    if rounds < 2:
        raise HTTPException(400, "rounds must be at least 2 (write + review)")
    if rounds > MAX_TOTAL_ROUNDS_CAP:
        raise HTTPException(400, f"rounds capped at {MAX_TOTAL_ROUNDS_CAP}")

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
        fallback_writer_model=fallback_writer_model,
        rounds=rounds,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        num_ctx=num_ctx,
        loop=loop,
    )
    return JSONResponse({"job_id": job.id, "review_id": review_session.id})


# Loop budget — multi-dimensional caps for the writer ↔ reviewer loop.
# Each dimension counts a different failure mode so a writer that's
# bad at code-correctness gets swapped on a different schedule than
# one that's bad at domain grounding. All operator-tunable; the
# defaults match the field-tested values from a few real review runs.
DEFAULT_MAX_TOTAL_ROUNDS = 10            # was bare `rounds`; cap raised below
MAX_TOTAL_ROUNDS_CAP = 12                # absolute upper bound on rounds input
MAX_SAME_WRITER_STATIC_FAILURES = 2      # static-gate fails before swap
MAX_SAME_WRITER_DOMAIN_FAILURES = 1      # reviewer-blocker rounds before swap
MAX_UNIQUE_BUILDERS = 3                  # primary + ≤2 fallbacks total
MAX_SAME_BLOCKER_REPEATS = 2             # same blocker text twice → abort
_FALLBACK_WRITER_THRESHOLD = MAX_SAME_WRITER_STATIC_FAILURES  # legacy alias


def _serialize_sections(sections: list[dict]) -> list[dict]:
    """Strip in-memory-only fields (the ReviewerVerdict dataclass) so
    the sections list is JSON-safe when emitted via SSE / metadata.
    The verdict already shaped the orchestrator's decisions; the
    client only needs role / model / text for rendering."""
    return [{k: v for k, v in s.items() if k != "verdict"} for s in sections]


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
    fallback_writer_model: str | None = None,
) -> None:
    def _persist(status: str, sections: list[dict]) -> None:
        """Save the current sections list to disk under the review session
        we created upfront. Idempotent — overwrites the file each time.

        ``sections`` is passed explicitly because the list lives inside
        ``_runner``'s scope; ``_persist`` is a sibling closure that
        otherwise wouldn't see it.
        """
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
        # Mutable across rounds — the gate path may swap the active
        # writer when the primary fails repeatedly. Captured by the
        # closures below so they always pick up the current value.
        active_writer = writer_model
        # Per-writer failure counters. Keyed by writer name so a swap
        # resets the budget for the new writer.
        writer_static_fails: dict[str, int] = {writer_model: 0}
        writer_domain_fails: dict[str, int] = {writer_model: 0}
        # Builders we've used so far, in swap order. Capped at
        # MAX_UNIQUE_BUILDERS so we don't churn through every model
        # the operator has installed.
        used_builders: list[str] = [writer_model]
        # Remaining fallbacks. fallback_writer_model accepts comma-
        # separated names so the operator can configure a chain
        # (e.g. "deepseek-coder-v2:latest,granite4") and the swap
        # logic pops the next one each time. Dedupe against the
        # primary writer so "qwen,granite,qwen" doesn't waste a
        # unique-builder slot on the same model twice.
        if fallback_writer_model:
            seen_chain: set[str] = {writer_model}
            fallback_chain: list[str] = []
            for m in fallback_writer_model.split(","):
                m = m.strip()
                if not m or m in seen_chain:
                    continue
                seen_chain.add(m)
                fallback_chain.append(m)
        else:
            fallback_chain = []
        # Reviewer blocker frequency. Normalised text → count across
        # ALL reviewer rounds; if any blocker hits MAX_SAME_BLOCKER_REPEATS
        # the orchestrator aborts because we're clearly stuck.
        blocker_counts: dict[str, int] = {}
        # Set when a hard-stop condition fires so the main loop breaks
        # at the next iteration boundary.
        abort_reason: str | None = None
        # Set true the first time we swap to a fallback so the next
        # writer round uses the fresh-start kickoff prompt instead of
        # the usual "fix your previous code" framing.
        fallback_kickoff_pending = False
        # Accumulated gate-failure messages so the fallback's kickoff
        # prompt can list every static issue the previous writer kept
        # tripping on — not just the latest one.
        gate_failure_history: list[str] = []

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
            composed = compose_system_prompt(system_prompt)
            if composed:
                msgs.append(ChatMessage(role="system", content=composed))
            return msgs

        def _emit_inline_section(role: str, model: str, text: str) -> None:
            """Render a non-LLM section (gates, fallback notice) in the
            UI by emitting section_start + a token chunk and appending
            to ``sections``. Mirrors the way live writer/reviewer
            sections show up on screen."""
            section_idx = len(sections)
            job_manager.emit_event(
                job.id,
                "section_start",
                {"index": section_idx, "role": role, "model": model},
                loop,
            )
            job_manager.append_chunk(job.id, text, loop)
            sections.append({"role": role, "model": model, "text": text})

        def _normalize_blocker(text: str) -> str:
            """Canonical key for blocker-repetition counting. casefold +
            collapse whitespace + truncate so trivial wording
            differences (and locale-quirky lowercase like Turkish
            dotted/dotless I) still match as the same blocker."""
            return " ".join(text.casefold().split())[:200]

        def gate_or_review(writer_text: str) -> None:
            """Run static-analysis gates on the writer's latest output.
            On failure: emit a `gates` section, bump
            writer_static_fails[active_writer], skip the reviewer.
            On success: run the reviewer; bump
            writer_domain_fails[active_writer] if there are any
            blockers; track each blocker's text frequency and abort
            the loop if any text repeats >= MAX_SAME_BLOCKER_REPEATS.
            """
            nonlocal abort_reason
            prompt_h = hash_prompt(prompt)
            # Gate 0: candidate_count runs on the FULL writer response
            # so its failure message can talk about fence syntax. If
            # the writer emitted zero or multiple python blocks, the
            # downstream gates would either run on empty code (giving
            # vacuous "OK"s) or on a concatenation of unrelated
            # programs (giving spurious failures), neither of which
            # gives the writer useful feedback.
            cc = gate_candidate_count(writer_text)
            if not cc.passed:
                writer_static_fails[active_writer] = writer_static_fails.get(active_writer, 0) + 1
                feedback = gates_format_for_writer([cc])
                gate_failure_history.append(feedback)
                _emit_inline_section("gates", "(static analysis)", feedback)
                outcomes_log.append(
                    review_id=review_id, writer=active_writer,
                    event="gate_fail", gate=cc.name, prompt_hash=prompt_h,
                )
                return

            blocks = extract_python_blocks(writer_text)
            code = "\n\n".join(blocks)
            results = run_gates(code) if code else []
            if results and not gates_all_passed(results):
                writer_static_fails[active_writer] = writer_static_fails.get(active_writer, 0) + 1
                feedback = gates_format_for_writer(results)
                gate_failure_history.append(feedback)
                _emit_inline_section("gates", "(static analysis)", feedback)
                first_fail = next((r for r in results if not r.passed), None)
                outcomes_log.append(
                    review_id=review_id, writer=active_writer,
                    event="gate_fail",
                    gate=first_fail.name if first_fail else "unknown",
                    prompt_hash=prompt_h,
                )
                return

            # All gates passed — log it before invoking the reviewer
            # so the model-stats CLI can compute a clean pass rate.
            outcomes_log.append(
                review_id=review_id, writer=active_writer,
                event="gate_pass", prompt_hash=prompt_h,
            )

            review_user = (
                f"{REVIEWER_INSTRUCTION}\n\n{REVIEWER_VERDICT_SCHEMA}\n\n"
                f"---\n\nOriginal task:\n{prompt}\n\n"
                f"---\n\nCode under review:\n\n{writer_text}"
            )
            msgs = base_messages() + [ChatMessage(role="user", content=review_user)]
            text = run_section(role="reviewer", model=reviewer_model, messages=msgs)
            verdict = parse_reviewer_verdict(text)

            # Domain-failure tracking. Any blockers from the reviewer
            # count as a domain failure for the current writer.
            if verdict.blockers:
                writer_domain_fails[active_writer] = (
                    writer_domain_fails.get(active_writer, 0) + 1
                )
                # Repeat tracking is per UNIQUE blocker, regardless of
                # which writer round produced it — same conceptual bug
                # surfacing twice means we're stuck.
                for b in verdict.blockers:
                    key = _normalize_blocker(b)
                    if not key:
                        continue
                    blocker_counts[key] = blocker_counts.get(key, 0) + 1
                    if blocker_counts[key] >= MAX_SAME_BLOCKER_REPEATS and abort_reason is None:
                        abort_reason = (
                            f"Same blocker reported {blocker_counts[key]} times: "
                            f"{b[:120]!r}"
                        )

            # Persist the cleaned prose (JSON block stripped) for the
            # human-facing review log; the verdict object is kept on the
            # in-memory section dict for the orchestrator to branch on.
            sections.append({
                "role": "reviewer",
                "model": reviewer_model,
                "text": verdict.display_text(),
                "verdict": verdict,
            })
            # Outcome log: writer attribution is the active_writer
            # (the model whose code the reviewer just looked at), not
            # the reviewer model.
            outcomes_log.append(
                review_id=review_id, writer=active_writer,
                event="review_block" if verdict.blockers else "review_pass",
                blocker_count=len(verdict.blockers) or None,
                prompt_hash=prompt_h,
            )

        def maybe_swap_writer(reason: str | None = None) -> None:
            """Pop the next builder off `fallback_chain` and engage it.

            Triggers:
              * reason=None — static-gate threshold met
                (writer_static_fails[active] >= MAX_SAME_WRITER_STATIC_FAILURES).
              * reason="domain" — reviewer-blocker threshold met
                (writer_domain_fails[active] >= MAX_SAME_WRITER_DOMAIN_FAILURES).
              * reason="reviewer_recommended" — reviewer's verdict
                next_action == "rebuild_different_writer". Bypasses
                the static / domain thresholds.

            Hard caps:
              * fallback_chain must be non-empty.
              * len(used_builders) < MAX_UNIQUE_BUILDERS — we don't
                churn through every model installed.
            """
            nonlocal active_writer, fallback_kickoff_pending
            if not fallback_chain:
                return
            if len(used_builders) >= MAX_UNIQUE_BUILDERS:
                return
            if reason is None and writer_static_fails.get(active_writer, 0) < MAX_SAME_WRITER_STATIC_FAILURES:
                return
            if reason == "domain" and writer_domain_fails.get(active_writer, 0) < MAX_SAME_WRITER_DOMAIN_FAILURES:
                return

            previous = active_writer
            active_writer = fallback_chain.pop(0)
            used_builders.append(active_writer)
            writer_static_fails.setdefault(active_writer, 0)
            writer_domain_fails.setdefault(active_writer, 0)
            # Tell the next writer-round to use the fresh-start kickoff
            # prompt instead of the usual "fix your previous code"
            # template — otherwise the fallback model just regurgitates
            # whatever broken pattern got us here.
            fallback_kickoff_pending = True

            if reason == "reviewer_recommended":
                why = (
                    "Reviewer recommended a different writer model — the "
                    "current primary keeps making the same domain-level "
                    "mistake. "
                )
            elif reason == "domain":
                why = (
                    f"Writer `{previous}` failed reviewer domain gate "
                    f"{writer_domain_fails.get(previous, 0)} time(s) "
                    f"(threshold: {MAX_SAME_WRITER_DOMAIN_FAILURES}). "
                )
            else:
                why = (
                    f"Writer `{previous}` failed static-analysis gates "
                    f"{writer_static_fails.get(previous, 0)} times "
                    f"(threshold: {MAX_SAME_WRITER_STATIC_FAILURES}). "
                )
            notice = (
                f"{why}Switching builder #{len(used_builders)}/"
                f"{MAX_UNIQUE_BUILDERS} to `{active_writer}` for the next "
                f"rebuild round."
            )
            _emit_inline_section("fallback", "(orchestrator)", notice)

        def revise_user_message(prev_writer_text: str, feedback_section: dict) -> str:
            """Build the writer's next prompt. Format depends on whether
            the previous slot was a reviewer (domain feedback) or a
            gates run (static-analysis feedback) — the writer needs
            different framing for each."""
            if feedback_section["role"] == "gates":
                return (
                    f"Your previous code failed static-analysis gates. "
                    f"Fix every issue listed below and resubmit the FULL "
                    f"corrected code in fenced markdown blocks. Do not add "
                    f"prose unrelated to fixing these issues.\n\n"
                    f"---\n\nOriginal task:\n{prompt}\n\n"
                    f"---\n\nGate failures:\n\n{feedback_section['text']}\n\n"
                    f"---\n\nYour previous code:\n\n{prev_writer_text}"
                )
            return (
                f"{REVISE_INSTRUCTION}\n\n---\n\nOriginal task:\n{prompt}\n\n"
                f"---\n\nYour previous code:\n\n{prev_writer_text}\n\n"
                f"---\n\nReviewer's notes:\n\n{feedback_section['text']}"
            )

        def fallback_kickoff_message(primary_name: str) -> str:
            """First prompt the fallback writer sees. Fresh-start
            framing: don't show the broken code (model would just
            patch the same bug); list every gate failure the primary
            kept hitting; require single fenced python block (no
            prose / setup blocks since those are what tripped the
            block extractor in the first place)."""
            joined = "\n\n".join(gate_failure_history[-3:]) or "(none recorded)"
            primary_static = writer_static_fails.get(primary_name, 0)
            primary_domain = writer_domain_fails.get(primary_name, 0)
            return (
                f"You are the fallback builder, taking over from "
                f"`{primary_name}` (static failures: {primary_static}, "
                f"reviewer-blocker rounds: {primary_domain}).\n\n"
                f"Original user request:\n{prompt}\n\n"
                f"Required output format:\n"
                f"- Return EXACTLY ONE fenced ```python ... ``` code block.\n"
                f"- Do NOT include markdown prose, headers, or shell "
                f"snippets outside the code block. Setup instructions, "
                f"if any, go inside the code as comments.\n"
                f"- Do NOT use wildcard imports (`from x import *`).\n"
                f"- Code must parse with ast.parse, import cleanly, "
                f"and not reference undefined names.\n"
                f"- Import every name you use (no `threading.Thread` "
                f"without `import threading`, etc.).\n\n"
                f"Static failures the previous builder kept producing — "
                f"avoid all of these:\n\n{joined}\n\n"
                f"Now produce a fresh implementation. Start from "
                f"scratch — do NOT patch the previous code; the "
                f"approach was wrong."
            )

        try:
            # Round 0 — writer produces initial code
            writer_msgs = base_messages() + [ChatMessage(role="user", content=prompt)]
            writer_text = run_section(role="writer", model=active_writer, messages=writer_msgs)
            sections.append({"role": "writer", "model": active_writer, "text": writer_text})

            # Round 1 — gates → reviewer (or gate failure stops here)
            gate_or_review(writer_text)

            # Subsequent rounds alternate writer revision / (gates → reviewer)
            for i in range(2, rounds):
                # Hard-abort check: same blocker text repeated past
                # threshold means we're stuck. Set by gate_or_review
                # after each reviewer round.
                if abort_reason is not None:
                    _emit_inline_section(
                        "fallback", "(orchestrator)",
                        f"Stopping: {abort_reason}. The loop isn't "
                        f"making progress — abort is cheaper than "
                        f"more rounds.",
                    )
                    outcomes_log.append(
                        review_id=review_id, writer=active_writer,
                        event="abort", prompt_hash=hash_prompt(prompt),
                        detail=abort_reason,
                    )
                    break
                if i % 2 == 0:
                    # Before the next rebuild, consult the latest
                    # reviewer verdict (if any). The reviewer's JSON
                    # `recommended_next_action` lets the orchestrator
                    # short-circuit clearly-finished reviews and engage
                    # the fallback for domain-level failures.
                    last = sections[-1] if sections else None
                    last_verdict: ReviewerVerdict | None = (
                        last.get("verdict") if last and last["role"] == "reviewer" else None
                    )
                    if last_verdict and last_verdict.next_action == "approve":
                        break  # reviewer says we're done
                    if last_verdict and last_verdict.next_action == "abort":
                        _emit_inline_section(
                            "fallback", "(orchestrator)",
                            "Reviewer indicated the task can't be completed as "
                            "specified (`safe_to_rebuild=false` or "
                            "`recommended_next_action=abort`). Stopping.",
                        )
                        break
                    if last_verdict and last_verdict.next_action == "rebuild_different_writer":
                        maybe_swap_writer(reason="reviewer_recommended")
                    elif last and last["role"] == "reviewer":
                        # Reviewer-blocker swap — kick fallback if the
                        # active writer has hit its domain-failure
                        # threshold (default: 1).
                        maybe_swap_writer(reason="domain")
                    else:
                        # Last slot was a `gates` fail — try the static
                        # threshold instead.
                        maybe_swap_writer()
                    # Pick the prompt: fresh-start kickoff if we just
                    # engaged the fallback (don't show it the broken
                    # code), otherwise the regular revise template.
                    if fallback_kickoff_pending:
                        revise_user = fallback_kickoff_message(writer_model)
                        fallback_kickoff_pending = False
                        # No base_messages() system prompt — the kickoff
                        # message is self-contained and includes its own
                        # output-format constraints. The system prompt
                        # would just dilute them.
                        msgs = [ChatMessage(role="user", content=revise_user)]
                    else:
                        prev_writer = next(
                            (s for s in reversed(sections) if s["role"] == "writer"),
                            sections[0],
                        )
                        revise_user = revise_user_message(prev_writer["text"], sections[-1])
                        msgs = base_messages() + [ChatMessage(role="user", content=revise_user)]
                    text = run_section(role="writer", model=active_writer, messages=msgs)
                    sections.append({"role": "writer", "model": active_writer, "text": text})
                else:
                    # Gates → reviewer for the latest writer output.
                    gate_or_review(sections[-1]["text"])

            _persist("done", sections)
            job_manager.finish(
                job.id,
                "done",
                loop,
                metadata={"kind": "review", "sections": _serialize_sections(sections), "review_id": review_id},
            )
        except _StopRequested as stop:
            _persist("stopped", sections)
            job_manager.finish(
                job.id,
                "stopped",
                loop,
                metadata={
                    "kind": "review",
                    "sections": _serialize_sections(sections),
                    "review_id": review_id,
                    "stopped_during": stop.model,
                },
            )
        except Exception as exc:
            logger.exception("Review job %s failed", job.id)
            _persist("error", sections)
            job_manager.finish(
                job.id,
                "error",
                loop,
                error=str(exc),
                metadata={"kind": "review", "sections": _serialize_sections(sections), "review_id": review_id},
            )

    threading.Thread(target=_runner, daemon=True, name=f"review-{job.id[:8]}").start()


class _StopRequested(Exception):
    def __init__(self, model: str) -> None:
        self.model = model


# ---------------------------------------------------------------------------
# Hardware detection + model recommendation
# ---------------------------------------------------------------------------

# Cached `ollama list` result. Every page render that hits a model
# dropdown calls _ollama_installed_models(); without a cache that's a
# fresh /api/tags HTTP round-trip per navigation, which Ollama can
# take a noticeable fraction of a second to answer when 10+ models
# are installed (the response includes metadata per model). 30 seconds
# is plenty fresh for "did I just pull a model?" UX, and the pull
# endpoint invalidates explicitly on success so the new model shows
# up immediately.
_installed_models_cache: "tuple[float, list[str]] | None" = None
_INSTALLED_MODELS_TTL_S = 30.0
# Last-known reachability state — populated alongside the model list
# so templates can render a "can't reach Ollama" banner without
# duplicating the discovery logic.
_ollama_status: dict = {"reachable": True, "error": None, "tried_urls": []}


def _invalidate_installed_models_cache() -> None:
    """Drop the cached model list. Called after a successful pull so
    the new model shows up in the next dropdown render without waiting
    for the TTL to expire."""
    global _installed_models_cache
    _installed_models_cache = None


def get_ollama_status() -> dict:
    """Snapshot of the last Ollama discovery attempt — what the templates
    use to decide whether to render the unreachable-banner. Populated
    inside _ollama_installed_models on every cache miss."""
    base_url = chat_service.settings.ollama_host or "http://127.0.0.1:11434"
    return {
        "reachable": bool(_ollama_status.get("reachable")),
        "error": _ollama_status.get("error"),
        "tried_urls": list(_ollama_status.get("tried_urls") or []),
        "configured_url": base_url,
        "model_count": len(_ollama_installed_models()),
    }


def _ollama_installed_models(*, force_refresh: bool = False) -> list[str]:
    """Names of models present in the user's Ollama daemon.

    Tries the ollama-python client first (reuses an already-open
    connection). On any failure — daemon not running, API drift in the
    pinned ollama package, surprise return type — falls back to a
    direct HTTP call against /api/tags. Logs each step at WARN so a
    silently-empty dropdown isn't a mystery: the operator can grep
    ollama-localllm.log / uvicorn output to see what failed.

    Cached with a short TTL so page-rendering hot paths don't hit the
    daemon on every navigation (was the cause of 5-10s lag on /review
    ↔ /chats round-trips).
    """
    import time
    global _installed_models_cache
    # Snapshot the tuple to a local before unpacking — a concurrent
    # invalidate (from the pull-thread completion path) could None
    # this out between the `is not None` check and the unpack,
    # raising TypeError.
    snapshot = _installed_models_cache
    if not force_refresh and snapshot is not None:
        ts, cached = snapshot
        if time.monotonic() - ts < _INSTALLED_MODELS_TTL_S:
            return cached

    base_url = chat_service.settings.ollama_host or "http://127.0.0.1:11434"
    tried_urls: list[str] = []
    last_error: str | None = None

    # Path 1 — ollama-python client (existing path, fastest).
    tried_urls.append(f"{base_url} (ollama-python)")
    try:
        resp = chat_service.adapters["granite"].client.list()
    except Exception as exc:
        last_error = str(exc)
        logger.warning(
            "Ollama list via python client failed (%s); falling back to /api/tags HTTP",
            exc,
        )
    else:
        names = _parse_ollama_list_response(resp)
        if names:
            _installed_models_cache = (time.monotonic(), names)
            _ollama_status.update(reachable=True, error=None, tried_urls=tried_urls)
            return names
        logger.warning(
            "Ollama python client returned empty / unrecognised payload (%r); "
            "falling back to /api/tags HTTP", type(resp).__name__,
        )
        last_error = f"empty payload ({type(resp).__name__})"

    # Path 2 — direct HTTP. Doesn't share the python client's response
    # parsing, so it survives ollama-python API changes between
    # versions we haven't pinned to. Use the raising variant so the
    # real error propagates to the UI banner instead of being
    # swallowed into "daemon reachable but empty".
    from app.services.ollama_models import installed_names_or_raise
    candidate_urls = [base_url]
    # Windows defaults `localhost` to ::1 first, but Ollama only binds
    # IPv4. If the operator's .env still has `localhost`, the python
    # client + first HTTP attempt both fail with WinError 10061.
    # Try the IPv4 loopback as a silent retry before giving up.
    if "localhost" in base_url and "127.0.0.1" not in base_url:
        candidate_urls.append(base_url.replace("localhost", "127.0.0.1"))

    names: list[str] = []
    last_err: Exception | None = None
    for url in candidate_urls:
        tried_urls.append(f"{url} (HTTP)")
        try:
            names = installed_names_or_raise(url)
            if names:
                if url != base_url:
                    logger.warning(
                        "Ollama unreachable at %s but reachable at %s — "
                        "set OLLAMA_HOST=%s in .env to silence the retry "
                        "and avoid the (~1s) IPv6 timeout per page render.",
                        base_url, url, url,
                    )
                break
        except Exception as exc:
            last_err = exc
            logger.warning("ollama list (%s) failed: %s", url, exc)
            continue

    reachable = bool(names)
    if not names:
        if last_err is not None:
            last_error = str(last_err)
            logger.warning(
                "Ollama list failed at %s: %s. "
                "If Ollama is running but bound to a non-default address, "
                "set OLLAMA_HOST in .env (e.g. OLLAMA_HOST=http://127.0.0.1:11434).",
                base_url, last_err,
            )
        else:
            last_error = "daemon reachable but reported zero models"
            logger.warning(
                "Ollama at %s returned no models — daemon running but empty?",
                base_url,
            )
    _installed_models_cache = (time.monotonic(), names)
    _ollama_status.update(
        reachable=reachable,
        error=last_error if not reachable else None,
        tried_urls=tried_urls,
    )
    return names


def _parse_ollama_list_response(resp) -> list[str]:
    """Pull model names out of whatever shape ``ollama.Client.list()``
    returned. Ollama-python's return type changed from dict to a Pydantic
    SubscriptableBaseModel between minor versions — handle both."""
    # Dict-style (older) or SubscriptableBaseModel.get("models").
    try:
        models = resp.get("models", []) if hasattr(resp, "get") else getattr(resp, "models", [])
    except Exception:
        models = []
    out: list[str] = []
    for entry in models or []:
        # Each entry might be a dict OR a Model object with .model / .name attrs.
        name = None
        if hasattr(entry, "get"):
            name = entry.get("name") or entry.get("model")
        if not name:
            name = getattr(entry, "model", None) or getattr(entry, "name", None)
        if name:
            out.append(str(name))
    return out


def _model_choices(*, include_auto: bool = True) -> list[str]:
    """Dropdown options for any model selector in the UI. Order:
    1. ``auto`` (preset router) — only when include_auto is True.
    2. The three preset slots (granite / gemma / qwen) — always
       present so existing flows / tests / smart-install presets
       keep working even if Ollama hasn't been queried yet.
    3. Every model name reported by Ollama, in install order.

    Slot names are kept distinct from raw model names ("granite" vs
    "granite4:latest") so the user can pick "granite" to follow the
    smart-install slot OR pin to "granite4:latest" directly."""
    choices: list[str] = []
    if include_auto:
        choices.append("auto")
    for slot in ("granite", "gemma", "qwen"):
        if slot not in choices:
            choices.append(slot)
    for name in _ollama_installed_models():
        if name not in choices:
            choices.append(name)
    return choices


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
            adapter = chat_service.get_adapter(model_key)
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
            "model_keys": _model_choices(include_auto=False),
            "summaries": _compare_summaries(),
            "tally": comparison_storage.winner_tally(),
            "ollama_status": get_ollama_status(),
            "loaded": None,
            "active_run": _find_active_compare_run(),
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
            "model_keys": _model_choices(include_auto=False),
            "summaries": _compare_summaries(),
            "tally": comparison_storage.winner_tally(),
            "ollama_status": get_ollama_status(),
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
    # Validate against the union of preset slots + actually-installed
    # Ollama models. ChatService.get_adapter builds adapters lazily,
    # but we still want fast feedback on a typo (e.g. "qewn3-coder")
    # rather than letting it fail mid-stream against Ollama's HTTP API.
    available = set(_model_choices(include_auto=False))
    unknown = [m for m in requested if m not in available]
    if unknown:
        raise HTTPException(400, f"unknown model(s): {', '.join(unknown)}")

    messages: list[ChatMessage] = []
    composed_system = compose_system_prompt(system_prompt)
    if composed_system:
        messages.append(ChatMessage(role="system", content=composed_system))
    messages.append(ChatMessage(role="user", content=prompt))
    chat_request = ChatRequest(messages=messages, stream=True)

    loop = asyncio.get_running_loop()
    run_id = new_comparison_run(prompt=prompt, system_prompt=system_prompt).id
    jobs: list[dict] = []
    for model_key in requested:
        job = job_manager.create(
            chat_id=f"compare:{run_id}",
            request_data={
                "kind": "compare",
                "run_id": run_id,
                "selection": model_key,
                "model_key": model_key,
                # Stash prompt + system_prompt + the full requested-model
                # list so /compare can rebuild the run state on reload
                # without consulting any other store.
                "prompt": prompt,
                "system_prompt": system_prompt,
                "all_model_keys": list(requested),
            },
        )
        _start_compare_thread(job, chat_request, model_key, loop)
        jobs.append({"model_key": model_key, "job_id": job.id})

    return JSONResponse({"run_id": run_id, "jobs": jobs})


def _find_active_compare_run() -> Optional[dict]:
    """Most-recent compare run with at least one job still streaming.

    Returns the run's id, the original prompt/system, and the per-model
    job list — enough for the /compare page to re-attach SSE streams
    and re-populate the form on reload.
    """
    by_run: dict[str, list[Job]] = {}
    for j in job_manager.list_with_chat_prefix("compare:"):
        if j.request_data.get("kind") != "compare":
            continue
        rid = str(j.request_data.get("run_id") or "")
        if not rid:
            continue
        by_run.setdefault(rid, []).append(j)
    candidates = [
        (rid, jobs) for rid, jobs in by_run.items()
        if any(j.status in ("pending", "streaming") for j in jobs)
    ]
    if not candidates:
        return None
    # Most recent first job wins if there are multiple in-progress runs
    # (rare — typically zero or one).
    candidates.sort(key=lambda pair: max(j.started_at for j in pair[1]))
    rid, jobs = candidates[-1]
    sample = jobs[0].request_data
    # Preserve the original column order so resumed columns match what
    # the user saw before navigating away.
    order: list[str] = list(sample.get("all_model_keys") or [])
    jobs.sort(key=lambda j: order.index(j.request_data["model_key"])
              if j.request_data["model_key"] in order else 0)
    return {
        "run_id": rid,
        "prompt": str(sample.get("prompt") or ""),
        "system_prompt": str(sample.get("system_prompt") or ""),
        "jobs": [
            {
                "model_key": j.request_data["model_key"],
                "job_id": j.id,
                "status": j.status,
            }
            for j in jobs
        ],
    }


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
            "ollama_status": get_ollama_status(),
        },
    )


@app.get("/api/hardware")
async def hardware_api() -> JSONResponse:
    hw = detect_hardware()
    rec = recommend_models(hw)
    return JSONResponse(hw_to_dict(hw, rec, installed=_ollama_installed_models()))


# ---------------------------------------------------------------------------
# Ollama model management — Pull missing models from the /hardware page
# ---------------------------------------------------------------------------

def _start_pull_thread(job: Job, model: str, loop: asyncio.AbstractEventLoop) -> None:
    """Run an Ollama `/api/pull` stream in a background thread, pumping
    each NDJSON progress event into the JobManager so SSE subscribers
    see it. Reuses the standard /api/jobs/{id}/stream path the chat /
    review pages already use — no separate streaming route needed."""
    from app.services.ollama_models import pull_model

    settings = chat_service.settings
    base_url = settings.ollama_host or "http://127.0.0.1:11434"

    def _runner() -> None:
        gen = pull_model(model, base_url=base_url)
        try:
            terminal_status = "done"
            error_text: str | None = None
            try:
                for event in gen:
                    if job_manager.is_stop_requested(job.id):
                        terminal_status = "stopped"
                        break
                    if "error" in event:
                        terminal_status = "error"
                        error_text = str(event["error"])
                        job_manager.emit_event(job.id, "pull_event", event, loop)
                        break
                    # Pumping as a structural event (not a token) so the
                    # frontend can render a progress bar from the
                    # `total` / `completed` fields without parsing prose.
                    job_manager.emit_event(job.id, "pull_event", event, loop)
            finally:
                # Close the generator explicitly so its underlying
                # urlopen response closes immediately on stop / break,
                # rather than leaking until GC.
                gen.close()
            if terminal_status == "done":
                _invalidate_installed_models_cache()
            job_manager.finish(
                job.id,
                terminal_status,
                loop,
                error=error_text,
                metadata={"kind": "ollama_pull", "model": model},
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("ollama pull failed for %s", model)
            job_manager.finish(
                job.id,
                "error",
                loop,
                error=str(exc),
                metadata={"kind": "ollama_pull", "model": model},
            )

    threading.Thread(
        target=_runner, daemon=True, name=f"pull-{model[:24]}",
    ).start()


@app.post("/api/ollama/pull")
async def start_ollama_pull(request: Request) -> JSONResponse:
    """Kick off an Ollama model pull as a JobManager job. Returns the
    job_id so the client can subscribe to /api/jobs/{job_id}/stream
    and receive `pull_event` SSE messages with progress."""
    from app.services.ollama_models import validate_model_name

    body = await request.json()
    raw = body.get("model") or ""
    try:
        model = validate_model_name(raw)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    # Refuse duplicate pulls — if a pull for this exact model is
    # already running we just return the existing job_id so the UI
    # reattaches its progress stream instead of doubling traffic.
    existing = job_manager.list_with_chat_prefix(f"pull:{model}")
    for j in existing:
        if j.status in ("pending", "streaming"):
            return JSONResponse({"job_id": j.id, "model": model, "resumed": True})

    job = job_manager.create(
        chat_id=f"pull:{model}",
        request_data={"kind": "ollama_pull", "model": model},
    )
    loop = asyncio.get_running_loop()
    _start_pull_thread(job, model, loop)
    return JSONResponse({"job_id": job.id, "model": model, "resumed": False})
