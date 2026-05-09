"""Server-side generation job tracker for the FastAPI web UI.

Each chat turn becomes a Job. Tokens are buffered per-job; SSE clients
subscribe and receive a replay of the current buffer plus live tokens
as they arrive. Disconnect/reconnect (browser tab change, refresh) is
non-fatal: the producer keeps generating, the user picks up where they
left off when they come back.

Jobs are in-memory only. Process restart loses any in-flight generation,
but the *chat history* survives via ChatStorage (which is JSON on disk).
The producer thread runs a synchronous Ollama generator; it pushes
chunks back through this manager via run_coroutine_threadsafe so the
asyncio side stays clean.
"""
from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Job:
    id: str
    chat_id: str
    request_data: dict[str, Any]
    text: str = ""
    status: str = "pending"  # pending | streaming | done | error | stopped
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Non-token events along with the text offset at which they fired.
    # Lets a late SSE subscriber replay structural events (e.g. section_start
    # on the review page) interleaved with their text slices instead of
    # missing them entirely because they fired before subscribe().
    checkpoints: list[tuple[str, Any, int]] = field(default_factory=list)


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        self._stop_flags: dict[str, bool] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ create / read

    def create(self, chat_id: str, request_data: dict[str, Any]) -> Job:
        job = Job(id=str(uuid.uuid4()), chat_id=chat_id, request_data=request_data)
        with self._lock:
            self._jobs[job.id] = job
            self._subscribers[job.id] = []
            self._stop_flags[job.id] = False
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_for_chat(self, chat_id: str) -> list[Job]:
        with self._lock:
            return [j for j in self._jobs.values() if j.chat_id == chat_id]

    def jobs_active_for_chat(self, chat_id: str) -> list[Job]:
        return [j for j in self.list_for_chat(chat_id) if j.status in ("pending", "streaming")]

    def list_with_chat_prefix(self, prefix: str) -> list[Job]:
        """All jobs whose chat_id starts with ``prefix``, sorted by start
        time. Used to find in-progress compare runs (chat_id = "compare:<id>")
        and agent runs (chat_id = "agent-<ts>") on page reload, so the UI
        can resume them across navigation."""
        with self._lock:
            jobs = [j for j in self._jobs.values() if j.chat_id.startswith(prefix)]
        jobs.sort(key=lambda j: j.started_at)
        return jobs

    # ------------------------------------------------------------------ stop signal

    def request_stop(self, job_id: str) -> bool:
        with self._lock:
            if job_id not in self._jobs:
                return False
            self._stop_flags[job_id] = True
            return True

    def is_stop_requested(self, job_id: str) -> bool:
        with self._lock:
            return self._stop_flags.get(job_id, False)

    # ------------------------------------------------------------------ subscribe / publish

    def subscribe(self, job_id: str) -> tuple[Optional[Job], Optional[asyncio.Queue]]:
        """Returns (job, queue). The queue receives ('token'|'done', payload)
        tuples produced by the generator thread. Caller is responsible for
        unsubscribe()."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None, None
            q: asyncio.Queue = asyncio.Queue()
            self._subscribers[job_id].append(q)
            return job, q

    def subscribe_with_snapshot(
        self, job_id: str,
    ) -> tuple[Optional[Job], Optional[asyncio.Queue], Optional[dict]]:
        """Like ``subscribe`` but ALSO returns an atomic snapshot of
        ``(text, checkpoints, status)`` taken under the lock. Lets the
        SSE replay path emit a frozen view instead of reading mutable
        job fields while the producer thread continues to mutate them
        — without this, replay can yield duplicate chunks (replay
        sees N tokens, then live mode delivers an event the producer
        emitted between snapshot and queue.put). Returns
        (job, queue, snapshot)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None, None, None
            q: asyncio.Queue = asyncio.Queue()
            self._subscribers[job_id].append(q)
            snapshot = {
                "text": job.text,
                "checkpoints": list(job.checkpoints),
                "status": job.status,
                "error": job.error,
                "metadata": dict(job.metadata),
            }
            return job, q, snapshot

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        with self._lock:
            subs = self._subscribers.get(job_id)
            if subs and queue in subs:
                subs.remove(queue)

    def _broadcast(
        self,
        job_id: str,
        event: str,
        payload: Any,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        with self._lock:
            subs = list(self._subscribers.get(job_id, []))
        # Run each subscriber in its own try/except: a closed event
        # loop (server shutting down) raises RuntimeError synchronously
        # from run_coroutine_threadsafe, and one bad subscriber
        # shouldn't kill the broadcast for the others.
        for q in subs:
            try:
                asyncio.run_coroutine_threadsafe(q.put((event, payload)), loop)
            except RuntimeError:
                # Loop closed; drop this subscriber so we don't keep
                # trying the same dead loop on every subsequent event.
                self.unsubscribe(job_id, q)

    # ------------------------------------------------------------------ producer side

    def append_chunk(
        self,
        job_id: str,
        chunk: str,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Producer thread calls this with each new chunk."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if job.status == "pending":
                job.status = "streaming"
            job.text += chunk
        self._broadcast(job_id, "token", {"chunk": chunk}, loop)

    def emit_event(
        self,
        job_id: str,
        event: str,
        payload: Any,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Emit an arbitrary SSE event (e.g. section_start for the review
        page's writer/reviewer transitions). Doesn't touch job.text, but DOES
        record a checkpoint so late subscribers can replay it."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.checkpoints.append((event, payload, len(job.text)))
        self._broadcast(job_id, event, payload, loop)

    def finish(
        self,
        job_id: str,
        status: str,
        loop: asyncio.AbstractEventLoop,
        error: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            job.status = status
            job.error = error
            job.finished_at = time.time()
            if metadata:
                job.metadata.update(metadata)
            payload = {
                "status": status,
                "error": error,
                "metadata": dict(job.metadata),
                "text": job.text,
            }
        self._broadcast(job_id, "done", payload, loop)
        return job
