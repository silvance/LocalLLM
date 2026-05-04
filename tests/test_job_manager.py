"""Tests for the JobManager. We exercise the public surface synchronously
where we can and use asyncio for the subscriber path."""
from __future__ import annotations

import asyncio

import pytest

from app.web.job_manager import JobManager


def test_create_returns_job_with_uuid() -> None:
    mgr = JobManager()
    job = mgr.create("chat-1", {"selection": "auto"})
    assert job.id
    assert job.chat_id == "chat-1"
    assert job.status == "pending"
    assert mgr.get(job.id) is job


def test_request_stop_sets_flag() -> None:
    mgr = JobManager()
    job = mgr.create("chat-1", {})
    assert mgr.is_stop_requested(job.id) is False
    assert mgr.request_stop(job.id) is True
    assert mgr.is_stop_requested(job.id) is True


def test_request_stop_unknown_returns_false() -> None:
    mgr = JobManager()
    assert mgr.request_stop("nonexistent") is False


def test_jobs_active_for_chat_filters_by_status() -> None:
    mgr = JobManager()
    mgr.create("chat-A", {})
    j2 = mgr.create("chat-A", {})
    mgr.create("chat-B", {})

    # Manually finish j2
    loop = asyncio.new_event_loop()
    try:
        mgr.finish(j2.id, "done", loop)
    finally:
        loop.close()

    active = mgr.jobs_active_for_chat("chat-A")
    assert len(active) == 1
    assert all(j.status in ("pending", "streaming") for j in active)


def test_subscribe_unknown_job_returns_none_pair() -> None:
    mgr = JobManager()
    job, q = mgr.subscribe("nope")
    assert job is None and q is None


def test_subscriber_receives_appended_chunks_and_done() -> None:
    """Producer (background thread emulation) appends chunks; subscriber
    receives them as ('token', payload) followed by ('done', payload)."""

    async def scenario() -> None:
        mgr = JobManager()
        job = mgr.create("c", {})
        loop = asyncio.get_running_loop()

        job_obj, queue = mgr.subscribe(job.id)
        assert job_obj is not None and queue is not None

        # Producer side: put two chunks then finish
        await asyncio.to_thread(mgr.append_chunk, job.id, "hello", loop)
        await asyncio.to_thread(mgr.append_chunk, job.id, " world", loop)
        await asyncio.to_thread(mgr.finish, job.id, "done", loop, None, {"k": 1})

        events: list[tuple[str, dict]] = []
        for _ in range(3):
            events.append(await asyncio.wait_for(queue.get(), timeout=1.0))

        assert events[0][0] == "token"
        assert events[0][1]["chunk"] == "hello"
        assert events[1][0] == "token"
        assert events[1][1]["chunk"] == " world"
        assert events[2][0] == "done"
        assert events[2][1]["status"] == "done"
        assert events[2][1]["text"] == "hello world"
        assert events[2][1]["metadata"] == {"k": 1}

        mgr.unsubscribe(job.id, queue)

    asyncio.run(scenario())


def test_text_buffer_persists_across_subscriptions() -> None:
    """A late subscriber should be able to see the current job buffer (via
    job.text) even if they joined after some chunks were emitted."""

    async def scenario() -> None:
        mgr = JobManager()
        job = mgr.create("c", {})
        loop = asyncio.get_running_loop()

        await asyncio.to_thread(mgr.append_chunk, job.id, "first ", loop)
        await asyncio.to_thread(mgr.append_chunk, job.id, "second", loop)

        late_job, late_queue = mgr.subscribe(job.id)
        assert late_job.text == "first second"

        await asyncio.to_thread(mgr.finish, job.id, "done", loop)
        evt = await asyncio.wait_for(late_queue.get(), timeout=1.0)
        assert evt[0] == "done"

    asyncio.run(scenario())


def test_emit_event_records_checkpoint() -> None:
    """Late subscribers (review page) need past section_start events to be
    replayable. emit_event captures (event, payload, text_offset) on the Job
    so the SSE handler can interleave them with text slices on replay."""
    async def scenario() -> None:
        mgr = JobManager()
        job = mgr.create("c", {})
        loop = asyncio.get_running_loop()

        await asyncio.to_thread(mgr.emit_event, job.id, "section_start", {"index": 0, "role": "writer"}, loop)
        await asyncio.to_thread(mgr.append_chunk, job.id, "writer output", loop)
        await asyncio.to_thread(mgr.emit_event, job.id, "section_start", {"index": 1, "role": "reviewer"}, loop)
        await asyncio.to_thread(mgr.append_chunk, job.id, "reviewer output", loop)

        # Late subscriber after both sections fired
        late_job, late_queue = mgr.subscribe(job.id)
        assert late_job is not None
        assert len(late_job.checkpoints) == 2

        # Checkpoints in order, with correct text offsets
        evt0, payload0, pos0 = late_job.checkpoints[0]
        evt1, payload1, pos1 = late_job.checkpoints[1]
        assert evt0 == "section_start" and payload0["role"] == "writer" and pos0 == 0
        assert evt1 == "section_start" and payload1["role"] == "reviewer"
        assert pos1 == len("writer output")
        assert late_job.text == "writer outputreviewer output"

        mgr.unsubscribe(job.id, late_queue)

    asyncio.run(scenario())


def test_unsubscribe_removes_only_target_queue() -> None:
    async def scenario() -> None:
        mgr = JobManager()
        job = mgr.create("c", {})
        loop = asyncio.get_running_loop()

        _, q1 = mgr.subscribe(job.id)
        _, q2 = mgr.subscribe(job.id)

        mgr.unsubscribe(job.id, q1)

        await asyncio.to_thread(mgr.append_chunk, job.id, "x", loop)
        # q2 still subscribed
        evt = await asyncio.wait_for(q2.get(), timeout=1.0)
        assert evt[0] == "token"
        # q1 should NOT receive (no items pushed; queue is empty)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(q1.get(), timeout=0.1)

    asyncio.run(scenario())
