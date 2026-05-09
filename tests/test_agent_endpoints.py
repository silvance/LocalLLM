"""Endpoint tests for the /agent page (online-connected agent loop).

Skipped in the minimal CI test job because chromadb (transitive via
app.web.app) and the agent's own deps may be missing. Runs locally /
on the full-dep build.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import pytest

chromadb = pytest.importorskip("chromadb")
fastapi_testclient = pytest.importorskip("fastapi.testclient")
ddgs = pytest.importorskip("ddgs")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    monkeypatch.setenv("LOCALLLM_DATA_DIR", str(tmp_path))
    import importlib

    from app import config
    config.get_settings.cache_clear()

    from app.web import app as web_app
    importlib.reload(web_app)
    from app.agent import routes as agent_routes
    from app.utils.agent_storage import AgentStorage
    agent_routes.agent_storage = AgentStorage(tmp_path / "agents")

    with TestClient(web_app.app) as c:
        yield c


# ---------------------------------------------------------------------------
# Resume across navigation
# ---------------------------------------------------------------------------

def test_agent_page_shows_active_job_after_navigation(
    monkeypatch: pytest.MonkeyPatch, client: TestClient,
) -> None:
    """An in-flight agent run must be surfaced on /agent reload so the
    JS can re-attach to its SSE stream instead of looking like nothing's
    happening."""
    from app.web import app as web_app
    from app.agent import routes as agent_routes

    # Force the deps probe to report all packages installed so the route
    # doesn't reject the request before we can simulate a long-running job.
    monkeypatch.setattr(agent_routes, "_missing_agent_deps", lambda: [])

    # Stub the run_agent loop so it never finishes — emulates the user
    # navigating away mid-research.
    started = {"flag": False}

    def _block(*_args, **_kwargs):
        started["flag"] = True
        # Sleep "forever" — the daemon thread will be reaped when the
        # process exits anyway.
        time.sleep(60)
        return ""

    monkeypatch.setattr(agent_routes, "run_agent", _block)

    r = client.post("/api/agent", json={
        "prompt": "agent resume probe",
        "model": "granite",
    })
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    # Give the thread a beat to start so its job status flips out of "pending"
    # — though pending is also non-terminal, so it'd still match either way.
    time.sleep(0.05)
    assert started["flag"] is True

    page = client.get("/agent")
    assert page.status_code == 200
    body = page.text
    assert "agent resume probe" in body
    assert job_id in body
    assert "activeJob" in body


def test_agent_page_keeps_finished_job_visible(
    monkeypatch: pytest.MonkeyPatch, client: TestClient,
) -> None:
    """Once the agent loop finishes, returning to /agent should still
    surface the most-recent run so the user can see its output."""
    from app.web import app as web_app
    from app.agent import routes as agent_routes

    monkeypatch.setattr(agent_routes, "_missing_agent_deps", lambda: [])

    def _instant_finish(*_args, **_kwargs):
        return "final synthesised answer"

    monkeypatch.setattr(agent_routes, "run_agent", _instant_finish)

    r = client.post("/api/agent", json={
        "prompt": "finished-job probe",
        "model": "granite",
    })
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    # Wait for the daemon thread to land its finish() call.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if web_app.job_manager.get(job_id).status in ("done", "error"):
            break
        time.sleep(0.02)

    page = client.get("/agent")
    assert page.status_code == 200
    body = page.text
    # Even though the job is done, /agent should still surface it.
    assert job_id in body
    assert "finished-job probe" in body


def test_agent_run_persists_to_history(
    monkeypatch: pytest.MonkeyPatch, client: TestClient,
) -> None:
    from app.web import app as web_app
    from app.agent import routes as agent_routes

    monkeypatch.setattr(agent_routes, "_missing_agent_deps", lambda: [])

    def _finish_with_events(*_args, **kwargs):
        on_event = kwargs["on_event"]
        on_event("step_start", {"iteration": 0})
        on_event("model_text", {"content": "I should search first."})
        on_event("tool_call", {"name": "web_search", "args": {"query": "localllm"}})
        on_event("tool_result", {"name": "web_search", "result": {"results": []}})
        on_event("done", {"iterations": 1, "answer": "saved final answer"})
        return "saved final answer"

    monkeypatch.setattr(agent_routes, "run_agent", _finish_with_events)

    r = client.post("/api/agent", json={
        "prompt": "history persistence probe",
        "model": "granite",
    })
    assert r.status_code == 200
    payload = r.json()
    job_id = payload["job_id"]
    agent_id = payload["agent_id"]

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        job = web_app.job_manager.get(job_id)
        if job and job.status in ("done", "error"):
            break
        time.sleep(0.02)

    saved = agent_routes.agent_storage.load(agent_id)
    assert saved is not None
    assert saved.prompt == "history persistence probe"
    assert saved.status == "done"
    assert saved.answer == "saved final answer"
    assert [e.kind for e in saved.events] == [
        "user_prompt", "step_start", "model_text", "tool_call", "tool_result", "final",
    ]

    page = client.get(f"/agent/{agent_id}")
    assert page.status_code == 200
    assert "history persistence probe" in page.text
    assert "saved final answer" in page.text


def test_agent_history_can_be_deleted(
    monkeypatch: pytest.MonkeyPatch, client: TestClient,
) -> None:
    from app.web import app as web_app
    from app.agent import routes as agent_routes

    monkeypatch.setattr(agent_routes, "_missing_agent_deps", lambda: [])
    monkeypatch.setattr(agent_routes, "run_agent", lambda *_args, **_kwargs: "answer")

    r = client.post("/api/agent", json={
        "prompt": "delete history probe",
        "model": "granite",
    })
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    agent_id = r.json()["agent_id"]

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        job = web_app.job_manager.get(job_id)
        if job and job.status in ("done", "error"):
            break
        time.sleep(0.02)

    assert agent_routes.agent_storage.load(agent_id) is not None
    delete = client.delete(f"/agent/{agent_id}")
    assert delete.status_code == 200
    assert agent_routes.agent_storage.load(agent_id) is None


def test_agent_session_accepts_followup(
    monkeypatch: pytest.MonkeyPatch, client: TestClient,
) -> None:
    from app.web import app as web_app
    from app.agent import routes as agent_routes

    monkeypatch.setattr(agent_routes, "_missing_agent_deps", lambda: [])
    seen_prompts: list[str] = []

    def _finish(*args, **kwargs):
        user_prompt = kwargs.get("user_prompt") or (args[0] if args else "")
        seen_prompts.append(user_prompt)
        return f"answer {len(seen_prompts)}"

    monkeypatch.setattr(agent_routes, "run_agent", _finish)

    first = client.post("/api/agent", json={
        "prompt": "initial web research",
        "model": "granite",
    })
    assert first.status_code == 200
    first_job = first.json()["job_id"]
    agent_id = first.json()["agent_id"]

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        job = web_app.job_manager.get(first_job)
        if job and job.status in ("done", "error"):
            break
        time.sleep(0.02)

    second = client.post(f"/api/agent/{agent_id}/messages", json={
        "prompt": "follow up question",
        "model": "granite",
    })
    assert second.status_code == 200
    second_job = second.json()["job_id"]

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        job = web_app.job_manager.get(second_job)
        if job and job.status in ("done", "error"):
            break
        time.sleep(0.02)

    saved = agent_routes.agent_storage.load(agent_id)
    assert saved is not None
    prompts = [e.payload.get("prompt") for e in saved.events if e.kind == "user_prompt"]
    finals = [e.payload.get("answer") for e in saved.events if e.kind == "final"]
    assert prompts == ["initial web research", "follow up question"]
    assert finals == ["answer 1", "answer 2"]
    assert "Prior transcript" in seen_prompts[1]
    assert "initial web research" in seen_prompts[1]
    assert "follow up question" in seen_prompts[1]
