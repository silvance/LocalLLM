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
    surface the most-recent run so the user can see its output. Agent
    has no on-disk persistence — this in-memory lookup is the only way
    to keep that work visible across navigation."""
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
