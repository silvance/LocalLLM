"""Endpoint tests for the model-pull route on /hardware. Skipped in
the minimal CI test job (chromadb / fastapi not installed)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import pytest

chromadb = pytest.importorskip("chromadb")
fastapi_testclient = pytest.importorskip("fastapi.testclient")

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


def test_pull_rejects_invalid_model_name(client: TestClient) -> None:
    """Command-injection / shell-meta-character names must 400, not
    reach the daemon."""
    r = client.post("/api/ollama/pull", json={"model": "evil; rm -rf /"})
    assert r.status_code == 400
    assert "invalid" in r.json()["detail"].lower()


def test_pull_rejects_empty_model(client: TestClient) -> None:
    r = client.post("/api/ollama/pull", json={"model": ""})
    assert r.status_code == 400


def test_pull_returns_job_id_and_kicks_off_thread(
    monkeypatch: pytest.MonkeyPatch, client: TestClient,
) -> None:
    """Happy path: valid name → /api/pull is called → JobManager has
    a streaming-or-done job with chat_id=`pull:<model>`."""
    from app.web import app as web_app

    captured = {"model": None}

    def _fake_thread(job, model, loop):
        captured["model"] = model
        web_app.job_manager.emit_event(
            job.id, "pull_event",
            {"status": "downloading", "total": 10, "completed": 5}, loop,
        )
        web_app.job_manager.finish(
            job.id, "done", loop,
            metadata={"kind": "ollama_pull", "model": model},
        )

    monkeypatch.setattr(web_app, "_start_pull_thread", _fake_thread)
    r = client.post("/api/ollama/pull", json={"model": "granite4:latest"})
    assert r.status_code == 200
    payload = r.json()
    assert payload["model"] == "granite4:latest"
    assert payload["job_id"]
    assert payload["resumed"] is False
    assert captured["model"] == "granite4:latest"


def test_pull_reattaches_when_already_running(
    monkeypatch: pytest.MonkeyPatch, client: TestClient,
) -> None:
    """Second POST for the same model should return the existing job_id
    instead of starting another download."""
    from app.web import app as web_app

    started = {"flag": 0}

    def _hang(job, model, loop):
        # Don't finish — stays in `pending` so the second request hits
        # the reattach path.
        started["flag"] += 1

    monkeypatch.setattr(web_app, "_start_pull_thread", _hang)
    r1 = client.post("/api/ollama/pull", json={"model": "granite4:latest"})
    r2 = client.post("/api/ollama/pull", json={"model": "granite4:latest"})
    assert r1.status_code == r2.status_code == 200
    assert r1.json()["job_id"] == r2.json()["job_id"]
    assert r2.json()["resumed"] is True
    assert started["flag"] == 1, "second pull should NOT have started a thread"


def test_pull_emits_pull_event_through_sse(
    monkeypatch: pytest.MonkeyPatch, client: TestClient,
) -> None:
    """End-to-end-ish: pull endpoint kicks off a thread, the same SSE
    machinery used by chat/review delivers progress events to the
    /hardware page client."""
    from app.web import app as web_app

    def _fake_thread(job, model, loop):
        web_app.job_manager.emit_event(
            job.id, "pull_event",
            {"status": "pulling manifest"}, loop,
        )
        web_app.job_manager.emit_event(
            job.id, "pull_event",
            {"status": "downloading", "total": 100, "completed": 100}, loop,
        )
        web_app.job_manager.finish(
            job.id, "done", loop,
            metadata={"kind": "ollama_pull", "model": model},
        )

    monkeypatch.setattr(web_app, "_start_pull_thread", _fake_thread)
    r = client.post("/api/ollama/pull", json={"model": "granite4"})
    job_id = r.json()["job_id"]
    # Wait for the thread to finish so the SSE replay sees the
    # checkpoints + final done event.
    time.sleep(0.05)

    with client.stream("GET", f"/api/jobs/{job_id}/stream") as resp:
        events = []
        for line in resp.iter_lines():
            if line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())
            if "event: done" in line or len(events) > 8:
                break
    assert "pull_event" in events
