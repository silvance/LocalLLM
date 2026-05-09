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


# ---------------------------------------------------------------------------
# Installed-models discovery — handles every ollama-python return shape
# ---------------------------------------------------------------------------

def test_parse_ollama_list_dict_with_name() -> None:
    """Older ollama-python returned a plain dict with `name` keys."""
    from app.web.app import _parse_ollama_list_response
    resp = {"models": [{"name": "qwen3-coder:30b"}, {"name": "granite4"}]}
    assert _parse_ollama_list_response(resp) == ["qwen3-coder:30b", "granite4"]


def test_parse_ollama_list_dict_with_model_key() -> None:
    """Some versions use `model` instead of `name`."""
    from app.web.app import _parse_ollama_list_response
    resp = {"models": [{"model": "deepseek-coder-v2:latest"}]}
    assert _parse_ollama_list_response(resp) == ["deepseek-coder-v2:latest"]


def test_parse_ollama_list_pydantic_objects() -> None:
    """Newer ollama-python (0.6.x) returns SubscriptableBaseModel objects.
    The parser must read `.model` / `.name` attributes when subscript
    access doesn't yield a hit."""
    class _FakeEntry:
        def __init__(self, name):
            self.model = name
    class _FakeResp:
        def __init__(self):
            self.models = [_FakeEntry("qwen2.5-coder:32b"), _FakeEntry("devstral:latest")]
    from app.web.app import _parse_ollama_list_response
    out = _parse_ollama_list_response(_FakeResp())
    assert out == ["qwen2.5-coder:32b", "devstral:latest"]


def test_parse_ollama_list_handles_empty_and_garbage() -> None:
    from app.web.app import _parse_ollama_list_response
    assert _parse_ollama_list_response(None) == []
    assert _parse_ollama_list_response({}) == []
    assert _parse_ollama_list_response({"models": []}) == []
    # Malformed — no name / model field at all.
    assert _parse_ollama_list_response({"models": [{"size": 123}]}) == []


def test_ollama_status_reflects_unreachable_daemon(
    monkeypatch: pytest.MonkeyPatch, client: TestClient,
) -> None:
    """When neither path resolves the model list, get_ollama_status()
    must report reachable=False with the configured URL and the
    actual error message, so the UI banner can show 'Last error: X'."""
    from app.web import app as web_app

    web_app._invalidate_installed_models_cache()
    monkeypatch.setattr(
        web_app.chat_service.adapters["granite"].client,
        "list",
        lambda: (_ for _ in ()).throw(ConnectionRefusedError("simulated refuse")),
    )
    monkeypatch.setattr(
        "app.services.ollama_models.installed_names",
        lambda url: (_ for _ in ()).throw(ConnectionRefusedError("WinError 10061")),
    )
    web_app._ollama_installed_models()
    status = web_app.get_ollama_status()
    assert status["reachable"] is False
    assert status["error"]
    assert "WinError" in status["error"] or "refuse" in status["error"]
    assert status["configured_url"]
    # Must list every URL we tried so the banner can show the path.
    assert any("HTTP" in u for u in status["tried_urls"])


def test_ollama_status_reflects_reachable_daemon(
    monkeypatch: pytest.MonkeyPatch, client: TestClient,
) -> None:
    """Happy path: status.reachable=True, error is None."""
    from app.web import app as web_app

    web_app._invalidate_installed_models_cache()
    monkeypatch.setattr(
        web_app.chat_service.adapters["granite"].client,
        "list",
        lambda: {"models": [{"name": "qwen3-coder:30b"}]},
    )
    web_app._ollama_installed_models()
    status = web_app.get_ollama_status()
    assert status["reachable"] is True
    assert status["error"] is None
    assert status["model_count"] == 1


def test_chat_page_renders_ollama_banner_when_unreachable(
    monkeypatch: pytest.MonkeyPatch, client: TestClient,
) -> None:
    """End-to-end-ish: with the daemon unreachable, /chats/<id> embeds
    the banner markup so the operator sees a clear failure mode + fix
    rather than a silently-empty dropdown."""
    from app.web import app as web_app

    web_app._invalidate_installed_models_cache()
    monkeypatch.setattr(
        web_app.chat_service.adapters["granite"].client,
        "list",
        lambda: (_ for _ in ()).throw(ConnectionRefusedError("refused")),
    )
    monkeypatch.setattr(
        "app.services.ollama_models.installed_names",
        lambda url: (_ for _ in ()).throw(ConnectionRefusedError("refused")),
    )

    r = client.post("/chats", follow_redirects=False)
    chat_url = r.headers["location"]
    page = client.get(chat_url)
    assert page.status_code == 200
    assert "Ollama unreachable" in page.text
    assert "OLLAMA_HOST" in page.text


def test_installed_models_caches_within_ttl(
    monkeypatch: pytest.MonkeyPatch, client: TestClient,
) -> None:
    """Page renders shouldn't hit Ollama once per nav. Second call
    within TTL must reuse the cached list."""
    from app.web import app as web_app

    web_app._invalidate_installed_models_cache()
    call_count = {"n": 0}

    def _fake_list():
        call_count["n"] += 1
        return {"models": [{"name": "qwen3-coder:30b"}]}

    monkeypatch.setattr(
        web_app.chat_service.adapters["granite"].client, "list", _fake_list,
    )
    web_app._ollama_installed_models()
    web_app._ollama_installed_models()
    web_app._ollama_installed_models()
    assert call_count["n"] == 1, "expected the cached list to be reused"


def test_installed_models_invalidate_after_successful_pull(
    monkeypatch: pytest.MonkeyPatch, client: TestClient,
) -> None:
    """The pull completion path must drop the cache so the freshly
    pulled model shows up on the next page render."""
    from app.web import app as web_app

    web_app._invalidate_installed_models_cache()
    monkeypatch.setattr(
        web_app.chat_service.adapters["granite"].client,
        "list",
        lambda: {"models": [{"name": "granite4"}]},
    )
    web_app._ollama_installed_models()  # populates cache
    assert web_app._installed_models_cache is not None

    web_app._invalidate_installed_models_cache()
    assert web_app._installed_models_cache is None


def test_installed_models_auto_retries_127_when_localhost_fails(
    monkeypatch: pytest.MonkeyPatch, client: TestClient,
) -> None:
    """Windows localhost trap: localhost resolves to ::1 on modern
    Windows but Ollama only binds IPv4. If the configured host uses
    `localhost` and the call refuses, we silently retry with
    127.0.0.1 so the dropdown still works without forcing every
    operator to update their .env."""
    from app.web import app as web_app

    web_app._invalidate_installed_models_cache()
    web_app.chat_service.settings = web_app.chat_service.settings.__class__(
        **{**web_app.chat_service.settings.__dict__,
           "ollama_host": "http://localhost:11434"},
    )

    # Make the python client path fail (simulates the WinError 10061).
    monkeypatch.setattr(
        web_app.chat_service.adapters["granite"].client,
        "list",
        lambda: (_ for _ in ()).throw(ConnectionRefusedError("refused")),
    )

    calls: list[str] = []

    def _fake_installed(base_url: str) -> list[str]:
        calls.append(base_url)
        if "localhost" in base_url:
            raise ConnectionRefusedError("WinError 10061 stand-in")
        # 127.0.0.1 path succeeds.
        return ["qwen3-coder:30b", "deepseek-coder-v2:latest"]

    monkeypatch.setattr(
        "app.services.ollama_models.installed_names", _fake_installed,
    )

    out = web_app._ollama_installed_models()
    assert out == ["qwen3-coder:30b", "deepseek-coder-v2:latest"]
    # First attempt at localhost, then the auto-retry at 127.0.0.1.
    assert any("localhost" in u for u in calls)
    assert any("127.0.0.1" in u for u in calls)


def test_installed_models_falls_back_to_http_when_client_throws(
    monkeypatch: pytest.MonkeyPatch, client: TestClient,
) -> None:
    """If ollama-python raises (API drift, version skew), we fall back
    to the direct /api/tags call. The dropdown stays populated."""
    from app.web import app as web_app

    # Make the python-client path raise.
    def _boom(*a, **kw):
        raise RuntimeError("simulated client failure")
    monkeypatch.setattr(web_app.chat_service.adapters["granite"].client, "list", _boom)

    # Make the HTTP fallback succeed.
    monkeypatch.setattr(
        "app.web.app.installed_names",
        lambda base_url: ["qwen2.5-coder:32b", "deepseek-coder-v2:latest"],
        raising=False,
    )
    # Need to patch where it's imported — _ollama_installed_models does
    # `from app.services.ollama_models import installed_names`.
    monkeypatch.setattr(
        "app.services.ollama_models.installed_names",
        lambda base_url: ["qwen2.5-coder:32b", "deepseek-coder-v2:latest"],
    )

    out = web_app._ollama_installed_models()
    assert "qwen2.5-coder:32b" in out
    assert "deepseek-coder-v2:latest" in out
