"""Endpoint tests for the /compare page (FastAPI rewrite of the legacy
Streamlit `pages/01_compare_models.py`).

Skipped in the minimal CI test job because chromadb is required to import
app.web.app. Runs locally / on the full-dep build.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import pytest

chromadb = pytest.importorskip("chromadb")
fastapi_testclient = pytest.importorskip("fastapi.testclient")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    """A TestClient with comparison storage rooted under tmp_path and the
    per-model thread stubbed so we don't try to reach Ollama."""
    monkeypatch.setenv("LOCALLLM_DATA_DIR", str(tmp_path))
    import importlib

    from app import config
    config.get_settings.cache_clear()

    from app.web import app as web_app
    importlib.reload(web_app)
    # Storage is module-level — re-point at tmp_path AFTER reload so the
    # data root from the reloaded module gets overwritten.
    from app.utils.comparison_storage import ComparisonStorage
    web_app.comparison_storage = ComparisonStorage(tmp_path / "comparisons")

    def _fake_thread(job, request_obj, model_key, loop):
        web_app.job_manager.append_chunk(job.id, f"out-from-{model_key}", loop)
        web_app.job_manager.finish(
            job.id, "done", loop,
            metadata={
                "model_key": model_key,
                "model_name": f"{model_key}-fake",
                "prompt_tokens": 10,
                "completion_tokens": 7,
                "elapsed_s": 0.5,
                "tokens_per_sec": 14.0,
            },
        )
    monkeypatch.setattr(web_app, "_start_compare_thread", _fake_thread)

    with TestClient(web_app.app) as c:
        yield c


# ---------------------------------------------------------------------------
# Page rendering
# ---------------------------------------------------------------------------

def test_compare_page_renders(client: TestClient) -> None:
    r = client.get("/compare")
    assert r.status_code == 200
    assert "Compare" in r.text or "compare" in r.text


# ---------------------------------------------------------------------------
# /api/compare — start
# ---------------------------------------------------------------------------

def test_start_compare_creates_one_job_per_model(client: TestClient) -> None:
    r = client.post("/api/compare", json={
        "prompt": "test prompt",
        "models": ["granite", "gemma"],
    })
    assert r.status_code == 200
    data = r.json()
    assert "run_id" in data
    assert len(data["jobs"]) == 2
    keys = {j["model_key"] for j in data["jobs"]}
    assert keys == {"granite", "gemma"}
    for j in data["jobs"]:
        assert j["job_id"]


def test_start_compare_rejects_empty_prompt(client: TestClient) -> None:
    r = client.post("/api/compare", json={"prompt": "  ", "models": ["granite"]})
    assert r.status_code == 400


def test_start_compare_rejects_no_models(client: TestClient) -> None:
    r = client.post("/api/compare", json={"prompt": "hi", "models": []})
    assert r.status_code == 400


def test_start_compare_rejects_unknown_model(client: TestClient) -> None:
    r = client.post("/api/compare", json={
        "prompt": "hi", "models": ["granite", "no-such-model"],
    })
    assert r.status_code == 400
    assert "no-such-model" in r.json()["detail"]


# ---------------------------------------------------------------------------
# /api/compare/save — persist
# ---------------------------------------------------------------------------

def test_save_persists_run_and_lists_in_sidebar(client: TestClient, tmp_path: Path) -> None:
    payload = {
        "run_id": "11111111-2222-3333-4444-555555555555",
        "prompt": "what is 2+2",
        "system_prompt": "",
        "outputs": [
            {
                "model_key": "granite", "model_name": "granite4",
                "text": "4", "prompt_tokens": 5, "completion_tokens": 1,
                "elapsed_s": 0.2, "tokens_per_sec": 5.0,
            },
            {
                "model_key": "gemma", "model_name": "gemma4",
                "text": "four", "prompt_tokens": 5, "completion_tokens": 1,
                "elapsed_s": 0.3, "tokens_per_sec": 3.3,
            },
        ],
        "winner": "granite",
    }
    r = client.post("/api/compare/save", json=payload)
    assert r.status_code == 200
    assert r.json()["id"] == payload["run_id"]

    # The saved run should appear on the index page sidebar.
    r2 = client.get("/compare")
    assert "what is 2+2" in r2.text


def test_save_drops_invalid_winner(client: TestClient) -> None:
    payload = {
        "run_id": "22222222-2222-3333-4444-555555555555",
        "prompt": "x",
        "system_prompt": "",
        "outputs": [
            {"model_key": "granite", "model_name": "g", "text": "ok"},
        ],
        "winner": "ghost-model",  # not in outputs
    }
    r = client.post("/api/compare/save", json=payload)
    assert r.status_code == 200

    detail = client.get(f"/compare/{payload['run_id']}")
    # Page should render; the body shouldn't claim ghost-model won.
    assert detail.status_code == 200
    assert "🏆" not in detail.text or "ghost-model" not in detail.text


def test_save_rejects_missing_fields(client: TestClient) -> None:
    r = client.post("/api/compare/save", json={"prompt": "x"})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# /compare/{run_id} — load saved run
# ---------------------------------------------------------------------------

def test_load_saved_run_renders_outputs(client: TestClient) -> None:
    payload = {
        "run_id": "33333333-2222-3333-4444-555555555555",
        "prompt": "explain entropy briefly",
        "system_prompt": "",
        "outputs": [
            {"model_key": "granite", "model_name": "granite4",
             "text": "Entropy is a measure of disorder.",
             "prompt_tokens": 3, "completion_tokens": 6, "elapsed_s": 0.4,
             "tokens_per_sec": 15.0},
        ],
        "winner": None,
    }
    client.post("/api/compare/save", json=payload)
    r = client.get(f"/compare/{payload['run_id']}")
    assert r.status_code == 200
    assert "explain entropy briefly" in r.text
    assert "Entropy is a measure" in r.text


def test_load_unknown_run_returns_404(client: TestClient) -> None:
    r = client.get("/compare/this-does-not-exist")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /compare/{run_id}
# ---------------------------------------------------------------------------

def test_delete_removes_run(client: TestClient) -> None:
    payload = {
        "run_id": "44444444-2222-3333-4444-555555555555",
        "prompt": "delete me",
        "system_prompt": "",
        "outputs": [{"model_key": "granite", "model_name": "g", "text": "x"}],
        "winner": None,
    }
    client.post("/api/compare/save", json=payload)
    assert client.get(f"/compare/{payload['run_id']}").status_code == 200

    r = client.delete(f"/compare/{payload['run_id']}")
    assert r.status_code == 200
    assert r.json() == {"ok": True}

    assert client.get(f"/compare/{payload['run_id']}").status_code == 404
