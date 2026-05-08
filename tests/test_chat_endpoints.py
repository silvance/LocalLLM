"""Endpoint tests for regenerate/edit on the chat page.

Skipped in the minimal CI test job (chromadb not installed there). Run
locally / on the full-dep build to exercise the FastAPI surface.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

import pytest

# Skip the whole module if chromadb (transitively required by app.web.app)
# isn't installed — the minimal CI test job pins to pure-Python deps.
chromadb = pytest.importorskip("chromadb")
fastapi_testclient = pytest.importorskip("fastapi.testclient")

from fastapi.testclient import TestClient  # noqa: E402

from app import web  # noqa: E402,F401  (kept for side-effect order)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    """A TestClient with chat storage rooted under tmp_path and the
    background generator stubbed so we don't try to reach Ollama."""
    monkeypatch.setenv("LOCALLLM_DATA_DIR", str(tmp_path))
    import importlib

    from app import config
    config.get_settings.cache_clear()

    from app.web import app as web_app
    importlib.reload(web_app)
    # Re-point the storage at tmp_path. The module-level singleton uses
    # `Path("data/chats")` (relative cwd), which would write into the test
    # runner's working directory — not what we want.
    from app.utils.chat_storage import ChatStorage
    web_app.chat_storage = ChatStorage(tmp_path / "chats")

    def _fake_thread(job, request_obj, selection, use_rag, loop):
        web_app.job_manager.append_chunk(job.id, "stubbed-response", loop)
        web_app.job_manager.finish(
            job.id, "done", loop,
            metadata={"selected_model": selection},
        )
        web_app._persist_assistant_message(job)

    monkeypatch.setattr(web_app, "_start_generation_thread", _fake_thread)

    with TestClient(web_app.app) as c:
        yield c


def _seed_chat(client: TestClient) -> str:
    """Create a chat with two user/assistant turns and return its id."""
    r = client.post("/chats", follow_redirects=False)
    chat_url = r.headers["location"]
    chat_id = chat_url.rsplit("/", 1)[-1]

    for content in ("first prompt", "second prompt"):
        r = client.post(
            f"/api/chats/{chat_id}/messages",
            json={"content": content, "model_selection": "auto"},
        )
        assert r.status_code == 200
        # Wait for the stubbed generation to persist.
        time.sleep(0.05)
    return chat_id


def _load_messages(tmp_path: Path, chat_id: str) -> list[dict]:
    """Read the on-disk chat file (under the test fixture's chats/ root)."""
    path = tmp_path / "chats" / f"{chat_id}.json"
    data = json.loads(path.read_text("utf-8"))
    return data["messages"]


# ---------------------------------------------------------------------------
# regenerate
# ---------------------------------------------------------------------------

def test_regenerate_drops_assistant_and_reruns(
    client: TestClient, tmp_path: Path
) -> None:
    chat_id = _seed_chat(client)
    msgs_before = _load_messages(tmp_path, chat_id)
    assert [m["role"] for m in msgs_before] == ["user", "assistant", "user", "assistant"]

    r = client.post(f"/api/chats/{chat_id}/messages/3/regenerate", json={})
    assert r.status_code == 200
    assert "job_id" in r.json()
    time.sleep(0.05)

    msgs_after = _load_messages(tmp_path, chat_id)
    assert [m["role"] for m in msgs_after] == ["user", "assistant", "user", "assistant"]
    # The first three messages survived; the fourth was regenerated.
    assert msgs_after[:3] == msgs_before[:3]


def test_regenerate_rejects_index_with_no_preceding_user(client: TestClient) -> None:
    chat_id = _seed_chat(client)
    # Index 0 has no preceding message at all — invalid.
    r = client.post(f"/api/chats/{chat_id}/messages/0/regenerate", json={})
    assert r.status_code == 400


def test_regenerate_rejects_out_of_range_index(client: TestClient) -> None:
    chat_id = _seed_chat(client)
    r = client.post(f"/api/chats/{chat_id}/messages/99/regenerate", json={})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# edit
# ---------------------------------------------------------------------------

def test_edit_replaces_user_msg_and_truncates_after(
    client: TestClient, tmp_path: Path
) -> None:
    chat_id = _seed_chat(client)
    # Edit message at index 2 ("second prompt"), drop everything after.
    r = client.post(
        f"/api/chats/{chat_id}/messages/2/edit",
        json={"content": "edited prompt"},
    )
    assert r.status_code == 200
    time.sleep(0.05)

    msgs = _load_messages(tmp_path, chat_id)
    # Original [0,1] survived, [2] replaced, [3] dropped, then a new
    # assistant turn from the stubbed generator.
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[2]["content"] == "edited prompt"


def test_edit_rejects_assistant_message(client: TestClient) -> None:
    chat_id = _seed_chat(client)
    r = client.post(
        f"/api/chats/{chat_id}/messages/1/edit",
        json={"content": "hijack"},
    )
    assert r.status_code == 400


def test_edit_rejects_empty_content(client: TestClient) -> None:
    chat_id = _seed_chat(client)
    r = client.post(
        f"/api/chats/{chat_id}/messages/2/edit",
        json={"content": "   "},
    )
    assert r.status_code == 400
