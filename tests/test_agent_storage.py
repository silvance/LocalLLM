import json
import tempfile
from pathlib import Path

import pytest

from app.utils.agent_storage import (
    AgentStorage,
    derive_title,
    new_session,
)


@pytest.fixture
def tmp_storage() -> AgentStorage:
    with tempfile.TemporaryDirectory() as td:
        yield AgentStorage(Path(td))


def test_new_session_carries_prompt_and_model() -> None:
    s = new_session("Research CVE patch state", "granite")
    assert s.id
    assert s.title == "Research CVE patch state"
    assert s.prompt == "Research CVE patch state"
    assert s.model == "granite"
    assert s.status == "pending"


def test_save_load_round_trip(tmp_storage: AgentStorage) -> None:
    s = new_session("Find upstream docs", "qwen")
    s.status = "done"
    s.answer = "final answer"
    s.job_id = "job-1"
    tmp_storage.save(s)
    tmp_storage.append_event(s.id, "tool_call", {"name": "web_search", "args": {"query": "x"}})

    loaded = tmp_storage.load(s.id)
    assert loaded is not None
    assert loaded.id == s.id
    assert loaded.answer == "final answer"
    assert loaded.events[0].kind == "tool_call"
    assert loaded.events[0].payload["args"]["query"] == "x"


def test_list_summaries_sorted_desc(tmp_storage: AgentStorage) -> None:
    older = {
        "id": "old", "title": "Old", "created_at": "2026-05-01T10:00:00.000+00:00",
        "updated_at": "2026-05-01T10:00:00.000+00:00",
        "prompt": "p", "model": "granite", "status": "done", "answer": "",
        "error": None, "job_id": None, "events": [],
    }
    newer = dict(older, id="new", title="New", updated_at="2026-05-03T10:00:00.000+00:00")
    (tmp_storage.base_dir / "old.json").write_text(json.dumps(older))
    (tmp_storage.base_dir / "new.json").write_text(json.dumps(newer))
    summaries = tmp_storage.list_summaries()
    assert [s.id for s in summaries] == ["new", "old"]


def test_path_traversal_blocked(tmp_storage: AgentStorage) -> None:
    with pytest.raises(ValueError):
        tmp_storage.load("../etc/passwd")


def test_delete_removes_file(tmp_storage: AgentStorage) -> None:
    s = new_session("delete me", "granite")
    tmp_storage.save(s)
    tmp_storage.delete(s.id)
    assert tmp_storage.load(s.id) is None


def test_derive_title_truncates() -> None:
    title = derive_title("x" * 200)
    assert len(title) <= 60
    assert title.endswith("…")
