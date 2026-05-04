import json
import tempfile
from pathlib import Path

import pytest

from app.utils.review_storage import (
    ReviewSection,
    ReviewSession,
    ReviewStorage,
    derive_title,
    new_session,
)


@pytest.fixture
def tmp_storage() -> ReviewStorage:
    with tempfile.TemporaryDirectory() as td:
        yield ReviewStorage(Path(td))


def test_new_session_carries_inputs() -> None:
    s = new_session("Build me a port scanner", "qwen", "gemma", 3)
    assert s.id and s.title == "Build me a port scanner"
    assert s.writer_model == "qwen" and s.reviewer_model == "gemma"
    assert s.rounds == 3


def test_save_load_round_trip(tmp_storage: ReviewStorage) -> None:
    s = new_session("Build a CTF console", "qwen", "gemma", 3)
    s.sections = [
        ReviewSection(role="writer", model="qwen", text="def foo(): pass"),
        ReviewSection(role="reviewer", model="gemma", text="No type hints"),
        ReviewSection(role="writer", model="qwen", text="def foo() -> None: pass"),
    ]
    s.status = "done"
    tmp_storage.save(s)
    loaded = tmp_storage.load(s.id)
    assert loaded is not None
    assert loaded.id == s.id
    assert loaded.status == "done"
    assert len(loaded.sections) == 3
    assert loaded.sections[1].role == "reviewer"


def test_summaries_sorted_desc(tmp_storage: ReviewStorage) -> None:
    older = {
        "id": "old", "title": "Old", "created_at": "2026-05-01T10:00:00.000+00:00",
        "updated_at": "2026-05-01T10:00:00.000+00:00",
        "prompt": "p", "writer_model": "qwen", "reviewer_model": "gemma",
        "rounds": 3, "status": "done", "sections": [],
    }
    newer = dict(older, id="new", title="New",
                 updated_at="2026-05-03T10:00:00.000+00:00")
    (tmp_storage.base_dir / "old.json").write_text(json.dumps(older))
    (tmp_storage.base_dir / "new.json").write_text(json.dumps(newer))
    summaries = tmp_storage.list_summaries()
    assert [s.id for s in summaries] == ["new", "old"]


def test_path_traversal_blocked(tmp_storage: ReviewStorage) -> None:
    with pytest.raises(ValueError):
        tmp_storage.load("../etc/passwd")


def test_delete_removes_file(tmp_storage: ReviewStorage) -> None:
    s = new_session("p", "qwen", "gemma", 3)
    tmp_storage.save(s)
    tmp_storage.delete(s.id)
    assert tmp_storage.load(s.id) is None


def test_derive_title_truncates() -> None:
    title = derive_title("x" * 200)
    assert len(title) <= 60
    assert title.endswith("…")
