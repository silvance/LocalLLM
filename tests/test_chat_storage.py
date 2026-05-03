import tempfile
from pathlib import Path

import pytest

from app.schemas.chat import ChatMessage
from app.utils.chat_storage import (
    ChatSession,
    ChatStorage,
    derive_title,
    new_session,
)


@pytest.fixture
def tmp_storage() -> ChatStorage:
    with tempfile.TemporaryDirectory() as td:
        yield ChatStorage(Path(td))


def test_new_session_has_uuid_and_timestamps() -> None:
    s = new_session()
    assert s.id
    assert s.created_at
    assert s.updated_at == s.created_at
    assert s.messages == []


def test_save_and_load_round_trip(tmp_storage: ChatStorage) -> None:
    s = new_session()
    s.messages = [
        ChatMessage(role="user", content="What is Kerberos?"),
        ChatMessage(role="assistant", content="An auth protocol."),
    ]
    tmp_storage.save(s)
    loaded = tmp_storage.load(s.id)
    assert loaded is not None
    assert loaded.id == s.id
    assert [m.content for m in loaded.messages] == ["What is Kerberos?", "An auth protocol."]


def test_save_derives_title_from_first_user_message(tmp_storage: ChatStorage) -> None:
    s = new_session()
    s.messages = [ChatMessage(role="user", content="How do I use scapy to send a SYN packet?")]
    tmp_storage.save(s)
    loaded = tmp_storage.load(s.id)
    assert loaded.title.startswith("How do I use scapy")


def test_summaries_sorted_by_updated_desc(tmp_storage: ChatStorage) -> None:
    """Sort logic: list_summaries reads updated_at directly off disk."""
    import json

    older = {
        "id": "older-id",
        "title": "Older",
        "created_at": "2026-05-01T10:00:00.000+00:00",
        "updated_at": "2026-05-01T10:00:00.000+00:00",
        "messages": [],
    }
    newer = {
        "id": "newer-id",
        "title": "Newer",
        "created_at": "2026-05-03T10:00:00.000+00:00",
        "updated_at": "2026-05-03T10:00:00.000+00:00",
        "messages": [],
    }
    (tmp_storage.base_dir / "older-id.json").write_text(
        json.dumps(older), encoding="utf-8"
    )
    (tmp_storage.base_dir / "newer-id.json").write_text(
        json.dumps(newer), encoding="utf-8"
    )
    summaries = tmp_storage.list_summaries()
    assert [s.id for s in summaries] == ["newer-id", "older-id"]


def test_delete_removes_the_file(tmp_storage: ChatStorage) -> None:
    s = new_session()
    tmp_storage.save(s)
    assert tmp_storage.load(s.id) is not None
    tmp_storage.delete(s.id)
    assert tmp_storage.load(s.id) is None


def test_path_traversal_blocked(tmp_storage: ChatStorage) -> None:
    with pytest.raises(ValueError):
        tmp_storage.load("../etc/passwd")
    with pytest.raises(ValueError):
        tmp_storage.load(".hidden")


def test_derive_title_skips_empty_and_uses_first_user_msg() -> None:
    msgs = [
        ChatMessage(role="system", content="You are a tool."),
        ChatMessage(role="user", content="   "),
        ChatMessage(role="user", content="Real question here"),
    ]
    assert derive_title(msgs) == "Real question here"


def test_derive_title_truncates_long_input() -> None:
    msgs = [ChatMessage(role="user", content="x" * 200)]
    title = derive_title(msgs)
    assert len(title) <= 60
    assert title.endswith("…")


def test_load_handles_corrupt_file(tmp_storage: ChatStorage) -> None:
    bad_path = tmp_storage.base_dir / "abc.json"
    bad_path.write_text("not valid json {{{", encoding="utf-8")
    # Listing tolerates the corrupt file by skipping it
    summaries = tmp_storage.list_summaries()
    assert all(s.id != "abc" for s in summaries)
    # Direct load returns None
    assert tmp_storage.load("abc") is None
