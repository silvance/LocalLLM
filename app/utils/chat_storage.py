"""On-disk JSON persistence for chat sessions.

Sessions live at <base_dir>/<chat_id>.json. We store only the fields needed
to round-trip a conversation — no pickling, no executable code in the
artifact, so a tampered file can't run code on load.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app.schemas.chat import ChatMessage
from app.utils.storage_ids import safe_storage_path


logger = logging.getLogger("localllm")


@dataclass
class ChatSession:
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[ChatMessage] = field(default_factory=list)


@dataclass(frozen=True)
class ChatSummary:
    """Lightweight metadata loaded for the sidebar list — no full messages."""
    id: str
    title: str
    updated_at: str
    message_count: int


_TITLE_FALLBACK = "New chat"
_TITLE_MAX_LEN = 60


def _now() -> str:
    # Millisecond precision so two saves in the same second still sort
    # deterministically by recency.
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


def derive_title(messages: list[ChatMessage]) -> str:
    """Use the first user message's first line, truncated, as the title."""
    for msg in messages:
        if msg.role != "user":
            continue
        text = msg.content.strip().splitlines()[0] if msg.content.strip() else ""
        if not text:
            continue
        text = re.sub(r"\s+", " ", text)
        if len(text) > _TITLE_MAX_LEN:
            text = text[: _TITLE_MAX_LEN - 1].rstrip() + "…"
        return text
    return _TITLE_FALLBACK


def new_session() -> ChatSession:
    now = _now()
    return ChatSession(
        id=str(uuid.uuid4()),
        title=_TITLE_FALLBACK,
        created_at=now,
        updated_at=now,
        messages=[],
    )


def _to_json(session: ChatSession) -> dict:
    return {
        "id": session.id,
        "title": session.title,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "messages": [{"role": m.role, "content": m.content} for m in session.messages],
    }


def _from_json(data: dict) -> ChatSession:
    msgs = [
        ChatMessage(role=m["role"], content=m["content"])
        for m in data.get("messages", [])
        if isinstance(m, dict) and "role" in m and "content" in m
    ]
    return ChatSession(
        id=str(data["id"]),
        title=str(data.get("title") or _TITLE_FALLBACK),
        created_at=str(data.get("created_at") or _now()),
        updated_at=str(data.get("updated_at") or _now()),
        messages=msgs,
    )


class ChatStorage:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, chat_id: str) -> Path:
        # Validate chat_id against a UUID-like whitelist AND confirm
        # the resolved path stays under base_dir. The previous guard
        # only rejected slash/backslash/leading-dot, but `C:foo` slips
        # through on Windows because pathlib treats `C:` as a drive
        # anchor — `Path("data/chats") / "C:evil"` resolves to
        # `C:evil`, which a malicious cross-origin request to
        # `DELETE /chats/C:evil` could then unlink.
        return safe_storage_path(self.base_dir, chat_id, kind="chat")

    def list_summaries(self) -> list[ChatSummary]:
        out: list[ChatSummary] = []
        for path in self.base_dir.glob("*.json"):
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                logger.warning("Skipping unreadable chat file %s: %s", path.name, exc)
                continue
            out.append(ChatSummary(
                id=str(data.get("id") or path.stem),
                title=str(data.get("title") or _TITLE_FALLBACK),
                updated_at=str(data.get("updated_at") or ""),
                message_count=len(data.get("messages") or []),
            ))
        out.sort(key=lambda s: s.updated_at, reverse=True)
        return out

    def load(self, chat_id: str) -> ChatSession | None:
        path = self._path(chat_id)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                return _from_json(json.load(f))
        except Exception as exc:
            logger.warning("Failed to load chat %s: %s", chat_id, exc)
            return None

    def save(self, session: ChatSession) -> None:
        session.updated_at = _now()
        if session.title == _TITLE_FALLBACK:
            session.title = derive_title(session.messages)
        path = self._path(session.id)
        # Atomic write: write to a temp file then rename.
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(_to_json(session), f, ensure_ascii=False, indent=2)
        tmp.replace(path)

    def delete(self, chat_id: str) -> None:
        path = self._path(chat_id)
        path.unlink(missing_ok=True)
