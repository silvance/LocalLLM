"""On-disk JSON persistence for online-agent research sessions.

The online agent is event-oriented rather than plain chat-oriented: a
run contains the original prompt, the selected model, structured tool
events, and the final answer. Persisting those events keeps research
sessions visible after page reloads and process restarts without trying
to squeeze them into the normal chat transcript shape.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.utils.storage_ids import safe_storage_path


logger = logging.getLogger("localllm")


@dataclass
class AgentEvent:
    kind: str
    payload: dict[str, Any]
    timestamp: str


@dataclass
class AgentSession:
    id: str
    title: str
    created_at: str
    updated_at: str
    prompt: str
    model: str
    status: str = "pending"  # pending | running | done | error | stopped
    answer: str = ""
    error: str | None = None
    job_id: str | None = None
    events: list[AgentEvent] = field(default_factory=list)


@dataclass(frozen=True)
class AgentSummary:
    id: str
    title: str
    updated_at: str
    model: str
    status: str
    event_count: int


_TITLE_FALLBACK = "New agent run"
_TITLE_MAX_LEN = 60


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


def derive_title(prompt: str) -> str:
    text = prompt.strip().splitlines()[0] if prompt.strip() else ""
    if not text:
        return _TITLE_FALLBACK
    text = re.sub(r"\s+", " ", text)
    if len(text) > _TITLE_MAX_LEN:
        text = text[: _TITLE_MAX_LEN - 1].rstrip() + "…"
    return text


def new_session(prompt: str, model: str) -> AgentSession:
    now = _now()
    return AgentSession(
        id=str(uuid.uuid4()),
        title=derive_title(prompt),
        created_at=now,
        updated_at=now,
        prompt=prompt,
        model=model,
    )


class AgentStorage:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, agent_id: str) -> Path:
        return safe_storage_path(self.base_dir, agent_id, kind="agent")

    def list_summaries(self) -> list[AgentSummary]:
        out: list[AgentSummary] = []
        for path in self.base_dir.glob("*.json"):
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                logger.warning("Skipping unreadable agent file %s: %s", path.name, exc)
                continue
            out.append(AgentSummary(
                id=str(data.get("id") or path.stem),
                title=str(data.get("title") or _TITLE_FALLBACK),
                updated_at=str(data.get("updated_at") or ""),
                model=str(data.get("model") or ""),
                status=str(data.get("status") or "pending"),
                event_count=len(data.get("events") or []),
            ))
        out.sort(key=lambda s: s.updated_at, reverse=True)
        return out

    def load(self, agent_id: str) -> AgentSession | None:
        path = self._path(agent_id)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                return _from_dict(json.load(f))
        except Exception as exc:
            logger.warning("Failed to load agent session %s: %s", agent_id, exc)
            return None

    def save(self, session: AgentSession) -> None:
        session.updated_at = _now()
        path = self._path(session.id)
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(asdict(session), f, ensure_ascii=False, indent=2)
        tmp.replace(path)

    def append_event(self, agent_id: str, kind: str, payload: dict[str, Any]) -> AgentSession | None:
        session = self.load(agent_id)
        if session is None:
            return None
        safe_payload = json.loads(json.dumps(payload, ensure_ascii=False, default=str))
        session.status = "running"
        session.events.append(AgentEvent(kind=kind, payload=safe_payload, timestamp=_now()))
        self.save(session)
        return session

    def delete(self, agent_id: str) -> None:
        self._path(agent_id).unlink(missing_ok=True)


def _from_dict(data: dict) -> AgentSession:
    events = [
        AgentEvent(
            kind=str(e.get("kind") or ""),
            payload=e.get("payload") if isinstance(e.get("payload"), dict) else {},
            timestamp=str(e.get("timestamp") or _now()),
        )
        for e in (data.get("events") or [])
        if isinstance(e, dict)
    ]
    return AgentSession(
        id=str(data["id"]),
        title=str(data.get("title") or _TITLE_FALLBACK),
        created_at=str(data.get("created_at") or _now()),
        updated_at=str(data.get("updated_at") or _now()),
        prompt=str(data.get("prompt") or ""),
        model=str(data.get("model") or ""),
        status=str(data.get("status") or "pending"),
        answer=str(data.get("answer") or ""),
        error=(str(data.get("error")) if data.get("error") else None),
        job_id=(str(data.get("job_id")) if data.get("job_id") else None),
        events=events,
    )
