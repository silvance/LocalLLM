"""On-disk JSON persistence for review-page runs (writer ↔ reviewer loops).

Same shape as chat_storage / comparison_storage: one JSON file per run,
atomic temp-rename writes, path-traversal guard, and never any pickle so
a tampered file can't execute code on load.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path


logger = logging.getLogger("localllm")


@dataclass
class ReviewSection:
    role: str  # "writer" | "reviewer"
    model: str
    text: str


@dataclass
class ReviewSession:
    id: str
    title: str
    created_at: str
    updated_at: str
    prompt: str
    writer_model: str
    reviewer_model: str
    rounds: int
    status: str = "pending"  # pending | done | stopped | error
    sections: list[ReviewSection] = field(default_factory=list)
    # Comma-separated chain of fallback writer model names that the
    # orchestrator will rotate to if the primary writer keeps failing.
    # Persisted so reloading a review session shows the operator the
    # actual config that was started, not "(none)" — UI used to silently
    # reset this dropdown on every page load.
    fallback_writer_model: str | None = None


@dataclass(frozen=True)
class ReviewSummary:
    id: str
    title: str
    updated_at: str
    rounds: int
    status: str


_TITLE_FALLBACK = "New review"
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


def new_session(
    prompt: str,
    writer_model: str,
    reviewer_model: str,
    rounds: int,
    fallback_writer_model: str | None = None,
) -> ReviewSession:
    now = _now()
    return ReviewSession(
        id=str(uuid.uuid4()),
        title=derive_title(prompt),
        created_at=now,
        updated_at=now,
        prompt=prompt,
        writer_model=writer_model,
        reviewer_model=reviewer_model,
        rounds=rounds,
        fallback_writer_model=fallback_writer_model or None,
    )


class ReviewStorage:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, review_id: str) -> Path:
        # Whitelist + resolve-and-contained check. See chat_storage._path
        # for the path-traversal rationale (Windows drive-letter trap).
        from app.utils.storage_ids import safe_storage_path
        return safe_storage_path(self.base_dir, review_id, kind="review")

    def list_summaries(self) -> list[ReviewSummary]:
        out: list[ReviewSummary] = []
        for path in self.base_dir.glob("*.json"):
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                logger.warning("Skipping unreadable review file %s: %s", path.name, exc)
                continue
            out.append(ReviewSummary(
                id=str(data.get("id") or path.stem),
                title=str(data.get("title") or _TITLE_FALLBACK),
                updated_at=str(data.get("updated_at") or ""),
                rounds=int(data.get("rounds") or 0),
                status=str(data.get("status") or "pending"),
            ))
        out.sort(key=lambda s: s.updated_at, reverse=True)
        return out

    def load(self, review_id: str) -> ReviewSession | None:
        path = self._path(review_id)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning("Failed to load review %s: %s", review_id, exc)
            return None
        sections = [
            ReviewSection(role=str(s["role"]), model=str(s["model"]), text=str(s.get("text", "")))
            for s in (data.get("sections") or [])
            if isinstance(s, dict) and "role" in s and "model" in s
        ]
        # ``fallback_writer_model`` was added later — pre-existing
        # session files won't have it. Fall through to None and
        # the UI degrades to "(none)" (which matches the prior
        # behavior for those old sessions).
        return ReviewSession(
            id=str(data["id"]),
            title=str(data.get("title") or _TITLE_FALLBACK),
            created_at=str(data.get("created_at") or _now()),
            updated_at=str(data.get("updated_at") or _now()),
            prompt=str(data.get("prompt") or ""),
            writer_model=str(data.get("writer_model") or ""),
            reviewer_model=str(data.get("reviewer_model") or ""),
            rounds=int(data.get("rounds") or 0),
            status=str(data.get("status") or "pending"),
            sections=sections,
            fallback_writer_model=str(data.get("fallback_writer_model") or "") or None,
        )

    def save(self, session: ReviewSession) -> None:
        session.updated_at = _now()
        path = self._path(session.id)
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(asdict(session), f, ensure_ascii=False, indent=2)
        tmp.replace(path)

    def delete(self, review_id: str) -> None:
        self._path(review_id).unlink(missing_ok=True)
