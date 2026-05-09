"""On-disk JSON persistence for Compare-page runs.

Each run captures the prompt, system prompt, per-model outputs + metrics,
and the user's winner pick. Stored as JSON (no pickle / no executable code)
under <base_dir>/<id>.json.
"""
from __future__ import annotations

import datetime as _dt
import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ModelOutput:
    model_key: str
    model_name: str
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    elapsed_s: float | None = None
    tokens_per_sec: float | None = None
    error: str | None = None


@dataclass
class ComparisonRun:
    id: str
    timestamp: str
    prompt: str
    system_prompt: str
    outputs: list[ModelOutput] = field(default_factory=list)
    winner: str | None = None  # model_key, if user voted


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


def new_run(prompt: str, system_prompt: str = "") -> ComparisonRun:
    return ComparisonRun(
        id=str(uuid.uuid4()),
        timestamp=_now(),
        prompt=prompt,
        system_prompt=system_prompt,
    )


class ComparisonStorage:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        # Whitelist + resolve-and-contained check. See chat_storage._path
        # for the path-traversal rationale (Windows drive-letter trap).
        from app.utils.storage_ids import safe_storage_path
        return safe_storage_path(self.base_dir, run_id, kind="comparison")

    def save(self, run: ComparisonRun) -> None:
        path = self._path(run.id)
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(asdict(run), f, ensure_ascii=False, indent=2)
        tmp.replace(path)

    def load(self, run_id: str) -> ComparisonRun | None:
        path = self._path(run_id)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None
        return _from_dict(data)

    def list_runs(self) -> list[ComparisonRun]:
        out: list[ComparisonRun] = []
        for p in self.base_dir.glob("*.json"):
            try:
                with p.open("r", encoding="utf-8") as f:
                    out.append(_from_dict(json.load(f)))
            except Exception:
                continue
        out.sort(key=lambda r: r.timestamp, reverse=True)
        return out

    def delete(self, run_id: str) -> None:
        self._path(run_id).unlink(missing_ok=True)

    def winner_tally(self) -> dict[str, int]:
        """Aggregate winner votes across all saved runs."""
        tally: dict[str, int] = {}
        for run in self.list_runs():
            if run.winner:
                tally[run.winner] = tally.get(run.winner, 0) + 1
        return tally


def _from_dict(data: dict) -> ComparisonRun:
    outputs = [
        ModelOutput(**{k: v for k, v in o.items() if k in ModelOutput.__annotations__})
        for o in data.get("outputs", [])
        if isinstance(o, dict)
    ]
    return ComparisonRun(
        id=str(data["id"]),
        timestamp=str(data.get("timestamp") or ""),
        prompt=str(data.get("prompt") or ""),
        system_prompt=str(data.get("system_prompt") or ""),
        outputs=outputs,
        winner=(data.get("winner") if data.get("winner") else None),
    )
