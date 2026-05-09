"""Shared path-safety helpers for the per-record storages.

Chat / review / comparison stores all map a user-controllable string
ID to a JSON file under a per-store base directory. The original
guards (reject ``/``, ``\\``, leading ``.``) miss the Windows drive-
letter case: ``Path("data/chats") / "C:foo.json"`` collapses to
``C:foo.json`` because pathlib treats ``C:`` as a drive anchor.

Two-layer check here:
  1. Whitelist regex on the raw ID — must match a UUID-like shape.
  2. Resolve the final path and assert it's still under base_dir.
     Defense-in-depth in case the regex misses something obvious
     in a future Python pathlib release.
"""
from __future__ import annotations

import re
from pathlib import Path


# UUIDs (with or without dashes), plus the few legacy IDs we've shipped
# (lowercase hex). Length cap keeps the disk footprint bounded too.
_VALID_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def validate_storage_id(id_: str, *, kind: str) -> None:
    """Raise ``ValueError`` if ``id_`` isn't a safe filename stem.

    ``kind`` is included in the error message so the caller doesn't
    have to wrap (e.g. "invalid chat id: ..." vs "invalid review id:").
    """
    if not isinstance(id_, str):
        raise ValueError(f"invalid {kind} id: not a string")
    if not _VALID_ID.fullmatch(id_):
        raise ValueError(f"invalid {kind} id: {id_!r}")
    if id_.startswith(".") or ".." in id_:
        raise ValueError(f"invalid {kind} id: {id_!r}")


def safe_storage_path(base_dir: Path, id_: str, *, suffix: str = ".json", kind: str = "record") -> Path:
    """Combine ``id_`` with ``base_dir`` after validating the id AND
    confirming the resolved path stays under ``base_dir``. Raises
    ``ValueError`` if either check fails."""
    validate_storage_id(id_, kind=kind)
    candidate = (base_dir / f"{id_}{suffix}").resolve()
    base_resolved = base_dir.resolve()
    try:
        candidate.relative_to(base_resolved)
    except ValueError as exc:
        raise ValueError(f"invalid {kind} id: {id_!r} resolves outside base") from exc
    return candidate
