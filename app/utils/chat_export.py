"""Markdown serialization of chat conversations for the export download button."""
from __future__ import annotations

import datetime as _dt

from app.schemas.chat import ChatMessage


_ROLE_LABELS = {
    "system": "System",
    "user": "User",
    "assistant": "Assistant",
}


def messages_to_markdown(messages: list[ChatMessage], title: str = "LocalLLM chat") -> str:
    if not messages:
        return f"# {title}\n\n*(empty conversation)*\n"
    parts: list[str] = [f"# {title}", ""]
    parts.append(f"*Exported {_dt.datetime.now().isoformat(timespec='seconds')}*")
    parts.append("")
    for msg in messages:
        label = _ROLE_LABELS.get(msg.role, msg.role.capitalize())
        parts.append(f"## {label}")
        parts.append("")
        parts.append(msg.content.rstrip())
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def export_filename(prefix: str = "localllm-chat") -> str:
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{ts}.md"
