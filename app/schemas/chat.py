from dataclasses import dataclass
from typing import Literal

ChatRole = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    role: ChatRole
    content: str


@dataclass
class ChatRequest:
    messages: list[ChatMessage]
    stream: bool = True
    temperature: float | None = None
    max_tokens: int | None = None


@dataclass
class ChatChunk:
    model_key: str
    model_name: str
    content: str = ""
    done: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_duration_ns: int | None = None
    eval_duration_ns: int | None = None
