from dataclasses import dataclass, field
from typing import Literal

ChatRole = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    role: ChatRole
    content: str


@dataclass
class ChatRequest:
    messages: list[ChatMessage]
    model_key: str = "granite"
    stream: bool = True
    temperature: float | None = None
    max_tokens: int | None = None


@dataclass
class ChatResponse:
    model_key: str
    model_name: str
    message: ChatMessage
    done: bool = True
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: int | None = None


@dataclass
class ChatChunk:
    model_key: str
    model_name: str
    content: str = ""
    done: bool = False