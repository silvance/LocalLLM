from dataclasses import dataclass
from typing import Literal

ModelKey = Literal["granite", "gemma", "qwen"]
ModelSelection = Literal["auto", "granite", "gemma", "qwen"]


@dataclass(frozen=True)
class ModelInfo:
    key: ModelKey
    name: str
    backend: str = "ollama"
