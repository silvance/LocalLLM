from dataclasses import dataclass
from typing import Literal

ModelKey = Literal["granite", "gemma"]
ModelSelection = Literal["auto", "granite", "gemma"]


@dataclass(frozen=True)
class ModelInfo:
    key: ModelKey
    name: str
    backend: str = "ollama"