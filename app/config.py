from dataclasses import dataclass
from functools import lru_cache
import os

from dotenv import load_dotenv

load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    app_env: str
    log_level: str

    ollama_host: str

    granite_model: str
    gemma_model: str
    qwen_model: str
    default_model: str

    default_stream: bool
    request_timeout: int
    temperature: float
    max_tokens: int
    num_ctx: int

    @property
    def model_map(self) -> dict[str, str]:
        return {
            "granite": self.granite_model,
            "gemma": self.gemma_model,
            "qwen": self.qwen_model,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        app_env=os.getenv("APP_ENV", "dev"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),

        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),

        granite_model=os.getenv("GRANITE_MODEL", "granite4"),
        gemma_model=os.getenv("GEMMA_MODEL", "gemma4"),
        qwen_model=os.getenv("QWEN_MODEL", "qwen3-coder:30b"),
        default_model=os.getenv("DEFAULT_MODEL", "auto"),

        default_stream=_get_bool("DEFAULT_STREAM", True),
        request_timeout=_get_int("REQUEST_TIMEOUT", 120),
        temperature=_get_float("TEMPERATURE", 0.2),
        max_tokens=_get_int("MAX_TOKENS", 1024),
        num_ctx=_get_int("NUM_CTX", 32768),
    )
