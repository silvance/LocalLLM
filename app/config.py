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
    temperature: float
    max_tokens: int
    num_ctx: int

    rag_enabled: bool
    rag_index_dir: str
    rag_collection: str
    rag_embedding_model: str
    rag_retrieval_k: int
    rag_max_context_chars: int
    rag_min_query_len: int
    rag_rerank_enabled: bool
    rag_rerank_model: str
    rag_rerank_pool: int

    default_system_prompt: str

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

        # Default to the explicit IPv4 loopback rather than `localhost`.
        # Modern Windows resolves `localhost` to `::1` first, but Ollama
        # only binds IPv4 — the connection refuses with WinError 10061
        # and the model dropdown ends up empty.
        ollama_host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),

        granite_model=os.getenv("GRANITE_MODEL", "granite4"),
        gemma_model=os.getenv("GEMMA_MODEL", "gemma4"),
        qwen_model=os.getenv("QWEN_MODEL", "qwen3-coder:30b"),
        default_model=os.getenv("DEFAULT_MODEL", "auto"),

        default_stream=_get_bool("DEFAULT_STREAM", True),
        temperature=_get_float("TEMPERATURE", 0.2),
        max_tokens=_get_int("MAX_TOKENS", 8192),
        num_ctx=_get_int("NUM_CTX", 32768),

        rag_enabled=_get_bool("RAG_ENABLED", False),
        rag_index_dir=os.getenv("RAG_INDEX_DIR", "data/index"),
        rag_collection=os.getenv("RAG_COLLECTION", "corpus"),
        rag_embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "nomic-embed-text"),
        # Default RAG tuned for the airgap target — wider net + reranker on.
        # The reranker adds 1-2 sec/query on the 3070 but noticeably better
        # citation relevance, which matters more than latency for the kind of
        # focused coding work this app exists for.
        rag_retrieval_k=_get_int("RAG_RETRIEVAL_K", 8),
        rag_max_context_chars=_get_int("RAG_MAX_CONTEXT_CHARS", 8000),
        rag_min_query_len=_get_int("RAG_MIN_QUERY_LEN", 8),
        rag_rerank_enabled=_get_bool("RAG_RERANK_ENABLED", True),
        rag_rerank_model=os.getenv("RAG_RERANK_MODEL", "granite4:tiny-h"),
        rag_rerank_pool=_get_int("RAG_RERANK_POOL", 20),

        default_system_prompt=os.getenv(
            "DEFAULT_SYSTEM_PROMPT",
            "You are LocalLLM, a local AI assistant for coding and "
            "security work (pentesting, digital forensics, CTF). Answer "
            "the user's actual request.\n\n"
            "Response rules:\n"
            "1. For normal questions, identity questions, planning, or "
            "explanations, answer in concise prose. Do not turn those "
            "answers into code.\n"
            "2. Only produce code when the user asks for code, commands, "
            "configuration, a script, or a concrete implementation.\n"
            "3. When you include code, wrap code in fenced markdown blocks "
            "(```language ... ```).\n"
            "4. Write file paths and identifiers as plain text or inline "
            "code (`like_this`). NEVER use markdown links like "
            "[scope.py](http://scope.py) — they are wrong.\n"
            "5. Before each code block, briefly verify the imports you use "
            "are real and the names you reference are defined.\n"
            "6. Python: add type hints to function signatures and return "
            "types. Use specific exception types in `except` clauses, never "
            "bare `except:`. Avoid `from x import *`.\n"
            "7. Public functions and classes get a one-line docstring "
            "stating their purpose.\n"
            "8. Be concise. If the user asks for a full skeleton, build "
            "it incrementally and self-check each piece compiles before "
            "moving on.",
        ),
    )
