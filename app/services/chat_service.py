from collections.abc import Generator
from dataclasses import dataclass, field

from app.config import get_settings
from app.models.ollama_adapter import OllamaAdapter
from app.schemas.chat import ChatChunk, ChatMessage, ChatRequest
from app.schemas.model import ModelKey, ModelSelection
from app.services.model_router import ModelRouter, RoutingDecision
from app.services.rag_service import RAGService, Retrieval


@dataclass
class ChatExecution:
    stream: Generator[ChatChunk, None, None]
    selected_model: ModelKey
    routing_decision: RoutingDecision | None = None
    retrievals: list[Retrieval] = field(default_factory=list)


_RAG_SYSTEM_PREFIX = (
    "Use the following retrieved context to answer if it is relevant. "
    "Cite source paths when applicable.\n\n"
)


class ChatService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.router = ModelRouter()
        self.rag = RAGService()
        # Three preset slots — what the auto-router chooses between and
        # what smart-install configures. Other adapters get built lazily
        # on first reference via get_adapter() so any installed Ollama
        # model is selectable without declaring it as a slot in config.
        self.adapters: dict[str, OllamaAdapter] = {
            "granite": OllamaAdapter("granite"),
            "gemma": OllamaAdapter("gemma"),
            "qwen": OllamaAdapter("qwen"),
        }

    def get_adapter(self, name: str) -> OllamaAdapter:
        """Resolve an adapter by slot name (granite/gemma/qwen) or raw
        Ollama model name (qwen2.5-coder:32b, deepseek-coder-v2:latest,
        ...). Caches subsequent lookups so the underlying ollama.Client
        is reused per model."""
        cached = self.adapters.get(name)
        if cached is not None:
            return cached
        adapter = OllamaAdapter(name)
        self.adapters[name] = adapter
        return adapter

    def _resolve_model(
        self,
        request: ChatRequest,
        selection: ModelSelection,
    ) -> tuple[str, RoutingDecision | None]:
        if selection == "auto":
            decision = self.router.route(request)
            return decision.selected_model, decision
        return selection, None

    def _augment_with_rag(
        self,
        request: ChatRequest,
    ) -> tuple[ChatRequest, list[Retrieval]]:
        """Replace the user's last message with one that prepends retrieved
        context. NOTE: only the latest message is augmented — earlier messages
        in session_state stay clean so the chat history doesn't accumulate
        retrieval noise."""
        if not request.messages:
            return request, []
        query = request.messages[-1].content
        retrievals = self.rag.retrieve(query)
        if not retrievals:
            return request, []
        context = self.rag.format_context(retrievals)
        if not context:
            return request, retrievals
        augmented = ChatMessage(
            role="user",
            content=f"{_RAG_SYSTEM_PREFIX}{context}\n\n---\n\nUser question: {query}",
        )
        new_request = ChatRequest(
            messages=list(request.messages[:-1]) + [augmented],
            stream=request.stream,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        return new_request, retrievals

    def stream_chat(
        self,
        request: ChatRequest,
        selection: ModelSelection = "auto",
        use_rag: bool = False,
    ) -> ChatExecution:
        retrievals: list[Retrieval] = []
        if use_rag:
            request, retrievals = self._augment_with_rag(request)

        model_key, decision = self._resolve_model(request, selection)
        return ChatExecution(
            stream=self.get_adapter(model_key).stream_chat(request),
            selected_model=model_key,
            routing_decision=decision,
            retrievals=retrievals,
        )

    def health_check(self) -> dict[str, bool]:
        return {key: a.health_check() for key, a in self.adapters.items()}
