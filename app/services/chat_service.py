from collections.abc import Generator
from dataclasses import dataclass, field

from app.config import get_settings
from app.models.ollama_adapter import OllamaAdapter
from app.schemas.chat import ChatChunk, ChatMessage, ChatRequest, ChatResponse
from app.schemas.model import ModelKey, ModelSelection
from app.services.model_router import ModelRouter, RoutingDecision
from app.services.rag_service import RAGService, Retrieval


@dataclass
class ChatExecution:
    response: ChatResponse | None = None
    stream: Generator[ChatChunk, None, None] | None = None
    routing_decision: RoutingDecision | None = None
    selected_model: ModelKey | None = None
    retrievals: list[Retrieval] = field(default_factory=list)


class ChatService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.router = ModelRouter()
        self.rag = RAGService()
        self.adapters: dict[ModelKey, OllamaAdapter] = {
            "granite": OllamaAdapter("granite"),
            "gemma": OllamaAdapter("gemma"),
            "qwen": OllamaAdapter("qwen"),
        }

    def _get_adapter(self, model_key: ModelKey) -> OllamaAdapter:
        adapter = self.adapters.get(model_key)
        if adapter is None:
            raise ValueError(f"Unsupported model_key: {model_key}")
        return adapter

    def _resolve_model(self, request: ChatRequest, selection: ModelSelection) -> tuple[ModelKey, RoutingDecision | None]:
        if selection == "auto":
            decision = self.router.route(request)
            return decision.selected_model, decision
        return selection, None

    def _augment_with_rag(self, request: ChatRequest) -> tuple[ChatRequest, list[Retrieval]]:
        if not request.messages:
            return request, []
        query = request.messages[-1].content
        retrievals = self.rag.retrieve(query)
        if not retrievals:
            return request, []
        context = self.rag.format_context(retrievals)
        if not context:
            return request, retrievals
        augmented_content = (
            "Use the following retrieved context to answer if it is relevant. "
            "Cite source paths when applicable.\n\n"
            f"{context}\n\n"
            "---\n\n"
            f"User question: {query}"
        )
        new_messages = list(request.messages[:-1]) + [
            ChatMessage(role="user", content=augmented_content)
        ]
        augmented = ChatRequest(
            messages=new_messages,
            model_key=request.model_key,
            stream=request.stream,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        return augmented, retrievals

    def chat(self, request: ChatRequest, selection: ModelSelection = "auto", use_rag: bool = False) -> ChatExecution:
        retrievals: list[Retrieval] = []
        if use_rag:
            request, retrievals = self._augment_with_rag(request)

        model_key, routing_decision = self._resolve_model(request, selection)
        adapter = self._get_adapter(model_key)
        response = adapter.chat(request)

        return ChatExecution(
            response=response,
            routing_decision=routing_decision,
            selected_model=model_key,
            retrievals=retrievals,
        )

    def stream_chat(self, request: ChatRequest, selection: ModelSelection = "auto", use_rag: bool = False) -> ChatExecution:
        retrievals: list[Retrieval] = []
        if use_rag:
            request, retrievals = self._augment_with_rag(request)

        model_key, routing_decision = self._resolve_model(request, selection)
        adapter = self._get_adapter(model_key)
        stream = adapter.stream_chat(request)

        return ChatExecution(
            stream=stream,
            routing_decision=routing_decision,
            selected_model=model_key,
            retrievals=retrievals,
        )

    def health_check(self) -> dict[str, bool]:
        return {
            model_key: adapter.health_check()
            for model_key, adapter in self.adapters.items()
        }
