from collections.abc import Generator
from dataclasses import dataclass

from app.config import get_settings
from app.models.gemma_adapter import GemmaAdapter
from app.models.granite_adapter import GraniteAdapter
from app.schemas.chat import ChatChunk, ChatRequest, ChatResponse
from app.schemas.model import ModelKey, ModelSelection
from app.services.model_router import ModelRouter, RoutingDecision


@dataclass
class ChatExecution:
    response: ChatResponse | None = None
    stream: Generator[ChatChunk, None, None] | None = None
    routing_decision: RoutingDecision | None = None
    selected_model: ModelKey | None = None


class ChatService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.router = ModelRouter()
        self.adapters = {
            "granite": GraniteAdapter(),
            "gemma": GemmaAdapter(),
        }

    def _get_adapter(self, model_key: ModelKey):
        adapter = self.adapters.get(model_key)
        if adapter is None:
            raise ValueError(f"Unsupported model_key: {model_key}")
        return adapter

    def _resolve_model(self, request: ChatRequest, selection: ModelSelection) -> tuple[ModelKey, RoutingDecision | None]:
        if selection == "auto":
            decision = self.router.route(request)
            return decision.selected_model, decision

        return selection, None

    def chat(self, request: ChatRequest, selection: ModelSelection = "auto") -> ChatExecution:
        model_key, routing_decision = self._resolve_model(request, selection)
        adapter = self._get_adapter(model_key)

        response = adapter.chat(request)

        return ChatExecution(
            response=response,
            routing_decision=routing_decision,
            selected_model=model_key,
        )

    def stream_chat(self, request: ChatRequest, selection: ModelSelection = "auto") -> ChatExecution:
        model_key, routing_decision = self._resolve_model(request, selection)
        adapter = self._get_adapter(model_key)

        stream = adapter.stream_chat(request)

        return ChatExecution(
            stream=stream,
            routing_decision=routing_decision,
            selected_model=model_key,
        )

    def health_check(self) -> dict[str, bool]:
        return {
            model_key: adapter.health_check()
            for model_key, adapter in self.adapters.items()
        }