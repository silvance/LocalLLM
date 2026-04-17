from abc import ABC, abstractmethod
from collections.abc import Generator

from app.schemas.chat import ChatRequest, ChatResponse, ChatChunk
from app.schemas.model import ModelInfo


class BaseModelAdapter(ABC):
    @abstractmethod
    def chat(self, request: ChatRequest) -> ChatResponse:
        pass

    @abstractmethod
    def stream_chat(self, request: ChatRequest) -> Generator[ChatChunk, None, None]:
        pass

    @abstractmethod
    def health_check(self) -> bool:
        pass

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        pass