import time
from collections.abc import Generator

from ollama import Client

from app.config import get_settings
from app.models.base import BaseModelAdapter
from app.schemas.chat import ChatChunk, ChatRequest, ChatResponse, ChatMessage
from app.schemas.model import ModelInfo


class GemmaAdapter(BaseModelAdapter):
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = Client(host=self.settings.ollama_host)
        self.model_key = "gemma"
        self.model_name = self.settings.model_map[self.model_key]

    def _build_messages(self, request: ChatRequest) -> list[dict[str, str]]:
        return [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in request.messages
        ]

    def chat(self, request: ChatRequest) -> ChatResponse:
        started = time.perf_counter()

        response = self.client.chat(
            model=self.model_name,
            messages=self._build_messages(request),
            stream=False,
            options={
                "temperature": request.temperature if request.temperature is not None else self.settings.temperature,
                "num_predict": request.max_tokens if request.max_tokens is not None else self.settings.max_tokens,
            },
        )

        latency_ms = int((time.perf_counter() - started) * 1000)
        content = response["message"]["content"]

        prompt_tokens = response.get("prompt_eval_count")
        completion_tokens = response.get("eval_count")
        total_tokens = None
        if prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens

        return ChatResponse(
            model_key=self.model_key,
            model_name=self.model_name,
            message=ChatMessage(role="assistant", content=content),
            done=response.get("done", True),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
        )

    def stream_chat(self, request: ChatRequest) -> Generator[ChatChunk, None, None]:
        stream = self.client.chat(
            model=self.model_name,
            messages=self._build_messages(request),
            stream=True,
            options={
                "temperature": request.temperature if request.temperature is not None else self.settings.temperature,
                "num_predict": request.max_tokens if request.max_tokens is not None else self.settings.max_tokens,
            },
        )

        for chunk in stream:
            content = chunk.get("message", {}).get("content", "")
            done = chunk.get("done", False)

            yield ChatChunk(
                model_key=self.model_key,
                model_name=self.model_name,
                content=content,
                done=done,
            )

    def health_check(self) -> bool:
        try:
            models_response = self.client.list()
            models = models_response.get("models", [])

            installed_names = set()

            for model in models:
                name = model.get("name") or model.get("model")
                if not name:
                    continue

                installed_names.add(name)

                if ":" in name:
                    installed_names.add(name.split(":")[0])

            target_names = {self.model_name}
            if ":" in self.model_name:
                target_names.add(self.model_name.split(":")[0])
            else:
                target_names.add(f"{self.model_name}:latest")

            return any(name in installed_names for name in target_names)

        except Exception:
            return False

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            key=self.model_key,
            name=self.model_name,
            backend="ollama",
        )