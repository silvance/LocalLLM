from collections.abc import Generator

from ollama import Client

from app.config import get_settings
from app.schemas.chat import ChatChunk, ChatRequest
from app.schemas.model import ModelKey


class OllamaAdapter:
    def __init__(self, model_key: ModelKey) -> None:
        self.settings = get_settings()
        self.client = Client(host=self.settings.ollama_host)
        self.model_key = model_key
        self.model_name = self.settings.model_map[model_key]

    def _options(self, request: ChatRequest) -> dict:
        return {
            "temperature": (
                request.temperature
                if request.temperature is not None
                else self.settings.temperature
            ),
            "num_predict": (
                request.max_tokens
                if request.max_tokens is not None
                else self.settings.max_tokens
            ),
            "num_ctx": (
                request.num_ctx
                if request.num_ctx is not None
                else self.settings.num_ctx
            ),
        }

    def stream_chat(self, request: ChatRequest) -> Generator[ChatChunk, None, None]:
        stream = self.client.chat(
            model=self.model_name,
            messages=[{"role": m.role, "content": m.content} for m in request.messages],
            stream=True,
            options=self._options(request),
        )
        for chunk in stream:
            done = chunk.get("done", False)
            yield ChatChunk(
                model_key=self.model_key,
                model_name=self.model_name,
                content=chunk.get("message", {}).get("content", ""),
                done=done,
                prompt_tokens=chunk.get("prompt_eval_count") if done else None,
                completion_tokens=chunk.get("eval_count") if done else None,
                total_duration_ns=chunk.get("total_duration") if done else None,
                eval_duration_ns=chunk.get("eval_duration") if done else None,
            )

    def health_check(self) -> bool:
        """Returns True if the configured model is registered in Ollama.
        Doesn't verify the model can actually load — that takes too long for a
        sidebar check."""
        try:
            response = self.client.list()
        except Exception:
            return False

        installed: set[str] = set()
        for model in response.get("models", []):
            name = model.get("name") or model.get("model") or ""
            if not name:
                continue
            installed.add(name)
            if ":" in name:
                installed.add(name.split(":")[0])

        target = {self.model_name}
        if ":" in self.model_name:
            target.add(self.model_name.split(":")[0])
        else:
            target.add(f"{self.model_name}:latest")
        return bool(target & installed)
