"""Side-by-side model comparison: run one prompt against multiple models and compare."""
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from app.schemas.chat import ChatMessage, ChatRequest
from app.services.chat_service import ChatService
from app.utils.logger import setup_logger


logger = setup_logger()
chat_service = ChatService()
all_keys = list(chat_service.adapters.keys())


def _format_tps(eval_count: int | None, eval_duration_ns: int | None) -> str:
    if not eval_count or not eval_duration_ns:
        return "—"
    seconds = eval_duration_ns / 1_000_000_000
    if seconds <= 0:
        return "—"
    return f"{eval_count / seconds:.1f} tok/s"


st.set_page_config(
    page_title="LocalLLM — Compare",
    page_icon="📊",
    layout="wide",
)

st.title("Compare Models")
st.caption("Run the same prompt against multiple models. Sequential — Ollama serializes on a single GPU.")

with st.sidebar:
    st.header("Comparison settings")
    selected_models = st.multiselect(
        "Models to run",
        options=all_keys,
        default=all_keys,
    )
    system_prompt = st.text_area(
        "System prompt (optional)",
        value="",
        height=120,
    )

prompt = st.text_area(
    "Prompt",
    height=220,
    placeholder="Paste a coding question, problem, or task here...",
)

run = st.button(
    "Run comparison",
    type="primary",
    disabled=not prompt or not selected_models,
)

if run:
    columns = st.columns(len(selected_models))

    response_placeholders: dict[str, st.delta_generator.DeltaGenerator] = {}
    metrics_placeholders: dict[str, st.delta_generator.DeltaGenerator] = {}

    for column, model_key in zip(columns, selected_models):
        with column:
            adapter = chat_service.adapters[model_key]
            st.subheader(f"{model_key}")
            st.caption(f"`{adapter.model_name}`")
            metrics_placeholders[model_key] = st.empty()
            response_placeholders[model_key] = st.empty()

    for model_key in selected_models:
        messages: list[ChatMessage] = []
        if system_prompt.strip():
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=prompt))

        request = ChatRequest(messages=messages, stream=True)

        adapter = chat_service.adapters[model_key]
        wall_started = time.perf_counter()
        full_response = ""
        prompt_tokens = None
        completion_tokens = None
        eval_duration_ns = None
        total_duration_ns = None

        metrics_placeholders[model_key].caption("⏳ running…")

        try:
            for chunk in adapter.stream_chat(request):
                if chunk.content:
                    full_response += chunk.content
                    response_placeholders[model_key].markdown(full_response)
                if chunk.done:
                    prompt_tokens = chunk.prompt_tokens
                    completion_tokens = chunk.completion_tokens
                    eval_duration_ns = chunk.eval_duration_ns
                    total_duration_ns = chunk.total_duration_ns

            wall_elapsed = time.perf_counter() - wall_started
            total_elapsed_s = (
                total_duration_ns / 1_000_000_000 if total_duration_ns else wall_elapsed
            )
            tps = _format_tps(completion_tokens, eval_duration_ns)
            prompt_tok_str = prompt_tokens if prompt_tokens is not None else "—"
            completion_tok_str = completion_tokens if completion_tokens is not None else "—"

            metrics_placeholders[model_key].caption(
                f"⏱ {total_elapsed_s:.1f}s · in {prompt_tok_str} · out {completion_tok_str} · {tps}"
            )

            logger.info(
                "Comparison run | model=%s | prompt_tokens=%s | completion_tokens=%s | total_s=%.2f",
                model_key,
                prompt_tokens,
                completion_tokens,
                total_elapsed_s,
            )

        except Exception as exc:
            logger.exception("Comparison run failed for %s", model_key)
            response_placeholders[model_key].error(f"Error: {exc}")
            metrics_placeholders[model_key].caption("❌ failed")
