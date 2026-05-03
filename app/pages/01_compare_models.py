"""Side-by-side model comparison: run one prompt against multiple models.

After a run, capture per-model output + metrics in session_state so the
user can vote on the winner and save the run to disk for later review.
"""
from contextlib import closing
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from app.schemas.chat import ChatMessage, ChatRequest
from app.services.chat_service import ChatService
from app.utils.comparison_storage import (
    ComparisonStorage,
    ModelOutput,
    new_run,
)
from app.utils.logger import setup_logger


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------

logger = setup_logger()


@st.cache_resource
def get_chat_service() -> ChatService:
    return ChatService()


@st.cache_resource
def get_storage() -> ComparisonStorage:
    return ComparisonStorage(Path("data/comparisons"))


chat_service = get_chat_service()
storage = get_storage()
all_keys = list(chat_service.adapters.keys())

st.set_page_config(
    page_title="LocalLLM — Compare",
    page_icon="📊",
    layout="wide",
)


def _format_tps(eval_count: int | None, eval_duration_ns: int | None) -> str:
    if not eval_count or not eval_duration_ns:
        return "—"
    seconds = eval_duration_ns / 1_000_000_000
    if seconds <= 0:
        return "—"
    return f"{eval_count / seconds:.1f} tok/s"


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

defaults = {
    "_compare_results": [],     # list[ModelOutput] — most recent run
    "_compare_prompt": "",      # prompt that produced _compare_results
    "_compare_system": "",      # system prompt that produced it
    "_compare_winner": None,    # picked winner model_key
    "_compare_saved_id": None,  # disk id once saved
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

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

    st.divider()

    with st.expander("Saved runs"):
        runs = storage.list_runs()
        if not runs:
            st.caption("No saved runs yet.")
        else:
            tally = storage.winner_tally()
            if tally:
                st.markdown("**Winner tally**")
                for model_key, count in sorted(tally.items(), key=lambda x: -x[1]):
                    st.caption(f"  {model_key}: {count}")
                st.divider()
            for r in runs[:10]:
                preview = r.prompt[:50] + ("…" if len(r.prompt) > 50 else "")
                marker = f" 🏆 {r.winner}" if r.winner else ""
                st.caption(f"`{r.timestamp[:16]}` · {preview}{marker}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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


def _execute_comparison(prompt: str, system_prompt: str, models: list[str]) -> list[ModelOutput]:
    columns = st.columns(len(models))

    # st.empty() returns a DeltaGenerator; we don't bind a strict type because
    # st.delta_generator is a private module.
    response_placeholders: dict[str, object] = {}
    metrics_placeholders: dict[str, object] = {}

    for column, model_key in zip(columns, models):
        with column:
            adapter = chat_service.adapters[model_key]
            st.subheader(model_key)
            st.caption(f"`{adapter.model_name}`")
            metrics_placeholders[model_key] = st.empty()
            response_placeholders[model_key] = st.empty()

    results: list[ModelOutput] = []
    for model_key in models:
        messages: list[ChatMessage] = []
        if system_prompt.strip():
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=prompt))

        request = ChatRequest(messages=messages, stream=True)
        adapter = chat_service.adapters[model_key]
        wall_started = time.perf_counter()

        full_response = ""
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        eval_duration_ns: int | None = None
        total_duration_ns: int | None = None

        metrics_placeholders[model_key].caption("⏳ running…")

        try:
            with closing(adapter.stream_chat(request)) as stream:
                for chunk in stream:
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
            tps_str = _format_tps(completion_tokens, eval_duration_ns)
            prompt_tok_str = prompt_tokens if prompt_tokens is not None else "—"
            completion_tok_str = completion_tokens if completion_tokens is not None else "—"

            metrics_placeholders[model_key].caption(
                f"⏱ {total_elapsed_s:.1f}s · in {prompt_tok_str} · out {completion_tok_str} · {tps_str}"
            )

            logger.info(
                "Comparison run | model=%s | prompt_tokens=%s | completion_tokens=%s | total_s=%.2f",
                model_key, prompt_tokens, completion_tokens, total_elapsed_s,
            )

            tps_value = (
                completion_tokens / (eval_duration_ns / 1_000_000_000)
                if completion_tokens and eval_duration_ns
                else None
            )

            results.append(ModelOutput(
                model_key=model_key,
                model_name=adapter.model_name,
                text=full_response,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                elapsed_s=total_elapsed_s,
                tokens_per_sec=tps_value,
            ))

        except Exception as exc:
            logger.exception("Comparison run failed for %s", model_key)
            response_placeholders[model_key].error(f"Error: {exc}")
            metrics_placeholders[model_key].caption("❌ failed")
            results.append(ModelOutput(
                model_key=model_key,
                model_name=adapter.model_name,
                text="",
                error=str(exc),
            ))

    return results


if run:
    results = _execute_comparison(prompt, system_prompt, selected_models)
    st.session_state._compare_results = results
    st.session_state._compare_prompt = prompt
    st.session_state._compare_system = system_prompt
    st.session_state._compare_winner = None
    st.session_state._compare_saved_id = None


# ---------------------------------------------------------------------------
# Post-run actions: pick winner + save
# ---------------------------------------------------------------------------

if st.session_state._compare_results and not run:
    # Re-render last results (so they survive button clicks like "Save run")
    columns = st.columns(len(st.session_state._compare_results))
    for column, output in zip(columns, st.session_state._compare_results):
        with column:
            st.subheader(output.model_key)
            st.caption(f"`{output.model_name}`")
            if output.error:
                st.error(f"Error: {output.error}")
            else:
                tps_str = (
                    f"{output.tokens_per_sec:.1f} tok/s"
                    if output.tokens_per_sec is not None
                    else "—"
                )
                elapsed_str = (
                    f"{output.elapsed_s:.1f}s" if output.elapsed_s is not None else "—"
                )
                in_str = output.prompt_tokens if output.prompt_tokens is not None else "—"
                out_str = output.completion_tokens if output.completion_tokens is not None else "—"
                st.caption(
                    f"⏱ {elapsed_str} · in {in_str} · out {out_str} · {tps_str}"
                )
                st.markdown(output.text or "_(empty response)_")

if st.session_state._compare_results:
    st.divider()

    pick_cols = st.columns([4, 1])
    with pick_cols[0]:
        keys = [o.model_key for o in st.session_state._compare_results if not o.error]
        if keys:
            current_winner = st.session_state._compare_winner
            options = [None] + keys
            selected_idx = options.index(current_winner) if current_winner in options else 0
            picked = st.radio(
                "Pick the winner",
                options=options,
                index=selected_idx,
                format_func=lambda x: "— no pick —" if x is None else f"🏆 {x}",
                horizontal=True,
                key="winner_radio",
            )
            st.session_state._compare_winner = picked
    with pick_cols[1]:
        save_label = "💾 Saved" if st.session_state._compare_saved_id else "💾 Save run"
        if st.button(
            save_label,
            type="primary",
            disabled=bool(st.session_state._compare_saved_id),
            use_container_width=True,
        ):
            run_record = new_run(
                prompt=st.session_state._compare_prompt,
                system_prompt=st.session_state._compare_system,
            )
            run_record.outputs = list(st.session_state._compare_results)
            run_record.winner = st.session_state._compare_winner
            try:
                storage.save(run_record)
                st.session_state._compare_saved_id = run_record.id
                st.success(f"Saved run `{run_record.id[:8]}…`")
                st.rerun()
            except Exception as exc:
                logger.exception("Failed to save comparison run")
                st.error(f"Save failed: {exc}")
