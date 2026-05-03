"""LocalLLM Streamlit UI.

Layout:
  - Sidebar: model selection, RAG toggle, system prompt, inference params,
    health check, export, clear
  - Main: empty-state example prompts OR chat history with per-last-assistant
    actions (regenerate / edit & retry), live streaming, RAG citations
"""
from contextlib import closing
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app.config import get_settings
from app.schemas.chat import ChatMessage, ChatRequest
from app.services.chat_service import ChatService
from app.utils.chat_export import export_filename, messages_to_markdown
from app.utils.logger import setup_logger


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------

logger = setup_logger()
settings = get_settings()


@st.cache_resource
def get_chat_service() -> ChatService:
    return ChatService()


chat_service = get_chat_service()

st.set_page_config(page_title="LocalLLM", page_icon="🤖", layout="wide")

EXAMPLE_PROMPTS = [
    "Write a Python script that uses scapy to send a SYN packet to a list of hosts.",
    "Explain Kerberos delegation and how to abuse unconstrained delegation.",
    "Walk me through analyzing a Windows memory image with Volatility 3.",
    "Compare BloodHound and SharpHound for AD enumeration workflows.",
    "Refactor a synchronous requests-based scanner to use asyncio + aiohttp.",
    "Write a YARA rule that matches on a custom PE section name.",
]

# Languages our RAG metadata may carry → Streamlit/Pygments lexer name
LANG_TO_LEXER = {
    "python": "python",
    "go": "go",
    "ruby": "ruby",
    "powershell": "powershell",
    "lua": "lua",
    "javascript": "javascript",
    "markdown": "markdown",
    "rst": "rst",
    "yaml": "yaml",
    "json": "json",
    "bash": "bash",
    "html": "html",
    "css": "css",
    "sql": "sql",
}


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "messages": [],
        "model_selection": "auto",
        "use_rag": settings.rag_enabled,
        "system_prompt": "",
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "num_ctx": settings.num_ctx,
        "_partial_response": "",
        "_regenerate": False,
        "_editing": False,
        "_pending_prompt": None,
        "_last_retrievals": [],
        "_last_routing": None,
        "_last_rag_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()

# Salvage a partial response if a previous run was interrupted (Stop / refresh).
if st.session_state._partial_response:
    st.session_state.messages.append(
        ChatMessage(
            role="assistant",
            content=st.session_state._partial_response.rstrip() + "\n\n_[stopped]_",
        )
    )
    st.session_state._partial_response = ""


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")

    selection_options = ["auto", "granite", "gemma", "qwen"]
    st.session_state.model_selection = st.selectbox(
        "Model",
        options=selection_options,
        index=selection_options.index(st.session_state.model_selection),
        help="auto routes by prompt content (code → qwen, general → gemma, simple → granite).",
    )

    rag_count = chat_service.rag.count()
    if rag_count:
        rag_label = f"Use RAG ({rag_count:,} chunks)"
        rag_disabled = False
    else:
        rag_label = "Use RAG"
        rag_disabled = True
    st.session_state.use_rag = st.checkbox(
        rag_label,
        value=st.session_state.use_rag and rag_count > 0,
        disabled=rag_disabled,
        help=f"Retrieval-augmented generation using `{settings.rag_embedding_model}`.",
    )
    if rag_disabled:
        st.caption("⚠️ no index — run `scripts/build_index.py`")

    st.session_state.system_prompt = st.text_area(
        "System prompt",
        value=st.session_state.system_prompt,
        height=120,
        placeholder="e.g. You are a senior pentester. Prefer Python over bash. Cite sources.",
        help="Prepended as a `system` role message on every turn. Empty = no system message.",
    )

    with st.expander("Inference parameters"):
        st.session_state.temperature = st.slider(
            "Temperature", 0.0, 2.0, float(st.session_state.temperature), 0.05,
        )
        st.session_state.max_tokens = st.number_input(
            "Max tokens (num_predict)",
            min_value=32,
            max_value=16384,
            value=int(st.session_state.max_tokens),
            step=64,
        )
        st.session_state.num_ctx = st.number_input(
            "Context window (num_ctx)",
            min_value=2048,
            max_value=262144,
            value=int(st.session_state.num_ctx),
            step=2048,
            help="Larger = more context, more VRAM. RAG retrieval needs at least 32K.",
        )

    st.divider()

    if st.button("Check model health", use_container_width=True):
        try:
            health = chat_service.health_check()
            for name, ready in health.items():
                if ready:
                    st.success(f"{name}: ready")
                else:
                    st.error(f"{name}: not available")
        except Exception as exc:
            logger.exception("Health check failed")
            st.error(f"Health check failed: {exc}")

    if st.session_state.messages:
        st.download_button(
            "📥 Export chat (.md)",
            data=messages_to_markdown(st.session_state.messages),
            file_name=export_filename(),
            mime="text/markdown",
            use_container_width=True,
        )

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state._editing = False
        st.session_state._regenerate = False
        st.session_state._partial_response = ""
        st.session_state._last_retrievals = []
        st.session_state._last_routing = None
        st.session_state._last_rag_error = None
        st.rerun()


# ---------------------------------------------------------------------------
# Main column
# ---------------------------------------------------------------------------

st.title("LocalLLM")
st.caption("Local Streamlit chat app powered by Ollama")


def _render_retrievals(retrievals: list) -> None:
    if not retrievals:
        return
    with st.expander(f"📚 Retrieved {len(retrievals)} sources", expanded=False):
        for r in retrievals:
            header_cols = st.columns([3, 2])
            with header_cols[0]:
                st.markdown(
                    f"**{r.source_id}** · *{r.language or 'n/a'}* · score `{r.score:.2f}`"
                )
            with header_cols[1]:
                # st.code gives a copy-on-hover button — easiest copy-path UX.
                st.code(r.file_path, language=None)
            snippet = r.document[:600] + ("…" if len(r.document) > 600 else "")
            st.code(snippet, language=LANG_TO_LEXER.get(r.language or "", None))


def _render_chat_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg.role):
            st.markdown(msg.content)


# Empty state with example prompt buttons
if not st.session_state.messages and not st.session_state._editing:
    st.info(
        "Empty conversation. Pick an example below or type your own prompt.",
        icon="💡",
    )
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLE_PROMPTS):
        with cols[i % 2]:
            if st.button(example, use_container_width=True, key=f"example_{i}"):
                st.session_state._pending_prompt = example
                st.rerun()

_render_chat_history()

# Surface the most recent retrievals + routing under the latest assistant
# message so they don't disappear on rerun (regenerate, edit, etc.).
last = st.session_state.messages[-1] if st.session_state.messages else None
if last and last.role == "assistant":
    if st.session_state._last_routing:
        d = st.session_state._last_routing
        st.caption(
            f"Auto-routed to: **{d.selected_model}** "
            f"(score={d.complexity_score}) — {d.reason}"
        )
    if st.session_state._last_rag_error:
        st.warning(f"⚠️ RAG: {st.session_state._last_rag_error}")
    _render_retrievals(st.session_state._last_retrievals)

    if not st.session_state._editing:
        action_cols = st.columns([1, 1, 6])
        if action_cols[0].button("🔄 Regenerate", key="regen_btn"):
            st.session_state.messages.pop()  # drop assistant
            st.session_state._regenerate = True
            st.session_state._last_retrievals = []
            st.session_state._last_routing = None
            st.session_state._last_rag_error = None
            st.rerun()
        if action_cols[1].button("✏️ Edit & retry", key="edit_btn"):
            st.session_state._editing = True
            st.rerun()


# Edit-mode UI
if st.session_state._editing:
    last_user_idx = next(
        (i for i in range(len(st.session_state.messages) - 1, -1, -1)
         if st.session_state.messages[i].role == "user"),
        None,
    )
    if last_user_idx is None:
        st.session_state._editing = False
        st.rerun()
    with st.container(border=True):
        st.caption("Edit the previous user message and resend")
        new_content = st.text_area(
            "Edit",
            value=st.session_state.messages[last_user_idx].content,
            height=160,
            label_visibility="collapsed",
            key="edit_textarea",
        )
        edit_cols = st.columns([1, 1, 4])
        if edit_cols[0].button("Save & retry", type="primary", key="edit_save"):
            st.session_state.messages[last_user_idx].content = new_content
            del st.session_state.messages[last_user_idx + 1:]  # drop later messages
            st.session_state._editing = False
            st.session_state._regenerate = True
            st.session_state._last_retrievals = []
            st.session_state._last_routing = None
            st.session_state._last_rag_error = None
            st.rerun()
        if edit_cols[1].button("Cancel", key="edit_cancel"):
            st.session_state._editing = False
            st.rerun()


# ---------------------------------------------------------------------------
# Input / generation
# ---------------------------------------------------------------------------

prompt = st.chat_input("Send a message")

# Resolve which path triggered generation this run
trigger: str | None = None
generation_messages: list[ChatMessage] | None = None

if prompt:
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    trigger = "new"
elif st.session_state._pending_prompt:
    pending = st.session_state._pending_prompt
    st.session_state._pending_prompt = None
    st.session_state.messages.append(ChatMessage(role="user", content=pending))
    trigger = "example"
elif st.session_state._regenerate:
    st.session_state._regenerate = False
    trigger = "regenerate"

if trigger:
    # Build the messages to send: optional system prompt + chat history.
    generation_messages = []
    if st.session_state.system_prompt.strip():
        generation_messages.append(
            ChatMessage(role="system", content=st.session_state.system_prompt)
        )
    generation_messages.extend(st.session_state.messages)

if generation_messages is not None:
    # Show the user's just-added message inline before streaming begins
    if trigger in ("new", "example"):
        with st.chat_message("user"):
            st.markdown(st.session_state.messages[-1].content)

    request = ChatRequest(
        messages=generation_messages,
        stream=True,
        temperature=st.session_state.temperature,
        max_tokens=st.session_state.max_tokens,
        num_ctx=st.session_state.num_ctx,
    )

    execution = chat_service.stream_chat(
        request=request,
        selection=st.session_state.model_selection,
        use_rag=st.session_state.use_rag,
    )

    st.session_state._last_routing = execution.routing_decision
    st.session_state._last_retrievals = execution.retrievals
    st.session_state._last_rag_error = (
        chat_service.rag.last_error if st.session_state.use_rag else None
    )

    with st.chat_message("assistant"):
        if execution.routing_decision is not None:
            st.caption(
                f"Auto-routed to: **{execution.selected_model}** "
                f"(score={execution.routing_decision.complexity_score}) — "
                f"{execution.routing_decision.reason}"
            )
        else:
            st.caption(f"Using model: **{execution.selected_model}**")

        if st.session_state._last_rag_error:
            st.warning(f"⚠️ RAG: {st.session_state._last_rag_error}")

        _render_retrievals(execution.retrievals)

        # Stop button — clicking it triggers a Streamlit rerun, which kills
        # the script mid-stream. The salvage block at the top of the file
        # then promotes the in-progress text in _partial_response into a
        # finalized assistant message.
        stop_placeholder = st.empty()
        stop_placeholder.button("⏹ Stop", key="stop_btn")

        response_placeholder = st.empty()
        full_response = ""

        try:
            with closing(execution.stream) as stream:
                for chunk in stream:
                    if chunk.content:
                        full_response += chunk.content
                        st.session_state._partial_response = full_response
                        response_placeholder.markdown(full_response)

            # Successful completion: clear the salvage flag and finalize.
            st.session_state._partial_response = ""
            stop_placeholder.empty()

            if not full_response.strip():
                full_response = "_(no response from model)_"
                response_placeholder.warning(full_response)

            st.session_state.messages.append(
                ChatMessage(role="assistant", content=full_response)
            )

            logger.info(
                "Chat completed | trigger=%s | model=%s | rag=%s | retrievals=%s | response_chars=%s",
                trigger,
                execution.selected_model,
                st.session_state.use_rag,
                len(execution.retrievals),
                len(full_response),
            )
            if execution.routing_decision is not None:
                logger.info(
                    "Routing | model=%s | score=%s | reason=%s",
                    execution.selected_model,
                    execution.routing_decision.complexity_score,
                    execution.routing_decision.reason,
                )

            # Rerun so the action bar (regenerate/edit) renders under the new
            # assistant message without stealing the streaming column.
            st.rerun()

        except Exception as exc:
            logger.exception("Chat request failed")
            response_placeholder.error(f"Error: {exc}")
            st.session_state._partial_response = ""
