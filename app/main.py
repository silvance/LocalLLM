"""LocalLLM Streamlit UI.

Layout:
  - Sidebar: chat session selector, model + RAG, system prompt, params,
    health dots, export, clear, delete-current
  - Main: empty-state example prompts OR chat history with regenerate /
    edit-and-retry actions, live streaming with first-token spinner,
    RAG citations, token-count + RAG-skipped notices below the input
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
from app.utils.chat_storage import (
    ChatSession,
    ChatStorage,
    derive_title,
    new_session,
)
from app.utils.logger import setup_logger
from app.utils.token_estimate import estimate_messages_tokens, estimate_tokens


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------

logger = setup_logger()
settings = get_settings()


@st.cache_resource
def get_chat_service() -> ChatService:
    return ChatService()


@st.cache_resource
def get_storage() -> ChatStorage:
    return ChatStorage(Path("data/chats"))


chat_service = get_chat_service()
storage = get_storage()

st.set_page_config(page_title="LocalLLM", page_icon="🤖", layout="wide")

EXAMPLE_PROMPTS = [
    "Write a Python script that uses scapy to send a SYN packet to a list of hosts.",
    "Explain Kerberos delegation and how to abuse unconstrained delegation.",
    "Walk me through analyzing a Windows memory image with Volatility 3.",
    "Compare BloodHound and SharpHound for AD enumeration workflows.",
    "Refactor a synchronous requests-based scanner to use asyncio + aiohttp.",
    "Write a YARA rule that matches on a custom PE section name.",
]

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


@st.cache_data(ttl=30, show_spinner=False)
def health_snapshot() -> dict[str, bool]:
    """30-second TTL — refreshes on the next interaction after expiry without
    forcing the user to click anything."""
    try:
        return chat_service.health_check()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Session state + multi-session helpers
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "messages": [],
        "current_chat_id": None,
        "_chat_created_at": "",
        "model_selection": "auto",
        "use_rag": settings.rag_enabled,
        "system_prompt": settings.default_system_prompt,
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
        "_last_user_query_was_short": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _clear_transient_state() -> None:
    st.session_state._partial_response = ""
    st.session_state._regenerate = False
    st.session_state._editing = False
    st.session_state._pending_prompt = None
    st.session_state._last_retrievals = []
    st.session_state._last_routing = None
    st.session_state._last_rag_error = None
    st.session_state._last_user_query_was_short = False


def _save_current_session() -> None:
    if not st.session_state.messages:
        return
    cid = st.session_state.current_chat_id
    if cid is None:
        return
    session = ChatSession(
        id=cid,
        title=derive_title(st.session_state.messages),
        created_at=st.session_state._chat_created_at or "",
        updated_at="",  # storage.save() overwrites this
        messages=list(st.session_state.messages),
    )
    try:
        storage.save(session)
    except Exception:
        logger.exception("Failed to save chat session %s", cid)


def _load_chat(chat_id: str) -> None:
    session = storage.load(chat_id)
    if not session:
        return
    st.session_state.current_chat_id = session.id
    st.session_state.messages = list(session.messages)
    st.session_state._chat_created_at = session.created_at
    _clear_transient_state()


def _start_new_chat() -> None:
    _save_current_session()
    s = new_session()
    st.session_state.current_chat_id = s.id
    st.session_state._chat_created_at = s.created_at
    st.session_state.messages = []
    _clear_transient_state()


def _delete_current_chat() -> None:
    cid = st.session_state.current_chat_id
    if cid:
        try:
            storage.delete(cid)
        except Exception:
            logger.exception("Failed to delete chat %s", cid)
    summaries = storage.list_summaries()
    if summaries:
        _load_chat(summaries[0].id)
    else:
        _start_new_chat()


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
    _save_current_session()

# Pick or create the active chat on first load.
if st.session_state.current_chat_id is None:
    summaries = storage.list_summaries()
    if summaries:
        _load_chat(summaries[0].id)
    else:
        _start_new_chat()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Sessions")

    summaries = storage.list_summaries()
    chat_options = ["__new__"] + [s.id for s in summaries]

    def _format_chat(opt: str) -> str:
        if opt == "__new__":
            return "+ New chat"
        s = next((x for x in summaries if x.id == opt), None)
        if s is None:
            return opt
        title = s.title if len(s.title) <= 38 else s.title[:37] + "…"
        return f"{title}  ({s.message_count})"

    current_idx = (
        chat_options.index(st.session_state.current_chat_id)
        if st.session_state.current_chat_id in chat_options
        else 0
    )
    selected = st.selectbox(
        "Active chat",
        options=chat_options,
        format_func=_format_chat,
        index=current_idx,
        label_visibility="collapsed",
    )
    if selected == "__new__":
        # Only act if user actually intended to create a new one (i.e. not
        # because the current chat is empty and we landed on __new__ due to
        # missing index). Trigger creation if the current chat already has
        # messages, otherwise stay put.
        if st.session_state.messages:
            _start_new_chat()
            st.rerun()
    elif selected != st.session_state.current_chat_id:
        _save_current_session()
        _load_chat(selected)
        st.rerun()

    nav_cols = st.columns(2)
    if nav_cols[0].button("➕ New", use_container_width=True, key="new_chat_btn"):
        _start_new_chat()
        st.rerun()
    if nav_cols[1].button(
        "🗑 Delete",
        use_container_width=True,
        disabled=not st.session_state.messages,
        key="del_chat_btn",
    ):
        _delete_current_chat()
        st.rerun()

    st.divider()
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
        help=(
            "Prepended as a `system` role message every turn. Defaults to a "
            "code-fencing reminder so models wrap code in ```language … ``` "
            "blocks instead of dumping bare prose. Edit or clear freely."
        ),
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

    with st.expander("Health"):
        health = health_snapshot()
        if not health:
            st.caption("could not reach Ollama")
        else:
            for key, ready in health.items():
                adapter = chat_service.adapters[key]
                icon = "🟢" if ready else "🔴"
                st.caption(f"{icon} `{adapter.model_name}`")
            st.caption("_auto-refreshes every 30 s_")

    if st.session_state.messages:
        st.download_button(
            "📥 Export chat (.md)",
            data=messages_to_markdown(st.session_state.messages),
            file_name=export_filename(),
            mime="text/markdown",
            use_container_width=True,
        )

    if st.button("Clear messages (keep session)", use_container_width=True):
        st.session_state.messages = []
        _clear_transient_state()
        _save_current_session()
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
    if st.session_state._last_user_query_was_short:
        st.caption(
            f"ℹ️ RAG skipped: query under {settings.rag_min_query_len} chars."
        )
    _render_retrievals(st.session_state._last_retrievals)

    if not st.session_state._editing:
        action_cols = st.columns([1, 1, 6])
        if action_cols[0].button("🔄 Regenerate", key="regen_btn"):
            st.session_state.messages.pop()
            st.session_state._regenerate = True
            st.session_state._last_retrievals = []
            st.session_state._last_routing = None
            st.session_state._last_rag_error = None
            st.session_state._last_user_query_was_short = False
            _save_current_session()
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
            del st.session_state.messages[last_user_idx + 1:]
            st.session_state._editing = False
            st.session_state._regenerate = True
            st.session_state._last_retrievals = []
            st.session_state._last_routing = None
            st.session_state._last_rag_error = None
            st.session_state._last_user_query_was_short = False
            _save_current_session()
            st.rerun()
        if edit_cols[1].button("Cancel", key="edit_cancel"):
            st.session_state._editing = False
            st.rerun()


# ---------------------------------------------------------------------------
# Input / generation
# ---------------------------------------------------------------------------

prompt = st.chat_input("Send a message")

# Token counter for what's about to be sent (system + history). Doesn't see
# the chat_input value live — Streamlit's chat_input doesn't expose it — but
# updates after each turn so the user can gauge headroom.
ctx_used = estimate_tokens(st.session_state.system_prompt) + estimate_messages_tokens(
    st.session_state.messages
)
ctx_budget = int(st.session_state.num_ctx)
ctx_pct = min(100, int(ctx_used * 100 / max(1, ctx_budget)))
st.caption(
    f"Conversation so far: ~{ctx_used:,} tokens / {ctx_budget:,} num_ctx ({ctx_pct}%)"
)

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
    last_user_text = next(
        (m.content for m in reversed(st.session_state.messages) if m.role == "user"),
        "",
    )
    st.session_state._last_user_query_was_short = (
        st.session_state.use_rag
        and len(last_user_text.strip()) < settings.rag_min_query_len
    )

    generation_messages = []
    if st.session_state.system_prompt.strip():
        generation_messages.append(
            ChatMessage(role="system", content=st.session_state.system_prompt)
        )
    generation_messages.extend(st.session_state.messages)

if generation_messages is not None:
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
        if st.session_state._last_user_query_was_short:
            st.caption(
                f"ℹ️ RAG skipped: query under {settings.rag_min_query_len} chars."
            )

        _render_retrievals(execution.retrievals)

        stop_placeholder = st.empty()
        stop_placeholder.button("⏹ Stop", key="stop_btn")

        # First-token spinner: replaces with streaming output once the first
        # non-empty chunk arrives.
        spinner_placeholder = st.empty()
        spinner_placeholder.info("⏳ Loading model / waiting for first token…")

        response_placeholder = st.empty()
        full_response = ""
        first_chunk_seen = False

        try:
            with closing(execution.stream) as stream:
                for chunk in stream:
                    if chunk.content:
                        if not first_chunk_seen:
                            spinner_placeholder.empty()
                            first_chunk_seen = True
                        full_response += chunk.content
                        st.session_state._partial_response = full_response
                        response_placeholder.markdown(full_response)

            spinner_placeholder.empty()
            st.session_state._partial_response = ""
            stop_placeholder.empty()

            if not full_response.strip():
                full_response = "_(no response from model)_"
                response_placeholder.warning(full_response)

            st.session_state.messages.append(
                ChatMessage(role="assistant", content=full_response)
            )
            _save_current_session()

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

            st.rerun()

        except Exception as exc:
            logger.exception("Chat request failed")
            spinner_placeholder.empty()
            response_placeholder.error(f"Error: {exc}")
            st.session_state._partial_response = ""
