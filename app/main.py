from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app.config import get_settings
from app.schemas.chat import ChatMessage, ChatRequest
from app.services.chat_service import ChatService
from app.utils.logger import setup_logger


logger = setup_logger()
settings = get_settings()
chat_service = ChatService()

st.set_page_config(
    page_title="LocalLLM",
    page_icon="🤖",
    layout="wide",
)

st.title("LocalLLM")
st.caption("Local Streamlit chat app powered by Ollama")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "model_selection" not in st.session_state:
    st.session_state.model_selection = "auto"
if "use_rag" not in st.session_state:
    st.session_state.use_rag = settings.rag_enabled

with st.sidebar:
    st.header("Settings")

    selection_options = ["auto", "granite", "gemma", "qwen"]
    selected_option = st.selectbox(
        "Choose model mode",
        options=selection_options,
        index=selection_options.index(st.session_state.model_selection),
        help="auto routes by prompt content (code → qwen, general → gemma, simple → granite).",
    )
    st.session_state.model_selection = selected_option

    rag_count = chat_service.rag.count()
    rag_label = f"Use RAG ({rag_count} chunks)" if rag_count else "Use RAG (no index — run scripts/build_index.py)"
    st.session_state.use_rag = st.checkbox(
        rag_label,
        value=st.session_state.use_rag and rag_count > 0,
        disabled=rag_count == 0,
        help=f"Retrieval-augmented generation using `{settings.rag_embedding_model}`.",
    )

    if st.button("Check model health"):
        try:
            health = chat_service.health_check()
            for model_name, is_ready in health.items():
                if is_ready:
                    st.success(f"{model_name}: ready")
                else:
                    st.error(f"{model_name}: not available")
        except Exception as exc:
            logger.exception("Health check failed")
            st.error(f"Health check failed: {exc}")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message.role):
        st.markdown(message.content)

prompt = st.chat_input("Send a message")

if prompt:
    user_message = ChatMessage(role="user", content=prompt)
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(prompt)

    request = ChatRequest(
        messages=st.session_state.messages,
        model_key="granite",
        stream=True,
    )

    execution = chat_service.stream_chat(
        request=request,
        selection=st.session_state.model_selection,
        use_rag=st.session_state.use_rag,
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

        if execution.retrievals:
            with st.expander(f"📚 Retrieved {len(execution.retrievals)} sources", expanded=False):
                for r in execution.retrievals:
                    st.markdown(
                        f"**`{r.source_id}:{r.file_path}`** "
                        f"({r.language or 'n/a'}, score={r.score:.2f})"
                    )
                    st.code(r.document[:600] + ("…" if len(r.document) > 600 else ""))

        response_placeholder = st.empty()
        full_response = ""

        try:
            for chunk in execution.stream:
                if chunk.content:
                    full_response += chunk.content
                    response_placeholder.markdown(full_response)

            if not full_response.strip():
                full_response = "No response was returned by the model."
                response_placeholder.warning(full_response)

            assistant_message = ChatMessage(role="assistant", content=full_response)
            st.session_state.messages.append(assistant_message)

            logger.info(
                "Chat completed | selection=%s | model=%s | rag=%s | retrievals=%s | prompt_chars=%s | response_chars=%s",
                st.session_state.model_selection,
                execution.selected_model,
                st.session_state.use_rag,
                len(execution.retrievals),
                len(prompt),
                len(full_response),
            )

            if execution.routing_decision is not None:
                logger.info(
                    "Routing | model=%s | score=%s | reason=%s",
                    execution.selected_model,
                    execution.routing_decision.complexity_score,
                    execution.routing_decision.reason,
                )

        except Exception as exc:
            logger.exception("Chat request failed")
            response_placeholder.error(f"Error: {exc}")
