from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app.schemas.chat import ChatMessage, ChatRequest
from app.services.chat_service import ChatService
from app.utils.logger import setup_logger


logger = setup_logger()
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

with st.sidebar:
    st.header("Settings")

    selection_options = ["auto", "granite", "gemma"]
    selected_option = st.selectbox(
        "Choose model mode",
        options=selection_options,
        index=selection_options.index(st.session_state.model_selection),
    )
    st.session_state.model_selection = selected_option

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
                "Chat completed | selection=%s | resolved_model=%s | prompt_chars=%s | response_chars=%s",
                st.session_state.model_selection,
                execution.selected_model,
                len(prompt),
                len(full_response),
            )

            if execution.routing_decision is not None:
                logger.info(
                    "Routing decision | model=%s | score=%s | reason=%s",
                    execution.selected_model,
                    execution.routing_decision.complexity_score,
                    execution.routing_decision.reason,
                )

        except Exception as exc:
            logger.exception("Chat request failed")
            response_placeholder.error(f"Error: {exc}")