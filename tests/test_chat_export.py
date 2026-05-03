from app.schemas.chat import ChatMessage
from app.utils.chat_export import export_filename, messages_to_markdown


def test_empty_conversation_renders_placeholder() -> None:
    out = messages_to_markdown([])
    assert "empty conversation" in out.lower()


def test_round_trips_user_assistant_alternation() -> None:
    msgs = [
        ChatMessage(role="user", content="Q1"),
        ChatMessage(role="assistant", content="A1"),
        ChatMessage(role="user", content="Q2"),
    ]
    out = messages_to_markdown(msgs)
    assert "## User" in out
    assert "## Assistant" in out
    # Order preserved
    assert out.index("Q1") < out.index("A1") < out.index("Q2")


def test_renders_system_message_distinctly() -> None:
    msgs = [
        ChatMessage(role="system", content="You are a pentester."),
        ChatMessage(role="user", content="hi"),
    ]
    out = messages_to_markdown(msgs)
    assert "## System" in out


def test_export_filename_has_md_extension_and_timestamp() -> None:
    name = export_filename()
    assert name.endswith(".md")
    assert "localllm-chat" in name
    assert len(name) > len("localllm-chat-.md")
