"""Smoke tests for the tools subsystem (the imports were broken until Step 0)."""
import pytest

from app.schemas.tools import ToolCall
from app.tools.builtin_tools import EchoTextTool, GetCurrentTimeTool
from app.tools.registry import ToolRegistry


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(GetCurrentTimeTool())
    reg.register(EchoTextTool())
    return reg


def test_listing_definitions(registry: ToolRegistry) -> None:
    names = [d.name for d in registry.list_definitions()]
    assert "get_current_time" in names
    assert "echo_text" in names


def test_register_duplicate_raises() -> None:
    reg = ToolRegistry()
    reg.register(EchoTextTool())
    with pytest.raises(ValueError, match="already registered"):
        reg.register(EchoTextTool())


def test_execute_round_trip(registry: ToolRegistry) -> None:
    result = registry.execute(ToolCall(name="echo_text", arguments={"text": "hello"}))
    assert result.success is True
    assert result.output == "hello"


def test_missing_required_param_returns_error(registry: ToolRegistry) -> None:
    result = registry.execute(ToolCall(name="echo_text", arguments={}))
    assert result.success is False
    assert "text" in (result.error or "")


def test_unexpected_param_returns_error(registry: ToolRegistry) -> None:
    result = registry.execute(
        ToolCall(name="echo_text", arguments={"text": "x", "extra": "y"})
    )
    assert result.success is False
    assert "extra" in (result.error or "")


def test_unknown_tool_returns_error(registry: ToolRegistry) -> None:
    result = registry.execute(ToolCall(name="nonexistent", arguments={}))
    assert result.success is False


def test_empty_properties_rejects_unexpected_args(registry: ToolRegistry) -> None:
    """get_current_time declares no properties — supplying any argument must
    fail validation. Prior to the fix this was silently accepted."""
    result = registry.execute(
        ToolCall(name="get_current_time", arguments={"surprise": "yes"})
    )
    assert result.success is False
    assert "surprise" in (result.error or "")
