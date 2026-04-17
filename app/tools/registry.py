from app.schemas.tool import ToolCall, ToolDefinition, ToolResult
from app.tools.base import BaseTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        definition = tool.definition()
        name = definition.name.strip()

        if not name:
            raise ValueError("Tool name cannot be empty.")

        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered.")

        self._tools[name] = tool

    def get(self, name: str) -> BaseTool:
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"Tool '{name}' is not registered.")
        return tool

    def list_definitions(self) -> list[ToolDefinition]:
        return [tool.definition() for tool in self._tools.values()]

    def execute(self, tool_call: ToolCall) -> ToolResult:
        try:
            tool = self.get(tool_call.name)
            tool.validate(tool_call.arguments)
            return tool.execute(tool_call.arguments)
        except Exception as exc:
            return ToolResult(
                name=tool_call.name,
                success=False,
                output=None,
                error=str(exc),
                metadata={},
            )