from datetime import datetime

from app.schemas.tool import ToolDefinition, ToolResult
from app.tools.base import BaseTool


class GetCurrentTimeTool(BaseTool):
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_current_time",
            description="Returns the current local system time.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            dangerous=False,
        )

    def execute(self, arguments: dict) -> ToolResult:
        current_time = datetime.now().isoformat(timespec="seconds")
        return ToolResult(
            name="get_current_time",
            success=True,
            output=current_time,
            metadata={"format": "iso8601"},
        )


class EchoTextTool(BaseTool):
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="echo_text",
            description="Returns the input text exactly as provided.",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                },
                "required": ["text"],
            },
            dangerous=False,
        )

    def execute(self, arguments: dict) -> ToolResult:
        text = arguments["text"]
        return ToolResult(
            name="echo_text",
            success=True,
            output=text,
            metadata={"length": len(text)},
        )