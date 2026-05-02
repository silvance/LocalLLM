from abc import ABC, abstractmethod

from app.schemas.tools import ToolDefinition, ToolResult


class BaseTool(ABC):
    @abstractmethod
    def definition(self) -> ToolDefinition:
        pass

    def validate(self, arguments: dict) -> None:
        required_params = self.definition().parameters.get("required", [])
        properties = self.definition().parameters.get("properties", {})

        for param in required_params:
            if param not in arguments:
                raise ValueError(f"Missing required parameter: {param}")

        for arg_name in arguments:
            if properties and arg_name not in properties:
                raise ValueError(f"Unexpected parameter: {arg_name}")

    @abstractmethod
    def execute(self, arguments: dict) -> ToolResult:
        pass