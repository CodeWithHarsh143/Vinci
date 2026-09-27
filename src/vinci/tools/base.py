from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    success: bool
    data: Any | None = None
    error: str | None = None
    duration_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


class BaseTool(ABC):
    def __init__(self, name: str, description: str, parameters_schema: dict) -> None:
        self.name: str = name
        self.description: str = description
        self.parameters_schema: dict = parameters_schema

    @abstractmethod
    async def execute(self, arguments: dict) -> ToolResult:
        pass
