from vinci.tools.base import BaseTool, ToolResult


class ToolsRegistry:
    def __init__(self) -> None:
        self.tools = {}
        self.available = []

    def register(self, tool: BaseTool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"Tool {tool.name} is already register")
        self.available.append(tool.name)
        self.tools[tool.name] = tool

    def list_schemas(self) -> dict:
        schemas = {}
        for tool_name, tool in self.tools.items():
            schemas[tool_name] = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters_schema": tool.parameters_schema,
                },
            }
        return schemas

    def get(self, tool_name: str) -> BaseTool | None:
        if tool_name in self.tools:
            return self.tools[tool_name]

    async def execute(self, tool_name: str, args: dict) -> ToolResult:
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Unknown tool: {tool_name}. Available:{''.join(self.available)}",
            )
        return await tool.execute(arguments=args)
