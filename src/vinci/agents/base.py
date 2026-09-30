from dataclasses import dataclass
import json
import time
from vinci.llm.client import LLMClient, CallResponse
from vinci.tools.base import ToolResult
from vinci.tools.registry import ToolsRegistry

MAX_RESULT_CHAR: int = 5000


@dataclass
class AgentResult:
    success: bool
    total_iterations: int
    total_tool_calls: int
    error: str | None = None
    final_answer: str | None = None
    durations_ms: float = 0.0


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        registry: ToolsRegistry,
        max_iterations: int = 30,
        time_limit: float = 10 * 60,
    ):
        self.llm = llm
        self.registry = registry
        self.max_iterations = max_iterations
        self.time_limit = time_limit

    def serialized(self, result: ToolResult) -> str:
        if result.success is True:
            return json.dumps({"content": result.data})
        return json.dumps({"content": {"error": result.error}})

    async def run(self, task: str) -> AgentResult:
        start = time.perf_counter()
        messages: list = [{"role": "user", "content": task}]
        iterations: int = 0
        tool_calls: int = 0

        while iterations < self.max_iterations:
            if (time.perf_counter() - start) > self.time_limit:
                return AgentResult(
                    success=False,
                    error="Timeout Error",
                    total_iterations=iterations,
                    total_tool_calls=tool_calls,
                    durations_ms=(time.perf_counter() - start) * 1000,
                )
            iterations += 1
            response: CallResponse = await self.llm.call(
                messages, self.registry.list_schemas()
            )

            if len(response.tool_calls) == 0:
                return AgentResult(
                    success=True,
                    total_tool_calls=tool_calls,
                    total_iterations=iterations,
                    final_answer=response.content,
                )
            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": response.tool_calls,
                }
            )
            for tool in response.tool_calls:
                tool_calls += 1
                result = await self.registry.execute(
                    tool_name=tool.name, args=tool.arguments
                )
                content = self.serialized(result)
                if len(content) > MAX_RESULT_CHAR:
                    content = content[:MAX_RESULT_CHAR] + "..truncated"
                messages.append(
                    {
                        "role": "tool",
                        "tool_id": tool.id,
                        "name": tool.name,
                        "content": content,
                    }
                )
        return AgentResult(
            success=False,
            error="Maximum iterations exceeded",
            total_iterations=iterations,
            total_tool_calls=tool_calls,
            durations_ms=(time.perf_counter() - start) * 1000,
        )
