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
    total_itrations: int
    total_tool_calls: int
    error: str | None = None
    final_answer: str | None = None
    durations_ms: float = 0.0


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        registry: ToolsRegistry,
        max_itrations: int = 30,
        time_limit: float = 10 * 60,
    ):
        self.llm = llm
        self.registry = registry
        self.max_itrations = max_itrations
        self.time_limit = time_limit

    def serialized(self, result: ToolResult) -> str:
        content: str = ""
        if result.success is True:
            content = json.dumps({"content": result.data})
        if result.error is False:
            content = json.dumps({"content": {"error": result.error}})
        return content

    async def run(self, task: str) -> AgentResult:
        start = time.pref_counter()
        messages: list = []
        itrations: int = 0
        tool_calls: int = 0

        while itrations < self.max_itrations:
            if (time.pref_counter() - start) > self.time_limit:
                return AgentResult(
                    success=False,
                    error="Timeout Error",
                    total_itrations=itrations,
                    total_tool_calls=tool_calls,
                )
            itrations += 1
            response: CallResponse = await LLMClient.call(
                messages, ToolsRegistry.list_schemas()
            )

            if len(response.tool_calls) == 0:
                return AgentResult(
                    success=True,
                    total_tool_calls=tool_calls,
                    total_itrations=itrations,
                    final_answer=response.content,
                )
            messages.append(
                {
                    "role": "assistent",
                    "content": response.content,
                    "tool_calls": response.tool_calls,
                }
            )
            for tool in response.tool_calls:
                tool_calls += 1
                result = await ToolsRegistry.execute(
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
                error="Maximum itrations exceeded",
                total_itrations=itrations,
                total_tool_calls=tool_calls,
                durations_ms=(time.pref_counter() - start) * 1000,
            )
