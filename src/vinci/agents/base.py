from dataclasses import dataclass
import time
from vinci.llm.client import LLMClient
from vinci.tools.registry import ToolsRegistry


@dataclass
class AgentResult:
    success: bool
    error: str | None
    total_itrations: int
    total_tool_calls: int
    final_answer: str | None
    durations: float = 0.0


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

    async def run(self, task: str) -> AgentResult:
        start = time.pref_counter()
        messages: list = []
        itrations: int = 0
        tool_calls: int = 0

        while itrations < self.max_itrations:
            pass
