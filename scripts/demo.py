"""End-to-end Vinci demo: user -> Agent -> real tools -> real LLM -> answer.

Usage:
    .venv/bin/python scripts/demo.py
"""

import asyncio
import os
from pathlib import Path

from vinci.agents.base import Agent
from vinci.llm.client import LLMClient
from vinci.tools.filesystem import ReadFileTool
from vinci.tools.registry import ToolsRegistry
from vinci.tools.searchTool import SearchCodeTool

SANDBOX = Path("/tmp/vinci_demo")
MODEL = os.environ.get("VINCI_MODEL", "gemini-3.8-flash")

SAMPLE_FILES = {
    "app.py": "def login(user):\n    return f'Welcome, {user}!'\n",
    "notes.txt": "Vinci demo sandbox.\nContains two sample files.\n",
}


async def main() -> int:
    SANDBOX.mkdir(parents=True, exist_ok=True)
    for name, content in SAMPLE_FILES.items():
        (SANDBOX / name).write_text(content)

    llm = LLMClient(model=MODEL)
    print(f"model:          {MODEL}")
    registry = ToolsRegistry()
    registry.register(ReadFileTool(SANDBOX))
    registry.register(SearchCodeTool(SANDBOX))
    agent = Agent(llm=llm, registry=registry, max_iterations=5)

    # ------------------------------------------------------------------
    # The agent runs in a daemon thread.  We start it, then wait for it to finish
    # and pull the result from the queue (the pattern required by the threaded
    # agent implementation).
    # ------------------------------------------------------------------
    agent.run("What does app.py do?")
    agent.join()  # blocks until the thread finishes
    try:
        result = agent.result_queue.get()  # blocking get; will succeed after join()
    except Exception as exc:  # catch any unexpected queue errors
        print(f"Demo failed: cannot retrieve result ({exc})")
        return 1

    # ------------------------------------------------------------------
    # Now we can inspect the result exactly as before.
    # ------------------------------------------------------------------
    print(f"success:        {result.success}")
    print(f"iterations:     {result.total_iterations}")
    print(f"tool calls:     {result.total_tool_calls}")
    print(f"duration (ms):  {result.duration_ms:.1f}")
    print(f"error:          {result.error}")
    print(f"final answer:\n{result.final_answer}")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
