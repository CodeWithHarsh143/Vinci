from vinci.agents.base import Agent
from vinci.llm.client import LLMClient
from vinci.tools.registry import ToolsRegistry

llm = LLMClient(model="gemini-3.8-flash")
registry = ToolsRegistry()
agent = Agent(llm=llm, registry=registry)

# Test the “@search:” path
agent.run("@search how does login user work?")
agent.join()
result = agent.result_queue.get()
assert result.success is True
