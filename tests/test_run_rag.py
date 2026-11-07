import asyncio
from vinci.agents.base import Agent
from vinci.llm.client import CallResponse, ToolCallRequest
import pytest
from vinci.agents.base import Agent
from vinci.llm.client import CallResponse, ToolCallRequest


# ----------------------------------------------------------------------
# Helper that runs the agent in the new threaded way and returns the result
# ----------------------------------------------------------------------
def _run_and_get_result(agent, task, timeout_sec=None):
    """Helper used by the tests.
    - agent.run(task) starts the thread,
    - agent.join() waits for the thread to finish,
    - agent.result_queue.get() returns the AgentResult."""
    agent.run(task)  # starts the thread, returns immediately
    agent.join()  # blocks until the thread finishes
    return (
        agent.result_queue.get()
    )  # gets the AgentResult that the thread placed in the queue


# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# 1️⃣  test_agent_rag_basic
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_agent_rag_basic():
    """Very small RAG test – the retriever returns a single snippet and the LLM
    replies with a final answer (no tool call)."""

    # ------------------------------------------------------------------
    # Fake LLM that always returns a final‑answer (no tool calls)
    # ------------------------------------------------------------------
    class FakeLLM:
        async def call(self, messages, tools):
            return CallResponse(
                content='The function greet(user) returns "Hello, " + user.',
                tool_calls=[],
            )

    class FakeRegistry:
        def list_schemas(self):
            return []

        async def execute(self, tool_name, args):
            raise AssertionError("No tool should be executed")

    # ------------------------------------------------------------------
    # Fake Retriever that returns a single snippet
    # ------------------------------------------------------------------
    class FakeRetriever:
        def query(self, query):
            return [
                {
                    "path": "demo.py",
                    "excerpt": 'def greet(user): return "Hello, " + user',
                }
            ]

    agent = Agent(llm=FakeLLM(), registry=FakeRegistry())
    # Monkeypatch the retriever into the agent so run_rag can use it
    agent.retriever = FakeRetriever()

    # ------------------------------------------------------------------
    # Run the RAG helper and check the result
    # ------------------------------------------------------------------
    result = await agent.run_rag("what is greet?")
    assert result.success is True
    assert result.final_answer == 'The function greet(user) returns "Hello, " + user.'
    assert result.total_iterations == 1
    assert result.total_tool_calls == 0


# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# 2️⃣  test_agent_rag_no_snippets
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_agent_rag_no_snippets():
    """When the retriever returns an empty list the agent should still return
    an AgentResult (success=False, error set)."""

    class FakeLLM:
        async def call(self, messages, tools):
            return CallResponse(
                content=None,
                tool_calls=[],
            )

    class FakeRegistry:
        def list_schemas(self):
            return []

        async def execute(self, tool_name, args):
            raise AssertionError("No tool should be executed")

    class FakeRetriever:
        def query(self, query):
            return []  # empty list → no snippets

    agent = Agent(llm=FakeLLM(), registry=FakeRegistry())
    agent.retriever = FakeRetriever()
    result = await agent.run_rag("what is the meaning of life?")
    assert result.success is False
    assert result.error is not None


# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# 3️⃣  test_agent_rag_cancel
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_agent_rag_cancel():

    class FakeLLM:
        async def call(self, messages, tools):
            pytest.fail("LLM should not be called after cancellation")

    class FakeRegistry:
        def list_schemas(self):
            return []

        async def execute(self, tool_name, args):
            pytest.fail("No tool should be executed")

    class FakeRetriever:
        def query(self, query):
            return [
                {
                    "path": "demo.py",
                    "excerpt": "def greet(): pass",
                }
            ]

    agent = Agent(llm=FakeLLM(), registry=FakeRegistry())
    agent.retriever = FakeRetriever()

    agent.cancel()

    result = await agent.run_rag("some query")

    assert result.success is False
    assert result.error is not None
