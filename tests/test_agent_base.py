import pytest
import asyncio
from vinci.agents.base import Agent
from vinci.llm.client import CallResponse, ToolCallRequest
from vinci.tools.base import ToolResult


# ----------------------------------------------------------------------
# Helper: run the agent and obtain the result from the queue.
# ----------------------------------------------------------------------
def _run_and_get_result(agent, task, timeout_sec=None):
    agent.run(task, timeout_sec=timeout_sec)  # run() starts the thread and returns immediately
    agent.join()                  # block until the thread finishes
    return agent.result_queue.get()   # get the AgentResult that the thread placed in the queue


# ----------------------------------------------------------------------
# Test 1 – agent returns a final answer without any tool calls
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_agent_returns_final_answer():
    class FakeLLM:
        async def call(self, messages, tools):
            return CallResponse(
                content="Task completed successfully",
                tool_calls=[],
            )

    class FakeRegistry:
        def list_schemas(self):
            return []

        async def execute(self, tool_name, args):
            raise AssertionError("No tool should be executed")

    agent = Agent(llm=FakeLLM(), registry=FakeRegistry())
    result = _run_and_get_result(agent, "Do something")

    assert result.success is True
    assert result.final_answer == "Task completed successfully"
    assert result.total_iterations == 1
    assert result.total_tool_calls == 0
    assert result.error is None
    assert result.duration_ms >= 0


# ----------------------------------------------------------------------
# Test 2 – agent executes a tool then returns a final answer
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_agent_executes_tool_then_returns_final_answer():
    class FakeLLM:
        def __init__(self):
            self.calls = 0

        async def call(self, messages, tools):
            self.calls += 1

            if self.calls == 1:
                return CallResponse(
                    content=None,
                    tool_calls=[
                        ToolCallRequest(
                            id="call_1",
                            name="read_file",
                            arguments={"path": "test.txt"},
                        )
                    ],
                )

            assert len(messages) == 3

            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "Read test.txt"

            assert messages[1]["role"] == "assistant"
            assert messages[1]["tool_calls"]

            assert messages[2]["role"] == "tool"
            assert messages[2]["tool_call_id"] == "call_1"

            return CallResponse(
                content="The file contains Hello",
                tool_calls=[],
            )

    class FakeRegistry:
        def list_schemas(self):
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                    },
                }
            ]

        async def execute(self, tool_name, args):
            assert tool_name == "read_file"
            assert args == {"path": "test.txt"}

            return ToolResult(
                success=True,
                data="Hello",
                error=None,
            )

    agent = Agent(llm=FakeLLM(), registry=FakeRegistry())
    result = _run_and_get_result(agent, "Read test.txt")

    assert result.success is True
    assert result.final_answer == "The file contains Hello"
    assert result.total_iterations == 2
    assert result.total_tool_calls == 1
    assert result.error is None


# ----------------------------------------------------------------------
# Test 3 – agent handles a tool error and returns the error message
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_agent_handles_tool_error():
    class FakeLLM:
        def __init__(self):
            self.calls = 0

        async def call(self, messages, tools):
            self.calls += 1

            if self.calls == 1:
                return CallResponse(
                    content=None,
                    tool_calls=[
                        ToolCallRequest(
                            id="call_1",
                            name="read_file",
                            arguments={"path": "missing.txt"},
                        )
                    ],
                )

            tool_message = messages[-1]

            assert tool_message["role"] == "tool"
            assert "File not found" in tool_message["content"]

            return CallResponse(
                content="The file does not exist.",
                tool_calls=[],
            )

    class FakeRegistry:
        def list_schemas(self):
            return []

        async def execute(self, tool_name, args):
            return ToolResult(
                success=False,
                data=None,
                error="File not found",
            )

    agent = Agent(llm=FakeLLM(), registry=FakeRegistry())
    result = _run_and_get_result(agent, "Read missing.txt")

    assert result.success is True
    assert result.final_answer == "The file does not exist."
    assert result.total_iterations == 2
    assert result.total_tool_calls == 1


# ----------------------------------------------------------------------
# Test 4 – agent can execute two different tools in one iteration
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_agent_can_execute_multiple_tools_in_one_iteration():
    class FakeLLM:
        def __init__(self):
            self.calls = 0

        async def call(self, messages, tools):
            self.calls += 1

            if self.calls == 1:
                return CallResponse(
                    content=None,
                    tool_calls=[
                        ToolCallRequest(
                            id="call_1",
                            name="tool_a",
                            arguments={"value": 1},
                        ),
                        ToolCallRequest(
                            id="call_2",
                            name="tool_b",
                            arguments={"value": 2},
                        ),
                    ],
                )

            assert len(messages) == 4

            return CallResponse(
                content="Both tools completed.",
                tool_calls=[],
            )

    class FakeRegistry:
        def list_schemas(self):
            return []

        async def execute(self, tool_name, args):
            if tool_name == "tool_a":
                return ToolResult(
                    success=True,
                    data="result_a",
                    error=None,
                )

            if tool_name == "tool_b":
                return ToolResult(
                    success=True,
                    data="result_b",
                    error=None,
                )

            raise AssertionError(f"Unexpected tool: {tool_name}")

    agent = Agent(llm=FakeLLM(), registry=FakeRegistry())
    result = _run_and_get_result(agent, "Run both tools")

    assert result.success is True
    assert result.final_answer == "Both tools completed."
    assert result.total_iterations == 2
    assert result.total_tool_calls == 2


# ----------------------------------------------------------------------
# Test 5 – agent stops after max_iterations
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_agent_stops_after_max_iterations():
    class FakeLLM:
        async def call(self, messages, tools):
            return CallResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="some_tool",
                        arguments={},
                    )
                ],
            )

    class FakeRegistry:
        def list_schemas(self):
            return []

        async def execute(self, tool_name, args):
            return ToolResult(
                success=True,
                data="result",
                error=None,
            )

    agent = Agent(llm=FakeLLM(), registry=FakeRegistry(), max_iterations=3)
    result = _run_and_get_result(agent, "Keep working")

    assert result.success is True
    assert result.error is None
    assert result.total_iterations == 3
    assert result.total_tool_calls == 3
    assert result.final_answer is None
    assert result.duration_ms >= 0


# ----------------------------------------------------------------------
# Test 6 – agent respects the global time_limit
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_agent_timeout():
    class FakeLLM:
        async def call(self, messages, tools):
            return CallResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="some_tool",
                        arguments={},
                    )
                ],
            )

    class FakeRegistry:
        def list_schemas(self):
            return []

        async def execute(self, tool_name, args):
            return ToolResult(
                success=True,
                data="result",
                error=None,
            )

    agent = Agent(llm=FakeLLM(), registry=FakeRegistry(), time_limit=0)
    result = _run_and_get_result(agent, "Do something")

    assert result.success is False
    assert result.error == "Timeout Error"
    assert result.total_iterations == 0
    assert result.total_tool_calls == 0
    assert result.duration_ms >= 0


@pytest.mark.asyncio
async def test_cancel():
    class FakeLLM:
        async def call(self, messages, tools):
            return CallResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="some_tool",
                        arguments={},
                    )
                ],
            )

    class FakeRegistry:
        def list_schemas(self):
            return []

        async def execute(self, tool_name, args):
            return ToolResult(
                success=True,
                data="result",
                error=None,
            )

    agent = Agent(llm=FakeLLM, registry=FakeRegistry)
    agent.run("some task")
    await asyncio.sleep(0.2)  # let the loop start
    agent.cancel()  # request cancellation
    agent.join()  # wait for thread to stop
    result = agent.result_queue.get()
    assert result.success is False
    assert result.error == "cancelled"
