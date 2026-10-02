import pytest

from vinci.agents.base import Agent, AgentResult
from vinci.llm.client import CallResponse, ToolCallRequest
from vinci.tools.base import ToolResult


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

    agent = Agent(
        llm=FakeLLM(),
        registry=FakeRegistry(),
    )

    result = await agent.run("Do something")

    assert result.success is True
    assert result.final_answer == "Task completed successfully"
    assert result.total_iterations == 1
    assert result.total_tool_calls == 0
    assert result.error is None
    assert result.durations_ms >= 0


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

    llm = FakeLLM()

    agent = Agent(
        llm=llm,
        registry=FakeRegistry(),
    )

    result = await agent.run("Read test.txt")

    assert result.success is True
    assert result.final_answer == "The file contains Hello"
    assert result.total_iterations == 2
    assert result.total_tool_calls == 1
    assert result.error is None


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

    agent = Agent(
        llm=FakeLLM(),
        registry=FakeRegistry(),
    )

    result = await agent.run("Read missing.txt")

    assert result.success is True
    assert result.final_answer == "The file does not exist."
    assert result.total_iterations == 2
    assert result.total_tool_calls == 1


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

    agent = Agent(
        llm=FakeLLM(),
        registry=FakeRegistry(),
    )

    result = await agent.run("Run both tools")

    assert result.success is True
    assert result.final_answer == "Both tools completed."
    assert result.total_iterations == 2
    assert result.total_tool_calls == 2


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

    agent = Agent(
        llm=FakeLLM(),
        registry=FakeRegistry(),
        max_iterations=3,
    )

    result = await agent.run("Keep working")

    assert result.success is False
    assert result.error == "Maximum iterations exceeded"
    assert result.total_iterations == 3
    assert result.total_tool_calls == 3
    assert result.final_answer is None
    assert result.durations_ms >= 0


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

    agent = Agent(
        llm=FakeLLM(),
        registry=FakeRegistry(),
        time_limit=0,
    )

    result = await agent.run("Do something")

    assert result.success is False
    assert result.error == "Timeout Error"
    assert result.total_iterations == 0
    assert result.total_tool_calls == 0
    assert result.durations_ms >= 0
