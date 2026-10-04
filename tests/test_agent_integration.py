import pytest

from vinci.agents.base import Agent
from vinci.tools.filesystem import ReadFileTool
from vinci.tools.registry import ToolsRegistry
from vinci.llm.client import CallResponse, ToolCallRequest


@pytest.fixture
def create_file(tmp_path) -> str:
    file = tmp_path / "test.txt"
    file.write_text("Hello")
    return str(file.relative_to(tmp_path))


@pytest.fixture
def read_file(tmp_path) -> ReadFileTool:
    return ReadFileTool(allowed_root=tmp_path)


@pytest.fixture
def registry(read_file: ReadFileTool) -> ToolsRegistry:
    registry = ToolsRegistry()
    registry.register(read_file)
    return registry


@pytest.mark.asyncio
async def test_agent_executes_tool_then_returns_final_answer(create_file, registry):
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
            assert messages[1]["tool_calls"][0] == {
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "test.txt"}'},
            }

            assert messages[2]["role"] == "tool"
            assert messages[2]["tool_call_id"] == "call_1"
            assert messages[2]["name"] == "read_file"
            assert messages[2]["content"] == '{"content": "Hello"}'

            return CallResponse(
                content="The file contains Hello",
                tool_calls=[],
            )

    llm = FakeLLM()
    agent = Agent(llm=llm, registry=registry)
    result = await agent.run("Read test.txt")

    assert result.success is True
    assert result.final_answer == "The file contains Hello"
    assert result.total_iterations == 2
    assert result.total_tool_calls == 1
    assert result.error is None


@pytest.mark.asyncio
async def test_agent_executes_tool_then_returns_file_not_foundError(registry):
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
            assert messages[1]["tool_calls"][0] == {
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "test.txt"}'},
            }

            assert messages[2]["role"] == "tool"
            assert messages[2]["tool_call_id"] == "call_1"
            assert messages[2]["name"] == "read_file"
            assert messages[2]["content"] == '{"content": {"error": "File not found"}}'

            return CallResponse(
                content="The file contains Hello",
                tool_calls=[],
            )

    llm = FakeLLM()
    agent = Agent(llm=llm, registry=registry)
    result = await agent.run("Read test.txt")

    assert result.success is True
    assert result.final_answer == "The file contains Hello"
    assert result.total_iterations == 2
    assert result.total_tool_calls == 1
    assert result.error is None
