from vinci.tools.base import BaseTool, ToolResult
from vinci.tools.filesystem import ReadFileTool
from vinci.tools.registry import ToolsRegistry

import pytest


class EchoTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="echo",
            description="Echoes back the input text",
            parameters_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to echo back"}
                },
                "required": ["text"],
            },
        )

    async def execute(self, arguments: dict) -> ToolResult:
        text = arguments.get("text", "")
        return ToolResult(success=True, data=text)


@pytest.fixture
def registry():
    return ToolsRegistry()


@pytest.fixture
def echo_tool():
    return EchoTool()


def test_register_and_get(registry, echo_tool):
    registry.register(echo_tool)

    assert registry.get("echo") is echo_tool


def test_get_unknown_returns_none(registry):
    assert registry.get("does_not_exist") is None


def test_register_duplicate_raises(registry, echo_tool):
    registry.register(echo_tool)

    with pytest.raises(ValueError, match="already register"):
        registry.register(echo_tool)


def test_list_schemas_empty_when_no_tools(registry):
    assert registry.list_schemas() == {}


def test_list_schemas_contains_registered_tool(registry, echo_tool):
    registry.register(echo_tool)

    schemas = registry.list_schemas()

    assert "echo" in schemas
    entry = schemas["echo"]
    assert entry["type"] == "function"
    assert entry["function"]["name"] == "echo"
    assert entry["function"]["description"] == "Echoes back the input text"
    assert (
        entry["function"]["parameters_schema"]
        == echo_tool.parameters_schema
    )


@pytest.mark.asyncio
async def test_execute_valid_tool_success(registry, echo_tool):
    registry.register(echo_tool)

    result = await registry.execute("echo", {"text": "hello"})

    assert result.success is True
    assert result.data == "hello"


@pytest.mark.asyncio
async def test_execute_unknown_tool_returns_error_with_available(
    registry, echo_tool
):
    registry.register(echo_tool)

    result = await registry.execute("foobar", {})

    assert result.success is False
    assert "Unknown tool: foobar" in result.error
    assert "echo" in result.error


@pytest.mark.asyncio
async def test_execute_propagates_tool_failure(tmp_path):
    registry = ToolsRegistry()
    registry.register(ReadFileTool(tmp_path))

    result = await registry.execute("read_file", {"path": "missing.txt"})

    assert result.success is False
    assert result.error == "File not found"


@pytest.mark.asyncio
async def test_execute_real_read_file_via_registry(tmp_path):
    registry = ToolsRegistry()
    registry.register(ReadFileTool(tmp_path))

    target = tmp_path / "hello.txt"
    target.write_text("Hello, Vinci")

    result = await registry.execute("read_file", {"path": "hello.txt"})

    assert result.success is True
    assert result.data == "Hello, Vinci"
