from ast import Return
from re import escape
from typing_extensions import deprecated
from unittest.mock import AsyncMock, Mock
import pytest
from vinci.schemas.agentsSchema import PlanModel
from vinci.llm.client import LLMClient, client
import asyncio


@pytest.fixture
def service():
    return LLMClient()


@pytest.mark.asyncio
async def test_generate_success(service, monkeypatch):
    response = Mock()
    response.choices = [Mock(message=Mock(content="Test is SuccessFull"))]

    create = AsyncMock(return_value=response)
    monkeypatch.setattr(
        client.chat.completions,
        "create",
        create,
    )
    result = await service.generate("Hello")

    assert result == "Test is SuccessFull"
    create.assert_awaited_once_with(
        model="gemini-3.6-flash",
        messages="Hello",
    )


@pytest.mark.asyncio
async def test_generate_api_failed(service, monkeypatch):
    service.generate = AsyncMock(side_effect=RuntimeError("LLM generation failed"))

    with pytest.raises(RuntimeError, match="LLM generation failed"):
        await service.generate("Hello")


@pytest.mark.asyncio
async def test_generate__cancelled_error(service):
    service.generate = AsyncMock(side_effect=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await service.generate("Call")


@pytest.mark.asyncio
async def test_generate_structured_success(service):
    service.generate = AsyncMock(
        return_value='{"plan":["test1","test2"],"dependencies":[[],["test"]],"estimated_time_minutes":35}'
    )
    result = await service.generate_structured("Call", PlanModel)

    assert isinstance(result, PlanModel)
    assert result.plan == ["test1", "test2"]
    assert result.dependencies == [[], ["test"]]
    assert result.estimated_time_minutes == 35
    service.generate.assert_awaited_once_with("Call")


@pytest.mark.asyncio
async def test_generate_structured_retry_then_success(service):
    service.generate = AsyncMock(
        side_effect=[
            "Invalid JSON",
            '{"plan":["SuccessFull"],"dependencies":[[]],"estimated_time_minutes":2}',
        ]
    )
    result = await service.generate_structured("Call", PlanModel)

    assert isinstance(result, PlanModel)
    assert result.plan == ["SuccessFull"]
    assert result.dependencies == [[]]
    assert result.estimated_time_minutes == 2
    assert service.generate.await_count == 2


@pytest.mark.asyncio
async def test_generate_structured_invalid_json(service):
    service.generate = AsyncMock(return_value="Invalid JSON")
    with pytest.raises(ValueError, match="Invalid JSON"):
        await service.generate_structured("Call", PlanModel)
    assert service.generate.await_count == 3


@pytest.mark.asyncio
async def test_generate_structured_invalid_structure(service):
    service.generate = AsyncMock(
        return_value='{"Plan":[],"dependencies":[],"estimated_time_minutes":"2"}'
    )

    with pytest.raises(ValueError, match="Invalid Structure from the LLM"):
        await service.generate_structured("Call", PlanModel)
    assert service.generate.await_count == 3


@pytest.mark.asyncio
async def test_generate_structured_invlaid_structure_then_success(service):
    service.generate = AsyncMock(
        side_effect=[
            '{"Plan":[],"dependencies":[],"estimated_time_minutes":"2"}',
            '{"plan":[],"dependencies":[],"estimated_time_minutes":"2"}',
        ]
    )
    result = await service.generate_structured("Call", PlanModel)

    assert isinstance(result, PlanModel)
    assert result.plan == []
    assert result.dependencies == []
    assert result.estimated_time_minutes == 2
    assert service.generate.await_count == 2


@pytest.mark.asyncio
async def test_generate_structured_api_failed(service):
    service.generate = AsyncMock(side_effect=RuntimeError("LLM generation failed"))

    with pytest.raises(RuntimeError, match="LLM generation failed"):
        await service.generate_structured("Call", PlanModel)


@pytest.mark.asyncio
async def test_generate_structured_cancelled_error(service):
    service.generate = AsyncMock(side_effect=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await service.generate_structured("Call", PlanModel)
