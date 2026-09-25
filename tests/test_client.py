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
    return LLMClient(api_key="test")


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
async def test_generate_provider_error(service, monkeypatch):
    create = AsyncMock(side_effect=Exception("API failed"))
    monkeypatch.setattr(
        client.chat.completions,
        "create",
        create,
    )
    service._friendly_error = Mock(return_value="Provider unavailable")
    result = await service.generate("Hello")
    assert result == (
        "⚠️ I couldn't generate an answer right now.\n\nProvider unavailable"
    )
    service._friendly_error.assert_called_once()


@pytest.mark.asyncio
async def test_generate_structured_success(service):
    service.generate = AsyncMock(
        return_value='{"plan":["test1","test2"],"dependencies":[[],["test"]],"estimated_time_minutes":45}'
    )
    result = await service.generate_structured("Call", PlanModel)

    assert isinstance(result, PlanModel)
    assert result.plan == ["test1", "test2"]
    assert result.dependencies == [[], ["test"]]
    assert result.estimated_time_minutes == 45
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
    assert service.generate.await_count == 4


@pytest.mark.asyncio
async def test_generate_structured_invlaid_structure(service):
    service.generate = AsyncMock(
        return_value='{"Plan":[],"dependencies":[],"estimated_time_minutes":"2"}'
    )

    with pytest.raises(ValueError, match="Invalid Structure from the LLM"):
        await service.generate_structured("Call", PlanModel)
    assert service.generate.await_count == 4


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
