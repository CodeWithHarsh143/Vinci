import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from openai import APIError

from vinci.core.exceptions import LLMAPIError
from vinci.llm.client import LLMClient, client
from vinci.schemas.agentsSchema import PlanModel


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
        messages=[{"role": "user", "content": "Hello"}],
    )


@pytest.mark.asyncio
async def test_generate_api_failed(service, monkeypatch):
    api_error = APIError(
        message="LLM API failed",
        request=Mock(),
        body=None,
    )

    create = AsyncMock(side_effect=api_error)

    monkeypatch.setattr(
        client.chat.completions,
        "create",
        create,
    )

    with pytest.raises(LLMAPIError):
        await service.generate("Hello")

    create.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_cancelled_error(service, monkeypatch):
    create = AsyncMock(side_effect=asyncio.CancelledError())

    monkeypatch.setattr(
        client.chat.completions,
        "create",
        create,
    )

    with pytest.raises(asyncio.CancelledError):
        await service.generate("Call")


@pytest.mark.asyncio
async def test_generate_empty_response(service, monkeypatch):
    response = Mock()
    response.choices = [Mock(message=Mock(content=None))]

    create = AsyncMock(return_value=response)

    monkeypatch.setattr(
        client.chat.completions,
        "create",
        create,
    )

    with pytest.raises(
        LLMAPIError,
        match="LLM returned an empty response",
    ):
        await service.generate("Hello")


@pytest.mark.asyncio
async def test_generate_structured_success(service):
    service.generate = AsyncMock(
        return_value=(
            '{"plan":["test1","test2"],'
            '"dependencies":[[],["test"]],'
            '"estimated_time_minutes":35}'
        )
    )

    result = await service.generate_structured(
        "Call",
        PlanModel,
    )

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
            ('{"plan":["SuccessFull"],"dependencies":[[]],"estimated_time_minutes":2}'),
        ]
    )

    result = await service.generate_structured(
        "Call",
        PlanModel,
    )

    assert isinstance(result, PlanModel)
    assert result.plan == ["SuccessFull"]
    assert result.dependencies == [[]]
    assert result.estimated_time_minutes == 2

    assert service.generate.await_count == 2


@pytest.mark.asyncio
async def test_generate_structured_invalid_json(service):
    service.generate = AsyncMock(return_value="Invalid JSON")

    with pytest.raises(
        ValueError,
        match="LLM returned invalid JSON",
    ):
        await service.generate_structured(
            "Call",
            PlanModel,
        )

    assert service.generate.await_count == 3


@pytest.mark.asyncio
async def test_generate_structured_invalid_structure(service):
    service.generate = AsyncMock(
        return_value=('{"Plan":[],"dependencies":[],"estimated_time_minutes":"2"}')
    )

    with pytest.raises(
        ValueError,
        match="LLM response does not match the expected structure",
    ):
        await service.generate_structured(
            "Call",
            PlanModel,
        )

    assert service.generate.await_count == 3


@pytest.mark.asyncio
async def test_generate_structured_invalid_structure_then_success(
    service,
):
    service.generate = AsyncMock(
        side_effect=[
            ('{"Plan":[],"dependencies":[],"estimated_time_minutes":"2"}'),
            ('{"plan":[],"dependencies":[],"estimated_time_minutes":"2"}'),
        ]
    )

    result = await service.generate_structured(
        "Call",
        PlanModel,
    )

    assert isinstance(result, PlanModel)
    assert result.plan == []
    assert result.dependencies == []
    assert result.estimated_time_minutes == 2

    assert service.generate.await_count == 2


@pytest.mark.asyncio
async def test_generate_structured_api_failed(service):
    service.generate = AsyncMock(side_effect=LLMAPIError())

    with pytest.raises(LLMAPIError):
        await service.generate_structured(
            "Call",
            PlanModel,
        )

    assert service.generate.await_count == 3


@pytest.mark.asyncio
async def test_generate_structured_cancelled_error(service):
    service.generate = AsyncMock(side_effect=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await service.generate_structured(
            "Call",
            PlanModel,
        )

    assert service.generate.await_count == 1


@pytest.mark.asyncio
async def test_generate_structured_connection_error(service, monkeypatch):

    connection_error = ConnectionError()

    create = AsyncMock(side_effect=connection_error)

    monkeypatch.setattr(
        client.chat.completions,
        "create",
        create,
    )

    with pytest.raises(LLMAPIError):
        await service.generate_structured("Call", PlanModel)
    assert create.await_count == 3


@pytest.mark.asyncio
async def test_generate_structured_LLMAPIError_then_success(service):
    service.generate = AsyncMock(
        side_effect=[
            LLMAPIError(),
            '{"plan": [], "dependencies": [], "estimated_time_minutes": "2"}',
        ]
    )
    result = await service.generate_structured("Call", PlanModel)
    assert isinstance(result, PlanModel)
    assert result.plan == []
    assert result.dependencies == []
    assert result.estimated_time_minutes == 2

    assert service.generate.await_count == 2
