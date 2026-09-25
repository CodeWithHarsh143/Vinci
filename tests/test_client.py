from ast import Return
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
