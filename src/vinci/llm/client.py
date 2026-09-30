import asyncio
import json
from dataclasses import dataclass

from openai import APIError, AsyncOpenAI
from pydantic import BaseModel, ValidationError

from vinci.config import settings
from vinci.core.exceptions import LLMAPIError


client = AsyncOpenAI(
    api_key=settings.gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments: dict


@dataclass
class CallResponse:
    content: str | None
    tool_calls: list[ToolCallRequest]


class LLMClient:
    def __init__(self, model: str = "gemini-3.6-flash") -> None:
        self.model = model

    async def generate(self, prompt: str) -> str:
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

            content = response.choices[0].message.content

            if content is None:
                raise LLMAPIError("LLM returned an empty response")

            return content

        except asyncio.CancelledError:
            raise

        except (APIError, ConnectionError, TimeoutError) as exc:
            raise LLMAPIError(f"API call failed: {exc}") from exc

    async def generate_structured(
        self,
        prompt: str,
        schema: type[BaseModel],
    ) -> BaseModel:
        max_retries = 3

        for retry in range(max_retries):
            try:
                data = await self.generate(prompt)

                structure_response = json.loads(data)

                return schema.model_validate(structure_response)

            except asyncio.CancelledError:
                raise

            except LLMAPIError:
                if retry == max_retries - 1:
                    raise

            except json.JSONDecodeError as exc:
                if retry == max_retries - 1:
                    raise ValueError("LLM returned invalid JSON") from exc

            except ValidationError as exc:
                if retry == max_retries - 1:
                    raise ValueError(
                        "LLM response does not match the expected structure"
                    ) from exc

        # This should never be reached because the final retry
        # always raises or returns.
        raise RuntimeError("Unexpected structured generation state")

    async def call(
        self,
        messages: list[dict],
        tools: list[dict] | dict | None = None,
    ) -> CallResponse:
        if isinstance(tools, dict):
            tools = list(tools.values())

        if not tools:
            tools = None

        try:
            kwargs: dict = {
                "model": self.model,
                "messages": messages,
            }

            if tools is not None:
                kwargs["tools"] = tools

            response = await client.chat.completions.create(**kwargs)

            message = response.choices[0].message

            tool_calls: list[ToolCallRequest] = []

            for tool_call in message.tool_calls or []:
                try:
                    arguments = json.loads(tool_call.function.arguments)

                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "LLM returned invalid tool arguments "
                        f"for '{tool_call.function.name}'"
                    ) from exc

                if not isinstance(arguments, dict):
                    raise TypeError(
                        "Tool arguments must be a JSON object "
                        f"for '{tool_call.function.name}'"
                    )

                tool_calls.append(
                    ToolCallRequest(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        arguments=arguments,
                    )
                )

            return CallResponse(
                content=message.content,
                tool_calls=tool_calls,
            )

        except asyncio.CancelledError:
            raise

        except (APIError, ConnectionError, TimeoutError) as exc:
            raise LLMAPIError(f"API call failed: {exc}") from exc
