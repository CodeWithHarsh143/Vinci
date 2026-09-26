from openai import APIError, AsyncOpenAI
from pydantic import BaseModel, ValidationError
import asyncio
import json
from vinci.config import settings
from vinci.core.exceptions import LLMAPIError


client = AsyncOpenAI(
    api_key=settings.gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


class LLMClient:
    def __init__(self, model: str = "gemini-3.6-flash") -> None:
        self.model = model

    async def generate(self, prompt: str) -> str:

        try:
            response = await client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}]
            )
            full_answer = response.choices[0].message.content
            if full_answer is None:
                raise LLMAPIError("LLM returned an empty response")
            return full_answer

        except asyncio.CancelledError:
            raise
        except (APIError, ConnectionError, TimeoutError) as exc:
            raise LLMAPIError(f"API call failed: {exc}") from exc

    async def generate_structured(
        self, prompt: str, schema: type[BaseModel]
    ) -> BaseModel:
        max_retry: int = 3
        for retry in range(max_retry):
            try:
                data: str = await self.generate(prompt)
                structure_response = json.loads(data)
                return schema.model_validate(structure_response)

            except LLMAPIError:
                if retry == max_retry - 1:
                    raise

            except asyncio.CancelledError:
                raise
            except json.JSONDecodeError as e:
                if retry == max_retry - 1:
                    raise ValueError("LLM returned invalid JSON") from e
            except ValidationError as e:
                if retry == max_retry - 1:
                    raise ValueError(
                        "LLM response does not match the expected structure"
                    ) from e
