from openai import AsyncOpenAI, APIStatusError
from pydantic import BaseModel, ValidationError
import asyncio
import json
from vinci.config import settings

client = AsyncOpenAI(
    api_key=settings.gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


class LLMClient:
    def __init__(self, api_key: str, model: str = "gemini-pro") -> None:
        self.api_key = api_key
        self.model = model

    def _friendly_error(self, exc: Exception) -> str:

        if isinstance(exc, APIStatusError):
            status = exc.status_code
            if status == 503:
                return (
                    "The AI model is currently experiencing high demand. "
                    "Please try again in a moment."
                )
            return f"The AI service returned an error ({status}). Please try again."
        if isinstance(exc, TimeoutError):
            return "The AI service timed out. Please try again."
        return "An unexpected error occurred while generating the answer."

    async def generate(self, prompt: str) -> str:

        try:
            response = await client.chat.completions.create(
                model="gemini-3.6-flash",
                messages=prompt,
            )
            full_answer = response.choices[0].message.content
            return full_answer

        except asyncio.CancelledError:
            # Client disconnected (stop button / navigation). The DB session is being
            # torn down at this point, so don't attempt any writes — just stop
            # generation and let the cancellation propagate cleanly. The partial
            # answer is already visible in the client UI.
            raise
        except Exception as exc:
            # Surface an upstream/provider or streaming failure as a readable token
            # instead of crashing the whole streamed response. The user message was
            # already persisted above, so record a short fallback answer too.
            message = self._friendly_error(exc)
            full_answer = f"⚠️ I couldn't generate an answer right now.\n\n{message}"
            return full_answer

    async def generate_structured(
        self, propmt: str, schema: type[BaseModel]
    ) -> BaseModel:
        retry: int = 3
        validated = None
        while retry >= 0:
            try:
                data: str = await self.generate(propmt)
                structure_response = json.loads(data)
                validated = schema.model_validate(structure_response)
            except json.JSONDecodeError as e:
                if retry == 0:
                    raise ValueError("Invalid JSON") from e
            except ValidationError as e:
                if retry == 0:
                    raise ValueError("Invalid Structure from the LLM") from e
            else:
                break
            finally:
                retry -= 1
        return validated
