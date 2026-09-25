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
    def __init__(self, model: str = "gemini-pro") -> None:
        self.model = model

    async def generate(self, prompt: str) -> str:

        try:
            response = await client.chat.completions.create(
                model="gemini-3.6-flash",
                messages=prompt,
            )
            full_answer = response.choices[0].message.content
            return full_answer

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise RuntimeError("LLM generation failed") from exc

    async def generate_structured(
        self, propmt: str, schema: type[BaseModel]
    ) -> BaseModel:
        retry: int = 3
        validated = None
        while retry > 0:
            try:
                data: str = await self.generate(propmt)
                structure_response = json.loads(data)
                validated = schema.model_validate(structure_response)

            except RuntimeError:
                if retry == 1:
                    raise

            except asyncio.CancelledError:
                raise
            except json.JSONDecodeError as e:
                if retry == 1:
                    raise ValueError("Invalid JSON") from e
            except ValidationError as e:
                if retry == 1:
                    raise ValueError("Invalid Structure from the LLM") from e
            else:
                break
            finally:
                retry -= 1
        return validated
