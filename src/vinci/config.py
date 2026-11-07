from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str = Field(default="")
    max_iterations: int = Field(default=30)
    time_limit: float = Field(default=10 * 60)
    vinci_reranker_model: str = Field(default="overlap")
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
