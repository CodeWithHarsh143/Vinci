from pydantic import BaseModel, Field


class ReadFileArgs(BaseModel):
    path: str = Field(description="Relative path to the file to read")
