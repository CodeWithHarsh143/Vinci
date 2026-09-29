from pydantic import BaseModel, Field


class ReadFileArgs(BaseModel):
    path: str = Field(description="Relative path to the file to read")


class SearchCodeArg(BaseModel):
    path: str = Field(description="Relative path of the file/Search Area", default=".")
    query: str = Field(description="Code/Text to Search")
    max_result: int = Field(description="Maximum result to be given", default=30)
