import time
import aiofiles
from pydantic import ValidationError

from vinci.tools.base import BaseTool, ToolResult
from vinci.schemas.toolsSchema import ReadFileArgs
from pathlib import Path

READFILE_NAME = "read_file"
READFILE_DESCRIPTION = "Reads a file from the repository and returns its content"
READFILE_MAX_SIZE = 1024 * 1024


class ReadFileTool(BaseTool):
    def __init__(self, allowed_root: Path, max_size: int = READFILE_MAX_SIZE):
        super().__init__(
            name=READFILE_NAME,
            description=READFILE_DESCRIPTION,
            parameters_schema=ReadFileArgs.model_json_schema(),
        )
        self.allowed_root = allowed_root
        self.max_size = max_size

    async def execute(self, arguments: dict) -> ToolResult:
        start = time.perf_counter()
        try:
            args = ReadFileArgs.model_validate(arguments)
            path: str = args.path
            resolved = (self.allowed_root / path).resolve()
            if not resolved.is_relative_to(self.allowed_root):
                return ToolResult(
                    success=False,
                    error="Path outside allowed root",
                    duration_ms=(time.perf_counter() - start),
                )
            size = 0
            chunks = []
            async with aiofiles.open(resolved, "r", encoding="utf-8") as file:
                while chunk := await file.read(64 * 1024):
                    size += len(chunk)
                    if size > self.max_size:
                        return ToolResult(
                            success=False,
                            error="File exceeded the maximum limit size",
                            duration_ms=(time.perf_counter() - start) * 1000,
                        )
                    chunks.append(chunk)
            file_content = "".join(chunks)
            end = time.perf_counter()
            return ToolResult(
                success=True,
                data=file_content,
                metadata={"size": size},
                duration_ms=(end - start) * 1000,
            )

        except ValidationError as e:
            return ToolResult(
                success=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=f"Wrong arguments structure: {e}",
            )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                error="File not found",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        except PermissionError:
            return ToolResult(
                success=False,
                error="Permission denied",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        except IsADirectoryError:
            return ToolResult(
                success=False,
                error="Path is a directory, not a file",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
