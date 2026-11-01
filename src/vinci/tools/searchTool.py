import asyncio
import time

from vinci.tools.base import BaseTool, ToolResult
from pathlib import Path
from vinci.schemas.toolsSchema import SearchCodeArg

SEARCHCODE_NAME = "search_code"
SEARCHCODE_DESCRIPTION = """
Search the codebase for a given text pattern or regular expression.

Use this tool to locate function definitions, class declarations, variable
references, imports, method calls, configuration values, or any other
code patterns across the project.

The search is performed recursively from the project root using ripgrep (rg).

Returns matching lines along with their file paths and line numbers.
Use the results to understand code structure, trace references, and identify
where specific functionality is implemented.

Use this tool when you need to find existing implementations or understand
how different parts of the codebase are connected.

If no matches are found, the result will indicate that no matching content
exists. Search errors will be reported separately.
"""


class SearchCodeTool(BaseTool):
    def __init__(self, allowed_root: Path):
        super().__init__(
            name=SEARCHCODE_NAME,
            description=SEARCHCODE_DESCRIPTION,
            parameters_schema=SearchCodeArg.model_json_schema(),
        )
        self.allowed_root = allowed_root

    async def execute(self, arguments: dict) -> ToolResult:
        start = time.perf_counter()
        try:
            args = SearchCodeArg.model_validate(arguments)
            path: str = args.path
            resolved = (self.allowed_root / path).resolve()
            if not resolved.is_relative_to(self.allowed_root):
                return ToolResult(
                    success=False,
                    error="Path outside allowed root",
                    duration_ms=(time.perf_counter() - start) * 1000,
                )
            proc = await asyncio.create_subprocess_exec(
                "rg",
                args.query,
                str(resolved),
                "--vimgrep",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
        except FileNotFoundError:
            return ToolResult(
                success=False,
                error="ripgrep binary not found (rg)",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        except PermissionError:
            return ToolResult(
                success=False,
                error="Permission denied",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            return ToolResult(
                success=False,
                error="Search timed out",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        except asyncio.CancelledError:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            raise

        except OSError as e:
            return ToolResult(
                success=False,
                error=f"Search failed: {e}",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        results = []
        if proc.returncode == 0:
            results = stdout.decode("utf-8", errors="replace").splitlines()[
                : args.max_result
            ]
        elif proc.returncode == 2:
            error = stderr.decode("utf-8", errors="replace").strip()
            return ToolResult(
                success=False,
                error=f"Search failed: {error}" if error else "Search failed",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        parser_result: list[dict] = []
        for res in results:
            file_path, line, column, file_content = res.split(":", 3)
            parser_result.append(
                {
                    "file": file_path,
                    "line": line,
                    "column": column,
                    "file_content": file_content,
                }
            )
        return ToolResult(
            success=True,
            data=parser_result,
            duration_ms=(time.perf_counter() - start) * 1000,
        )
