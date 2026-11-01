from vinci.tools.base import ToolResult
import os, re
from vinci.tools.filesystem import ReadFileTool, READFILE_MAX_SIZE
from vinci.tools.searchTool import SearchCodeTool
from pathlib import Path

MAX_SNIPPETS = 2  # how many snippets to return
SNIPPET_MAX_LINES = 10  # how many lines per file to show


class Retriever:
    def __init__(self, base_dir: str = ".") -> None:
        self.base_dir = base_dir
        self.read_tool = ReadFileTool(
            allowed_root=Path(self.base_dir), max_size=READFILE_MAX_SIZE
        )
        self.search_tool = SearchCodeTool(allowed_root=Path(self.base_dir))

    async def read_snippet(
        self, path: str, start_line: int = 0, line_count: int = SNIPPET_MAX_LINES
    ) -> dict:
        try:
            result: ToolResult = await self.read_tool.execute({"path": path})
            if result.success and result.data:
                lines = result.data.splitlines()
                excerpt = "\n".join(lines[start_line : start_line + line_count])
                return {"path": path, "excerpt": excerpt, "score": 1.0}
        except Exception:
            return {"path": path, "excerpt": "", "score": 0.0}

        return {"path": path, "excerpt": "", "score": 0.0}

    async def _query(
        self,
        query: str = "",
    ) -> list[dict]:
        results: list[dict] = []
        search_result: ToolResult = await self.search_tool.execute({"query": query})
        if search_result.success:
            for line in (search_result.data or [])[:MAX_SNIPPETS]:
                path: str = line["file"] or ""
                start_line: int = line["line"] or 0
                results.append(await self.read_snippet(path, start_line))
        if len(results) < MAX_SNIPPETS:
            # naive scan of all .py files under base_dir
            pattern = re.compile(re.escape(query), re.IGNORECASE)
            for root, _, files in os.walk(self.base_dir):
                for fname in files:
                    if fname.endswith(".py"):
                        full = os.path.join(root, fname)
                        try:
                            r = await self.read_tool.execute({"path": full})
                            if r.success and r.data:
                                lines = r.data.splitlines()
                                for i, line in enumerate(lines):
                                    if pattern.search(line):
                                        results.append(
                                            await self.read_snippet(full, start_line=i)
                                        )
                                        break
                        except Exception:
                            pass
                        if len(results) >= MAX_SNIPPETS:
                            break

        # sort by score (simple: higher score first) and truncate
        results.sort(key=lambda d: d["score"], reverse=True)
        return results[:MAX_SNIPPETS]
