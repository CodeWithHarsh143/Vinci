from vinci.tools.base import BaseTool
from pathlib import Path
from vinci.tools.filesystem import READFILE_NAME
from vinci.schemas.toolsSchema import SearchCodeArg

SEARCHCODE_NAME = "search_code"
SEARCHCODE_DESCRIPTION = ""


class SearchCodeTool(BaseTool):
    def __init__(self, allowed_root: Path):
        super().__init__(
            name=SEARCHCODE_NAME,
            description=SEARCHCODE_DESCRIPTION,
            parameters_schema=SearchCodeArg.model_json_schema(),
        )
