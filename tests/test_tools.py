from pathlib import Path

from vinci.tools.filesystem import ReadFileTool
import pytest


@pytest.fixture
def read_file(tmp_path):
    return ReadFileTool(tmp_path)


@pytest.mark.asyncio
async def test_valid_file(tmp_path, read_file):
    file = tmp_path / "text.txt"
    file.write_text("Hello, Vinci")
    file = file.relative_to(tmp_path)
    result = await read_file.execute({"path": str(file)})
    assert result.success
    assert result.data == "Hello, Vinci"


@pytest.mark.asyncio
async def test_not_allowed_file(tmp_path):
    read_file = ReadFileTool(Path("/home/harsh/Documents/Vinci"))

    result = await read_file.execute({"path": "../../../tmp/some-file.txt"})

    assert result.success is False
    assert result.error == "Path outside allowed root"


@pytest.mark.asyncio
async def test_not_existent_file(tmp_path, read_file):
    file = tmp_path / "text.txt"
    file = file.relative_to(tmp_path)
    result = await read_file.execute({"path": str(file)})
    assert result.success == False
    assert result.error == "File not found"


@pytest.mark.asyncio
async def test_directory_found_error(tmp_path, read_file):
    directory = tmp_path / "folder"
    directory.mkdir()
    relative_path = directory.relative_to(tmp_path)
    result = await read_file.execute({"path": str(relative_path)})
    assert result.success == False
    assert result.error == "Path is a directory, not a file"


@pytest.mark.asyncio
async def test_max_file_size(tmp_path, read_file):
    file = tmp_path / "large.txt"
    file.write_bytes(b"x" * (1024 * 1024 + 1))
    relative_path = file.relative_to(tmp_path)
    result = await read_file.execute({"path": str(relative_path)})
    assert result.success == False
    assert result.error == "File exceeded the maximum limit size"
