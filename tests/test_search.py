from vinci.tools.searchTool import SearchCodeTool

import asyncio
import pytest


@pytest.fixture
def search_code(tmp_path):
    return SearchCodeTool(tmp_path)


@pytest.mark.asyncio
async def test_search_finds_match(tmp_path, search_code):
    target = tmp_path / "app.py"
    target.write_text("def login(user):\n    return user\n")

    result = await search_code.execute({"path": str(tmp_path), "query": "login"})

    assert result.success is True
    assert len(result.data) == 1
    match = result.data[0]
    assert match["file"].endswith("app.py")
    assert match["line"] == "1"
    assert "login" in match["file_content"]


@pytest.mark.asyncio
async def test_search_no_matches_returns_empty(tmp_path, search_code):
    target = tmp_path / "app.py"
    target.write_text("def login(user):\n")

    result = await search_code.execute(
        {"path": str(tmp_path), "query": "zzz_no_match_xyz"}
    )

    assert result.success is True
    assert result.data == []


@pytest.mark.asyncio
async def test_search_path_outside_allowed_root(tmp_path):
    search_code = SearchCodeTool(tmp_path)

    result = await search_code.execute(
        {"path": "../../../etc", "query": "login"}
    )

    assert result.success is False
    assert result.error == "Path outside allowed root"


@pytest.mark.asyncio
async def test_search_max_results_trims(tmp_path, search_code):
    target = tmp_path / "big.py"
    target.write_text("\n".join(f"marker line {i}" for i in range(10)) + "\n")

    result = await search_code.execute(
        {"path": str(tmp_path), "query": "marker", "max_result": 3}
    )

    assert result.success is True
    assert len(result.data) == 3


@pytest.mark.asyncio
async def test_search_invalid_regex_returns_error(tmp_path, search_code):
    target = tmp_path / "app.py"
    target.write_text("def login(user):\n")

    result = await search_code.execute({"path": str(tmp_path), "query": "["})

    assert result.success is False
    assert "Search failed" in result.error


@pytest.mark.asyncio
async def test_search_rg_missing_returns_error(tmp_path, search_code, monkeypatch):
    async def _missing(*args, **kwargs):
        raise FileNotFoundError("rg")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _missing)

    result = await search_code.execute(
        {"path": str(tmp_path), "query": "login"}
    )

    assert result.success is False
    assert result.error == "ripgrep binary not found (rg)"
