"""Tests for pai_agent_sdk.toolsets.core.filesystem.glob module."""

from contextlib import AsyncExitStack
from pathlib import Path
from unittest.mock import MagicMock

from pydantic_ai import RunContext

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.environment.local import LocalEnvironment
from pai_agent_sdk.toolsets.core.filesystem.glob import GlobTool


def test_glob_tool_attributes(agent_context: AgentContext) -> None:
    """Should have correct name and description."""
    assert GlobTool.name == "glob_tool"
    assert "glob pattern" in GlobTool.description
    tool = GlobTool(agent_context)
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context
    instruction = tool.get_instruction(mock_run_ctx)
    assert instruction is not None


async def test_glob_find_files(tmp_path: Path) -> None:
    """Should find files matching pattern."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GlobTool(ctx)

        # Create test files
        (tmp_path / "file1.py").write_text("content")
        (tmp_path / "file2.py").write_text("content")
        (tmp_path / "file3.txt").write_text("content")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="*.py")
        assert len(result) == 2
        assert any("file1.py" in r for r in result)
        assert any("file2.py" in r for r in result)


async def test_glob_recursive_pattern(tmp_path: Path) -> None:
    """Should find files recursively with ** pattern."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GlobTool(ctx)

        # Create nested structure
        (tmp_path / "subdir").mkdir()
        (tmp_path / "file.py").write_text("content")
        (tmp_path / "subdir" / "nested.py").write_text("content")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="**/*.py")
        assert len(result) >= 2


async def test_glob_no_matches(tmp_path: Path) -> None:
    """Should return empty list when no matches."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GlobTool(ctx)

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="*.nonexistent")
        assert result == []


async def test_glob_specific_extension(tmp_path: Path) -> None:
    """Should match specific file extensions."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GlobTool(ctx)

        (tmp_path / "test.json").write_text("{}")
        (tmp_path / "test.yaml").write_text("key: value")
        (tmp_path / "test.txt").write_text("text")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="*.json")
        assert len(result) == 1
        assert "test.json" in result[0]


async def test_glob_empty_directory(tmp_path: Path) -> None:
    """Should return empty list for empty directory."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GlobTool(ctx)

        # tmp_path is empty, no files created
        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="*.py")
        assert result == []


async def test_glob_matches_directories(tmp_path: Path) -> None:
    """Should include directories in glob results when pattern matches."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GlobTool(ctx)

        (tmp_path / "mydir").mkdir()
        (tmp_path / "myfile.txt").write_text("content")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="my*")
        assert len(result) == 2
        assert any("mydir" in r for r in result)
        assert any("myfile.txt" in r for r in result)
