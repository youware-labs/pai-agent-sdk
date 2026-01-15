"""Tests for pai_agent_sdk.toolsets.core.filesystem.ls module."""

from contextlib import AsyncExitStack
from pathlib import Path
from unittest.mock import MagicMock

from inline_snapshot import snapshot
from pydantic_ai import RunContext

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.environment.local import LocalEnvironment
from pai_agent_sdk.toolsets.core.filesystem.ls import ListTool


def test_list_tool_attributes(agent_context: AgentContext) -> None:
    """Should have correct name and description."""
    assert ListTool.name == "ls"
    assert "List directory" in ListTool.description
    tool = ListTool()
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context
    instruction = tool.get_instruction(mock_run_ctx)
    assert instruction is not None


async def test_ls_list_directory(tmp_path: Path) -> None:
    """Should list directory contents."""
    async with AsyncExitStack() as stack:
        # Create a clean subdirectory to avoid tmp_dir pollution
        test_dir = tmp_path / "test_list"
        test_dir.mkdir()

        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[test_dir], default_path=test_dir, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ListTool()

        # Create files and directories in the clean test directory
        (test_dir / "file1.txt").write_text("content")
        (test_dir / "file2.py").write_text("content")
        (test_dir / "subdir").mkdir()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, path=".")
        assert result["success"] is True
        assert result["count"] == 3
        names = [e["name"] for e in result["entries"]]
        assert "file1.txt" in names
        assert "file2.py" in names
        assert "subdir" in names


async def test_ls_file_info(tmp_path: Path) -> None:
    """Should include file info (size, modified) for files."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ListTool()

        (tmp_path / "test.txt").write_text("hello")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, path=".")
        file_entry = next(e for e in result["entries"] if e["name"] == "test.txt")
        assert file_entry["type"] == "file"
        assert "size" in file_entry
        assert file_entry["size"] == 5
        assert "modified" in file_entry


async def test_ls_directory_type(tmp_path: Path) -> None:
    """Should identify directories correctly."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ListTool()

        (tmp_path / "subdir").mkdir()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, path=".")
        dir_entry = next(e for e in result["entries"] if e["name"] == "subdir")
        assert dir_entry["type"] == "directory"


async def test_ls_directory_not_found(tmp_path: Path) -> None:
    """Should return error when directory not found."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ListTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, path="nonexistent")
        assert result["success"] is False
        assert result["error"] == snapshot("Directory not found: nonexistent")


async def test_ls_path_is_file(tmp_path: Path) -> None:
    """Should return error when path is a file."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ListTool()

        (tmp_path / "test.txt").write_text("content")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, path="test.txt")
        assert result["success"] is False
        assert result["error"] == snapshot("Path is not a directory: test.txt")


async def test_ls_with_ignore_pattern(tmp_path: Path) -> None:
    """Should ignore files matching ignore patterns."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ListTool()

        (tmp_path / "keep.txt").write_text("content")
        (tmp_path / "ignore.pyc").write_text("content")
        (tmp_path / "__pycache__").mkdir()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, path=".", ignore=["*.pyc", "__pycache__"])
        names = [e["name"] for e in result["entries"]]
        assert "keep.txt" in names
        assert "ignore.pyc" not in names
        assert "__pycache__" not in names


async def test_ls_empty_directory(tmp_path: Path) -> None:
    """Should handle empty directory."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ListTool()

        (tmp_path / "empty").mkdir()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, path="empty")
        assert result["success"] is True
        assert result["count"] == 0
        assert result["entries"] == []
