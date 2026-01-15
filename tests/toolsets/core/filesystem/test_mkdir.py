"""Tests for pai_agent_sdk.toolsets.core.filesystem.mkdir module."""

from contextlib import AsyncExitStack
from pathlib import Path
from unittest.mock import MagicMock

from pydantic_ai import RunContext

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.environment.local import LocalEnvironment
from pai_agent_sdk.toolsets.core.filesystem.mkdir import MkdirTool


def test_mkdir_tool_attributes(agent_context: AgentContext) -> None:
    """Should have correct name and description."""
    assert MkdirTool.name == "mkdir"
    assert "directories" in MkdirTool.description
    tool = MkdirTool()
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context
    instruction = tool.get_instruction(mock_run_ctx)
    assert instruction is not None


async def test_mkdir_single_directory(tmp_path: Path) -> None:
    """Should create single directory."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = MkdirTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, paths=["newdir"])
        assert result["success"] is True
        assert result["summary"]["successful"] == 1
        assert (tmp_path / "newdir").is_dir()


async def test_mkdir_multiple_directories(tmp_path: Path) -> None:
    """Should create multiple directories."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = MkdirTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, paths=["dir1", "dir2", "dir3"])
        assert result["success"] is True
        assert result["summary"]["total"] == 3
        assert result["summary"]["successful"] == 3
        assert (tmp_path / "dir1").is_dir()
        assert (tmp_path / "dir2").is_dir()
        assert (tmp_path / "dir3").is_dir()


async def test_mkdir_nested_with_parents(tmp_path: Path) -> None:
    """Should create nested directories with parents=True."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = MkdirTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, paths=["a/b/c"], parents=True)
        assert result["success"] is True
        assert (tmp_path / "a" / "b" / "c").is_dir()


async def test_mkdir_nested_without_parents(tmp_path: Path) -> None:
    """Should fail to create nested directories without parents=True."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = MkdirTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, paths=["x/y/z"], parents=False)
        assert result["success"] is False
        assert result["summary"]["failed"] == 1


async def test_mkdir_empty_paths(tmp_path: Path) -> None:
    """Should return error when no paths provided."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = MkdirTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, paths=[])
        assert result["success"] is False
        assert "No paths provided" in result["message"]


async def test_mkdir_mixed_results(tmp_path: Path) -> None:
    """Should handle mixed success/failure results."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = MkdirTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        # First one succeeds, second fails (nested without parents)
        result = await tool.call(mock_run_ctx, paths=["gooddir", "bad/nested/dir"], parents=False)
        assert result["success"] is False
        assert result["summary"]["successful"] == 1
        assert result["summary"]["failed"] == 1


async def test_mkdir_result_structure(tmp_path: Path) -> None:
    """Should return correct result structure."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = MkdirTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, paths=["testdir"])
        assert "success" in result
        assert "message" in result
        assert "results" in result
        assert "summary" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["path"] == "testdir"
        assert result["results"][0]["success"] is True
