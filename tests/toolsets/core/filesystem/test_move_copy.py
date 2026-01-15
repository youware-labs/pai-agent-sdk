"""Tests for pai_agent_sdk.toolsets.core.filesystem.move_copy module."""

from contextlib import AsyncExitStack
from pathlib import Path
from unittest.mock import MagicMock

from pydantic_ai import RunContext

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.environment.local import LocalEnvironment
from pai_agent_sdk.toolsets.core.filesystem.move_copy import CopyTool, MoveTool


def test_move_tool_attributes(agent_context: AgentContext) -> None:
    """Should have correct name and description."""
    assert MoveTool.name == "move"
    assert "Move" in MoveTool.description
    tool = MoveTool()
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context
    instruction = tool.get_instruction(mock_run_ctx)
    assert instruction is not None


def test_copy_tool_attributes(agent_context: AgentContext) -> None:
    """Should have correct name and description."""
    assert CopyTool.name == "copy"
    assert "Copy" in CopyTool.description


# --- MoveTool tests ---


async def test_move_file(tmp_path: Path) -> None:
    """Should move file to new location."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = MoveTool()

        (tmp_path / "source.txt").write_text("content")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pairs=[{"src": "source.txt", "dst": "dest.txt"}])
        assert len(result) == 1
        assert result[0]["success"] is True
        assert not (tmp_path / "source.txt").exists()
        assert (tmp_path / "dest.txt").read_text() == "content"


async def test_move_source_not_found(tmp_path: Path) -> None:
    """Should return error when source not found."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = MoveTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pairs=[{"src": "nonexistent.txt", "dst": "dest.txt"}])
        assert result[0]["success"] is False
        assert "not found" in result[0]["message"]


async def test_move_dest_exists_no_overwrite(tmp_path: Path) -> None:
    """Should fail when destination exists without overwrite."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = MoveTool()

        (tmp_path / "source.txt").write_text("source")
        (tmp_path / "dest.txt").write_text("existing")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pairs=[{"src": "source.txt", "dst": "dest.txt"}])
        assert result[0]["success"] is False
        assert "exists" in result[0]["message"]


async def test_move_dest_exists_with_overwrite(tmp_path: Path) -> None:
    """Should overwrite when overwrite=True."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = MoveTool()

        (tmp_path / "source.txt").write_text("new content")
        (tmp_path / "dest.txt").write_text("old content")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pairs=[{"src": "source.txt", "dst": "dest.txt"}], overwrite=True)
        assert result[0]["success"] is True
        assert (tmp_path / "dest.txt").read_text() == "new content"


async def test_move_multiple_pairs(tmp_path: Path) -> None:
    """Should handle multiple move operations."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = MoveTool()

        (tmp_path / "file1.txt").write_text("content1")
        (tmp_path / "file2.txt").write_text("content2")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(
            mock_run_ctx,
            pairs=[
                {"src": "file1.txt", "dst": "moved1.txt"},
                {"src": "file2.txt", "dst": "moved2.txt"},
            ],
        )
        assert len(result) == 2
        assert all(r["success"] for r in result)


async def test_move_directory(tmp_path: Path) -> None:
    """Should move directory."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = MoveTool()

        (tmp_path / "srcdir").mkdir()
        (tmp_path / "srcdir" / "file.txt").write_text("content")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pairs=[{"src": "srcdir", "dst": "dstdir"}])
        assert result[0]["success"] is True
        assert not (tmp_path / "srcdir").exists()
        assert (tmp_path / "dstdir" / "file.txt").read_text() == "content"


# --- CopyTool tests ---


async def test_copy_file(tmp_path: Path) -> None:
    """Should copy file to new location."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = CopyTool()

        (tmp_path / "source.txt").write_text("content")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pairs=[{"src": "source.txt", "dst": "copy.txt"}])
        assert result[0]["success"] is True
        assert (tmp_path / "source.txt").exists()  # Original still exists
        assert (tmp_path / "copy.txt").read_text() == "content"


async def test_copy_source_not_found(tmp_path: Path) -> None:
    """Should return error when source not found."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = CopyTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pairs=[{"src": "nonexistent.txt", "dst": "copy.txt"}])
        assert result[0]["success"] is False
        assert "not found" in result[0]["message"]


async def test_copy_source_is_directory(tmp_path: Path) -> None:
    """Should fail when source is directory (files only)."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = CopyTool()

        (tmp_path / "srcdir").mkdir()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pairs=[{"src": "srcdir", "dst": "copydir"}])
        assert result[0]["success"] is False
        assert "not a file" in result[0]["message"]


async def test_copy_dest_exists_no_overwrite(tmp_path: Path) -> None:
    """Should fail when destination exists without overwrite."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = CopyTool()

        (tmp_path / "source.txt").write_text("source")
        (tmp_path / "dest.txt").write_text("existing")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pairs=[{"src": "source.txt", "dst": "dest.txt"}])
        assert result[0]["success"] is False
        assert "exists" in result[0]["message"]


async def test_copy_dest_exists_with_overwrite(tmp_path: Path) -> None:
    """Should overwrite when overwrite=True."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = CopyTool()

        (tmp_path / "source.txt").write_text("new content")
        (tmp_path / "dest.txt").write_text("old content")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pairs=[{"src": "source.txt", "dst": "dest.txt"}], overwrite=True)
        assert result[0]["success"] is True
        assert (tmp_path / "dest.txt").read_text() == "new content"
        assert (tmp_path / "source.txt").exists()  # Original still exists


async def test_copy_multiple_pairs(tmp_path: Path) -> None:
    """Should handle multiple copy operations."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = CopyTool()

        (tmp_path / "file1.txt").write_text("content1")
        (tmp_path / "file2.txt").write_text("content2")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(
            mock_run_ctx,
            pairs=[
                {"src": "file1.txt", "dst": "copy1.txt"},
                {"src": "file2.txt", "dst": "copy2.txt"},
            ],
        )
        assert len(result) == 2
        assert all(r["success"] for r in result)
        assert (tmp_path / "file1.txt").exists()
        assert (tmp_path / "file2.txt").exists()
        assert (tmp_path / "copy1.txt").exists()
        assert (tmp_path / "copy2.txt").exists()
