"""Tests for pai_agent_sdk.toolsets.core.filesystem.replace module."""

from contextlib import AsyncExitStack
from pathlib import Path
from unittest.mock import MagicMock

from inline_snapshot import snapshot
from pydantic_ai import RunContext

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.environment.local import LocalEnvironment
from pai_agent_sdk.toolsets.core.filesystem.replace import ReplaceTool


def test_replace_tool_attributes(agent_context: AgentContext) -> None:
    """Should have correct name and description."""
    assert ReplaceTool.name == "replace"
    assert "Write or overwrite" in ReplaceTool.description
    tool = ReplaceTool()
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context
    instruction = tool.get_instruction(mock_run_ctx)
    assert instruction is not None


async def test_replace_create_new_file(tmp_path: Path) -> None:
    """Should create new file."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ReplaceTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, file_path="new_file.txt", content="Hello World")
        assert result == snapshot("Successfully wrote to file: new_file.txt")
        assert (tmp_path / "new_file.txt").read_text() == "Hello World"


async def test_replace_overwrite_file(tmp_path: Path) -> None:
    """Should overwrite existing file."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ReplaceTool()

        (tmp_path / "test.txt").write_text("old content")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, file_path="test.txt", content="new content")
        assert result == snapshot("Successfully wrote to file: test.txt")
        assert (tmp_path / "test.txt").read_text() == "new content"


async def test_replace_append_mode(tmp_path: Path) -> None:
    """Should append to file in append mode."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ReplaceTool()

        (tmp_path / "test.txt").write_text("Hello ")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, file_path="test.txt", content="World", mode="a")
        assert result == snapshot("Successfully wrote to file: test.txt")
        assert (tmp_path / "test.txt").read_text() == "Hello World"


async def test_replace_invalid_mode(tmp_path: Path) -> None:
    """Should return error for invalid mode."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ReplaceTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, file_path="test.txt", content="content", mode="x")
        assert result == snapshot("Error: Invalid mode 'x'. Only 'w' and 'a' are supported.")


async def test_replace_create_with_subdirectory(tmp_path: Path) -> None:
    """Should create file in subdirectory."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ReplaceTool()

        (tmp_path / "subdir").mkdir()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, file_path="subdir/test.txt", content="content")
        assert result == snapshot("Successfully wrote to file: subdir/test.txt")
        assert (tmp_path / "subdir" / "test.txt").read_text() == "content"
