"""Tests for pai_agent_sdk.toolsets.core.filesystem.edit module."""

from contextlib import AsyncExitStack
from pathlib import Path
from unittest.mock import MagicMock

from inline_snapshot import snapshot
from pydantic_ai import RunContext

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.environment.local import LocalEnvironment
from pai_agent_sdk.toolsets.core.filesystem._types import EditItem
from pai_agent_sdk.toolsets.core.filesystem.edit import EditTool, MultiEditTool


def test_edit_tool_attributes(agent_context: AgentContext) -> None:
    """Should have correct name and description."""
    assert EditTool.name == "edit"
    assert "string replacement" in EditTool.description
    tool = EditTool()
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context
    instruction = tool.get_instruction(mock_run_ctx)
    assert instruction is not None


def test_multi_edit_tool_attributes(agent_context: AgentContext) -> None:
    """Should have correct name and description."""
    assert MultiEditTool.name == "multi_edit"
    assert "multiple find-and-replace" in MultiEditTool.description


# --- EditTool tests ---


async def test_edit_create_new_file(tmp_path: Path) -> None:
    """Should create new file when old_string is empty."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = EditTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, file_path="new_file.txt", old_string="", new_string="Hello World")
        assert result == snapshot("Successfully created new file: new_file.txt")
        assert (tmp_path / "new_file.txt").read_text() == "Hello World"


async def test_edit_create_file_already_exists(tmp_path: Path) -> None:
    """Should return error when trying to create file that already exists."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = EditTool()

        # Create existing file
        (tmp_path / "existing.txt").write_text("existing content")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, file_path="existing.txt", old_string="", new_string="new content")
        assert result == snapshot("Error: File already exists: existing.txt. Use `replace` tool to overwrite.")


async def test_edit_file_not_found(tmp_path: Path) -> None:
    """Should return error when file not found."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = EditTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, file_path="nonexistent.txt", old_string="foo", new_string="bar")
        assert result == snapshot("Error: File not found: nonexistent.txt")


async def test_edit_path_is_directory(tmp_path: Path) -> None:
    """Should return error when path is a directory."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = EditTool()

        (tmp_path / "testdir").mkdir()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, file_path="testdir", old_string="foo", new_string="bar")
        assert result == snapshot("Error: Path is a directory, not a file: testdir")


async def test_edit_text_not_found(tmp_path: Path) -> None:
    """Should return error when old_string not found in file."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = EditTool()

        (tmp_path / "test.txt").write_text("Hello World")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, file_path="test.txt", old_string="foo", new_string="bar")
        assert result == snapshot("Error: Text not found. Ensure exact match including whitespace and indentation.")


async def test_edit_single_replacement(tmp_path: Path) -> None:
    """Should replace single occurrence."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = EditTool()

        (tmp_path / "test.txt").write_text("Hello World")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, file_path="test.txt", old_string="World", new_string="Universe")
        assert result == snapshot("Successfully edited file: test.txt")
        assert (tmp_path / "test.txt").read_text() == "Hello Universe"


async def test_edit_multiple_occurrences_error(tmp_path: Path) -> None:
    """Should return error when text appears multiple times without replace_all."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = EditTool()

        (tmp_path / "test.txt").write_text("foo bar foo baz foo")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, file_path="test.txt", old_string="foo", new_string="qux")
        assert result == snapshot("Error: Text appears 3 times. Add more context or use replace_all=true.")


async def test_edit_replace_all(tmp_path: Path) -> None:
    """Should replace all occurrences when replace_all is True."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = EditTool()

        (tmp_path / "test.txt").write_text("foo bar foo baz foo")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(
            mock_run_ctx, file_path="test.txt", old_string="foo", new_string="qux", replace_all=True
        )
        assert result == snapshot("Successfully edited file: test.txt")
        assert (tmp_path / "test.txt").read_text() == "qux bar qux baz qux"


# --- MultiEditTool tests ---


async def test_multi_edit_empty_edits(tmp_path: Path) -> None:
    """Should return error when no edits provided."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = MultiEditTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, file_path="test.txt", edits=[])
        assert result == snapshot("Error: At least one edit operation must be provided.")


async def test_multi_edit_create_new_file(tmp_path: Path) -> None:
    """Should create new file when first edit has empty old_string."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = MultiEditTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        edits = [
            EditItem(old_string="", new_string="Hello World", replace_all=False),
            EditItem(old_string="World", new_string="Universe", replace_all=False),
        ]
        result = await tool.call(mock_run_ctx, file_path="new_file.txt", edits=edits)
        assert result == snapshot("Successfully created new file with 2 edits: new_file.txt")
        assert (tmp_path / "new_file.txt").read_text() == "Hello Universe"


async def test_multi_edit_file_already_exists(tmp_path: Path) -> None:
    """Should return error when creating file that exists."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = MultiEditTool()

        (tmp_path / "existing.txt").write_text("content")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        edits = [EditItem(old_string="", new_string="new content", replace_all=False)]
        result = await tool.call(mock_run_ctx, file_path="existing.txt", edits=edits)
        assert result == snapshot("Error: File already exists: existing.txt. Use `replace` tool to overwrite.")


async def test_multi_edit_file_not_found(tmp_path: Path) -> None:
    """Should return error when file not found."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = MultiEditTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        edits = [EditItem(old_string="foo", new_string="bar", replace_all=False)]
        result = await tool.call(mock_run_ctx, file_path="nonexistent.txt", edits=edits)
        assert result == snapshot("Error: File not found: nonexistent.txt")


async def test_multi_edit_path_is_directory(tmp_path: Path) -> None:
    """Should return error when path is directory."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = MultiEditTool()

        (tmp_path / "testdir").mkdir()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        edits = [EditItem(old_string="foo", new_string="bar", replace_all=False)]
        result = await tool.call(mock_run_ctx, file_path="testdir", edits=edits)
        assert result == snapshot("Error: Path is a directory, not a file: testdir")


async def test_multi_edit_sequential_edits(tmp_path: Path) -> None:
    """Should apply edits sequentially."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = MultiEditTool()

        (tmp_path / "test.txt").write_text("aaa bbb ccc")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        edits = [
            EditItem(old_string="aaa", new_string="xxx", replace_all=False),
            EditItem(old_string="bbb", new_string="yyy", replace_all=False),
            EditItem(old_string="ccc", new_string="zzz", replace_all=False),
        ]
        result = await tool.call(mock_run_ctx, file_path="test.txt", edits=edits)
        assert result == snapshot("Successfully applied 3 edits to file: test.txt")
        assert (tmp_path / "test.txt").read_text() == "xxx yyy zzz"


async def test_multi_edit_text_not_found(tmp_path: Path) -> None:
    """Should return error when edit text not found."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = MultiEditTool()

        (tmp_path / "test.txt").write_text("Hello World")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        edits = [
            EditItem(old_string="Hello", new_string="Hi", replace_all=False),
            EditItem(old_string="nonexistent", new_string="bar", replace_all=False),
        ]
        result = await tool.call(mock_run_ctx, file_path="test.txt", edits=edits)
        assert result == snapshot("Error: Edit 2: Text not found. Ensure exact match.")


async def test_multi_edit_multiple_occurrences_error(tmp_path: Path) -> None:
    """Should return error when edit has multiple matches without replace_all."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = MultiEditTool()

        (tmp_path / "test.txt").write_text("foo foo foo")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        edits = [EditItem(old_string="foo", new_string="bar", replace_all=False)]
        result = await tool.call(mock_run_ctx, file_path="test.txt", edits=edits)
        assert result == snapshot("Error: Edit 1: Text appears 3 times. Use replace_all=true.")


async def test_multi_edit_with_replace_all(tmp_path: Path) -> None:
    """Should replace all occurrences when replace_all is True."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = MultiEditTool()

        (tmp_path / "test.txt").write_text("foo bar foo baz foo")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        edits = [EditItem(old_string="foo", new_string="qux", replace_all=True)]
        result = await tool.call(mock_run_ctx, file_path="test.txt", edits=edits)
        assert result == snapshot("Successfully applied 1 edits to file: test.txt")
        assert (tmp_path / "test.txt").read_text() == "qux bar qux baz qux"
