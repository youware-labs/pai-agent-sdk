"""Tests for pai_agent_sdk.toolsets.core.enhance.todo module."""

import json
from contextlib import AsyncExitStack
from pathlib import Path
from unittest.mock import MagicMock

from inline_snapshot import snapshot
from pydantic_ai import RunContext

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.environment.local import LocalEnvironment
from pai_agent_sdk.toolsets.core.enhance.todo import (
    TodoItem,
    TodoReadTool,
    TodoWriteTool,
    _get_todo_file_name,
)


def test_todo_item_creation() -> None:
    """Should create a valid TodoItem."""
    item = TodoItem(
        id="TASK-1",
        content="Test task",
        status="pending",
        priority="high",
    )
    assert item.id == "TASK-1"
    assert item.content == "Test task"
    assert item.status == "pending"
    assert item.priority == "high"


def test_todo_item_status_values() -> None:
    """Should accept valid status values."""
    for status in ["pending", "in_progress", "completed"]:
        item = TodoItem(id="1", content="test", status=status, priority="medium")
        assert item.status == status


def test_todo_item_priority_values() -> None:
    """Should accept valid priority values."""
    for priority in ["high", "medium", "low"]:
        item = TodoItem(id="1", content="test", status="pending", priority=priority)
        assert item.priority == priority


def test_todo_read_tool_attributes(agent_context: AgentContext) -> None:
    """Should have correct name, description and instruction."""
    assert TodoReadTool.name == "to_do_read"
    assert TodoReadTool.description == snapshot("Read the current session's to-do list.")
    # Test get_instruction with a mock context
    tool = TodoReadTool()
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context
    instruction = tool.get_instruction(mock_run_ctx)
    assert instruction is not None
    assert "<todo-read-guidelines>" in instruction


def test_todo_read_tool_initialization(agent_context: AgentContext) -> None:
    """Should initialize without context."""
    tool = TodoReadTool()
    assert tool.name == "to_do_read"


def test_todo_read_tool_is_available(agent_context: AgentContext, mock_run_ctx) -> None:
    """Should be available by default."""
    tool = TodoReadTool()
    assert tool.is_available(mock_run_ctx) is True


async def test_todo_read_tool_no_file(tmp_path: Path) -> None:
    """Should return message when no file exists."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = TodoReadTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx)
        assert result == "No TO-DO file found"


async def test_todo_read_tool_empty_file(tmp_path: Path) -> None:
    """Should return message when file is empty."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = TodoReadTool()

        # Create empty file in tmp_dir
        todo_file = env.tmp_dir / _get_todo_file_name(ctx.run_id)
        todo_file.write_text("")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx)
        assert result == "No TO-DOs found"


async def test_todo_read_tool_valid_file(tmp_path: Path) -> None:
    """Should return JSON string of TodoItems when file is valid."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = TodoReadTool()

        # Create valid file in tmp_dir
        todos = [
            {"id": "TASK-1", "content": "Task 1", "status": "pending", "priority": "high"},
            {"id": "TASK-2", "content": "Task 2", "status": "completed", "priority": "low"},
        ]
        todo_file = env.tmp_dir / _get_todo_file_name(ctx.run_id)
        todo_file.write_text(json.dumps(todos))

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert len(parsed) == 2
        assert parsed[0]["id"] == "TASK-1"
        assert parsed[1]["status"] == "completed"


async def test_todo_read_tool_corrupted_file(tmp_path: Path) -> None:
    """Should return error message and delete corrupted file."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = TodoReadTool()

        # Create corrupted file in tmp_dir
        todo_file = env.tmp_dir / _get_todo_file_name(ctx.run_id)
        todo_file.write_text("not valid json {{{")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx)
        assert result == "Error reading to_do file, please try again."
        assert not todo_file.exists()


def test_todo_write_tool_attributes(agent_context: AgentContext) -> None:
    """Should have correct name, description and instruction."""
    assert TodoWriteTool.name == "to_do_write"
    assert TodoWriteTool.description == snapshot("Replace the session's to-do list with an updated list.")
    # Test get_instruction with a mock context
    write_tool = TodoWriteTool()
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context
    instruction = write_tool.get_instruction(mock_run_ctx)
    assert instruction is not None
    assert "<todo-write-guidelines>" in instruction


def test_todo_write_tool_initialization(agent_context: AgentContext) -> None:
    """Should initialize without context."""
    tool = TodoWriteTool()
    assert tool.name == "to_do_write"


def test_todo_write_tool_is_available(agent_context: AgentContext, mock_run_ctx) -> None:
    """Should be available by default."""
    tool = TodoWriteTool()
    assert tool.is_available(mock_run_ctx) is True


async def test_todo_write_tool_write_todos(tmp_path: Path) -> None:
    """Should write todos to file and return JSON string."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = TodoWriteTool()

        todos = [
            TodoItem(id="TASK-1", content="Task 1", status="pending", priority="high"),
            TodoItem(id="TASK-2", content="Task 2", status="in_progress", priority="medium"),
        ]

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, to_dos=todos)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert len(parsed) == 2

        # Verify file was created in tmp_dir
        todo_file = env.tmp_dir / _get_todo_file_name(ctx.run_id)
        assert todo_file.exists()
        content = json.loads(todo_file.read_text())
        assert len(content) == 2


async def test_todo_write_tool_clear_todos(tmp_path: Path) -> None:
    """Should clear todos when empty list is passed."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = TodoWriteTool()

        # First create a file in tmp_dir
        todo_file = env.tmp_dir / _get_todo_file_name(ctx.run_id)
        todo_file.write_text('[{"id": "1", "content": "test", "status": "pending", "priority": "low"}]')

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, to_dos=[])
        assert result == "TO-DO list cleared successfully."
        assert not todo_file.exists()


async def test_todo_write_tool_overwrite_existing(tmp_path: Path) -> None:
    """Should overwrite existing file and return JSON string."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = TodoWriteTool()

        # Create initial file in tmp_dir
        todo_file = env.tmp_dir / _get_todo_file_name(ctx.run_id)
        todo_file.write_text('[{"id": "OLD", "content": "old", "status": "pending", "priority": "low"}]')

        new_todos = [
            TodoItem(id="NEW", content="New task", status="completed", priority="high"),
        ]

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, to_dos=new_todos)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed[0]["id"] == "NEW"

        # Verify file was overwritten
        content = json.loads(todo_file.read_text())
        assert len(content) == 1
        assert content[0]["id"] == "NEW"


async def test_todo_write_and_read_integration(tmp_path: Path) -> None:
    """Should be able to write and then read todos."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        write_tool = TodoWriteTool()
        read_tool = TodoReadTool()

        todos = [
            TodoItem(id="TASK-1", content="Integration test", status="pending", priority="high"),
        ]

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        # Write
        await write_tool.call(mock_run_ctx, to_dos=todos)

        # Read
        result = await read_tool.call(mock_run_ctx)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["id"] == "TASK-1"
        assert parsed[0]["content"] == "Integration test"


def test_enhance_module_exports() -> None:
    """Should export expected symbols from enhance module."""
    from pai_agent_sdk.toolsets.core import enhance

    assert hasattr(enhance, "ThinkingTool")
    assert hasattr(enhance, "TodoItem")
    assert hasattr(enhance, "TodoReadTool")
    assert hasattr(enhance, "TodoWriteTool")
    assert hasattr(enhance, "tools")
    # 4 task tools are now enabled by default
    assert len(enhance.tools) == 4
    # Also check new task tools are exported
    assert hasattr(enhance, "TaskCreateTool")
    assert hasattr(enhance, "TaskGetTool")
    assert hasattr(enhance, "TaskUpdateTool")
    assert hasattr(enhance, "TaskListTool")
