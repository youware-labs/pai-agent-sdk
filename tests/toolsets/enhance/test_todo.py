"""Tests for pai_agent_sdk.toolsets.enhance.todo module."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic_ai import RunContext

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.toolsets.enhance.todo import (
    TodoItem,
    TodoReadTool,
    TodoWriteTool,
)

# --- TodoItem tests ---


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


# --- TodoReadTool tests ---


def test_todo_read_tool_attributes() -> None:
    """Should have correct name and description."""
    assert TodoReadTool.name == "to_do_read"
    assert "to-do" in TodoReadTool.description.lower() or "todo" in TodoReadTool.description.lower()
    assert TodoReadTool.instruction is None


def test_todo_read_tool_initialization() -> None:
    """Should initialize with context."""
    ctx = AgentContext()
    tool = TodoReadTool(ctx)
    assert tool.ctx is ctx


def test_todo_read_tool_is_available() -> None:
    """Should be available by default."""
    assert TodoReadTool.is_available() is True


@pytest.mark.asyncio
async def test_todo_read_tool_no_file(tmp_path: Path) -> None:
    """Should return message when no file exists."""
    async with AgentContext(tmp_base_dir=tmp_path) as ctx:
        tool = TodoReadTool(ctx)

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx)
        assert result == "No TO-DO file found"


@pytest.mark.asyncio
async def test_todo_read_tool_empty_file(tmp_path: Path) -> None:
    """Should return message when file is empty."""
    async with AgentContext(tmp_base_dir=tmp_path) as ctx:
        tool = TodoReadTool(ctx)

        # Create empty file
        todo_file = ctx.tmp_dir / f"TO-DO-{ctx.run_id}.json"
        todo_file.write_text("")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx)
        assert result == "No TO-DOs found"


@pytest.mark.asyncio
async def test_todo_read_tool_valid_file(tmp_path: Path) -> None:
    """Should return list of TodoItems when file is valid."""
    async with AgentContext(tmp_base_dir=tmp_path) as ctx:
        tool = TodoReadTool(ctx)

        # Create valid file
        todos = [
            {"id": "TASK-1", "content": "Task 1", "status": "pending", "priority": "high"},
            {"id": "TASK-2", "content": "Task 2", "status": "completed", "priority": "low"},
        ]
        todo_file = ctx.tmp_dir / f"TO-DO-{ctx.run_id}.json"
        todo_file.write_text(json.dumps(todos))

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].id == "TASK-1"
        assert result[1].status == "completed"


@pytest.mark.asyncio
async def test_todo_read_tool_corrupted_file(tmp_path: Path) -> None:
    """Should return error message and delete corrupted file."""
    async with AgentContext(tmp_base_dir=tmp_path) as ctx:
        tool = TodoReadTool(ctx)

        # Create corrupted file
        todo_file = ctx.tmp_dir / f"TO-DO-{ctx.run_id}.json"
        todo_file.write_text("not valid json {{{")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx)
        assert result == "Error reading to_do file, please try again."
        assert not todo_file.exists()


# --- TodoWriteTool tests ---


def test_todo_write_tool_attributes() -> None:
    """Should have correct name and description."""
    assert TodoWriteTool.name == "to_do_write"
    assert "to-do" in TodoWriteTool.description.lower() or "todo" in TodoWriteTool.description.lower()
    assert TodoWriteTool.instruction is None


def test_todo_write_tool_initialization() -> None:
    """Should initialize with context."""
    ctx = AgentContext()
    tool = TodoWriteTool(ctx)
    assert tool.ctx is ctx


def test_todo_write_tool_is_available() -> None:
    """Should be available by default."""
    assert TodoWriteTool.is_available() is True


@pytest.mark.asyncio
async def test_todo_write_tool_write_todos(tmp_path: Path) -> None:
    """Should write todos to file."""
    async with AgentContext(tmp_base_dir=tmp_path) as ctx:
        tool = TodoWriteTool(ctx)

        todos = [
            TodoItem(id="TASK-1", content="Task 1", status="pending", priority="high"),
            TodoItem(id="TASK-2", content="Task 2", status="in_progress", priority="medium"),
        ]

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, to_dos=todos)
        assert isinstance(result, list)
        assert len(result) == 2

        # Verify file was created
        todo_file = ctx.tmp_dir / f"TO-DO-{ctx.run_id}.json"
        assert todo_file.exists()
        content = json.loads(todo_file.read_text())
        assert len(content) == 2


@pytest.mark.asyncio
async def test_todo_write_tool_clear_todos(tmp_path: Path) -> None:
    """Should clear todos when empty list is passed."""
    async with AgentContext(tmp_base_dir=tmp_path) as ctx:
        tool = TodoWriteTool(ctx)

        # First create a file
        todo_file = ctx.tmp_dir / f"TO-DO-{ctx.run_id}.json"
        todo_file.write_text('[{"id": "1", "content": "test", "status": "pending", "priority": "low"}]')

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, to_dos=[])
        assert result == "TO-DO list cleared successfully."
        assert not todo_file.exists()


@pytest.mark.asyncio
async def test_todo_write_tool_overwrite_existing(tmp_path: Path) -> None:
    """Should overwrite existing file."""
    async with AgentContext(tmp_base_dir=tmp_path) as ctx:
        tool = TodoWriteTool(ctx)

        # Create initial file
        todo_file = ctx.tmp_dir / f"TO-DO-{ctx.run_id}.json"
        todo_file.write_text('[{"id": "OLD", "content": "old", "status": "pending", "priority": "low"}]')

        new_todos = [
            TodoItem(id="NEW", content="New task", status="completed", priority="high"),
        ]

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, to_dos=new_todos)
        assert isinstance(result, list)
        assert result[0].id == "NEW"

        # Verify file was overwritten
        content = json.loads(todo_file.read_text())
        assert len(content) == 1
        assert content[0]["id"] == "NEW"


@pytest.mark.asyncio
async def test_todo_write_and_read_integration(tmp_path: Path) -> None:
    """Should be able to write and then read todos."""
    async with AgentContext(tmp_base_dir=tmp_path) as ctx:
        write_tool = TodoWriteTool(ctx)
        read_tool = TodoReadTool(ctx)

        todos = [
            TodoItem(id="TASK-1", content="Integration test", status="pending", priority="high"),
        ]

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        # Write
        await write_tool.call(mock_run_ctx, to_dos=todos)

        # Read
        result = await read_tool.call(mock_run_ctx)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].id == "TASK-1"
        assert result[0].content == "Integration test"


# --- Module exports tests ---


def test_enhance_module_exports() -> None:
    """Should export expected symbols from enhance module."""
    from pai_agent_sdk.toolsets import enhance

    assert hasattr(enhance, "ThinkingTool")
    assert hasattr(enhance, "TodoItem")
    assert hasattr(enhance, "TodoReadTool")
    assert hasattr(enhance, "TodoWriteTool")
    assert hasattr(enhance, "tools")
    assert len(enhance.tools) == 3
