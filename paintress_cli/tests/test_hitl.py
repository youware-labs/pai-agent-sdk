"""Tests for paintress_cli.hitl module."""

from __future__ import annotations

import asyncio

import pytest
from paintress_cli.hitl import ApprovalManager, ApprovalResult, ApprovalState, ApprovalUI
from paintress_cli.rendering import RichRenderer
from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ToolCallPart

# =============================================================================
# ApprovalResult Tests
# =============================================================================


def test_approval_result_approved():
    """Test ApprovalResult for approved decision."""
    result = ApprovalResult(approved=True)
    assert result.approved is True
    assert result.reason is None


def test_approval_result_rejected():
    """Test ApprovalResult for rejected decision."""
    result = ApprovalResult(approved=False, reason="Too risky")
    assert result.approved is False
    assert result.reason == "Too risky"


# =============================================================================
# ApprovalState Tests
# =============================================================================


def test_approval_state_default():
    """Test ApprovalState default values."""
    state = ApprovalState()
    assert state.pending_approvals == []
    assert state.current_index == 0
    assert state.is_pending is False


def test_approval_state_with_approvals():
    """Test ApprovalState with pending approvals."""
    tool1 = ToolCallPart(tool_name="shell", args="{}", tool_call_id="call-1")
    tool2 = ToolCallPart(tool_name="edit", args="{}", tool_call_id="call-2")

    state = ApprovalState(
        pending_approvals=[tool1, tool2],
        current_index=1,
        is_pending=True,
    )

    assert len(state.pending_approvals) == 2
    assert state.current_index == 1
    assert state.is_pending is True


# =============================================================================
# ApprovalManager Tests
# =============================================================================


def test_approval_manager_init():
    """Test ApprovalManager initialization."""
    manager = ApprovalManager()

    assert not manager.is_pending
    assert manager.pending_approvals == []
    assert manager.current_index == 0
    assert manager.current_tool is None
    assert manager.total_count == 0


def test_approval_manager_start_approval_flow():
    """Test starting approval flow."""
    manager = ApprovalManager()

    tool = ToolCallPart(tool_name="shell", args="{}", tool_call_id="call-1")
    deferred = DeferredToolRequests()
    deferred.approvals = [tool]

    result = manager.start_approval_flow(deferred)

    assert result is True
    assert manager.is_pending
    assert len(manager.pending_approvals) == 1
    assert manager.current_tool == tool


def test_approval_manager_start_empty_flow():
    """Test starting approval flow with no approvals."""
    manager = ApprovalManager()

    deferred = DeferredToolRequests()
    deferred.approvals = []

    result = manager.start_approval_flow(deferred)

    assert result is False
    assert not manager.is_pending


def test_approval_manager_approve():
    """Test approving a tool."""
    manager = ApprovalManager()

    # Set up event manually
    manager._event = asyncio.Event()
    manager.approve()

    assert manager._result is not None
    assert manager._result.approved is True
    assert manager._event.is_set()


def test_approval_manager_reject():
    """Test rejecting a tool."""
    manager = ApprovalManager()

    manager._event = asyncio.Event()
    manager.reject("Not safe")

    assert manager._result is not None
    assert manager._result.approved is False
    assert manager._result.reason == "Not safe"


def test_approval_manager_advance():
    """Test advancing to next tool."""
    manager = ApprovalManager()

    tool1 = ToolCallPart(tool_name="shell", args="{}", tool_call_id="call-1")
    tool2 = ToolCallPart(tool_name="edit", args="{}", tool_call_id="call-2")
    deferred = DeferredToolRequests()
    deferred.approvals = [tool1, tool2]

    manager.start_approval_flow(deferred)

    assert manager.current_index == 0
    assert manager.advance() is True
    assert manager.current_index == 1
    assert manager.advance() is False


def test_approval_manager_reset():
    """Test resetting approval state."""
    manager = ApprovalManager()

    tool = ToolCallPart(tool_name="shell", args="{}", tool_call_id="call-1")
    deferred = DeferredToolRequests()
    deferred.approvals = [tool]

    manager.start_approval_flow(deferred)
    manager.reset()

    assert not manager.is_pending
    assert manager.pending_approvals == []
    assert manager.current_index == 0


def test_approval_manager_reset_with_waiting():
    """Test reset signals waiting coroutine."""
    manager = ApprovalManager()
    manager._event = asyncio.Event()

    # Event should not be set before reset
    assert not manager._event.is_set()

    manager.reset()

    # After reset, the state should be cleared
    assert not manager.is_pending
    assert manager._event is None


@pytest.mark.asyncio
async def test_approval_manager_wait_for_decision():
    """Test waiting for user decision."""
    manager = ApprovalManager()

    async def approve_after_delay():
        await asyncio.sleep(0.01)
        manager.approve()

    # Start approval task
    task = asyncio.create_task(approve_after_delay())  # noqa: RUF006, F841

    result = await manager.wait_for_decision()

    assert result.approved is True


@pytest.mark.asyncio
async def test_approval_manager_collect_approvals_empty():
    """Test collecting approvals with empty list."""
    manager = ApprovalManager()

    deferred = DeferredToolRequests()
    deferred.approvals = []

    results = await manager.collect_approvals(deferred)

    assert len(results.approvals) == 0


# =============================================================================
# ApprovalUI Tests
# =============================================================================


def test_approval_ui_init():
    """Test ApprovalUI initialization."""
    renderer = RichRenderer(width=100)
    ui = ApprovalUI(renderer, code_theme="monokai")
    assert ui is not None


def test_approval_ui_render_approval_panel():
    """Test rendering approval panel."""
    renderer = RichRenderer(width=100)
    ui = ApprovalUI(renderer)

    tool = ToolCallPart(
        tool_name="shell",
        args='{"command": "rm -rf /"}',
        tool_call_id="call-1",
    )

    result = ui.render_approval_panel(tool, index=1, total=3)

    assert "Tool 1 of 3" in result
    assert "shell" in result
    assert "command" in result
    assert "Approval Required" in result


def test_approval_ui_render_approval_result_approved():
    """Test rendering approved result."""
    renderer = RichRenderer(width=100)
    ui = ApprovalUI(renderer)

    result = ui.render_approval_result("shell", approved=True)

    assert "Approved" in result
    assert "shell" in result


def test_approval_ui_render_approval_result_rejected():
    """Test rendering rejected result."""
    renderer = RichRenderer(width=100)
    ui = ApprovalUI(renderer)

    result = ui.render_approval_result("edit", approved=False, reason="Too dangerous")

    assert "Rejected" in result
    assert "edit" in result
    assert "Too dangerous" in result


def test_approval_ui_render_approval_header():
    """Test rendering approval header."""
    renderer = RichRenderer(width=100)
    ui = ApprovalUI(renderer)

    result = ui.render_approval_header(5)

    assert "5" in result
    assert "approval" in result.lower()


def test_approval_ui_format_args_dict():
    """Test formatting dict arguments."""
    renderer = RichRenderer(width=100)
    ui = ApprovalUI(renderer)

    args = {"file_path": "test.py", "content": "Hello"}
    result = ui._format_args(args)

    assert "file_path" in result
    assert "test.py" in result


def test_approval_ui_format_args_json_string():
    """Test formatting JSON string arguments."""
    renderer = RichRenderer(width=100)
    ui = ApprovalUI(renderer)

    args = '{"key": "value"}'
    result = ui._format_args(args)

    assert "key" in result
    assert "value" in result


def test_approval_ui_format_args_truncation():
    """Test truncation of long arguments."""
    renderer = RichRenderer(width=100)
    ui = ApprovalUI(renderer)

    args = {"long_content": "x" * 1000}
    result = ui._format_args(args, max_str_len=100)

    assert "more chars" in result


def test_approval_ui_format_args_line_limit():
    """Test line limiting."""
    renderer = RichRenderer(width=100)
    ui = ApprovalUI(renderer)

    args = {f"key_{i}": f"value_{i}" for i in range(50)}
    result = ui._format_args(args, max_lines=10)

    assert "more lines" in result
