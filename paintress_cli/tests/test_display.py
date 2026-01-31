"""Tests for paintress_cli.display module.

Tests cover:
- ToolCallState and RenderDirective enums
- ToolCallInfo data class
- ToolCallTracker class
- RichRenderer class
- ToolMessage class
- EventRenderer class
"""

from __future__ import annotations

import time

from paintress_cli.display import (
    EventRenderer,
    RenderDirective,
    RichRenderer,
    ToolCallInfo,
    ToolCallState,
    ToolCallTracker,
    ToolMessage,
)

# =============================================================================
# Enum Tests
# =============================================================================


def test_tool_call_state_values():
    """Test ToolCallState enum values."""
    assert ToolCallState.CALLING == "calling"
    assert ToolCallState.COMPLETE == "complete"
    assert ToolCallState.RENDERED == "rendered"


def test_render_directive_values():
    """Test RenderDirective enum values."""
    assert RenderDirective.CALLING == "calling"
    assert RenderDirective.COMPLETE == "complete"
    assert RenderDirective.TEXT == "text"
    assert RenderDirective.RESULT == "result"


# =============================================================================
# ToolCallInfo Tests
# =============================================================================


def test_tool_call_info_creation():
    """Test ToolCallInfo creation with required fields."""
    info = ToolCallInfo(
        tool_call_id="test-123",
        name="grep",
        state=ToolCallState.CALLING,
        start_time=time.time(),
    )
    assert info.tool_call_id == "test-123"
    assert info.name == "grep"
    assert info.state == ToolCallState.CALLING
    assert info.args is None
    assert info.end_time is None
    assert info.result is None


def test_tool_call_info_with_args():
    """Test ToolCallInfo with args as dict."""
    info = ToolCallInfo(
        tool_call_id="test-456",
        name="edit",
        args={"file_path": "test.py", "old_string": "a", "new_string": "b"},
        state=ToolCallState.CALLING,
        start_time=time.time(),
    )
    assert info.args == {"file_path": "test.py", "old_string": "a", "new_string": "b"}


def test_tool_call_info_duration_running():
    """Test duration calculation for running tool."""
    start = time.time() - 1.5  # Started 1.5 seconds ago
    info = ToolCallInfo(
        tool_call_id="test-789",
        name="shell",
        state=ToolCallState.CALLING,
        start_time=start,
    )
    duration = info.duration()
    assert 1.4 < duration < 2.0  # Should be around 1.5 seconds


def test_tool_call_info_duration_completed():
    """Test duration calculation for completed tool."""
    start = time.time() - 2.0
    end = start + 1.5
    info = ToolCallInfo(
        tool_call_id="test-abc",
        name="view",
        state=ToolCallState.COMPLETE,
        start_time=start,
        end_time=end,
    )
    duration = info.duration()
    assert abs(duration - 1.5) < 0.01  # Should be exactly 1.5 seconds


def test_tool_call_info_is_special_tool():
    """Test is_special_tool identification."""
    special_tools = [
        "edit",
        "thinking",
        "to_do_read",
        "to_do_write",
        "multi_edit",
        "task_create",
        "task_get",
        "task_update",
        "task_list",
    ]
    normal_tools = ["grep", "shell", "view", "glob", "search"]

    for tool_name in special_tools:
        info = ToolCallInfo(
            tool_call_id=f"test-{tool_name}",
            name=tool_name,
            state=ToolCallState.CALLING,
            start_time=time.time(),
        )
        assert info.is_special_tool(), f"{tool_name} should be special"

    for tool_name in normal_tools:
        info = ToolCallInfo(
            tool_call_id=f"test-{tool_name}",
            name=tool_name,
            state=ToolCallState.CALLING,
            start_time=time.time(),
        )
        assert not info.is_special_tool(), f"{tool_name} should not be special"


# =============================================================================
# ToolCallTracker Tests
# =============================================================================


def test_tracker_start_call():
    """Test starting a tool call."""
    tracker = ToolCallTracker()
    tracker.start_call("call-1", "grep", {"pattern": "test"})

    assert "call-1" in tracker.tool_calls
    assert tracker.tool_calls["call-1"].name == "grep"
    assert tracker.tool_calls["call-1"].args == {"pattern": "test"}
    assert tracker.tool_calls["call-1"].state == ToolCallState.CALLING
    assert "call-1" in tracker.call_order


def test_tracker_complete_call():
    """Test completing a tool call."""
    tracker = ToolCallTracker()
    tracker.start_call("call-1", "grep")
    tracker.complete_call("call-1", result="Found 5 matches")

    info = tracker.tool_calls["call-1"]
    assert info.state == ToolCallState.COMPLETE
    assert info.result == "Found 5 matches"
    assert info.end_time is not None


def test_tracker_complete_nonexistent():
    """Test completing a non-existent call does nothing."""
    tracker = ToolCallTracker()
    tracker.complete_call("nonexistent")  # Should not raise
    assert "nonexistent" not in tracker.tool_calls


def test_tracker_mark_rendered():
    """Test marking a call as rendered."""
    tracker = ToolCallTracker()
    tracker.start_call("call-1", "view")
    tracker.complete_call("call-1")
    tracker.mark_rendered("call-1")

    assert tracker.tool_calls["call-1"].state == ToolCallState.RENDERED


def test_tracker_get_calling_tools():
    """Test getting tools in CALLING state."""
    tracker = ToolCallTracker()
    tracker.start_call("call-1", "grep")
    tracker.start_call("call-2", "view")
    tracker.complete_call("call-2")

    calling = tracker.get_calling_tools()
    assert len(calling) == 1
    assert calling[0].tool_call_id == "call-1"


def test_tracker_get_completed_tools():
    """Test getting tools in COMPLETE state."""
    tracker = ToolCallTracker()
    tracker.start_call("call-1", "grep")
    tracker.start_call("call-2", "view")
    tracker.start_call("call-3", "shell")
    tracker.complete_call("call-2")
    tracker.complete_call("call-3")
    tracker.mark_rendered("call-3")

    completed = tracker.get_completed_tools()
    assert len(completed) == 1
    assert completed[0].tool_call_id == "call-2"


def test_tracker_has_active_calls():
    """Test checking for active calls."""
    tracker = ToolCallTracker()
    assert not tracker.has_active_calls()

    tracker.start_call("call-1", "grep")
    assert tracker.has_active_calls()

    tracker.complete_call("call-1")
    assert tracker.has_active_calls()  # COMPLETE is still "active"

    tracker.mark_rendered("call-1")
    assert not tracker.has_active_calls()


def test_tracker_clear():
    """Test clearing tracker state."""
    tracker = ToolCallTracker()
    tracker.start_call("call-1", "grep")
    tracker.start_call("call-2", "view")

    tracker.clear()

    assert len(tracker.tool_calls) == 0
    assert len(tracker.call_order) == 0


def test_tracker_preserves_order():
    """Test that call order is preserved."""
    tracker = ToolCallTracker()
    tracker.start_call("third", "c")
    tracker.start_call("first", "a")
    tracker.start_call("second", "b")

    assert tracker.call_order == ["third", "first", "second"]


# =============================================================================
# RichRenderer Tests
# =============================================================================


def test_renderer_init_default_width():
    """Test renderer initialization with default width."""
    renderer = RichRenderer()
    assert renderer._width == 120


def test_renderer_init_custom_width():
    """Test renderer initialization with custom width."""
    renderer = RichRenderer(width=80)
    assert renderer._width == 80


def test_renderer_render_text():
    """Test rendering plain text."""
    renderer = RichRenderer(width=80)
    result = renderer.render_text("Hello World")
    assert "Hello World" in result


def test_renderer_render_text_with_style():
    """Test rendering styled text."""
    renderer = RichRenderer(width=80)
    result = renderer.render_text("Error message", style="bold red")
    # Should contain ANSI codes
    assert "\x1b[" in result or "Error message" in result


def test_renderer_render_markdown():
    """Test rendering markdown content."""
    renderer = RichRenderer(width=80)
    result = renderer.render_markdown("# Title\n\nSome **bold** text.")
    # Should contain the text content
    assert "Title" in result
    assert "bold" in result


def test_renderer_render_panel():
    """Test rendering a panel."""
    renderer = RichRenderer(width=80)
    result = renderer.render_panel("Content", title="Test Panel", border_style="blue")
    assert "Content" in result
    assert "Test Panel" in result


def test_renderer_width_override():
    """Test that render width can be overridden per call."""
    renderer = RichRenderer(width=120)
    result1 = renderer.render_text(
        "Test",
    )
    result2 = renderer.render("Test", width=40)
    # Both should contain the text
    assert "Test" in result1
    assert "Test" in result2


# =============================================================================
# ToolMessage Tests
# =============================================================================


def test_tool_message_creation():
    """Test ToolMessage creation."""
    msg = ToolMessage(
        tool_call_id="msg-123",
        name="grep",
        args={"pattern": "test"},
        content="Found 3 matches",
    )
    assert msg.tool_call_id == "msg-123"
    assert msg.name == "grep"
    assert msg.args == {"pattern": "test"}
    assert msg.content == "Found 3 matches"


def test_tool_message_to_panel():
    """Test ToolMessage.to_panel() produces a Panel."""
    msg = ToolMessage(
        tool_call_id="msg-456",
        name="view",
        args={"file_path": "test.py"},
        content="file content here",
    )
    panel = msg.to_panel()
    # Panel should have title containing tool name
    assert "view" in str(panel.title)


def test_tool_message_to_inline_text():
    """Test ToolMessage.to_inline_text() produces Text."""
    msg = ToolMessage(
        tool_call_id="msg-789",
        name="grep",
        args={"pattern": "test"},
        content="Found 5 matches",
    )
    text = msg.to_inline_text(duration=1.5)
    # Text should contain tool name
    assert "grep" in text.plain


def test_tool_message_to_inline_text_error():
    """Test ToolMessage.to_inline_text() with error result."""
    msg = ToolMessage(
        tool_call_id="msg-err",
        name="shell",
        args={"command": "invalid"},
        content="Tool execution error: Command failed",
    )
    text = msg.to_inline_text(duration=0.5)
    # Should indicate error
    assert "Error" in text.plain or "shell" in text.plain


def test_tool_message_format_inline_args_dict():
    """Test formatting dict args for inline display."""
    msg = ToolMessage(
        tool_call_id="test",
        name="test",
        args={"key1": "value1", "key2": "value2"},
    )
    result, truncated = msg._format_inline_args_with_info(max_length=100)
    assert "key1" in result
    assert "value1" in result
    assert truncated == 0


def test_tool_message_format_inline_args_truncation():
    """Test truncation of long args."""
    msg = ToolMessage(
        tool_call_id="test",
        name="test",
        args={"long_key": "x" * 200},
    )
    result, truncated = msg._format_inline_args_with_info(max_length=50)
    assert len(result) <= 50
    assert truncated > 0


def test_tool_message_format_inline_output_json():
    """Test formatting JSON output."""
    msg = ToolMessage(
        tool_call_id="test",
        name="test",
        content='{"status": "ok", "count": 5}',
    )
    result, _ = msg._format_inline_output_with_info(max_lines=5, max_line_length=100)
    assert "status" in result or "ok" in result


def test_tool_message_format_inline_output_plain_text():
    """Test formatting plain text output."""
    msg = ToolMessage(
        tool_call_id="test",
        name="test",
        content="Line 1\nLine 2\nLine 3",
    )
    result, truncated = msg._format_inline_output_with_info(max_lines=5, max_line_length=100)
    assert "Line 1" in result


def test_tool_message_special_panel_edit():
    """Test special panel for edit tool."""
    msg = ToolMessage(
        tool_call_id="edit-1",
        name="edit",
        args={
            "file_path": "test.py",
            "old_string": "def old():",
            "new_string": "def new():",
        },
        content="File edited successfully",
    )
    panel = msg.to_special_panel(code_theme="monokai")
    assert panel is not None
    # Should have green border for edit
    assert panel.border_style == "green"


def test_tool_message_special_panel_thinking():
    """Test special panel for thinking tool."""
    msg = ToolMessage(
        tool_call_id="think-1",
        name="thinking",
        args={"thought": "I need to analyze this problem..."},
        content="Thought recorded",
    )
    panel = msg.to_special_panel()
    assert panel is not None
    # Should have magenta border for thinking
    assert panel.border_style == "magenta"


def test_tool_message_special_panel_task():
    """Test special panel for task tools."""
    msg = ToolMessage(
        tool_call_id="task-1",
        name="task_list",
        content="#1 [pending] Do something\n#2 [completed] Done task",
    )
    panel = msg.to_special_panel()
    assert panel is not None
    # Should have cyan border for task
    assert panel.border_style == "cyan"


def test_tool_message_generate_clean_diff():
    """Test diff generation for edit operations."""
    msg = ToolMessage(tool_call_id="test", name="edit")

    old_string = "line1\nline2\nline3"
    new_string = "line1\nmodified\nline3"

    diff = msg._generate_clean_diff_content(old_string, new_string)
    # Should show the change
    assert "-line2" in diff or "modified" in diff


# =============================================================================
# EventRenderer Tests
# =============================================================================


def test_event_renderer_init():
    """Test EventRenderer initialization."""
    renderer = EventRenderer(width=100, code_theme="monokai")
    assert renderer._code_theme == "monokai"
    assert renderer._tracker is not None


def test_event_renderer_clear():
    """Test clearing EventRenderer state."""
    renderer = EventRenderer()
    renderer._tracker.start_call("test", "grep")
    renderer._current_text = "Some text"
    renderer._current_thinking = "Some thinking"

    renderer.clear()

    assert len(renderer._tracker.tool_calls) == 0
    assert renderer._current_text == ""
    assert renderer._current_thinking == ""


def test_event_renderer_thinking():
    """Test thinking content management."""
    renderer = EventRenderer()

    renderer.start_thinking("Initial thought")
    assert renderer.get_current_thinking() == "Initial thought"

    renderer.update_thinking(" more thinking")
    assert renderer.get_current_thinking() == "Initial thought more thinking"


def test_event_renderer_render_thinking():
    """Test rendering thinking content."""
    renderer = EventRenderer(width=80)
    result = renderer.render_thinking("This is my analysis...")
    # Should contain the thinking text and be formatted as blockquote
    assert "analysis" in result
    assert ">" in result  # Blockquote prefix


def test_event_renderer_render_tool_call_start():
    """Test rendering tool call start."""
    renderer = EventRenderer()
    result = renderer.render_tool_call_start("grep", "call-123")
    assert "Calling" in result
    assert "grep" in result


def test_event_renderer_render_tool_call_complete_normal():
    """Test rendering normal tool completion."""
    renderer = EventRenderer(width=100)
    msg = ToolMessage(
        tool_call_id="call-1",
        name="grep",
        args={"pattern": "test"},
        content="Found 3 matches",
    )
    result = renderer.render_tool_call_complete(msg, duration=1.2)
    assert "grep" in result
    assert "Complete" in result


def test_event_renderer_render_tool_call_complete_special():
    """Test rendering special tool completion (uses panel)."""
    renderer = EventRenderer(width=100)
    msg = ToolMessage(
        tool_call_id="call-2",
        name="edit",
        args={"file_path": "test.py", "old_string": "a", "new_string": "b"},
        content="File edited",
    )
    result = renderer.render_tool_call_complete(msg, duration=0.5)
    # Special tools use panel format
    assert "edit" in result


def test_event_renderer_render_markdown():
    """Test markdown rendering."""
    renderer = EventRenderer()
    result = renderer.render_markdown("# Header\n\nParagraph text")
    assert "Header" in result


def test_event_renderer_render_compact_start():
    """Test compact start notification."""
    renderer = EventRenderer()
    result = renderer.render_compact_start(50)
    assert "50" in result
    assert "compact" in result.lower()


def test_event_renderer_render_compact_complete():
    """Test compact complete panel."""
    renderer = EventRenderer()
    result = renderer.render_compact_complete(100, 30, "Summary text")
    assert "100" in result
    assert "30" in result
    assert "70%" in result  # 70% reduction


def test_event_renderer_render_compact_failed():
    """Test compact failed notification."""
    renderer = EventRenderer()
    result = renderer.render_compact_failed("Context too short")
    assert "failed" in result.lower()
    assert "Context too short" in result


def test_event_renderer_render_handoff_start():
    """Test handoff start notification."""
    renderer = EventRenderer()
    result = renderer.render_handoff_start(75)
    assert "75" in result
    assert "handoff" in result.lower()


def test_event_renderer_render_handoff_complete():
    """Test handoff complete panel."""
    renderer = EventRenderer()
    result = renderer.render_handoff_complete("Preserved context summary")
    assert "Preserved context summary" in result


def test_event_renderer_render_handoff_failed():
    """Test handoff failed notification."""
    renderer = EventRenderer()
    result = renderer.render_handoff_failed("Export failed")
    assert "failed" in result.lower()
    assert "Export failed" in result


def test_event_renderer_render_steering_injected():
    """Test steering injected panel."""
    renderer = EventRenderer()
    result = renderer.render_steering_injected(["Focus on tests", "Skip docs"])
    assert "2" in result  # 2 messages
    assert "Focus on tests" in result


def test_event_renderer_tracker_property():
    """Test tracker property access."""
    renderer = EventRenderer()
    tracker = renderer.tracker

    tracker.start_call("test", "grep")

    # Should be the same instance
    assert renderer.tracker is tracker
    assert "test" in renderer.tracker.tool_calls


# =============================================================================
# Integration Tests
# =============================================================================


def test_full_tool_lifecycle():
    """Test complete tool call lifecycle through tracker and renderer."""
    renderer = EventRenderer(width=100)

    # Start tool
    start_output = renderer.render_tool_call_start("view", "call-1")
    renderer.tracker.start_call("call-1", "view", {"file_path": "test.py"})

    assert "Calling" in start_output
    assert renderer.tracker.has_active_calls()

    # Complete tool
    renderer.tracker.complete_call("call-1", result="file content")
    msg = ToolMessage(
        tool_call_id="call-1",
        name="view",
        args={"file_path": "test.py"},
        content="file content",
    )

    complete_output = renderer.render_tool_call_complete(msg, duration=0.3)
    assert "view" in complete_output

    # Mark rendered
    renderer.tracker.mark_rendered("call-1")
    assert not renderer.tracker.has_active_calls()


def test_multiple_tools_parallel():
    """Test multiple parallel tool calls."""
    tracker = ToolCallTracker()

    # Start multiple tools
    tracker.start_call("call-1", "grep")
    tracker.start_call("call-2", "view")
    tracker.start_call("call-3", "glob")

    assert len(tracker.get_calling_tools()) == 3

    # Complete in different order
    tracker.complete_call("call-2")
    tracker.complete_call("call-1")

    calling = tracker.get_calling_tools()
    completed = tracker.get_completed_tools()

    assert len(calling) == 1
    assert calling[0].name == "glob"
    assert len(completed) == 2
