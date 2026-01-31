"""Tests for paintress_cli.streaming module."""

from __future__ import annotations

from paintress_cli.rendering import RichRenderer
from paintress_cli.streaming import (
    StreamEventHandler,
    SubagentState,
    SubagentTracker,
    TextStreamer,
    ThinkingStreamer,
    is_text_delta,
    is_text_start,
    is_thinking_delta,
    is_thinking_start,
)
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
)

# =============================================================================
# TextStreamer Tests
# =============================================================================


def test_text_streamer_init():
    """Test TextStreamer initialization."""
    renderer = RichRenderer(width=80)
    streamer = TextStreamer(renderer, code_theme="monokai")

    assert streamer.text == ""
    assert streamer.line_index is None
    assert not streamer.is_active


def test_text_streamer_start():
    """Test starting text streaming."""
    renderer = RichRenderer(width=80)
    streamer = TextStreamer(renderer)

    result = streamer.start("Hello", line_index=5)

    assert streamer.text == "Hello"
    assert streamer.line_index == 5
    assert streamer.is_active
    assert result == "Hello"


def test_text_streamer_update():
    """Test updating text stream."""
    renderer = RichRenderer(width=80)
    streamer = TextStreamer(renderer)

    streamer.start("# Header")
    result = streamer.update("\n\nParagraph text")

    assert "Header" in streamer.text
    assert "Paragraph" in streamer.text
    assert "Header" in result  # Markdown rendered


def test_text_streamer_finalize():
    """Test finalizing text stream."""
    renderer = RichRenderer(width=80)
    streamer = TextStreamer(renderer)

    streamer.start("Some text")
    streamer.update(" more")
    result = streamer.finalize()

    assert "Some text more" in result
    assert streamer.text == ""
    assert streamer.line_index is None
    assert not streamer.is_active


def test_text_streamer_custom_width():
    """Test TextStreamer with custom width callback."""
    renderer = RichRenderer(width=120)
    width_value = [100]  # Mutable for callback

    streamer = TextStreamer(renderer, get_width=lambda: width_value[0])
    streamer.start("Test")

    width_value[0] = 50
    result = streamer.update(" content")

    # Should use the callback width
    assert result is not None


# =============================================================================
# ThinkingStreamer Tests
# =============================================================================


def test_thinking_streamer_init():
    """Test ThinkingStreamer initialization."""
    renderer = RichRenderer(width=80)
    streamer = ThinkingStreamer(renderer)

    assert streamer.thinking == ""
    assert streamer.line_index is None
    assert not streamer.is_active


def test_thinking_streamer_start():
    """Test starting thinking streaming."""
    renderer = RichRenderer(width=80)
    streamer = ThinkingStreamer(renderer)

    result = streamer.start("Analyzing...", line_index=10)

    assert streamer.thinking == "Analyzing..."
    assert streamer.line_index == 10
    assert streamer.is_active
    assert ">" in result  # Blockquote prefix
    assert "Analyzing" in result


def test_thinking_streamer_update():
    """Test updating thinking stream."""
    renderer = RichRenderer(width=80)
    streamer = ThinkingStreamer(renderer)

    streamer.start("First thought")
    result = streamer.update("\nSecond thought")

    assert "First thought" in streamer.thinking
    assert "Second thought" in streamer.thinking
    assert ">" in result


def test_thinking_streamer_finalize():
    """Test finalizing thinking stream."""
    renderer = RichRenderer(width=80)
    streamer = ThinkingStreamer(renderer)

    streamer.start("Thinking")
    result = streamer.finalize()

    assert "Thinking" in result
    assert streamer.thinking == ""
    assert streamer.line_index is None
    assert not streamer.is_active


# =============================================================================
# SubagentState Tests
# =============================================================================


def test_subagent_state_creation():
    """Test SubagentState dataclass."""
    state = SubagentState(
        agent_id="explorer",
        agent_name="Explorer Agent",
        line_index=5,
    )

    assert state.agent_id == "explorer"
    assert state.agent_name == "Explorer Agent"
    assert state.line_index == 5
    assert state.tool_names == []


def test_subagent_state_with_tools():
    """Test SubagentState with tool names."""
    state = SubagentState(
        agent_id="debugger",
        agent_name="Debugger",
        line_index=10,
        tool_names=["grep", "view"],
    )

    assert state.tool_names == ["grep", "view"]


# =============================================================================
# SubagentTracker Tests
# =============================================================================


def test_subagent_tracker_init():
    """Test SubagentTracker initialization."""
    renderer = RichRenderer(width=80)
    tracker = SubagentTracker(renderer)

    assert not tracker.has_state("test")


def test_subagent_tracker_start():
    """Test starting subagent tracking."""
    renderer = RichRenderer(width=80)
    tracker = SubagentTracker(renderer)

    result = tracker.start("explorer", "Explorer Agent", line_index=5)

    assert tracker.has_state("explorer")
    state = tracker.get_state("explorer")
    assert state is not None
    assert state.agent_id == "explorer"
    assert state.line_index == 5
    assert "explorer" in result
    assert "Running" in result


def test_subagent_tracker_add_tool():
    """Test adding tool to subagent tracking."""
    renderer = RichRenderer(width=80)
    tracker = SubagentTracker(renderer)

    tracker.start("explorer", "Explorer", line_index=5)
    result = tracker.add_tool("explorer", "grep")

    state = tracker.get_state("explorer")
    assert state is not None
    assert "grep" in state.tool_names
    assert result is not None
    assert "grep" in result


def test_subagent_tracker_add_tool_untracked():
    """Test adding tool to untracked agent returns None."""
    renderer = RichRenderer(width=80)
    tracker = SubagentTracker(renderer)

    result = tracker.add_tool("unknown", "grep")

    assert result is None


def test_subagent_tracker_complete_success():
    """Test completing subagent tracking successfully."""
    renderer = RichRenderer(width=80)
    tracker = SubagentTracker(renderer)

    tracker.start("explorer", "Explorer", line_index=5)
    tracker.add_tool("explorer", "grep")

    summary, line_idx = tracker.complete(
        "explorer",
        success=True,
        duration_seconds=2.5,
        request_count=3,
        result_preview="Found relevant files",
    )

    assert not tracker.has_state("explorer")
    assert line_idx == 5
    assert "Done" in summary
    assert "2.5s" in summary
    assert "3 reqs" in summary


def test_subagent_tracker_complete_failure():
    """Test completing subagent tracking with failure."""
    renderer = RichRenderer(width=80)
    tracker = SubagentTracker(renderer)

    tracker.start("debugger", "Debugger", line_index=10)

    summary, line_idx = tracker.complete(
        "debugger",
        success=False,
        duration_seconds=1.0,
        error="Connection timeout",
    )

    assert "Failed" in summary
    assert "Connection timeout" in summary


def test_subagent_tracker_complete_untracked():
    """Test completing untracked agent."""
    renderer = RichRenderer(width=80)
    tracker = SubagentTracker(renderer)

    summary, line_idx = tracker.complete(
        "unknown",
        success=True,
        duration_seconds=1.0,
    )

    assert line_idx is None
    assert "Done" in summary


def test_subagent_tracker_clear():
    """Test clearing tracker."""
    renderer = RichRenderer(width=80)
    tracker = SubagentTracker(renderer)

    tracker.start("a", "A", 1)
    tracker.start("b", "B", 2)

    tracker.clear()

    assert not tracker.has_state("a")
    assert not tracker.has_state("b")


# =============================================================================
# StreamEventHandler Tests
# =============================================================================


def test_event_handler_init():
    """Test StreamEventHandler initialization."""
    handler = StreamEventHandler()
    assert handler is not None


def test_event_handler_on():
    """Test registering event handlers."""
    handler = StreamEventHandler()
    calls = []

    handler.on(FunctionToolCallEvent, lambda e, aid: calls.append((e, aid)))

    # Create a mock tool call event
    from pai_agent_sdk.context import StreamEvent

    tool_part = ToolCallPart(tool_name="grep", args="{}", tool_call_id="call-1")
    event = FunctionToolCallEvent(part=tool_part)
    stream_event = StreamEvent(agent_id="main", agent_name="Main", event=event)

    handler.handle(stream_event)

    assert len(calls) == 1
    assert calls[0][1] == "main"


def test_event_handler_on_any():
    """Test global event handler."""
    handler = StreamEventHandler()
    calls = []

    handler.on_any(lambda se: calls.append(se))

    from pai_agent_sdk.context import StreamEvent

    tool_part = ToolCallPart(tool_name="view", args="{}", tool_call_id="call-2")
    event = FunctionToolCallEvent(part=tool_part)
    stream_event = StreamEvent(agent_id="sub", agent_name="Sub", event=event)

    handler.handle(stream_event)

    assert len(calls) == 1
    assert calls[0].agent_id == "sub"


def test_event_handler_clear():
    """Test clearing handlers."""
    handler = StreamEventHandler()
    calls = []

    handler.on(FunctionToolCallEvent, lambda e, aid: calls.append(1))
    handler.on_any(lambda se: calls.append(2))

    handler.clear()

    from pai_agent_sdk.context import StreamEvent

    tool_part = ToolCallPart(tool_name="grep", args="{}", tool_call_id="call-3")
    event = FunctionToolCallEvent(part=tool_part)
    stream_event = StreamEvent(agent_id="main", agent_name="Main", event=event)

    handler.handle(stream_event)

    assert len(calls) == 0


# =============================================================================
# Event Type Helper Tests
# =============================================================================


def test_is_text_start():
    """Test is_text_start helper."""
    text_event = PartStartEvent(index=0, part=TextPart(content="Hello"))
    thinking_event = PartStartEvent(index=0, part=ThinkingPart(content="Thinking"))

    assert is_text_start(text_event)
    assert not is_text_start(thinking_event)


def test_is_thinking_start():
    """Test is_thinking_start helper."""
    text_event = PartStartEvent(index=0, part=TextPart(content="Hello"))
    thinking_event = PartStartEvent(index=0, part=ThinkingPart(content="Thinking"))

    assert not is_thinking_start(text_event)
    assert is_thinking_start(thinking_event)


def test_is_text_delta():
    """Test is_text_delta helper."""
    text_delta = PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="more"))
    thinking_delta = PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="thought"))

    assert is_text_delta(text_delta)
    assert not is_text_delta(thinking_delta)


def test_is_thinking_delta():
    """Test is_thinking_delta helper."""
    text_delta = PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="more"))
    thinking_delta = PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="thought"))

    assert not is_thinking_delta(text_delta)
    assert is_thinking_delta(thinking_delta)
