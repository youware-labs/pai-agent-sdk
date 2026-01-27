"""Tests for events, session, and hooks modules."""

from __future__ import annotations

from paintress_cli.events import ContextUpdateEvent
from paintress_cli.session import TUIContext

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.events import BusMessageInfo, MessageReceivedEvent

# =============================================================================
# Event Tests
# =============================================================================


def test_context_update_event():
    """Test ContextUpdateEvent."""
    event = ContextUpdateEvent(
        event_id="ctx-1",
        total_tokens=50000,
        context_window_size=200000,
    )
    assert event.total_tokens == 50000
    assert event.context_window_size == 200000


def test_message_received_event():
    """Test creating message received event with messages list."""
    event = MessageReceivedEvent(
        event_id="msg-abc",
        messages=[
            BusMessageInfo(content="Focus on the UI...", rendered_content="Focus on the UI...", source="user"),
        ],
    )
    assert event.event_id == "msg-abc"
    assert len(event.messages) == 1
    assert event.messages[0].source == "user"
    assert event.messages[0].rendered_content == "Focus on the UI..."


def test_message_received_event_inherits_from_agent_event():
    """Test that event inherits from AgentEvent."""
    from pai_agent_sdk.events import AgentEvent

    event = MessageReceivedEvent(event_id="test", messages=[])
    assert isinstance(event, AgentEvent)


# =============================================================================
# Session Tests
# =============================================================================


def test_tui_context_inherits_from_agent_context():
    """Test that TUIContext inherits from AgentContext."""
    ctx = TUIContext()
    assert isinstance(ctx, AgentContext)


def test_tui_context_has_message_bus():
    """Test that TUIContext has message bus (inherited)."""
    ctx = TUIContext()
    assert ctx.message_bus is not None
    assert hasattr(ctx.message_bus, "send")


def test_tui_context_send_message():
    """Test sending message via TUIContext."""
    ctx = TUIContext()
    msg = ctx.send_message("Focus on performance", source="user")
    assert msg.content == "Focus on performance"
    assert msg.source == "user"


def test_tui_context_message_bus_pending():
    """Test message bus pending check."""
    ctx = TUIContext()
    ctx.message_bus.subscribe("main")
    ctx.send_message("Test message", source="user", target="main")
    assert ctx.message_bus.has_pending("main")


def test_tui_context_create_subagent_context():
    """Test that subagent context is AgentContext."""
    ctx = TUIContext()
    sub_ctx = ctx.create_subagent_context("search")

    # Should be AgentContext (subagent shares message_bus)
    assert isinstance(sub_ctx, AgentContext)
    # Should share message_bus with parent
    assert sub_ctx.message_bus is ctx.message_bus
