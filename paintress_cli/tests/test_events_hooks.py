"""Tests for events, session, and hooks modules."""

from __future__ import annotations

import pytest
from paintress_cli.events import (
    AgentPhase,
    AgentPhaseEvent,
    ProcessExitedEvent,
    ProcessOutputEvent,
    ProcessStartedEvent,
)
from paintress_cli.hooks import emit_phase_event
from paintress_cli.session import TUIContext

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.events import BusMessageInfo, MessageReceivedEvent

# =============================================================================
# Event Tests
# =============================================================================


class TestAgentPhaseEvent:
    """Tests for AgentPhaseEvent."""

    def test_create_phase_event(self):
        """Test creating a phase event."""
        event = AgentPhaseEvent(
            event_id="test-123",
            phase=AgentPhase.GENERATING,
            agent_id="agent-1",
            agent_name="main",
            node_type="ModelRequestNode",
            details="Calling claude-4-sonnet",
        )
        assert event.event_id == "test-123"
        assert event.phase == AgentPhase.GENERATING
        assert event.agent_name == "main"

    def test_phase_enum_values(self):
        """Test all phase enum values."""
        assert AgentPhase.IDLE == "idle"
        assert AgentPhase.GENERATING == "generating"
        assert AgentPhase.EXECUTING == "executing"
        assert AgentPhase.COMPLETED == "completed"


class TestMessageReceivedEvent:
    """Tests for MessageReceivedEvent (from SDK)."""

    def test_create_event(self):
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

    def test_inherits_from_agent_event(self):
        """Test that event inherits from AgentEvent."""
        from pai_agent_sdk.events import AgentEvent

        event = MessageReceivedEvent(event_id="test", messages=[])
        assert isinstance(event, AgentEvent)
        assert isinstance(event, AgentEvent)


class TestProcessEvents:
    """Tests for process lifecycle events."""

    def test_process_started(self):
        """Test ProcessStartedEvent."""
        event = ProcessStartedEvent(
            event_id="proc-1",
            process_id="proc-1",
            command="npm run build",
            pid=12345,
        )
        assert event.process_id == "proc-1"
        assert event.pid == 12345

    def test_process_output(self):
        """Test ProcessOutputEvent."""
        event = ProcessOutputEvent(
            event_id="out-1",
            process_id="proc-1",
            output="Build complete",
            is_stderr=False,
        )
        assert event.output == "Build complete"
        assert event.is_stderr is False

    def test_process_exited(self):
        """Test ProcessExitedEvent."""
        event = ProcessExitedEvent(
            event_id="exit-1",
            process_id="proc-1",
            exit_code=0,
        )
        assert event.exit_code == 0


# =============================================================================
# Session Tests
# =============================================================================


class TestTUIContext:
    """Tests for TUIContext."""

    def test_inherits_from_agent_context(self):
        """Test that TUIContext inherits from AgentContext."""
        ctx = TUIContext()
        assert isinstance(ctx, AgentContext)

    def test_has_message_bus(self):
        """Test that TUIContext has message bus (inherited)."""
        ctx = TUIContext()
        assert ctx.message_bus is not None
        assert hasattr(ctx.message_bus, "send")

    def test_send_message(self):
        """Test sending message via TUIContext."""
        ctx = TUIContext()
        msg = ctx.send_message("Focus on performance", source="user")
        assert msg.content == "Focus on performance"
        assert msg.source == "user"

    def test_message_bus_pending(self):
        """Test message bus pending check."""
        ctx = TUIContext()
        ctx.message_bus.subscribe("main")
        ctx.send_message("Test message", source="user", target="main")
        assert ctx.message_bus.has_pending("main")

    def test_create_subagent_context_returns_agent_context(self):
        """Test that subagent context is AgentContext."""
        ctx = TUIContext()
        sub_ctx = ctx.create_subagent_context("search")

        # Should be AgentContext (subagent shares message_bus)
        assert isinstance(sub_ctx, AgentContext)
        # Should share message_bus with parent
        assert sub_ctx.message_bus is ctx.message_bus


# =============================================================================
# Hooks Tests
# =============================================================================


class TestEmitPhaseEvent:
    """Tests for emit_phase_event utility."""

    @pytest.mark.asyncio
    async def test_emit_phase_event(self):
        """Test emitting a phase event."""
        ctx = TUIContext()
        ctx._stream_queue_enabled = True

        await emit_phase_event(
            ctx,
            phase=AgentPhase.GENERATING,
            agent_id="main",
            agent_name="main",
            node_type="ModelRequestNode",
            details="Generating...",
        )

        # Check event was emitted (emit_event uses _agent_id as key)
        queue = ctx.agent_stream_queues[ctx._agent_id]
        assert not queue.empty()

        event = queue.get_nowait()
        assert isinstance(event, AgentPhaseEvent)
        assert event.phase == AgentPhase.GENERATING
        assert event.agent_name == "main"
