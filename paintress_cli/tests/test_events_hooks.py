"""Tests for events, session, and hooks modules."""

from __future__ import annotations

import pytest
from paintress_cli.events import (
    AgentPhase,
    AgentPhaseEvent,
    ProcessExitedEvent,
    ProcessOutputEvent,
    ProcessStartedEvent,
    SteeringInjectedEvent,
)
from paintress_cli.hooks import emit_phase_event
from paintress_cli.session import TUIContext, create_steering_filter, render_steering_messages
from paintress_cli.steering import SteeringMessage
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart

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


class TestSteeringInjectedEvent:
    """Tests for SteeringInjectedEvent."""

    def test_create_event(self):
        """Test creating steering injected event."""
        event = SteeringInjectedEvent(
            event_id="steer-abc",
            message_count=2,
            content="Focus on the UI...",
        )
        assert event.event_id == "steer-abc"
        assert event.message_count == 2
        assert event.content == "Focus on the UI..."

    def test_inherits_from_agent_event(self):
        """Test that event inherits from AgentEvent."""
        from pai_agent_sdk.events import AgentEvent

        event = SteeringInjectedEvent(event_id="test")
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


class TestRenderSteeringMessages:
    """Tests for render_steering_messages function."""

    def test_render_single_message(self):
        """Test rendering a single steering message."""
        messages = [SteeringMessage(message_id="steer-1", prompt="Focus on tests")]
        parts = render_steering_messages(messages)

        assert len(parts) == 1
        assert isinstance(parts[0], UserPromptPart)
        assert "<steering>" in parts[0].content
        assert "Focus on tests" in parts[0].content
        assert "<system-reminder>" in parts[0].content

    def test_render_multiple_messages(self):
        """Test rendering multiple steering messages."""
        messages = [
            SteeringMessage(message_id="steer-1", prompt="Message 1"),
            SteeringMessage(message_id="steer-2", prompt="Message 2"),
        ]
        parts = render_steering_messages(messages)

        content = parts[0].content
        assert "Message 1" in content
        assert "Message 2" in content


class TestTUIContext:
    """Tests for TUIContext."""

    def test_has_steering_manager(self):
        """Test that TUIContext has steering manager."""
        ctx = TUIContext()
        assert ctx.steering_manager is not None
        assert hasattr(ctx.steering_manager, "enqueue")

    def test_steering_manager_is_local(self):
        """Test steering manager is LocalSteeringManager."""
        from paintress_cli.steering import LocalSteeringManager

        ctx = TUIContext()
        assert isinstance(ctx.steering_manager, LocalSteeringManager)

    def test_get_history_processors_includes_steering(self):
        """Test that steering filter is in history processors."""
        ctx = TUIContext()
        processors = ctx.get_history_processors()

        # Should have parent processors + steering
        assert len(processors) > 0
        # Last processor should be a callable (the steering filter)
        assert callable(processors[-1])

    @pytest.mark.asyncio
    async def test_inject_steering_into_request(self):
        """Test steering injection into message history."""
        ctx = TUIContext()
        # Enable stream queue for event emission
        ctx._stream_queue_enabled = True

        # Enqueue a steering message
        await ctx.steering_manager.enqueue("Focus on performance")

        # Create message history with a request
        history: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content="Build an app")])]

        # Mock RunContext with proper typing
        class MockRunContext:
            deps = ctx

        mock_ctx = MockRunContext()

        # Inject steering using factory-created filter
        steering_filter = create_steering_filter(ctx)
        result = await steering_filter(mock_ctx, history)  # type: ignore[arg-type]

        # Verify injection
        assert len(result) == 1
        assert isinstance(result[0], ModelRequest)
        assert len(result[0].parts) == 2  # Original + steering
        part = result[0].parts[1]
        assert isinstance(part, UserPromptPart)
        assert "<steering>" in part.content
        assert "Focus on performance" in part.content

        # Verify buffer was consumed
        assert not ctx.steering_manager.has_pending()

        # Verify event was emitted (emit_event uses _agent_id as key)
        queue = ctx.agent_stream_queues[ctx._agent_id]
        assert not queue.empty()
        event = queue.get_nowait()
        assert isinstance(event, SteeringInjectedEvent)
        assert event.message_count == 1

    @pytest.mark.asyncio
    async def test_no_injection_on_empty_buffer(self):
        """Test no injection when buffer is empty."""
        ctx = TUIContext()

        history: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content="Build an app")])]

        class MockRunContext:
            deps = ctx

        steering_filter = create_steering_filter(ctx)
        result = await steering_filter(MockRunContext(), history)  # type: ignore[arg-type]

        # Should be unchanged
        assert isinstance(result[0], ModelRequest)
        assert len(result[0].parts) == 1

    @pytest.mark.asyncio
    async def test_no_injection_on_response(self):
        """Test no injection on model response (only on request)."""
        ctx = TUIContext()

        await ctx.steering_manager.enqueue("Focus on tests")

        # History ends with a response
        history: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="Build an app")]),
            ModelResponse(parts=[TextPart(content="I will build...")], model_name="test"),
        ]

        class MockRunContext:
            deps = ctx

        steering_filter = create_steering_filter(ctx)
        result = await steering_filter(MockRunContext(), history)  # type: ignore[arg-type]

        # Should be unchanged
        assert len(result) == 2
        # Message should still be pending
        assert ctx.steering_manager.has_pending()

    def test_create_subagent_context_returns_agent_context(self):
        """Test that subagent context is plain AgentContext."""
        from pai_agent_sdk.context import AgentContext

        ctx = TUIContext()
        sub_ctx = ctx.create_subagent_context("search")

        # Should be AgentContext, not TUIContext
        # (subagents don't get steering)
        assert isinstance(sub_ctx, AgentContext)


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
