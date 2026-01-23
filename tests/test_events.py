"""Tests for pai_agent_sdk.events module."""

from dataclasses import dataclass
from datetime import datetime

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.events import AgentEvent, CompactCompleteEvent, CompactFailedEvent, CompactStartEvent

# =============================================================================
# AgentEvent Tests
# =============================================================================


def test_agent_event_creation() -> None:
    """AgentEvent should be created with event_id and auto timestamp."""
    event = AgentEvent(event_id="test-123")
    assert event.event_id == "test-123"
    assert isinstance(event.timestamp, datetime)


def test_agent_event_custom_timestamp() -> None:
    """AgentEvent should accept custom timestamp."""
    custom_time = datetime(2024, 1, 1, 12, 0, 0)
    event = AgentEvent(event_id="test-123", timestamp=custom_time)
    assert event.timestamp == custom_time


# =============================================================================
# CompactStartEvent Tests
# =============================================================================


def test_compact_start_event_creation() -> None:
    """CompactStartEvent should be created with required fields."""
    event = CompactStartEvent(event_id="compact-001", message_count=50)
    assert event.event_id == "compact-001"
    assert event.message_count == 50
    assert isinstance(event.timestamp, datetime)


def test_compact_start_event_default_message_count() -> None:
    """CompactStartEvent should default message_count to 0."""
    event = CompactStartEvent(event_id="compact-001")
    assert event.message_count == 0


# =============================================================================
# CompactCompleteEvent Tests
# =============================================================================


def test_compact_complete_event_creation() -> None:
    """CompactCompleteEvent should be created with all fields."""
    event = CompactCompleteEvent(
        event_id="compact-001",
        summary_markdown="# Summary\nCompacted content here.",
        original_message_count=100,
        compacted_message_count=5,
    )
    assert event.event_id == "compact-001"
    assert event.summary_markdown == "# Summary\nCompacted content here."
    assert event.original_message_count == 100
    assert event.compacted_message_count == 5


def test_compact_complete_event_defaults() -> None:
    """CompactCompleteEvent should have sensible defaults."""
    event = CompactCompleteEvent(event_id="compact-001")
    assert event.summary_markdown == ""
    assert event.original_message_count == 0
    assert event.compacted_message_count == 0


# =============================================================================
# CompactFailedEvent Tests
# =============================================================================


def test_compact_failed_event_creation() -> None:
    """CompactFailedEvent should be created with error and message_count."""
    event = CompactFailedEvent(
        event_id="compact-001",
        error="API rate limit exceeded",
        message_count=50,
    )
    assert event.event_id == "compact-001"
    assert event.error == "API rate limit exceeded"
    assert event.message_count == 50


def test_compact_failed_event_defaults() -> None:
    """CompactFailedEvent should have sensible defaults."""
    event = CompactFailedEvent(event_id="compact-001")
    assert event.error == ""
    assert event.message_count == 0


# =============================================================================
# AgentContext.emit_event Tests
# =============================================================================


async def test_emit_event_puts_event_in_queue(agent_context: AgentContext) -> None:
    """emit_event should put event into the agent's stream queue."""
    # Enable stream queue for testing
    agent_context._stream_queue_enabled = True

    event = CompactStartEvent(event_id="test-001", message_count=10)

    await agent_context.emit_event(event)

    queue = agent_context.agent_stream_queues[agent_context._agent_id]
    assert not queue.empty()

    received = await queue.get()
    assert received is event
    assert received.event_id == "test-001"
    assert received.message_count == 10


async def test_emit_event_multiple_events(agent_context: AgentContext) -> None:
    """emit_event should queue multiple events in order."""
    # Enable stream queue for testing
    agent_context._stream_queue_enabled = True

    start_event = CompactStartEvent(event_id="test-001", message_count=50)
    complete_event = CompactCompleteEvent(
        event_id="test-001",
        summary_markdown="Done",
        original_message_count=50,
        compacted_message_count=3,
    )

    await agent_context.emit_event(start_event)
    await agent_context.emit_event(complete_event)

    queue = agent_context.agent_stream_queues[agent_context._agent_id]

    first = await queue.get()
    assert isinstance(first, CompactStartEvent)
    assert first.event_id == "test-001"

    second = await queue.get()
    assert isinstance(second, CompactCompleteEvent)
    assert second.event_id == "test-001"


async def test_emit_event_failed_event(agent_context: AgentContext) -> None:
    """emit_event should handle CompactFailedEvent."""
    # Enable stream queue for testing
    agent_context._stream_queue_enabled = True

    start_event = CompactStartEvent(event_id="test-fail", message_count=30)
    failed_event = CompactFailedEvent(event_id="test-fail", error="Model unavailable", message_count=30)

    await agent_context.emit_event(start_event)
    await agent_context.emit_event(failed_event)

    queue = agent_context.agent_stream_queues[agent_context._agent_id]

    first = await queue.get()
    assert isinstance(first, CompactStartEvent)

    second = await queue.get()
    assert isinstance(second, CompactFailedEvent)
    assert second.error == "Model unavailable"
    assert second.message_count == 30


async def test_emit_event_noop_when_disabled(agent_context: AgentContext) -> None:
    """emit_event should be no-op when stream queue is disabled."""
    # Ensure stream queue is disabled (default)
    assert not agent_context._stream_queue_enabled

    event = CompactStartEvent(event_id="test-noop", message_count=10)
    await agent_context.emit_event(event)

    # Queue should remain empty since emit_event is no-op
    queue = agent_context.agent_stream_queues[agent_context._agent_id]
    assert queue.empty()


# =============================================================================
# Event Correlation Tests
# =============================================================================


def test_event_id_correlation() -> None:
    """Start and Complete events should be correlated by event_id."""
    event_id = "correlation-test-abc"

    start = CompactStartEvent(event_id=event_id, message_count=100)
    complete = CompactCompleteEvent(
        event_id=event_id,
        summary_markdown="Summary",
        original_message_count=100,
        compacted_message_count=5,
    )

    assert start.event_id == complete.event_id


def test_event_id_correlation_with_failed() -> None:
    """Start and Failed events should be correlated by event_id."""
    event_id = "fail-correlation-xyz"

    start = CompactStartEvent(event_id=event_id, message_count=80)
    failed = CompactFailedEvent(event_id=event_id, error="Timeout", message_count=80)

    assert start.event_id == failed.event_id


# =============================================================================
# Custom Event Subclass Test
# =============================================================================


def test_custom_event_subclass() -> None:
    """Users should be able to create custom event subclasses."""

    @dataclass
    class CustomAgentEvent(AgentEvent):
        custom_field: str = ""

    event = CustomAgentEvent(event_id="custom-001", custom_field="my value")
    assert event.event_id == "custom-001"
    assert event.custom_field == "my value"
    assert isinstance(event.timestamp, datetime)
