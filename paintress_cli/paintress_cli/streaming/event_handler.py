"""Stream event handling.

Provides StreamEventHandler for dispatching and processing agent stream events.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
)

from pai_agent_sdk.context import StreamEvent
from pai_agent_sdk.events import (
    CompactCompleteEvent,
    CompactFailedEvent,
    CompactStartEvent,
    HandoffCompleteEvent,
    HandoffFailedEvent,
    HandoffStartEvent,
    MessageReceivedEvent,
    ModelRequestStartEvent,
    SubagentCompleteEvent,
    SubagentStartEvent,
    ToolCallsStartEvent,
)

if TYPE_CHECKING:
    pass


class EventHandlerCallback(Protocol):
    """Protocol for event handler callbacks."""

    def __call__(self, event: StreamEvent) -> None:
        """Handle a stream event."""
        ...


class StreamEventHandler:
    """Dispatches stream events to appropriate handlers.

    Provides a clean separation between event routing and handling logic.
    Handlers are registered per event type and called when matching events arrive.
    """

    def __init__(self) -> None:
        """Initialize StreamEventHandler."""
        self._handlers: dict[type, list[Callable[[Any, str], None]]] = {}
        self._global_handlers: list[Callable[[StreamEvent], None]] = []

    def on(self, event_type: type, handler: Callable[[Any, str], None]) -> None:
        """Register a handler for a specific event type.

        Args:
            event_type: Type of event to handle.
            handler: Callback receiving (event, agent_id).
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def on_any(self, handler: Callable[[StreamEvent], None]) -> None:
        """Register a handler for all events.

        Args:
            handler: Callback receiving the full StreamEvent.
        """
        self._global_handlers.append(handler)

    def handle(self, stream_event: StreamEvent) -> None:
        """Handle a stream event by dispatching to registered handlers.

        Args:
            stream_event: The stream event to handle.
        """
        event = stream_event.event
        agent_id = stream_event.agent_id

        # Call global handlers first
        for handler in self._global_handlers:
            handler(stream_event)

        # Call type-specific handlers
        event_type = type(event)
        if event_type in self._handlers:
            for handler in self._handlers[event_type]:
                handler(event, agent_id)

    def clear(self) -> None:
        """Clear all registered handlers."""
        self._handlers.clear()
        self._global_handlers.clear()


# Event type constants for easy registration
EVENT_TYPES = {
    # Text streaming
    "part_start": PartStartEvent,
    "part_delta": PartDeltaEvent,
    # Tool events
    "tool_call": FunctionToolCallEvent,
    "tool_result": FunctionToolResultEvent,
    # Lifecycle events
    "model_request_start": ModelRequestStartEvent,
    "tool_calls_start": ToolCallsStartEvent,
    # SDK events
    "compact_start": CompactStartEvent,
    "compact_complete": CompactCompleteEvent,
    "compact_failed": CompactFailedEvent,
    "handoff_start": HandoffStartEvent,
    "handoff_complete": HandoffCompleteEvent,
    "handoff_failed": HandoffFailedEvent,
    # Subagent events
    "subagent_start": SubagentStartEvent,
    "subagent_complete": SubagentCompleteEvent,
    # Message bus
    "message_received": MessageReceivedEvent,
}


def is_text_start(event: PartStartEvent) -> bool:
    """Check if this is a text part start event."""
    return isinstance(event.part, TextPart)


def is_thinking_start(event: PartStartEvent) -> bool:
    """Check if this is a thinking part start event."""
    return isinstance(event.part, ThinkingPart)


def is_text_delta(event: PartDeltaEvent) -> bool:
    """Check if this is a text part delta event."""
    return isinstance(event.delta, TextPartDelta)


def is_thinking_delta(event: PartDeltaEvent) -> bool:
    """Check if this is a thinking part delta event."""
    return isinstance(event.delta, ThinkingPartDelta)
