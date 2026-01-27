"""TUI-specific event types for paintress-cli.

All TUI events extend pai_agent_sdk.events.AgentEvent to integrate with
the SDK's agent_stream_queues mechanism. This allows TUI events to flow
through the same channel as SDK events (compact, handoff, etc.).

Events are emitted via AgentContext.emit_event() and consumed by stream_agent().

Note: Steering functionality now uses SDK's MessageReceivedEvent from
pai_agent_sdk.events. The TUI sends steering messages via ctx.send_message()
and receives MessageReceivedEvent when they are injected.
"""

from __future__ import annotations

from dataclasses import dataclass

from pai_agent_sdk.events import AgentEvent


@dataclass
class ContextUpdateEvent(AgentEvent):
    """Real-time context usage update for status bar.

    Contains the current total tokens from message history for context window calculation.

    Attributes:
        total_tokens: Current total tokens used.
        context_window_size: Maximum context window size.
    """

    total_tokens: int = 0
    context_window_size: int = 0
