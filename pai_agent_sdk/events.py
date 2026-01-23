"""Custom agent events for sideband streaming.

This module defines custom events that agents can emit via the sideband
stream channel (agent_stream_queues) to communicate status and results
to consumers without interrupting the main agent flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class AgentEvent:
    """Base class for custom agent events (sideband channel).

    Attributes:
        event_id: Unique identifier to correlate related events (e.g., start/complete pairs).
        timestamp: When the event was created.
    """

    event_id: str
    timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# Compact Events
# =============================================================================


@dataclass
class CompactStartEvent(AgentEvent):
    """Emitted when context compaction starts.

    Attributes:
        message_count: Number of messages before compaction.
    """

    message_count: int = 0


@dataclass
class CompactCompleteEvent(AgentEvent):
    """Emitted when context compaction completes successfully.

    Attributes:
        summary_markdown: The compacted summary in markdown format.
        original_message_count: Number of messages before compaction.
        compacted_message_count: Number of messages after compaction.
    """

    summary_markdown: str = ""
    original_message_count: int = 0
    compacted_message_count: int = 0


@dataclass
class CompactFailedEvent(AgentEvent):
    """Emitted when context compaction fails.

    Attributes:
        error: Error message describing the failure.
        message_count: Number of messages that were attempted to compact.
    """

    error: str = ""
    message_count: int = 0


# =============================================================================
# Handoff Events
# =============================================================================


@dataclass
class HandoffStartEvent(AgentEvent):
    """Emitted when context handoff starts.

    Attributes:
        message_count: Number of messages before handoff.
    """

    message_count: int = 0


@dataclass
class HandoffCompleteEvent(AgentEvent):
    """Emitted when context handoff completes successfully.

    Attributes:
        handoff_content: The actual handoff content/summary being passed to next context.
        original_message_count: Number of messages before handoff.
    """

    handoff_content: str = ""
    original_message_count: int = 0


@dataclass
class HandoffFailedEvent(AgentEvent):
    """Emitted when context handoff fails.

    Attributes:
        error: Error message describing the failure.
        message_count: Number of messages at failure time.
    """

    error: str = ""
    message_count: int = 0


# =============================================================================
# Subagent Lifecycle Events
# =============================================================================


@dataclass
class SubagentStartEvent(AgentEvent):
    """Emitted when a subagent starts execution.

    This event is emitted by the delegate tool when a subagent begins processing.
    Consumers can use this to display a progress indicator for the subagent.

    Attributes:
        agent_id: Unique identifier for this subagent instance (e.g., "explorer-a7b9").
        agent_name: Human-readable subagent name (e.g., "explorer").
        prompt_preview: First N characters of the prompt sent to subagent.
    """

    agent_id: str = ""
    agent_name: str = ""
    prompt_preview: str = ""


@dataclass
class SubagentCompleteEvent(AgentEvent):
    """Emitted when a subagent completes execution.

    This event is emitted by the delegate tool when a subagent finishes,
    regardless of success or failure. Consumers can use this to update
    the progress indicator to show completion status.

    Attributes:
        agent_id: Unique identifier for this subagent instance.
        agent_name: Human-readable subagent name.
        success: Whether the subagent completed successfully.
        request_count: Number of LLM requests the subagent made during execution.
        result_preview: First N characters of the subagent's output.
        error: Error message if success is False.
        duration_seconds: How long the subagent ran.
    """

    agent_id: str = ""
    agent_name: str = ""
    success: bool = True
    request_count: int = 0
    result_preview: str = ""
    error: str = ""
    duration_seconds: float = 0.0
