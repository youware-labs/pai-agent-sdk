"""Custom agent events for sideband streaming.

This module defines custom events that agents can emit via the sideband
stream channel (agent_stream_queues) to communicate status and results
to consumers without interrupting the main agent flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai import DeferredToolResults
    from pydantic_ai.messages import UserContent


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


# =============================================================================
# Message Bus Events
# =============================================================================


@dataclass
class BusMessageInfo:
    """Info about a single bus message.

    Attributes:
        content: Original message content (before template rendering).
        rendered_content: Rendered message content (template already applied).
        source: Who sent the message (e.g., "user", agent_id).
        target: Who should receive the message (agent_id, or None for broadcast).
    """

    content: str
    rendered_content: str
    source: str
    target: str | None = None


@dataclass
class MessageReceivedEvent(AgentEvent):
    """Emitted when bus messages are received and injected into conversation.

    This event is emitted by the bus_message filter when pending messages
    are consumed and injected. Consumers can use this to display
    incoming messages in the UI.

    Attributes:
        messages: List of received message info.
    """

    messages: list[BusMessageInfo] = field(default_factory=list)


# =============================================================================
# Agent Lifecycle Events
# =============================================================================

# Node type for agent graph traversal
NodeType = Literal["user_prompt", "model_request", "call_tools", "end"]


@dataclass
class AgentExecutionStartEvent(AgentEvent):
    """Emitted when agent execution begins (before first node).

    This event marks the start of an agent run. Use it for:
    - Initializing progress indicators
    - Starting execution timers
    - Logging agent invocations

    Attributes:
        user_prompt: The user prompt passed to the agent (str or multimodal content).
        deferred_tool_results: Results from deferred tool calls, if any.
        message_history_count: Number of messages in provided history.
    """

    user_prompt: str | Sequence[UserContent] | None = None
    deferred_tool_results: DeferredToolResults | None = None
    message_history_count: int = 0


@dataclass
class AgentExecutionCompleteEvent(AgentEvent):
    """Emitted when agent execution completes successfully.

    This event marks successful completion of an agent run. Use it for:
    - Finalizing progress indicators
    - Recording execution metrics
    - Logging completion status

    Attributes:
        total_loops: Total number of model request loops executed.
        total_duration_seconds: Total execution time.
        final_message_count: Number of messages after execution.
    """

    total_loops: int = 0
    total_duration_seconds: float = 0.0
    final_message_count: int = 0


@dataclass
class AgentExecutionFailedEvent(AgentEvent):
    """Emitted when agent execution fails with an error.

    This event is emitted when an exception occurs during agent execution.
    Use it for error tracking and user notification.

    Attributes:
        error: Error message describing the failure.
        error_type: Type name of the exception (e.g., "UsageLimitExceeded").
        total_loops: Number of loops completed before failure.
        total_duration_seconds: Time elapsed before failure.
    """

    error: str = ""
    error_type: str = ""
    total_loops: int = 0
    total_duration_seconds: float = 0.0


# =============================================================================
# Loop Events
# =============================================================================


@dataclass
class LoopStartEvent(AgentEvent):
    """Emitted when agent starts a new loop iteration.

    A "loop" is one model_request and its subsequent tool executions (if any).
    Loop index increments each time a new model request begins.

    Use this event for:
    - Displaying loop progress (e.g., "Loop 3/10")
    - Implementing client-side loop limits
    - Debugging infinite loop scenarios

    Attributes:
        loop_index: Zero-based loop iteration number.
        message_count: Number of messages in history at loop start.
    """

    loop_index: int = 0
    message_count: int = 0


# =============================================================================
# Node Events
# =============================================================================


@dataclass
class NodeStartEvent(AgentEvent):
    """Emitted when a graph node starts processing.

    Use this event for:
    - Displaying current phase (e.g., "Thinking..." vs "Running tools...")
    - Fine-grained progress tracking

    Attributes:
        node_type: Type of node being processed.
        loop_index: Current loop iteration number (None for user_prompt node).
    """

    node_type: NodeType = "model_request"
    loop_index: int | None = None


@dataclass
class NodeCompleteEvent(AgentEvent):
    """Emitted when a graph node completes processing.

    Contains node-specific details based on node_type:
    - model_request: has_tool_calls

    Attributes:
        node_type: Type of node that completed.
        loop_index: Current loop iteration number (None for user_prompt node).
        duration_seconds: Time spent processing this node.
        has_tool_calls: Whether model response contains tool calls (model_request only).
    """

    node_type: NodeType = "model_request"
    loop_index: int | None = None
    duration_seconds: float = 0.0

    # model_request specific
    has_tool_calls: bool = False


# =============================================================================
# Type Aliases
# =============================================================================

# Union of all lifecycle events for type hints
LifecycleEvent = (
    AgentExecutionStartEvent
    | AgentExecutionCompleteEvent
    | AgentExecutionFailedEvent
    | LoopStartEvent
    | NodeStartEvent
    | NodeCompleteEvent
)
