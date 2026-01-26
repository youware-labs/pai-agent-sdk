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
from enum import Enum

from pai_agent_sdk.events import AgentEvent

# -----------------------------------------------------------------------------
# Agent Phase Events (for status bar)
# -----------------------------------------------------------------------------


class AgentPhase(str, Enum):
    """Agent execution phase for status display.

    Used to drive the status bar indicator during agent execution.
    """

    IDLE = "idle"  # Waiting for input
    GENERATING = "generating"  # ModelRequestNode - calling LLM
    EXECUTING = "executing"  # CallToolsNode - running tools
    COMPLETED = "completed"  # EndNode - finished


@dataclass
class AgentPhaseEvent(AgentEvent):
    """Emitted when agent transitions between execution phases.

    This event is emitted via pre_node_hook when the agent graph
    transitions to a new node type.

    Attributes:
        phase: Current execution phase.
        agent_id: Unique identifier for the agent.
        agent_name: Human-readable agent name.
        node_type: Internal node type name (for debugging).
        details: Optional details like "Calling claude-4-sonnet".
    """

    phase: AgentPhase = AgentPhase.IDLE
    agent_id: str = ""
    agent_name: str = ""
    node_type: str = ""
    details: str = ""


# -----------------------------------------------------------------------------
# Process Lifecycle Events
# -----------------------------------------------------------------------------


@dataclass
class ProcessStartedEvent(AgentEvent):
    """Emitted when a subprocess starts.

    Attributes:
        process_id: Unique identifier for the process.
        command: Command being executed.
        pid: Operating system process ID.
    """

    process_id: str = ""
    command: str = ""
    pid: int = 0


@dataclass
class ProcessOutputEvent(AgentEvent):
    """Emitted when subprocess produces output.

    Attributes:
        process_id: Unique identifier for the process.
        output: The output text.
        is_stderr: True if output is from stderr.
    """

    process_id: str = ""
    output: str = ""
    is_stderr: bool = False


@dataclass
class ProcessExitedEvent(AgentEvent):
    """Emitted when subprocess exits.

    Attributes:
        process_id: Unique identifier for the process.
        exit_code: Process exit code.
    """

    process_id: str = ""
    exit_code: int = 0


# -----------------------------------------------------------------------------
# Context Management Events
# -----------------------------------------------------------------------------


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
