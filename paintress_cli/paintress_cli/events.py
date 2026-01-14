"""TUI-specific event types for paintress-cli.

All TUI events extend pai_agent_sdk.events.AgentEvent to integrate with
the SDK's agent_stream_queues mechanism. This allows TUI events to flow
through the same channel as SDK events (compact, handoff, etc.).

Events are emitted via AgentContext.emit_event() and consumed by stream_agent().
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
# Steering Events
# -----------------------------------------------------------------------------


@dataclass
class SteeringInjectedEvent(AgentEvent):
    """Emitted when steering messages are injected into agent context.

    This event is emitted by TUIContext._inject_steering after successfully
    injecting user guidance into the message history.

    Attributes:
        message_count: Number of messages injected.
        content: Full content of all injected messages for user audit.
    """

    message_count: int = 0
    content: str = ""


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
