"""Agent lifecycle hooks for TUI integration.

This module provides hooks for:
- Phase events (emitting AgentPhaseEvent based on node transitions)
- Other lifecycle hooks as needed

Note: Steering injection is now handled by TUIContext._inject_steering
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from paintress_cli.events import AgentPhase, AgentPhaseEvent

if TYPE_CHECKING:
    from pai_agent_sdk.context import AgentContext


async def emit_phase_event(
    ctx: AgentContext,
    phase: AgentPhase,
    agent_id: str = "",
    agent_name: str = "",
    node_type: str = "",
    details: str = "",
) -> None:
    """Emit an AgentPhaseEvent to the context's stream queue.

    Utility function for emitting phase events from pre_node_hook.

    Args:
        ctx: The agent context.
        phase: Current execution phase.
        agent_id: Agent identifier.
        agent_name: Agent name.
        node_type: Internal node type.
        details: Optional details.
    """
    event = AgentPhaseEvent(
        event_id=f"phase-{uuid.uuid4().hex[:8]}",
        phase=phase,
        agent_id=agent_id,
        agent_name=agent_name,
        node_type=node_type,
        details=details,
    )
    await ctx.emit_event(event)
