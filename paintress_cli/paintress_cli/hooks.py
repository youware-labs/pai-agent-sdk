"""Agent lifecycle hooks for TUI integration.

This module provides hooks for:
- Phase events (emitting AgentPhaseEvent based on node transitions)
- Context update events (emitting ContextUpdateEvent after each node)
- Other lifecycle hooks as needed

Note: Steering injection is now handled by TUIContext._inject_steering
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from paintress_cli.events import AgentPhase, AgentPhaseEvent, ContextUpdateEvent

if TYPE_CHECKING:
    from pai_agent_sdk.agents.main import NodeHookContext
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


async def emit_context_update(hook_ctx: NodeHookContext[Any, Any]) -> None:
    """Emit ContextUpdateEvent after each node completion.

    This hook is designed to be used as post_node_hook in stream_agent.
    It reads the current usage from the run and emits an event to update
    the status bar's context percentage.

    Args:
        hook_ctx: The node hook context containing run and agent info.
    """
    if hook_ctx.run is None:
        return

    # Get current usage
    usage = hook_ctx.run.usage()
    total_tokens = usage.total_tokens

    # Get our TUIContext from run.ctx.deps
    ctx = hook_ctx.run.ctx.deps
    if ctx is None:
        return

    # Get context window size from model config
    model_cfg = getattr(ctx, "model_cfg", None)
    if model_cfg and hasattr(model_cfg, "context_window"):
        window_size = model_cfg.context_window or 200000
    else:
        window_size = 200000

    # Emit event via output_queue
    from pai_agent_sdk.context import StreamEvent

    event = ContextUpdateEvent(
        event_id=f"context-{uuid.uuid4().hex[:8]}",
        total_tokens=total_tokens,
        context_window_size=window_size,
    )
    await hook_ctx.output_queue.put(
        StreamEvent(
            agent_id=hook_ctx.agent_info.agent_id,
            agent_name=hook_ctx.agent_info.agent_name,
            event=event,
        )
    )
