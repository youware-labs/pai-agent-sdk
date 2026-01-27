"""Agent lifecycle hooks for TUI integration.

This module provides hooks for:
- Context update events (emitting ContextUpdateEvent after each node)

Note: Steering injection is now handled by TUIContext._inject_steering
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from pai_agent_sdk.context import AgentContext, StreamEvent
from pai_agent_sdk.utils import get_latest_request_usage
from paintress_cli.events import ContextUpdateEvent

if TYPE_CHECKING:
    from pai_agent_sdk.agents.main import NodeHookContext


async def emit_context_update(hook_ctx: NodeHookContext[AgentContext, Any]) -> None:
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
    usage = get_latest_request_usage(hook_ctx.run.all_messages())
    if not usage:
        return
    total_tokens = usage.total_tokens

    # Get our TUIContext from run.ctx.deps
    ctx = hook_ctx.run.ctx.deps.user_deps

    # Get context window size from model config
    model_cfg = ctx.model_cfg
    if model_cfg:
        window_size = model_cfg.context_window or 200000
    else:
        window_size = 200000

    # Emit event via output_queue
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
