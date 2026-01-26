"""Output guards for agent execution.

This module provides output validators that are automatically attached
to agents created via create_agent and subagent factory functions.
"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelRetry

from pai_agent_sdk.context import AgentContext

OutputT = TypeVar("OutputT")
AgentDepsT = TypeVar("AgentDepsT", bound=AgentContext)


async def message_bus_guard(ctx: RunContext[AgentContext], output: OutputT) -> OutputT:
    """Output guard that checks for pending bus messages.

    This guard triggers a ModelRetry when there are pending messages
    in the message bus for the current agent. This ensures the agent
    processes any user messages or inter-agent communication before
    completing.

    Args:
        ctx: Run context containing AgentContext with message_bus.
        output: The output from the agent (any type).

    Returns:
        The output unchanged if no pending messages.

    Raises:
        ModelRetry: If there are pending bus messages.
    """
    agent_id = ctx.deps._agent_id
    if ctx.deps.message_bus.has_pending(agent_id):
        raise ModelRetry(
            "<system-reminder>There are pending messages in your message bus. Please address them before completing.</system-reminder>"
        )

    return output


def attach_message_bus_guard(agent: Agent[AgentDepsT, Any]) -> None:
    """Attach message bus guard to an agent.

    This function adds the message_bus_guard as an output validator
    to the given agent. It should be called after agent creation.

    Args:
        agent: The agent to attach the guard to.
    """

    @agent.output_validator
    async def _guard(ctx: RunContext[AgentDepsT], output: Any) -> Any:
        return await message_bus_guard(ctx, output)  # type: ignore[arg-type]
