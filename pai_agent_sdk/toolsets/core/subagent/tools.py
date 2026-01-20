"""Subagent management tools.

Tools for managing and inspecting subagents that may have been compressed
or forgotten from context. These tools allow the agent to access subagent
metadata stored in AgentContext.

Available tools:
- SubagentInfoTool: List all known subagents with their metadata
"""

from collections.abc import Sequence
from typing import NotRequired

from pydantic_ai import RunContext
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from typing_extensions import TypedDict

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.toolsets.core.base import BaseTool


def _extract_first_user_prompt(messages: Sequence[ModelMessage]) -> str | None:
    """Extract the first UserPromptPart content from message history.

    Finds the first ModelRequest in the message list, then returns the content
    of its first UserPromptPart. This is useful for getting a hint about what
    the subagent was asked to do.

    Args:
        messages: Sequence of ModelMessage objects (typically subagent history).

    Returns:
        The text content of the first UserPromptPart, or None if not found.
    """
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    content = part.content
                    if isinstance(content, str):
                        return content
                    # For multipart content, find first str part
                    for item in content:
                        if isinstance(item, str):
                            return item
                    return None
    return None


class SubagentEntry(TypedDict):
    """Information about a single subagent."""

    agent_id: str
    agent_name: str
    parent_agent_id: str | None
    has_history: bool
    history_length: NotRequired[int]
    hint: NotRequired[str]


class SubagentInfoResult(TypedDict):
    """Result from SubagentInfoTool.call()."""

    subagents: list[SubagentEntry]
    total_count: NotRequired[int]
    message: NotRequired[str]


def _has_subagent_info(ctx: AgentContext) -> bool:
    """Check if there is any subagent information available."""
    return bool(ctx.agent_registry) or bool(ctx.subagent_history)


def _format_agent_info(agent_id: str, ctx: AgentContext) -> SubagentEntry:
    """Format agent info from registry."""
    info = ctx.agent_registry.get(agent_id)
    if info:
        return SubagentEntry(
            agent_id=info.agent_id,
            agent_name=info.agent_name,
            parent_agent_id=info.parent_agent_id,
            has_history=agent_id in ctx.subagent_history,
        )
    # Agent not in registry but has history
    return SubagentEntry(
        agent_id=agent_id,
        agent_name="unknown",
        parent_agent_id=None,
        has_history=True,
    )


class SubagentInfoTool(BaseTool):
    """Tool to list all known subagents.

    This tool helps the agent discover subagents that may have been
    created earlier in the conversation but compressed from context.
    """

    name = "subagent_info"
    description = (
        "List all known subagents and their metadata. "
        "Use this to discover subagents that may have been forgotten from context."
    )

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        """Only available when there is subagent information."""
        return _has_subagent_info(ctx.deps)

    async def call(self, ctx: RunContext[AgentContext]) -> SubagentInfoResult:
        """List all known subagents.

        Returns:
            Dictionary containing list of subagents with their metadata.
        """
        deps = ctx.deps
        all_agent_ids = set(deps.agent_registry.keys()) | set(deps.subagent_history.keys())

        # Filter out the main agent
        subagent_ids = [aid for aid in all_agent_ids if aid != deps.run_id]

        if not subagent_ids:
            return SubagentInfoResult(subagents=[], message="No subagents found.")

        subagents: list[SubagentEntry] = []
        for agent_id in subagent_ids:
            entry = _format_agent_info(agent_id, deps)
            # Add history summary and hint
            if agent_id in deps.subagent_history:
                history = deps.subagent_history[agent_id]
                entry["history_length"] = len(history)
                hint = _extract_first_user_prompt(history)
                if hint:
                    entry["hint"] = hint
            subagents.append(entry)

        return SubagentInfoResult(
            subagents=subagents,
            total_count=len(subagents),
        )


# Export list of tool classes
tools: list[type[BaseTool]] = [
    SubagentInfoTool,
]
