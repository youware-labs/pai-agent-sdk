"""Usage tracking models for agent token consumption.

This module provides models for tracking token usage from both main agents
and internal agent calls (sub-agents, image understanding, video understanding, etc.).
"""

from __future__ import annotations

from pydantic import BaseModel
from pydantic_ai.usage import RunUsage


class InternalUsage(BaseModel):
    """Usage record with model information for internal agent calls.

    This model captures token usage along with the model identifier,
    enabling accurate cost tracking and usage attribution per model.

    Example::

        from pai_agent_sdk.usage import InternalUsage

        # Return from internal agent functions
        async def get_image_description(...) -> tuple[str, InternalUsage]:
            result = await agent.run(...)
            return result.output, InternalUsage(
                model_id="openai:gpt-4o",
                usage=result.usage(),
            )
    """

    model_id: str
    """Model identifier that generated this usage (e.g., 'openai:gpt-4o', 'anthropic:claude-sonnet-4')."""

    usage: RunUsage
    """Token usage from this call."""


class ExtraUsageRecord(BaseModel):
    """Record of extra usage from tool calls or filters.

    This model captures additional token usage that occurs outside the main
    agent run, such as from sub-agents, filters, or tool calls that invoke
    other models.
    """

    uuid: str
    """Unique identifier for this usage record (e.g., tool_call_id or generated UUID)."""

    agent: str
    """Agent name that generated this usage (e.g., 'compact', 'search', 'image_understanding')."""

    model_id: str
    """Model identifier that generated this usage (e.g., 'openai:gpt-4o', 'anthropic:claude-sonnet-4')."""

    usage: RunUsage
    """Token usage from this call."""
