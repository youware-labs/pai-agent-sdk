"""Thinking tool for agent reasoning.

This tool allows the agent to think about something without obtaining new information
or making changes. Useful for complex reasoning or caching memory.
"""

from typing import Annotated, Any

from pydantic import Field
from pydantic_ai import RunContext

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.toolsets.base import BaseTool


class ThinkingTool(BaseTool):
    """Tool for agent to think and reason."""

    name = "thinking"
    description = (
        "Use the tool to think about something. It will not obtain new information or change the database, "
        "but just append the thought to the log. Use it when complex reasoning or some cache memory is needed. "
        "For task planning and management, use the to_do tool instead."
    )
    instruction: str | None = None

    async def call(
        self,
        ctx: RunContext[AgentContext],
        thought: Annotated[
            str,
            Field(
                description="A thought to think or plan about in markdown format. Use the same language as the user."
            ),
        ],
    ) -> dict[str, Any]:
        return {"thought": thought}
