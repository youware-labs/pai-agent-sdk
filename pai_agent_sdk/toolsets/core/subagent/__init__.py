"""Subagent tool creation utilities.

This module provides utilities for creating tools that wrap subagent calls,
automatically handling subagent context creation and usage tracking.

The factory automatically:
- Creates a subagent context via enter_subagent(name, agent_id=tool_call_id)
- Records RunUsage to ctx.deps.extra_usage[tool_call_id]
- Converts output to string for LLM consumption

Example::

    from pydantic_ai.usage import RunUsage

    from pai_agent_sdk.context import AgentContext
    from pai_agent_sdk.toolsets.core.subagent import create_subagent_tool

    async def search(ctx: AgentContext, query: str) -> tuple[str, RunUsage]:
        # ctx is already a subagent context (auto-created by factory)
        agent = get_search_agent()
        result = await agent.run(query, deps=ctx)
        return result.output, result.usage()

    SearchTool = create_subagent_tool(
        name="search",
        description="Search the web for information",
        call_func=search,
    )

    # Use with Toolset
    toolset = Toolset(tools=[SearchTool])

For streaming, use the parent context's stream queue::

    async def search_with_stream(ctx: AgentContext, query: str) -> tuple[str, RunUsage]:
        # tool_call_id is stored in ctx.parent_run_id (set by enter_subagent)
        # Stream events can be sent to parent's subagent_stream_queues
        agent = get_search_agent()
        result = await agent.run(query, deps=ctx)
        return result.output, result.usage()
"""

from pai_agent_sdk.toolsets.core.base import BaseTool
from pai_agent_sdk.toolsets.core.subagent.factory import (
    AvailabilityCheckFunc,
    SubagentCallFunc,
    create_subagent_call_func,
    create_subagent_tool,
)
from pai_agent_sdk.toolsets.core.subagent.tools import (
    SubagentInfoTool,
)
from pai_agent_sdk.toolsets.core.subagent.unified import (
    create_unified_subagent_tool,
    get_available_subagent_names,
)

# Management tools for inspecting subagent state
tools: list[type[BaseTool]] = [
    SubagentInfoTool,
]

__all__ = [
    "AvailabilityCheckFunc",
    "SubagentCallFunc",
    "SubagentInfoTool",
    "create_subagent_call_func",
    "create_subagent_tool",
    "create_unified_subagent_tool",
    "get_available_subagent_names",
    "tools",
]
