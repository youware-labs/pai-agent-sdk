"""MCP server configuration and approval hook utilities.

This module provides:
- MCPServerSpec: Configuration model for MCP servers
- ProcessToolCallback: Type alias for tool call hook
- create_mcp_approval_hook: Factory for creating approval hooks

Example:
    Building an MCP server with approval hook::

        from pydantic_ai.mcp import MCPServerStdio
        from pai_agent_sdk.mcp import create_mcp_approval_hook

        server = MCPServerStdio(
            command="uvx",
            args=["mcp-server-filesystem"],
            tool_prefix="filesystem",
            process_tool_call=create_mcp_approval_hook("filesystem"),
        )
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai import ApprovalRequired, RunContext

from pai_agent_sdk._logger import get_logger

if TYPE_CHECKING:
    from pydantic_ai.mcp import CallToolFunc, ToolResult

    from pai_agent_sdk.context import AgentContext

logger = get_logger(__name__)

# Type alias matching pydantic-ai's ProcessToolCallback
ProcessToolCallback = Callable[
    ["RunContext[Any]", "CallToolFunc", str, dict[str, Any]],
    Awaitable["ToolResult"],
]


class MCPServerSpec(BaseModel):
    """MCP server specification.

    Configuration model for MCP servers, independent of transport implementation.
    Can be used by CLI or other integrations to define MCP server configurations.

    Example:
        Stdio transport::

            spec = MCPServerSpec(
                transport="stdio",
                command="uvx",
                args=["mcp-server-filesystem"],
            )

        Streamable HTTP transport::

            spec = MCPServerSpec(
                transport="streamable_http",
                url="http://localhost:8000/mcp",
            )
    """

    transport: Literal["stdio", "streamable_http"] = "stdio"
    """Transport type: stdio or streamable_http."""

    # stdio transport fields
    command: str | None = None
    """Command for stdio transport."""

    args: list[str] = Field(default_factory=list)
    """Command arguments for stdio transport."""

    env: dict[str, str] = Field(default_factory=dict)
    """Environment variables for the server."""

    # streamable_http transport fields
    url: str | None = None
    """URL for streamable_http transport."""

    headers: dict[str, str] = Field(default_factory=dict)
    """Headers for streamable_http transport."""


def create_mcp_approval_hook(server_name: str) -> ProcessToolCallback:
    """Create a process_tool_call hook for MCP tool approval.

    The hook checks if the server is in ctx.deps.need_user_approve_mcps.
    If approval is required and not yet approved, raises ApprovalRequired.
    pydantic-ai catches this and produces DeferredToolRequests.

    Args:
        server_name: MCP server name (used as tool_prefix).

    Returns:
        ProcessToolCallback that implements approval logic.

    Example:
        >>> from pydantic_ai.mcp import MCPServerStdio
        >>> server = MCPServerStdio(
        ...     command="uvx",
        ...     args=["mcp-server-filesystem"],
        ...     tool_prefix="filesystem",
        ...     process_tool_call=create_mcp_approval_hook("filesystem"),
        ... )
    """

    async def hook(
        ctx: RunContext[AgentContext],
        call_tool: CallToolFunc,
        name: str,
        tool_args: dict[str, Any],
    ) -> ToolResult:
        # Check if this MCP server needs approval
        if server_name in ctx.deps.need_user_approve_mcps and not ctx.tool_call_approved:
            full_name = f"{server_name}_{name}"
            logger.debug("MCP tool %r requires approval", full_name)
            raise ApprovalRequired(metadata={"mcp_server": server_name, "mcp_tool": name, "full_name": full_name})

        # Approved or no approval needed - execute the tool
        return await call_tool(name, tool_args, None)

    return hook
