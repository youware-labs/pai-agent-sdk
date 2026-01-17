"""MCP Server construction from configuration.

This module provides functions to build pydantic-ai MCPServer instances
from MCPConfig. It's a pure data transformation layer with no I/O.

Example:
    from paintress_cli.config import ConfigManager
    from paintress_cli.mcp import build_mcp_servers

    config_manager = ConfigManager()
    mcp_config = config_manager.load_mcp_config()
    servers = build_mcp_servers(mcp_config)

    # Use with agent
    agent = Agent(..., toolsets=servers)
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.shared.message import SessionMessage
from pydantic_ai.mcp import MCPServer, MCPServerStdio, MCPServerStreamableHTTP

from pai_agent_sdk.mcp import create_mcp_approval_hook
from paintress_cli.config import MCPConfig, MCPServerConfig
from paintress_cli.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class QuietMCPServerStdio(MCPServerStdio):
    """MCPServerStdio variant that suppresses server stderr output.

    MCP server subprocesses write logs to stderr which clutters the TUI.
    This class redirects stderr to /dev/null (or NUL on Windows).
    """

    @asynccontextmanager
    async def client_streams(
        self,
    ) -> AsyncIterator[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
        ]
    ]:
        """Override to pass errlog parameter that suppresses stderr."""
        server = StdioServerParameters(command=self.command, args=list(self.args), env=self.env, cwd=self.cwd)
        # Open /dev/null (or NUL on Windows) for subprocess stderr
        null_path = "NUL" if sys.platform == "win32" else "/dev/null"
        with open(null_path, "w") as devnull:
            async with stdio_client(server=server, errlog=devnull) as (read_stream, write_stream):
                yield read_stream, write_stream


def build_mcp_server(
    name: str,
    config: MCPServerConfig,
    need_approval: bool = False,
) -> MCPServer | None:
    """Build a single MCPServer instance from configuration.

    Args:
        name: Server name (used as tool_prefix).
        config: Server configuration.
        need_approval: Whether all tools from this server need approval.

    Returns:
        MCPServer instance, or None if configuration is invalid.
    """
    # Create approval hook if needed
    process_tool_call = create_mcp_approval_hook(name) if need_approval else None

    match config.transport:
        case "stdio":
            if not config.command:
                logger.warning(
                    "MCP server '%s' has stdio transport but no command, skipping",
                    name,
                )
                return None

            logger.debug(
                "Building MCPServerStdio: %s (command=%s, approval=%s)",
                name,
                config.command,
                need_approval,
            )
            return QuietMCPServerStdio(
                command=config.command,
                args=config.args,
                env=config.env or None,
                tool_prefix=name,
                process_tool_call=process_tool_call,
            )

        case "streamable_http":
            if not config.url:
                logger.warning(
                    "MCP server '%s' has streamable_http transport but no url, skipping",
                    name,
                )
                return None

            logger.debug(
                "Building MCPServerStreamableHTTP: %s (url=%s, approval=%s)",
                name,
                config.url,
                need_approval,
            )
            return MCPServerStreamableHTTP(
                url=config.url,
                headers=config.headers or None,
                tool_prefix=name,
                process_tool_call=process_tool_call,
            )

        case _:
            logger.warning(
                "MCP server '%s' has unknown transport type: %s, skipping",
                name,
                config.transport,
            )
            return None


def build_mcp_servers(
    mcp_config: MCPConfig,
    need_approval_mcps: list[str] | None = None,
) -> list[MCPServer]:
    """Build MCPServer instances from MCPConfig.

    Args:
        mcp_config: MCP configuration containing server definitions.
        need_approval_mcps: Server names that require approval for all tools.

    Returns:
        List of MCPServer instances ready for use with pydantic-ai Agent.
        Invalid configurations are skipped with warnings.
    """
    servers: list[MCPServer] = []
    need_approval_mcps = need_approval_mcps or []

    for name, config in mcp_config.servers.items():
        need_approval = name in need_approval_mcps
        server = build_mcp_server(name, config, need_approval=need_approval)
        if server is not None:
            servers.append(server)
            logger.info("Added MCP server: %s (%s, approval=%s)", name, config.transport, need_approval)

    logger.debug("Built %d MCP servers from config", len(servers))
    return servers
