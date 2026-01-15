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

from typing import TYPE_CHECKING

from pydantic_ai.mcp import MCPServer, MCPServerStdio, MCPServerStreamableHTTP

from paintress_cli.config import MCPConfig, MCPServerConfig
from paintress_cli.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


def build_mcp_server(name: str, config: MCPServerConfig) -> MCPServer | None:
    """Build a single MCPServer instance from configuration.

    Args:
        name: Server name (used as tool_prefix).
        config: Server configuration.

    Returns:
        MCPServer instance, or None if configuration is invalid.
    """
    match config.transport:
        case "stdio":
            if not config.command:
                logger.warning(
                    "MCP server '%s' has stdio transport but no command, skipping",
                    name,
                )
                return None

            logger.debug(
                "Building MCPServerStdio: %s (command=%s)",
                name,
                config.command,
            )
            return MCPServerStdio(
                command=config.command,
                args=config.args,
                env=config.env or None,
                tool_prefix=name,
            )

        case "streamable_http":
            if not config.url:
                logger.warning(
                    "MCP server '%s' has streamable_http transport but no url, skipping",
                    name,
                )
                return None

            logger.debug(
                "Building MCPServerStreamableHTTP: %s (url=%s)",
                name,
                config.url,
            )
            return MCPServerStreamableHTTP(
                url=config.url,
                headers=config.headers or None,
                tool_prefix=name,
            )

        case _:
            logger.warning(
                "MCP server '%s' has unknown transport type: %s, skipping",
                name,
                config.transport,
            )
            return None


def build_mcp_servers(mcp_config: MCPConfig) -> list[MCPServer]:
    """Build MCPServer instances from MCPConfig.

    Args:
        mcp_config: MCP configuration containing server definitions.

    Returns:
        List of MCPServer instances ready for use with pydantic-ai Agent.
        Invalid configurations are skipped with warnings.
    """
    servers: list[MCPServer] = []

    for name, config in mcp_config.servers.items():
        server = build_mcp_server(name, config)
        if server is not None:
            servers.append(server)
            logger.info("Added MCP server: %s (%s)", name, config.transport)

    logger.debug("Built %d MCP servers from config", len(servers))
    return servers
