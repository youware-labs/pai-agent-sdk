"""Agent runtime factory for paintress-cli.

This module provides factory functions to create AgentRuntime configured
for TUI usage. It wraps the SDK's create_agent() with TUI-specific
configuration and integrates Browser and MCP toolsets.

Example:
    from paintress_cli.runtime import create_tui_runtime
    from paintress_cli.browser import BrowserManager
    from paintress_cli.config import ConfigManager

    config_manager = ConfigManager()
    config = config_manager.load_config()
    mcp_config = config_manager.load_mcp_config()

    async with BrowserManager(config.browser) as browser:
        runtime = create_tui_runtime(
            config=config,
            mcp_config=mcp_config,
            browser_manager=browser,
        )
        async with runtime:
            # Use runtime.agent, runtime.ctx, runtime.env
            pass
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic_ai import ModelSettings
from pydantic_ai.mcp import MCPServer

from pai_agent_sdk.agents.main import AgentRuntime, create_agent
from pai_agent_sdk.context import ModelConfig, ToolConfig
from pai_agent_sdk.presets import resolve_model_settings
from paintress_cli.browser import BrowserManager
from paintress_cli.config import MCPConfig, PaintressConfig
from paintress_cli.environment import TUIEnvironment
from paintress_cli.logging import get_logger
from paintress_cli.mcp import build_mcp_servers
from paintress_cli.session import TUIContext

if TYPE_CHECKING:
    from pydantic_ai.toolsets import AbstractToolset

logger = get_logger(__name__)


def create_tui_runtime(
    config: PaintressConfig,
    mcp_config: MCPConfig | None = None,
    browser_manager: BrowserManager | None = None,
    *,
    working_dir: Path | None = None,
    system_prompt: str | None = None,
) -> AgentRuntime[TUIContext, str]:
    """Create AgentRuntime configured for TUI.

    This function wraps the SDK's create_agent() with TUI-specific
    configuration, integrating:
    - TUIEnvironment with ProcessManager
    - TUIContext with SteeringManager
    - MCP servers from configuration
    - Browser toolset if available

    Args:
        config: Paintress CLI configuration.
        mcp_config: MCP server configuration. If None, no MCP servers are added.
        browser_manager: Optional browser manager. If available and started,
            its toolset will be included.
        working_dir: Working directory for the environment. Defaults to cwd.
        system_prompt: Custom system prompt. If None, uses default.

    Returns:
        AgentRuntime configured for TUI usage. Use as async context manager.

    Example:
        runtime = create_tui_runtime(config, mcp_config, browser)
        async with runtime:
            async with stream_agent(runtime, "Hello") as stream:
                async for event in stream:
                    print(event)
    """
    # Collect toolsets
    toolsets: list[AbstractToolset[Any] | MCPServer] = []

    # Add MCP servers
    if mcp_config:
        mcp_servers = build_mcp_servers(mcp_config)
        toolsets.extend(mcp_servers)
        logger.info("Added %d MCP servers to runtime", len(mcp_servers))

    # Add browser toolset if available
    if browser_manager and browser_manager.is_available:
        browser_toolset = browser_manager.get_browser_toolset()
        if browser_toolset:
            toolsets.append(browser_toolset)
            logger.info("Added browser toolset (cdp_url=%s)", browser_manager.cdp_url)

    # Environment configuration
    env_kwargs: dict[str, Any] = {}
    if working_dir:
        env_kwargs["default_path"] = working_dir
        env_kwargs["allowed_paths"] = [working_dir]
    else:
        cwd = Path.cwd()
        env_kwargs["default_path"] = cwd
        env_kwargs["allowed_paths"] = [cwd]

    # Model configuration
    model_cfg = ModelConfig()

    # Resolve model settings from preset name or dict
    model_settings = resolve_model_settings(config.general.model_settings)
    if model_settings:
        logger.debug(f"Using model settings: {config.general.model_settings} -> {model_settings}")

    # Tool configuration
    tool_config = ToolConfig()
    if config.tools.need_approval:
        logger.debug("Tools requiring approval: %s", config.tools.need_approval)

    # Create runtime using SDK factory
    runtime = create_agent(
        model=config.general.model or None,
        model_settings=cast(ModelSettings, model_settings),
        env=TUIEnvironment,
        env_kwargs=env_kwargs,
        context_type=TUIContext,
        model_cfg=model_cfg,
        tool_config=tool_config,
        toolsets=toolsets if toolsets else None,
        system_prompt=system_prompt,
        need_user_approve_tools=config.tools.need_approval or None,
        include_builtin_subagents=True,
    )

    logger.info(
        "Created TUI runtime: model=%s, toolsets=%d",
        config.general.model,
        len(toolsets),
    )

    return runtime
