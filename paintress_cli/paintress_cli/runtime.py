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

from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic_ai import DeferredToolRequests, ModelSettings, TextOutput
from pydantic_ai.mcp import MCPServer
from pydantic_ai.output import OutputSpec

from pai_agent_sdk.agents.main import AgentRuntime, create_agent
from pai_agent_sdk.context import ModelCapability, ModelConfig, ToolConfig
from pai_agent_sdk.presets import resolve_model_cfg, resolve_model_settings
from pai_agent_sdk.subagents import SubagentConfig, load_subagents_from_dir
from pai_agent_sdk.toolsets.core.content import tools as content_tools
from pai_agent_sdk.toolsets.core.context import tools as context_tools
from pai_agent_sdk.toolsets.core.document import tools as document_tools
from pai_agent_sdk.toolsets.core.enhance import tools as enhance_tools
from pai_agent_sdk.toolsets.core.filesystem import tools as filesystem_tools
from pai_agent_sdk.toolsets.core.multimodal import tools as multimodal_tools
from pai_agent_sdk.toolsets.core.shell import tools as shell_tools
from pai_agent_sdk.toolsets.core.subagent import tools as subagent_tools
from pai_agent_sdk.toolsets.core.web import tools as web_tools
from paintress_cli.browser import BrowserManager
from paintress_cli.config import ConfigManager, MCPConfig, PaintressConfig, SubagentsConfig
from paintress_cli.environment import TUIEnvironment
from paintress_cli.logging import get_logger
from paintress_cli.mcp import build_mcp_servers
from paintress_cli.session import TUIContext
from paintress_cli.steering import steering_output_guard

if TYPE_CHECKING:
    from pydantic_ai.toolsets import AbstractToolset

logger = get_logger(__name__)


def _resolve_model_cfg(model_cfg_input: str | dict[str, Any] | None) -> ModelConfig:
    """Resolve model_cfg from preset name or dict to ModelConfig instance.

    Handles conversion of capabilities from list[str] to set[ModelCapability].

    Args:
        model_cfg_input: Preset name (e.g., 'claude_200k'), dict, or None.

    Returns:
        ModelConfig instance.
    """
    if model_cfg_input is None:
        return ModelConfig()

    # Use SDK's resolve_model_cfg to get dict
    cfg_dict = resolve_model_cfg(model_cfg_input)
    if cfg_dict is None:
        return ModelConfig()

    # Convert capabilities from list[str] to set[ModelCapability] if present
    if "capabilities" in cfg_dict:
        caps = cfg_dict["capabilities"]
        if isinstance(caps, (list, set)):
            cfg_dict["capabilities"] = {ModelCapability(c) if isinstance(c, str) else c for c in caps}

    return ModelConfig(**cfg_dict)


def _load_system_prompt(config: PaintressConfig) -> str:
    """Load system prompt from config or built-in default.

    Priority:
    1. Custom file path from config.general.system_prompt_file
    2. Built-in default from templates/system_prompt.md
    """
    if config.general.system_prompt_file:
        prompt_path = Path(config.general.system_prompt_file).expanduser()
        if prompt_path.exists():
            logger.debug("Loading system prompt from: %s", prompt_path)
            return prompt_path.read_text(encoding="utf-8")
        logger.warning("System prompt file not found: %s, using default", prompt_path)

    # Load built-in default
    template_files = resources.files("paintress_cli").joinpath("templates")
    default_prompt = template_files.joinpath("system_prompt.md").read_text(encoding="utf-8")
    logger.debug("Using built-in system prompt")
    return default_prompt


def _load_subagent_configs(
    subagents_config: SubagentsConfig,
    config_dir: Path | None = None,
) -> list[SubagentConfig]:
    """Load subagent configs from user config directory.

    Subagents are loaded from ~/.config/youware-labs/paintress-cli/subagents/
    and filtered based on the disabled list in config.

    Args:
        subagents_config: Subagents configuration with disabled list and overrides.
        config_dir: Config directory (defaults to ConfigManager.DEFAULT_CONFIG_DIR).

    Returns:
        List of SubagentConfig objects.
    """
    if config_dir is None:
        config_dir = ConfigManager.DEFAULT_CONFIG_DIR

    subagents_dir = config_dir / "subagents"
    if not subagents_dir.exists():
        logger.debug("Subagents directory not found: %s", subagents_dir)
        return []

    # Load all subagents from directory
    all_configs = load_subagents_from_dir(subagents_dir)
    logger.debug("Found %d subagents in %s", len(all_configs), subagents_dir)

    # Filter out disabled subagents
    disabled_set = set(subagents_config.disabled)
    enabled_configs: list[SubagentConfig] = []

    for name, cfg in all_configs.items():
        if name in disabled_set:
            logger.debug("Subagent disabled: %s", name)
            continue

        # Apply overrides if any
        if name in subagents_config.overrides:
            override = subagents_config.overrides[name]
            if override.model is not None:
                cfg = cfg.model_copy(update={"model": override.model})
            if override.model_settings is not None:
                cfg = cfg.model_copy(update={"model_settings": override.model_settings})
            logger.debug("Applied overrides for subagent: %s", name)

        enabled_configs.append(cfg)

    logger.info("Loaded %d subagents (disabled: %d)", len(enabled_configs), len(disabled_set))
    return enabled_configs


def create_tui_runtime(
    config: PaintressConfig,
    mcp_config: MCPConfig | None = None,
    browser_manager: BrowserManager | None = None,
    *,
    working_dir: Path | None = None,
    system_prompt: str | None = None,
) -> AgentRuntime[TUIContext, str | DeferredToolRequests]:
    """Create AgentRuntime configured for TUI.

    This function wraps the SDK's create_agent() with TUI-specific
    configuration, integrating:
    - TUIEnvironment with ProcessManager
    - TUIContext with SteeringManager
    - MCP servers from configuration
    - Browser toolset if available
    - Steering guard for output validation

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
    # Include global config dir in allowed_paths so agent can modify configs directly
    global_config_dir = ConfigManager.DEFAULT_CONFIG_DIR
    env_kwargs: dict[str, Any] = {}
    if working_dir:
        env_kwargs["default_path"] = working_dir
        env_kwargs["allowed_paths"] = [working_dir, global_config_dir]
    else:
        cwd = Path.cwd()
        env_kwargs["default_path"] = cwd
        env_kwargs["allowed_paths"] = [cwd, global_config_dir]

    # Model configuration - resolve from preset name or dict
    model_cfg = _resolve_model_cfg(config.general.model_cfg)
    if config.general.model_cfg:
        logger.debug(f"Using model_cfg: {config.general.model_cfg}")

    # Resolve model settings from preset name or dict
    model_settings = resolve_model_settings(config.general.model_settings)
    if model_settings:
        logger.debug(f"Using model settings: {config.general.model_settings} -> {model_settings}")

    # Tool configuration
    tool_config = ToolConfig()
    if config.tools.need_approval:
        logger.debug("Tools requiring approval: %s", config.tools.need_approval)

    # Load subagent configs from user config directory
    subagent_configs = _load_subagent_configs(config.subagents)

    # Load system prompt
    effective_system_prompt = system_prompt or _load_system_prompt(config)

    # Configure output type with steering guard
    # This ensures agent retries when user adds steering messages during output
    output_type: OutputSpec[str | DeferredToolRequests] = [
        TextOutput(steering_output_guard),
        DeferredToolRequests,
    ]
    # Create runtime using SDK factory
    runtime = create_agent(
        model=config.general.model or None,
        model_settings=cast(ModelSettings, model_settings),
        output_type=output_type,
        env=TUIEnvironment,
        env_kwargs=env_kwargs,
        context_type=TUIContext,
        model_cfg=model_cfg,
        tools=[
            *content_tools,
            *context_tools,
            *document_tools,
            *enhance_tools,
            *filesystem_tools,
            *multimodal_tools,
            *shell_tools,
            *web_tools,
            *subagent_tools,
        ],
        tool_config=tool_config,
        toolsets=toolsets if toolsets else None,
        system_prompt=effective_system_prompt,
        need_user_approve_tools=config.tools.need_approval or None,
        subagent_configs=subagent_configs if subagent_configs else None,
        include_builtin_subagents=False,
    )

    logger.info(
        "Created TUI runtime: model=%s, toolsets=%d",
        config.general.model,
        len(toolsets),
    )

    return runtime
