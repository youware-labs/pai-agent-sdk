"""Subagent tool creation from configuration.

This module provides functions to create subagent tools from SubagentConfig objects.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic_ai import Agent, RunContext
from pydantic_ai._agent_graph import HistoryProcessor

from pai_agent_sdk.agents.models import infer_model
from pai_agent_sdk.context import AgentContext, ModelConfig
from pai_agent_sdk.presets import INHERIT, resolve_model_cfg, resolve_model_settings
from pai_agent_sdk.subagents.config import (
    SubagentConfig,
    load_subagent_from_file,
    load_subagents_from_dir,
    parse_subagent_markdown,
)
from pai_agent_sdk.toolsets.core.base import BaseTool, Toolset
from pai_agent_sdk.toolsets.core.subagent import (
    create_subagent_call_func,
    create_subagent_tool,
)

if TYPE_CHECKING:
    from pydantic_ai import ModelSettings
    from pydantic_ai.models import Model


def _resolve_model(config: SubagentConfig, model: str | Model | None) -> str | Model:
    """Resolve effective model from config and fallback."""
    if config.model is not None and config.model != INHERIT:
        return config.model
    if model is not None:
        return model
    return "test"  # Placeholder, actual model passed at runtime


def _resolve_model_settings(
    config: SubagentConfig, model_settings: ModelSettings | dict[str, Any] | str | None
) -> dict[str, Any] | None:
    """Resolve effective model settings from config and fallback."""
    if config.model_settings is not None and config.model_settings != INHERIT:
        return resolve_model_settings(config.model_settings)
    if model_settings is not None:
        return resolve_model_settings(model_settings)
    return None


def _resolve_model_cfg(config: SubagentConfig, model_cfg: ModelConfig | None) -> ModelConfig | None:
    """Resolve effective ModelConfig from config and fallback.

    Resolution order:
    1. config.model_cfg is not None and != 'inherit' -> resolve to ModelConfig
    2. Otherwise use model_cfg fallback (inherit from parent)
    """
    resolved = resolve_model_cfg(config.model_cfg)
    if resolved is not None:
        return ModelConfig(**resolved)
    return model_cfg


def _collect_tools(config: SubagentConfig) -> list[str] | None:
    """Collect all tools (required + optional) from config."""
    if config.tools is None and config.optional_tools is None:
        return None
    all_tools: list[str] = []
    if config.tools:
        all_tools.extend(config.tools)
    if config.optional_tools:
        all_tools.extend(config.optional_tools)
    return all_tools


def create_subagent_tool_from_config(
    config: SubagentConfig,
    parent_toolset: Toolset[Any],
    *,
    model: str | Model | None = None,
    model_settings: ModelSettings | dict[str, Any] | str | None = None,
    history_processors: Sequence[HistoryProcessor[AgentContext]] | None = None,
    model_cfg: ModelConfig | None = None,
) -> type[BaseTool]:
    """Create a subagent tool from a SubagentConfig.

    Args:
        config: The parsed subagent configuration.
        parent_toolset: The parent toolset to derive tools from.
        model: Fallback model. Used if config.model is 'inherit' or None.
        model_settings: Fallback model settings. Used if config.model_settings is 'inherit' or None.
        history_processors: History processors to use for the subagent.
        model_cfg: Fallback ModelConfig. Used if config.model_cfg is None.

    Returns:
        A BaseTool subclass that wraps the subagent.
    """
    effective_model = _resolve_model(config, model)
    resolved_settings = _resolve_model_settings(config, model_settings)
    resolved_model_cfg = _resolve_model_cfg(config, model_cfg)
    all_tools = _collect_tools(config)

    sub_toolset = parent_toolset.subset(all_tools, include_auto_inherit=True)

    subagent: Agent[AgentContext, str] = Agent(
        model=infer_model(effective_model),
        system_prompt=config.system_prompt,
        toolsets=[sub_toolset],
        model_settings=resolved_settings,  # type: ignore[arg-type]
        deps_type=AgentContext,
        history_processors=history_processors,
        name=config.name,
    )

    required_tools = config.tools

    def check_tools_available(ctx: RunContext[AgentContext]) -> bool:
        if required_tools is None:
            return True
        return all(parent_toolset.is_tool_available(name, ctx) for name in required_tools)

    return create_subagent_tool(
        name=config.name,
        description=config.description,
        call_func=create_subagent_call_func(subagent, model_cfg=resolved_model_cfg),
        instruction=config.instruction,
        availability_check=check_tools_available,
    )


def create_subagent_tool_from_markdown(
    content: str | Path,
    parent_toolset: Toolset[Any],
    *,
    model: str | Model | None = None,
    model_settings: dict[str, Any] | str | None = None,
    history_processors: Sequence[HistoryProcessor[AgentContext]] | None = None,
    model_cfg: ModelConfig | None = None,
) -> type[BaseTool]:
    """Create a subagent tool from markdown content or file path.

    This is the main convenience function for creating subagent tools.

    Args:
        content: Markdown string or path to markdown file.
        parent_toolset: The parent toolset to derive tools from.
        model: Fallback model. Used if config.model is 'inherit' or None.
        model_settings: Fallback model settings. Used if config.model_settings is 'inherit' or None.
        history_processors: History processors to use for the subagent.
        model_cfg: Fallback ModelConfig. Used if config.model_cfg is None.

    Returns:
        A BaseTool subclass that wraps the subagent.

    Example::

        # From file
        DebuggerTool = create_subagent_tool_from_markdown(
            "pai_agent_sdk/subagents/debugger.md",
            parent_toolset=main_toolset,
            model="anthropic:claude-sonnet-4",
        )

        # From string
        SearchTool = create_subagent_tool_from_markdown(
            '''
            ---
            name: search
            description: Search for information
            model_settings: anthropic_low
            ---
            You are a search agent...
            ''',
            parent_toolset=main_toolset,
        )
    """
    if isinstance(content, Path) or (isinstance(content, str) and Path(content).exists()):
        config = load_subagent_from_file(content)
    else:
        config = parse_subagent_markdown(content)

    return create_subagent_tool_from_config(
        config,
        parent_toolset,
        model=model,
        model_settings=model_settings,
        history_processors=history_processors,
        model_cfg=model_cfg,
    )


def load_subagent_tools_from_dir(
    dir_path: Path | str,
    parent_toolset: Toolset[Any],
    *,
    model: str | Model | None = None,
    model_settings: dict[str, Any] | str | None = None,
    history_processors: Sequence[HistoryProcessor[AgentContext]] | None = None,
    model_cfg: ModelConfig | None = None,
) -> list[type[BaseTool]]:
    """Load all subagent tools from a directory.

    Scans the directory for .md files and creates a subagent tool for each.

    Args:
        dir_path: Path to the directory containing markdown files.
        parent_toolset: The parent toolset to derive tools from.
        model: Fallback model for all subagents.
        model_settings: Fallback model settings for all subagents.
        history_processors: History processors to use for all subagents.
        model_cfg: Fallback ModelConfig for all subagents.

    Returns:
        List of BaseTool subclasses.

    Example::

        subagent_tools = load_subagent_tools_from_dir(
            "pai_agent_sdk/subagents",
            parent_toolset=main_toolset,
            model="anthropic:claude-sonnet-4",
            model_settings="anthropic_medium",
        )
    """
    configs = load_subagents_from_dir(dir_path)
    tools: list[type[BaseTool]] = []

    for config in configs.values():
        tool = create_subagent_tool_from_config(
            config,
            parent_toolset,
            model=model,
            model_settings=model_settings,
            history_processors=history_processors,
            model_cfg=model_cfg,
        )
        tools.append(tool)

    return tools
