"""Subagent tool creation from configuration.

This module provides functions to create subagent tools from SubagentConfig objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.subagents.config import (
    INHERIT,
    SubagentConfig,
    load_subagent_from_file,
    load_subagents_from_dir,
    parse_subagent_markdown,
)
from pai_agent_sdk.subagents.presets import resolve_model_settings
from pai_agent_sdk.toolsets.core.base import BaseTool, Toolset
from pai_agent_sdk.toolsets.core.subagent import create_subagent_call_func, create_subagent_tool

if TYPE_CHECKING:
    from pydantic_ai.models import Model


def create_subagent_tool_from_config(
    config: SubagentConfig,
    parent_toolset: Toolset[Any],
    *,
    model: str | Model | None = None,
    model_settings: dict[str, Any] | str | None = None,
) -> type[BaseTool]:
    """Create a subagent tool from a SubagentConfig.

    Args:
        config: The parsed subagent configuration.
        parent_toolset: The parent toolset to derive tools from.
        model: Fallback model. Used if config.model is 'inherit' or None.
        model_settings: Fallback model settings. Used if config.model_settings is 'inherit' or None.

    Returns:
        A BaseTool subclass that wraps the subagent.
    """
    from pydantic_ai import Agent

    # Resolve model
    # Priority: config.model (if not inherit/None) > function arg model
    if config.model is not None and config.model != INHERIT:
        effective_model = config.model
    elif model is not None:
        effective_model = model
    else:
        # Will be resolved at runtime - use a placeholder
        effective_model = "test"  # Placeholder, actual model passed at runtime

    # Resolve model_settings
    # Priority: config.model_settings (if not inherit/None) > function arg model_settings
    resolved_settings: dict[str, Any] | None = None
    if config.model_settings is not None and config.model_settings != INHERIT:
        resolved_settings = resolve_model_settings(config.model_settings)
    elif model_settings is not None:
        resolved_settings = resolve_model_settings(model_settings)

    # Create subset toolset
    sub_toolset = parent_toolset.subset(config.tools)

    # Create the subagent
    subagent: Agent[AgentContext, str] = Agent(
        model=effective_model,
        system_prompt=config.system_prompt,
        toolsets=[sub_toolset],
        model_settings=resolved_settings,  # type: ignore[arg-type]
        deps_type=AgentContext,
    )

    # Create the tool
    return create_subagent_tool(
        name=config.name,
        description=config.description,
        call_func=create_subagent_call_func(subagent),
        instruction=config.instruction,
    )


def create_subagent_tool_from_markdown(
    content: str | Path,
    parent_toolset: Toolset[Any],
    *,
    model: str | Model | None = None,
    model_settings: dict[str, Any] | str | None = None,
) -> type[BaseTool]:
    """Create a subagent tool from markdown content or file path.

    This is the main convenience function for creating subagent tools.

    Args:
        content: Markdown string or path to markdown file.
        parent_toolset: The parent toolset to derive tools from.
        model: Fallback model. Used if config.model is 'inherit' or None.
        model_settings: Fallback model settings. Used if config.model_settings is 'inherit' or None.

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
    )


def load_subagent_tools_from_dir(
    dir_path: Path | str,
    parent_toolset: Toolset[Any],
    *,
    model: str | Model | None = None,
    model_settings: dict[str, Any] | str | None = None,
) -> list[type[BaseTool]]:
    """Load all subagent tools from a directory.

    Scans the directory for .md files and creates a subagent tool for each.

    Args:
        dir_path: Path to the directory containing markdown files.
        parent_toolset: The parent toolset to derive tools from.
        model: Fallback model for all subagents.
        model_settings: Fallback model settings for all subagents.

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
        )
        tools.append(tool)

    return tools
