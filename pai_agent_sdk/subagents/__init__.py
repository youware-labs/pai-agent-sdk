"""Subagent configuration and creation utilities.

This module provides utilities for creating subagents from markdown configuration files.
Each markdown file contains YAML frontmatter with subagent metadata and the system prompt
as the body content.

YAML Frontmatter Schema::

    ---
    name: debugger
    description: Debug code issues
    instruction: Use this tool when debugging
    tools:  # Optional - list of tool names from parent toolset
      - grep
      - view
      - edit
    model: inherit  # Optional - 'inherit' or model name
    model_settings: anthropic_medium  # Optional - preset name, 'inherit', or dict
    ---

    System prompt content here...

Usage::

    from pai_agent_sdk.subagents import (
        parse_subagent_markdown,
        create_subagent_tool_from_markdown,
        load_subagent_tools_from_dir,
        get_builtin_subagent_configs,
        load_builtin_subagent_tools,
    )
    from pai_agent_sdk.toolsets.core.base import Toolset

    # Parse config from markdown string
    config = parse_subagent_markdown(markdown_content)

    # Create single subagent tool
    DebuggerTool = create_subagent_tool_from_markdown(
        "path/to/debugger.md",
        parent_toolset=main_toolset,
        model="anthropic:claude-sonnet-4",
    )

    # Load all builtin subagents
    subagent_tools = load_builtin_subagent_tools(
        parent_toolset=main_toolset,
        model="anthropic:claude-sonnet-4",
    )
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic_ai._agent_graph import HistoryProcessor

from pai_agent_sdk.context import AgentContext, ModelConfig
from pai_agent_sdk.presets import INHERIT
from pai_agent_sdk.subagents.config import (
    SubagentConfig,
    load_subagent_from_file,
    load_subagents_from_dir,
    parse_subagent_markdown,
)
from pai_agent_sdk.subagents.factory import (
    create_subagent_tool_from_config,
    create_subagent_tool_from_markdown,
    load_subagent_tools_from_dir,
)
from pai_agent_sdk.toolsets.core.base import BaseTool, Toolset

# Lazy imports to avoid circular dependency with toolsets.core.subagent
# These are re-exported via __getattr__ below
_LAZY_IMPORTS = (
    "create_subagent_call_func",
    "create_subagent_tool",
    "create_unified_subagent_tool",
    "get_available_subagent_names",
)

if TYPE_CHECKING:
    from pydantic_ai.models import Model

    # Type stubs for lazy imports (actual imports via __getattr__)
    from pai_agent_sdk.toolsets.core.subagent import (
        create_subagent_call_func as create_subagent_call_func,
    )
    from pai_agent_sdk.toolsets.core.subagent import (
        create_subagent_tool as create_subagent_tool,
    )
    from pai_agent_sdk.toolsets.core.subagent import (
        create_unified_subagent_tool as create_unified_subagent_tool,
    )
    from pai_agent_sdk.toolsets.core.subagent import (
        get_available_subagent_names as get_available_subagent_names,
    )

_HERE = Path(__file__).parent
PRESET_SUBAGNENTS_DIR = _HERE / "presets"


def get_builtin_subagent_configs() -> dict[str, SubagentConfig]:
    """Get all builtin subagent configurations from pai_agent_sdk/subagents/*.md.

    Returns:
        Dict mapping subagent names to their configurations.

    Example::

        configs = get_builtin_subagent_configs()
        for name, config in configs.items():
            print(f"{name}: {config.description}")
    """
    return load_subagents_from_dir(PRESET_SUBAGNENTS_DIR)


def load_builtin_subagent_tools(
    parent_toolset: Toolset[Any],
    *,
    model: str | Model | None = None,
    model_settings: dict[str, Any] | str | None = None,
    history_processors: Sequence[HistoryProcessor[AgentContext]] | None = None,
    model_cfg: ModelConfig | None = None,
) -> list[type[BaseTool]]:
    """Load all builtin subagent tools from pai_agent_sdk/subagents/*.md.

    This is a convenience function that loads all predefined subagent configurations
    and creates tools from them.

    Args:
        parent_toolset: The parent toolset to derive tools from.
        model: Fallback model for all subagents.
        model_settings: Fallback model settings for all subagents.
        history_processors: History processors for all subagents.
        model_cfg: Fallback ModelConfig for all subagents.

    Returns:
        List of BaseTool subclasses.

    Example::

        from pai_agent_sdk.subagents import load_builtin_subagent_tools
        from pai_agent_sdk.toolsets.core.base import Toolset

        main_toolset = Toolset(tools=[ViewTool, EditTool, GrepTool])
        subagent_tools = load_builtin_subagent_tools(
            parent_toolset=main_toolset,
            model="anthropic:claude-sonnet-4",
            model_settings="anthropic_medium",
        )
    """
    return load_subagent_tools_from_dir(
        PRESET_SUBAGNENTS_DIR,
        parent_toolset,
        model=model,
        model_settings=model_settings,
        history_processors=history_processors,
        model_cfg=model_cfg,
    )


def load_unified_subagent_tool_from_dir(
    dir_path: Path | str,
    parent_toolset: Toolset[Any],
    *,
    name: str = "delegate",
    description: str = "Delegate task to a specialized subagent",
    model: str | Model | None = None,
    model_settings: dict[str, Any] | str | None = None,
    history_processors: Sequence[HistoryProcessor[AgentContext]] | None = None,
    model_cfg: ModelConfig | None = None,
) -> type[BaseTool]:
    """Load all subagent configs from a directory and create a unified tool.

    This combines all subagents from the directory into a single "delegate" tool
    that selects subagents by name parameter.

    Args:
        dir_path: Path to directory containing .md subagent files.
        parent_toolset: The parent toolset to derive tools from.
        name: Tool name (default: "delegate").
        description: Tool description shown to the model.
        model: Fallback model for subagents with model="inherit".
        model_settings: Fallback model settings for subagents.
        history_processors: History processors for all subagents.
        model_cfg: Fallback ModelConfig for subagents.

    Returns:
        A BaseTool subclass that delegates to subagents by name.

    Example::

        DelegateTool = load_unified_subagent_tool_from_dir(
            "~/.config/myapp/subagents",
            parent_toolset,
            model="anthropic:claude-sonnet-4",
        )
    """
    from pai_agent_sdk.toolsets.core.subagent import create_unified_subagent_tool

    configs = load_subagents_from_dir(dir_path)
    return create_unified_subagent_tool(
        list(configs.values()),
        parent_toolset,
        name=name,
        description=description,
        model=model,
        model_settings=model_settings,
        history_processors=history_processors,
        model_cfg=model_cfg,
    )


def load_builtin_unified_subagent_tool(
    parent_toolset: Toolset[Any],
    *,
    name: str = "delegate",
    description: str = "Delegate task to a specialized subagent",
    model: str | Model | None = None,
    model_settings: dict[str, Any] | str | None = None,
    history_processors: Sequence[HistoryProcessor[AgentContext]] | None = None,
    model_cfg: ModelConfig | None = None,
) -> type[BaseTool]:
    """Load all builtin subagents as a single unified tool.

    This is a convenience function that creates a unified "delegate" tool
    from all predefined subagent configurations.

    Args:
        parent_toolset: The parent toolset to derive tools from.
        name: Tool name (default: "delegate").
        description: Tool description shown to the model.
        model: Fallback model for subagents with model="inherit".
        model_settings: Fallback model settings for subagents.
        history_processors: History processors for all subagents.
        model_cfg: Fallback ModelConfig for subagents.

    Returns:
        A BaseTool subclass that delegates to builtin subagents by name.

    Example::

        from pai_agent_sdk.subagents import load_builtin_unified_subagent_tool

        DelegateTool = load_builtin_unified_subagent_tool(
            parent_toolset,
            model="anthropic:claude-sonnet-4",
        )
        # Can call: delegate(subagent_name="debugger", prompt="Fix this error...")
    """
    return load_unified_subagent_tool_from_dir(
        PRESET_SUBAGNENTS_DIR,
        parent_toolset,
        name=name,
        description=description,
        model=model,
        model_settings=model_settings,
        history_processors=history_processors,
        model_cfg=model_cfg,
    )


def __getattr__(name: str) -> Any:
    """Lazy import for symbols from toolsets.core.subagent to avoid circular imports."""
    if name in _LAZY_IMPORTS:
        from pai_agent_sdk.toolsets.core.subagent import (
            create_subagent_call_func,
            create_subagent_tool,
            create_unified_subagent_tool,
            get_available_subagent_names,
        )

        _lazy_exports = {
            "create_subagent_call_func": create_subagent_call_func,
            "create_subagent_tool": create_subagent_tool,
            "create_unified_subagent_tool": create_unified_subagent_tool,
            "get_available_subagent_names": get_available_subagent_names,
        }
        return _lazy_exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "INHERIT",
    "SubagentConfig",
    "create_subagent_call_func",
    "create_subagent_tool",
    "create_subagent_tool_from_config",
    "create_subagent_tool_from_markdown",
    "create_unified_subagent_tool",
    "get_available_subagent_names",
    "get_builtin_subagent_configs",
    "load_builtin_subagent_tools",
    "load_builtin_unified_subagent_tool",
    "load_subagent_from_file",
    "load_subagent_tools_from_dir",
    "load_subagents_from_dir",
    "load_unified_subagent_tool_from_dir",
    "parse_subagent_markdown",
]
