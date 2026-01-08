"""Tests for subagents module initialization and builtin functions."""

from __future__ import annotations

from pai_agent_sdk.subagents import (
    INHERIT,
    ModelSettingsPreset,
    SubagentConfig,
    create_subagent_call_func,
    create_subagent_tool,
    get_builtin_subagent_configs,
    get_model_settings,
    list_presets,
    load_subagent_from_file,
    load_subagents_from_dir,
    parse_subagent_markdown,
    resolve_model_settings,
)


def test_public_api_exports() -> None:
    """Test that all public API is exported from __init__."""
    # Constants
    assert INHERIT == "inherit"

    # Config
    assert SubagentConfig is not None
    assert parse_subagent_markdown is not None
    assert load_subagent_from_file is not None
    assert load_subagents_from_dir is not None

    # Presets
    assert ModelSettingsPreset is not None
    assert get_model_settings is not None
    assert resolve_model_settings is not None
    assert list_presets is not None

    # Factory re-exports
    assert create_subagent_call_func is not None
    assert create_subagent_tool is not None


def test_get_builtin_subagent_configs() -> None:
    """Test that builtin configs can be loaded."""
    configs = get_builtin_subagent_configs()

    # Should return dict
    assert isinstance(configs, dict)

    # Should have known builtin subagents
    expected_names = {"debugger", "code-reviewer", "explorer", "searcher"}
    assert set(configs.keys()) == expected_names

    # All values should be SubagentConfig
    for name, config in configs.items():
        assert isinstance(name, str)
        assert isinstance(config, SubagentConfig)
        assert config.name == name
        assert config.description is not None
        assert config.system_prompt is not None


def test_get_builtin_subagent_configs_content() -> None:
    """Test that builtin configs have expected content."""
    configs = get_builtin_subagent_configs()

    # Debugger should have instruction
    debugger = configs["debugger"]
    assert "error" in debugger.instruction.lower() or "debug" in debugger.instruction.lower()

    # Explorer should have system_prompt mentioning exploration tools
    explorer = configs["explorer"]
    assert "grep_tool" in explorer.system_prompt or "glob_tool" in explorer.system_prompt

    # Searcher should have system_prompt mentioning search tools
    searcher = configs["searcher"]
    assert "search_with_tavily" in searcher.system_prompt or "search_with_google" in searcher.system_prompt


def test_load_builtin_subagent_tools_import() -> None:
    """Test that load_builtin_subagent_tools is exported."""
    from pai_agent_sdk.subagents import load_builtin_subagent_tools

    assert load_builtin_subagent_tools is not None


def test_load_builtin_subagent_tools(agent_context) -> None:
    """Test that load_builtin_subagent_tools loads all preset subagents."""
    from pai_agent_sdk.subagents import (
        get_builtin_subagent_configs,
        load_builtin_subagent_tools,
    )
    from pai_agent_sdk.toolsets.core.base import BaseTool, Toolset

    # Create mock tools that the subagents need
    class GrepTool(BaseTool):
        name = "grep_tool"
        description = "Search file contents"

        async def call(self, ctx, pattern: str) -> str:
            return f"grep: {pattern}"

    class ViewTool(BaseTool):
        name = "view"
        description = "View file contents"

        async def call(self, ctx, path: str) -> str:
            return f"view: {path}"

    class LsTool(BaseTool):
        name = "ls"
        description = "List directory"

        async def call(self, ctx, path: str = ".") -> str:
            return f"ls: {path}"

    class SearchTavilyTool(BaseTool):
        name = "search_with_tavily"
        description = "Search with Tavily"

        async def call(self, ctx, query: str) -> str:
            return f"tavily: {query}"

    class SearchGoogleTool(BaseTool):
        name = "search_with_google"
        description = "Search with Google"

        async def call(self, ctx, query: str) -> str:
            return f"google: {query}"

    class VisitWebpageTool(BaseTool):
        name = "visit_webpage"
        description = "Visit a webpage"

        async def call(self, ctx, url: str) -> str:
            return f"visit: {url}"

    # Create parent toolset with all tools needed by presets
    parent_toolset = Toolset(
        agent_context,
        tools=[GrepTool, ViewTool, LsTool, SearchTavilyTool, SearchGoogleTool, VisitWebpageTool],
    )

    # Load builtin subagent tools
    tools = load_builtin_subagent_tools(
        parent_toolset,
        model="test",
        model_settings={"temperature": 0.5},
    )

    # Should return list
    assert isinstance(tools, list)

    # Should have same number as builtin configs
    configs = get_builtin_subagent_configs()
    assert len(tools) == len(configs)

    # All should be BaseTool subclasses
    for tool_cls in tools:
        assert isinstance(tool_cls, type)
        assert issubclass(tool_cls, BaseTool)

    # Should have correct names matching configs
    tool_names = {tool_cls.name for tool_cls in tools}
    config_names = set(configs.keys())
    assert tool_names == config_names


def test_load_builtin_subagent_tools_with_preset_model_settings(agent_context) -> None:
    """Test load_builtin_subagent_tools with preset name for model_settings."""
    from pai_agent_sdk.subagents import load_builtin_subagent_tools
    from pai_agent_sdk.toolsets.core.base import BaseTool, Toolset

    # Minimal tools
    class GrepTool(BaseTool):
        name = "grep_tool"
        description = "Search"

        async def call(self, ctx, pattern: str) -> str:
            return pattern

    class ViewTool(BaseTool):
        name = "view"
        description = "View"

        async def call(self, ctx, path: str) -> str:
            return path

    class LsTool(BaseTool):
        name = "ls"
        description = "List"

        async def call(self, ctx) -> str:
            return "."

    class SearchTool(BaseTool):
        name = "search_with_tavily"
        description = "Search"

        async def call(self, ctx, query: str) -> str:
            return query

    class VisitTool(BaseTool):
        name = "visit_webpage"
        description = "Visit"

        async def call(self, ctx, url: str) -> str:
            return url

    parent_toolset = Toolset(
        agent_context,
        tools=[GrepTool, ViewTool, LsTool, SearchTool, VisitTool],
    )

    # Should work with preset name string
    tools = load_builtin_subagent_tools(
        parent_toolset,
        model="test",
        model_settings="anthropic_medium",
    )

    assert len(tools) == 4
    for tool_cls in tools:
        assert issubclass(tool_cls, BaseTool)
