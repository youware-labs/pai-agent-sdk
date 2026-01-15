"""Tests for subagents module initialization and builtin functions."""

from __future__ import annotations

from pai_agent_sdk.presets import (
    ModelSettingsPreset,
    get_model_settings,
    list_presets,
    resolve_model_settings,
)
from pai_agent_sdk.subagents import (
    INHERIT,
    SubagentConfig,
    create_subagent_call_func,
    create_subagent_tool,
    get_builtin_subagent_configs,
    load_subagent_from_file,
    load_subagents_from_dir,
    parse_subagent_markdown,
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
    assert "grep" in explorer.system_prompt or "glob" in explorer.system_prompt

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
        name = "grep"
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
        name = "grep"
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


# --- create_agent subagent integration tests ---


async def test_create_agent_with_subagent_configs(agent_context) -> None:
    """Test create_agent with custom subagent_configs."""
    from pai_agent_sdk.agents.main import create_agent
    from pai_agent_sdk.subagents import SubagentConfig
    from pai_agent_sdk.toolsets.core.base import BaseTool

    class DummyTool(BaseTool):
        name = "dummy_tool"
        description = "A dummy tool"

        async def call(self, ctx, message: str = "hello") -> str:
            return f"dummy: {message}"

    config = SubagentConfig(
        name="custom_subagent",
        description="A custom subagent",
        system_prompt="You are a custom subagent.",
        tools=["dummy_tool"],
    )

    async with create_agent(
        "test",
        tools=[DummyTool],
        subagent_configs=[config],
        compact_model="test",
    ) as runtime:
        # Verify subagent tool was added
        assert runtime.core_toolset is not None
        assert "dummy_tool" in runtime.core_toolset.tool_names
        assert "custom_subagent" in runtime.core_toolset.tool_names


async def test_create_agent_with_include_builtin_subagents(agent_context) -> None:
    """Test create_agent with include_builtin_subagents=True."""
    from pai_agent_sdk.agents.main import create_agent
    from pai_agent_sdk.subagents import get_builtin_subagent_configs
    from pai_agent_sdk.toolsets.core.base import BaseTool

    # Create tools needed by builtin subagents
    class GlobTool(BaseTool):
        name = "glob"
        description = "Find files"

        async def call(self, ctx, pattern: str) -> str:
            return pattern

    class GrepTool(BaseTool):
        name = "grep"
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
        name = "search"
        description = "Search"

        async def call(self, ctx, query: str) -> str:
            return query

    async with create_agent(
        "test",
        tools=[GlobTool, GrepTool, ViewTool, LsTool, SearchTool],
        include_builtin_subagents=True,
        compact_model="test",
    ) as runtime:
        # Verify builtin subagent tools were added
        assert runtime.core_toolset is not None
        builtin_names = set(get_builtin_subagent_configs().keys())
        for name in builtin_names:
            assert name in runtime.core_toolset.tool_names


async def test_create_agent_with_both_custom_and_builtin_subagents() -> None:
    """Test create_agent with both custom and builtin subagents."""
    from pai_agent_sdk.agents.main import create_agent
    from pai_agent_sdk.subagents import SubagentConfig, get_builtin_subagent_configs
    from pai_agent_sdk.toolsets.core.base import BaseTool

    class GlobTool(BaseTool):
        name = "glob"
        description = "Find files"

        async def call(self, ctx, pattern: str) -> str:
            return pattern

    class GrepTool(BaseTool):
        name = "grep"
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
        name = "search"
        description = "Search"

        async def call(self, ctx, query: str) -> str:
            return query

    custom_config = SubagentConfig(
        name="my_custom_agent",
        description="My custom agent",
        system_prompt="You are my custom agent.",
    )

    async with create_agent(
        "test",
        tools=[GlobTool, GrepTool, ViewTool, LsTool, SearchTool],
        subagent_configs=[custom_config],
        include_builtin_subagents=True,
        compact_model="test",
    ) as runtime:
        assert runtime.core_toolset is not None
        # Check custom subagent
        assert "my_custom_agent" in runtime.core_toolset.tool_names
        # Check builtin subagents
        builtin_names = set(get_builtin_subagent_configs().keys())
        for name in builtin_names:
            assert name in runtime.core_toolset.tool_names


async def test_create_agent_no_subagents_by_default() -> None:
    """Test that create_agent does not include subagents by default."""
    from pai_agent_sdk.agents.main import create_agent
    from pai_agent_sdk.subagents import get_builtin_subagent_configs
    from pai_agent_sdk.toolsets.core.base import BaseTool

    class DummyTool(BaseTool):
        name = "dummy_tool"
        description = "A dummy tool"

        async def call(self, ctx) -> str:
            return "dummy"

    async with create_agent(
        "test",
        tools=[DummyTool],
        compact_model="test",
    ) as runtime:
        assert runtime.core_toolset is not None
        # Only dummy_tool should be present
        assert runtime.core_toolset.tool_names == ["dummy_tool"]
        # No builtin subagents
        builtin_names = set(get_builtin_subagent_configs().keys())
        for name in builtin_names:
            assert name not in runtime.core_toolset.tool_names
