"""Tests for unified subagent tool creation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_ai import RunContext

from pai_agent_sdk.subagents import (
    SubagentConfig,
    create_unified_subagent_tool,
    get_available_subagent_names,
    load_builtin_unified_subagent_tool,
    load_unified_subagent_tool_from_dir,
)
from pai_agent_sdk.toolsets.core.base import BaseTool, Toolset

# =============================================================================
# Test fixtures and mock tools
# =============================================================================


class GrepTool(BaseTool):
    """Test grep tool."""

    name = "grep"
    description = "Search file contents"

    async def call(self, ctx: RunContext, pattern: str) -> str:
        return f"grep: {pattern}"


class ViewTool(BaseTool):
    """Test view tool."""

    name = "view"
    description = "View file contents"

    async def call(self, ctx: RunContext, path: str) -> str:
        return f"view: {path}"


class LsTool(BaseTool):
    """Test ls tool."""

    name = "ls"
    description = "List directory"

    async def call(self, ctx: RunContext, path: str = ".") -> str:
        return f"ls: {path}"


class SearchTool(BaseTool):
    """Test search tool."""

    name = "search"
    description = "Search the web"

    async def call(self, ctx: RunContext, query: str) -> str:
        return f"search: {query}"


class DynamicTool(BaseTool):
    """Test tool with dynamic availability."""

    name = "dynamic_tool"
    description = "This tool has dynamic availability"
    _available = True

    def is_available(self, ctx: RunContext) -> bool:
        return DynamicTool._available

    async def call(self, ctx: RunContext) -> str:
        return "dynamic"


# =============================================================================
# Basic creation tests
# =============================================================================


def test_create_unified_tool_basic(mock_run_ctx) -> None:
    """Should create a single tool from multiple configs."""
    configs = [
        SubagentConfig(name="agent1", description="Agent 1", system_prompt="You are agent 1"),
        SubagentConfig(name="agent2", description="Agent 2", system_prompt="You are agent 2"),
    ]
    parent_toolset = Toolset(tools=[GrepTool, ViewTool])

    tool_cls = create_unified_subagent_tool(configs, parent_toolset, model="test")

    assert issubclass(tool_cls, BaseTool)
    assert tool_cls.name == "delegate"


def test_create_unified_tool_custom_name(mock_run_ctx) -> None:
    """Should support custom tool name."""
    configs = [SubagentConfig(name="agent1", description="...", system_prompt="...")]
    parent_toolset = Toolset(tools=[])

    tool_cls = create_unified_subagent_tool(
        configs,
        parent_toolset,
        name="call_specialist",
        model="test",
    )

    assert tool_cls.name == "call_specialist"


def test_create_unified_tool_custom_description(mock_run_ctx) -> None:
    """Should support custom description."""
    configs = [SubagentConfig(name="agent1", description="...", system_prompt="...")]
    parent_toolset = Toolset(tools=[])

    tool_cls = create_unified_subagent_tool(
        configs,
        parent_toolset,
        description="My custom delegate tool",
        model="test",
    )

    assert tool_cls.description == "My custom delegate tool"


def test_create_unified_tool_empty_configs_raises() -> None:
    """Should raise error when no configs provided."""
    parent_toolset = Toolset(tools=[])

    with pytest.raises(ValueError, match="At least one SubagentConfig"):
        create_unified_subagent_tool([], parent_toolset, model="test")


# =============================================================================
# Availability tests
# =============================================================================


def test_unified_tool_available_when_at_least_one_subagent_available(mock_run_ctx) -> None:
    """Tool should be available if at least one subagent is available."""
    configs = [
        SubagentConfig(name="a1", description="...", system_prompt="...", tools=["grep"]),
        SubagentConfig(name="a2", description="...", system_prompt="...", tools=["missing"]),
    ]
    parent_toolset = Toolset(tools=[GrepTool])  # Only grep available

    tool_cls = create_unified_subagent_tool(configs, parent_toolset, model="test")
    tool = tool_cls()

    assert tool.is_available(mock_run_ctx) is True


def test_unified_tool_unavailable_when_all_subagents_unavailable(mock_run_ctx) -> None:
    """Tool should be unavailable if all subagents are unavailable."""
    configs = [
        SubagentConfig(name="a1", description="...", system_prompt="...", tools=["missing1"]),
        SubagentConfig(name="a2", description="...", system_prompt="...", tools=["missing2"]),
    ]
    parent_toolset = Toolset(tools=[GrepTool])

    tool_cls = create_unified_subagent_tool(configs, parent_toolset, model="test")
    tool = tool_cls()

    assert tool.is_available(mock_run_ctx) is False


def test_unified_tool_available_when_no_required_tools(mock_run_ctx) -> None:
    """Tool should be available when subagents have no required tools."""
    configs = [
        SubagentConfig(name="a1", description="...", system_prompt="...", tools=None),
    ]
    parent_toolset = Toolset(tools=[])

    tool_cls = create_unified_subagent_tool(configs, parent_toolset, model="test")
    tool = tool_cls()

    assert tool.is_available(mock_run_ctx) is True


def test_unified_tool_availability_dynamic(mock_run_ctx) -> None:
    """Availability should be checked dynamically."""
    DynamicTool._available = True
    configs = [
        SubagentConfig(name="a1", description="...", system_prompt="...", tools=["dynamic_tool"]),
    ]
    parent_toolset = Toolset(tools=[DynamicTool])

    tool_cls = create_unified_subagent_tool(configs, parent_toolset, model="test")
    tool = tool_cls()

    assert tool.is_available(mock_run_ctx) is True

    DynamicTool._available = False
    assert tool.is_available(mock_run_ctx) is False

    # Restore
    DynamicTool._available = True


# =============================================================================
# Instruction generation tests
# =============================================================================


def test_unified_tool_instruction_contains_available_subagents(mock_run_ctx) -> None:
    """Instruction should list available subagents."""
    configs = [
        SubagentConfig(
            name="debugger",
            description="Debug issues",
            system_prompt="...",
            instruction="Use for debugging errors.",
            tools=["grep"],
        ),
    ]
    parent_toolset = Toolset(tools=[GrepTool])

    tool_cls = create_unified_subagent_tool(configs, parent_toolset, model="test")
    tool = tool_cls()

    instruction = tool.get_instruction(mock_run_ctx)

    assert instruction is not None
    assert "debugger" in instruction
    assert "debugging errors" in instruction


def test_unified_tool_instruction_excludes_unavailable_subagents(mock_run_ctx) -> None:
    """Instruction should not include unavailable subagents."""
    configs = [
        SubagentConfig(
            name="available_agent",
            description="...",
            system_prompt="...",
            instruction="Available agent instruction",
            tools=["grep"],
        ),
        SubagentConfig(
            name="unavailable_agent",
            description="...",
            system_prompt="...",
            instruction="Unavailable agent instruction",
            tools=["missing_tool"],
        ),
    ]
    parent_toolset = Toolset(tools=[GrepTool])

    tool_cls = create_unified_subagent_tool(configs, parent_toolset, model="test")
    tool = tool_cls()

    instruction = tool.get_instruction(mock_run_ctx)

    assert instruction is not None
    assert "available_agent" in instruction
    assert "unavailable_agent" not in instruction


def test_unified_tool_instruction_none_when_no_subagents_available(mock_run_ctx) -> None:
    """Instruction should be None when no subagents available."""
    configs = [
        SubagentConfig(name="a1", description="...", system_prompt="...", tools=["missing"]),
    ]
    parent_toolset = Toolset(tools=[])

    tool_cls = create_unified_subagent_tool(configs, parent_toolset, model="test")
    tool = tool_cls()

    instruction = tool.get_instruction(mock_run_ctx)

    assert instruction is None


def test_unified_tool_instruction_format(mock_run_ctx) -> None:
    """Instruction should use XML-like format for subagent sections."""
    configs = [
        SubagentConfig(
            name="helper",
            description="...",
            system_prompt="...",
            instruction="Help with tasks",
        ),
    ]
    parent_toolset = Toolset(tools=[])

    tool_cls = create_unified_subagent_tool(configs, parent_toolset, model="test")
    tool = tool_cls()

    instruction = tool.get_instruction(mock_run_ctx)

    assert '<subagent name="helper">' in instruction
    assert "</subagent>" in instruction


# =============================================================================
# Parameter type tests
# =============================================================================


def test_unified_tool_subagent_name_is_literal_type() -> None:
    """subagent_name parameter should be Literal type with correct values."""
    configs = [
        SubagentConfig(name="agent_a", description="...", system_prompt="..."),
        SubagentConfig(name="agent_b", description="...", system_prompt="..."),
    ]
    parent_toolset = Toolset(tools=[])

    tool_cls = create_unified_subagent_tool(configs, parent_toolset, model="test")

    # Use the helper function
    names = get_available_subagent_names(tool_cls)
    assert set(names) == {"agent_a", "agent_b"}


def test_get_available_subagent_names() -> None:
    """get_available_subagent_names should extract names from tool class."""
    configs = [
        SubagentConfig(name="debugger", description="...", system_prompt="..."),
        SubagentConfig(name="explorer", description="...", system_prompt="..."),
        SubagentConfig(name="searcher", description="...", system_prompt="..."),
    ]
    parent_toolset = Toolset(tools=[])

    tool_cls = create_unified_subagent_tool(configs, parent_toolset, model="test")
    names = get_available_subagent_names(tool_cls)

    assert set(names) == {"debugger", "explorer", "searcher"}


# =============================================================================
# Call behavior tests
# =============================================================================


@pytest.mark.asyncio
async def test_unified_tool_call_unknown_subagent_returns_error(mock_run_ctx) -> None:
    """Calling unknown subagent should return error message."""
    configs = [
        SubagentConfig(name="helper", description="...", system_prompt="You help."),
    ]
    parent_toolset = Toolset(tools=[])

    tool_cls = create_unified_subagent_tool(configs, parent_toolset, model="test")
    tool = tool_cls()

    # Call with unknown name (bypassing type check)
    result = await tool.call(mock_run_ctx, subagent_name="nonexistent", prompt="Help me")

    assert "error" in result.lower()
    assert "nonexistent" in result


@pytest.mark.asyncio
async def test_unified_tool_call_unavailable_subagent_returns_error(mock_run_ctx) -> None:
    """Calling unavailable subagent should return error message."""
    configs = [
        SubagentConfig(
            name="restricted",
            description="...",
            system_prompt="...",
            tools=["missing_tool"],
        ),
    ]
    parent_toolset = Toolset(tools=[])

    tool_cls = create_unified_subagent_tool(configs, parent_toolset, model="test")
    tool = tool_cls()

    result = await tool.call(mock_run_ctx, subagent_name="restricted", prompt="Do something")

    assert "not available" in result.lower() or "error" in result.lower()
    assert "missing_tool" in result


@pytest.mark.asyncio
async def test_unified_tool_call_available_subagent_succeeds(mock_run_ctx) -> None:
    """Calling available subagent should succeed."""
    configs = [
        SubagentConfig(name="helper", description="...", system_prompt="You help."),
    ]
    parent_toolset = Toolset(tools=[])

    tool_cls = create_unified_subagent_tool(configs, parent_toolset, model="test")
    tool = tool_cls()

    # Set tool_call_id to a real string (required for usage tracking)
    mock_run_ctx.tool_call_id = "test-call-123"

    result = await tool.call(mock_run_ctx, subagent_name="helper", prompt="Help me")

    # Should return formatted result with id and response
    assert "<id>" in result
    assert "helper" in result


# =============================================================================
# Convenience function tests
# =============================================================================


def test_load_unified_from_dir(tmp_path: Path, mock_run_ctx) -> None:
    """Should load configs from directory and create unified tool."""
    (tmp_path / "a1.md").write_text("""---
name: agent1
description: First agent
---
Prompt 1
""")
    (tmp_path / "a2.md").write_text("""---
name: agent2
description: Second agent
---
Prompt 2
""")
    parent_toolset = Toolset(tools=[])

    tool_cls = load_unified_subagent_tool_from_dir(tmp_path, parent_toolset, model="test")

    assert tool_cls.name == "delegate"
    names = get_available_subagent_names(tool_cls)
    assert set(names) == {"agent1", "agent2"}


def test_load_unified_from_dir_custom_name(tmp_path: Path, mock_run_ctx) -> None:
    """Should support custom name when loading from directory."""
    (tmp_path / "a1.md").write_text("""---
name: agent1
description: First
---
Prompt
""")
    parent_toolset = Toolset(tools=[])

    tool_cls = load_unified_subagent_tool_from_dir(
        tmp_path,
        parent_toolset,
        name="specialist",
        model="test",
    )

    assert tool_cls.name == "specialist"


def test_load_builtin_unified_tool(mock_run_ctx) -> None:
    """Should load builtin subagents as unified tool."""
    parent_toolset = Toolset(tools=[GrepTool, ViewTool, LsTool, SearchTool])

    tool_cls = load_builtin_unified_subagent_tool(parent_toolset, model="test")

    assert tool_cls.name == "delegate"

    # Should have builtin subagent names
    names = get_available_subagent_names(tool_cls)
    assert "debugger" in names or "explorer" in names  # At least some builtins


def test_load_builtin_unified_tool_custom_name(mock_run_ctx) -> None:
    """Should support custom name for builtin unified tool."""
    parent_toolset = Toolset(tools=[GrepTool, ViewTool, LsTool, SearchTool])

    tool_cls = load_builtin_unified_subagent_tool(
        parent_toolset,
        name="subagent",
        model="test",
    )

    assert tool_cls.name == "subagent"


# =============================================================================
# Optional tools tests
# =============================================================================


def test_unified_tool_with_optional_tools(mock_run_ctx) -> None:
    """Subagent with optional tools should be available if required tools exist."""
    configs = [
        SubagentConfig(
            name="flexible",
            description="...",
            system_prompt="...",
            tools=["grep"],  # required
            optional_tools=["missing"],  # optional, missing
        ),
    ]
    parent_toolset = Toolset(tools=[GrepTool])

    tool_cls = create_unified_subagent_tool(configs, parent_toolset, model="test")
    tool = tool_cls()

    # Should be available because required tools exist
    assert tool.is_available(mock_run_ctx) is True


def test_unified_tool_only_optional_tools(mock_run_ctx) -> None:
    """Subagent with only optional_tools should always be available."""
    configs = [
        SubagentConfig(
            name="flexible",
            description="...",
            system_prompt="...",
            tools=None,  # no required
            optional_tools=["missing"],  # optional only
        ),
    ]
    parent_toolset = Toolset(tools=[])

    tool_cls = create_unified_subagent_tool(configs, parent_toolset, model="test")
    tool = tool_cls()

    assert tool.is_available(mock_run_ctx) is True


# =============================================================================
# Toolset.with_subagents(unified=True) tests
# =============================================================================


def test_toolset_with_subagents_unified_true(mock_run_ctx) -> None:
    """Toolset.with_subagents(unified=True) should create a single delegate tool."""
    configs = [
        SubagentConfig(name="agent1", description="Agent 1", system_prompt="You are agent 1"),
        SubagentConfig(name="agent2", description="Agent 2", system_prompt="You are agent 2"),
    ]
    parent_toolset = Toolset(tools=[GrepTool, ViewTool])

    toolset_with_subs = parent_toolset.with_subagents(configs, model="test", unified=True)

    # Should have original tools + 1 unified delegate tool
    tool_names = list(toolset_with_subs._tool_classes.keys())
    assert "grep" in tool_names
    assert "view" in tool_names
    assert "delegate" in tool_names
    assert len(tool_names) == 3  # grep, view, delegate


def test_toolset_with_subagents_unified_false(mock_run_ctx) -> None:
    """Toolset.with_subagents(unified=False) should create individual tools per subagent."""
    configs = [
        SubagentConfig(name="agent1", description="Agent 1", system_prompt="You are agent 1"),
        SubagentConfig(name="agent2", description="Agent 2", system_prompt="You are agent 2"),
    ]
    parent_toolset = Toolset(tools=[GrepTool, ViewTool])

    toolset_with_subs = parent_toolset.with_subagents(configs, model="test", unified=False)

    # Should have original tools + 2 individual subagent tools
    tool_names = list(toolset_with_subs._tool_classes.keys())
    assert "grep" in tool_names
    assert "view" in tool_names
    assert "agent1" in tool_names
    assert "agent2" in tool_names
    assert len(tool_names) == 4  # grep, view, agent1, agent2


def test_toolset_with_subagents_default_is_not_unified(mock_run_ctx) -> None:
    """Toolset.with_subagents() default behavior should create individual tools."""
    configs = [
        SubagentConfig(name="agent1", description="Agent 1", system_prompt="You are agent 1"),
    ]
    parent_toolset = Toolset(tools=[GrepTool])

    toolset_with_subs = parent_toolset.with_subagents(configs, model="test")

    # Should have original tool + individual subagent tool
    tool_names = list(toolset_with_subs._tool_classes.keys())
    assert "grep" in tool_names
    assert "agent1" in tool_names
    assert "delegate" not in tool_names


# =============================================================================
# auto_inherit tests
# =============================================================================


class AutoInheritTool(BaseTool):
    """Test tool with auto_inherit=True."""

    name = "auto_tool"
    description = "Auto-inherit tool"
    auto_inherit = True

    async def call(self, ctx: RunContext) -> str:
        return "auto"


class ManualTool(BaseTool):
    """Test tool with auto_inherit=False (default)."""

    name = "manual_tool"
    description = "Manual tool"
    # auto_inherit = False is default

    async def call(self, ctx: RunContext) -> str:
        return "manual"


def test_subset_include_auto_inherit_true() -> None:
    """subset(include_auto_inherit=True) should include auto_inherit tools."""
    parent_toolset = Toolset(tools=[GrepTool, AutoInheritTool, ManualTool])

    # Subset with only grep, but include auto_inherit tools
    sub_toolset = parent_toolset.subset(["grep"], include_auto_inherit=True)

    tool_names = list(sub_toolset._tool_classes.keys())
    assert "grep" in tool_names
    assert "auto_tool" in tool_names  # auto_inherit=True, included
    assert "manual_tool" not in tool_names  # auto_inherit=False, not included


def test_subset_include_auto_inherit_false() -> None:
    """subset(include_auto_inherit=False) should NOT include auto_inherit tools."""
    parent_toolset = Toolset(tools=[GrepTool, AutoInheritTool, ManualTool])

    # Subset with only grep, don't include auto_inherit tools
    sub_toolset = parent_toolset.subset(["grep"], include_auto_inherit=False)

    tool_names = list(sub_toolset._tool_classes.keys())
    assert "grep" in tool_names
    assert "auto_tool" not in tool_names
    assert "manual_tool" not in tool_names


def test_subset_default_does_not_include_auto_inherit() -> None:
    """subset() default should NOT include auto_inherit tools."""
    parent_toolset = Toolset(tools=[GrepTool, AutoInheritTool])

    sub_toolset = parent_toolset.subset(["grep"])

    tool_names = list(sub_toolset._tool_classes.keys())
    assert "grep" in tool_names
    assert "auto_tool" not in tool_names


def test_unified_subagent_includes_auto_inherit_tools(mock_run_ctx) -> None:
    """Unified subagent tool should include auto_inherit tools automatically."""
    configs = [
        SubagentConfig(
            name="agent1",
            description="Agent 1",
            system_prompt="You are agent 1",
            tools=["grep"],  # Only explicitly requests grep
        ),
    ]
    parent_toolset = Toolset(tools=[GrepTool, AutoInheritTool])

    # create_unified_subagent_tool uses subset with include_auto_inherit=True
    tool_cls = create_unified_subagent_tool(configs, parent_toolset, model="test")

    # The subagent should have access to both grep and auto_tool
    assert tool_cls is not None
