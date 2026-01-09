"""Tests for subagent factory with availability checking."""

from __future__ import annotations

from pai_agent_sdk.subagents import SubagentConfig, create_subagent_tool_from_config
from pai_agent_sdk.toolsets.core.base import BaseTool, Toolset


class GrepTool(BaseTool):
    """Test grep tool."""

    name = "grep_tool"
    description = "Search file contents"

    async def call(self, ctx, pattern: str) -> str:
        return f"grep: {pattern}"


class ViewTool(BaseTool):
    """Test view tool."""

    name = "view"
    description = "View file contents"

    async def call(self, ctx, path: str) -> str:
        return f"view: {path}"


class UnavailableTool(BaseTool):
    """Test tool that is never available."""

    name = "unavailable_tool"
    description = "This tool is never available"

    def is_available(self) -> bool:
        return False

    async def call(self, ctx) -> str:
        return "should not be called"


class DynamicTool(BaseTool):
    """Test tool with dynamic availability."""

    name = "dynamic_tool"
    description = "This tool has dynamic availability"
    _available = True

    def is_available(self) -> bool:
        return DynamicTool._available

    async def call(self, ctx) -> str:
        return "dynamic"


class TestSubagentToolAvailability:
    """Tests for subagent tool availability checking."""

    def test_subagent_available_when_all_tools_exist(self, agent_context) -> None:
        """Subagent should be available when all required tools exist and are available."""
        parent_toolset = Toolset(agent_context, tools=[GrepTool, ViewTool])

        config = SubagentConfig(
            name="test_subagent",
            description="Test subagent",
            system_prompt="You are a test agent",
            tools=["grep_tool", "view"],
        )

        tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
        tool_instance = tool_cls(agent_context)

        assert tool_instance.is_available() is True

    def test_subagent_unavailable_when_tool_missing(self, agent_context) -> None:
        """Subagent should be unavailable when a required tool is missing."""
        parent_toolset = Toolset(agent_context, tools=[GrepTool])  # ViewTool missing

        config = SubagentConfig(
            name="test_subagent",
            description="Test subagent",
            system_prompt="You are a test agent",
            tools=["grep_tool", "view"],  # requires view which is missing
        )

        tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
        tool_instance = tool_cls(agent_context)

        assert tool_instance.is_available() is False

    def test_subagent_unavailable_when_tool_not_available(self, agent_context) -> None:
        """Subagent should be unavailable when a required tool exists but is_available=False."""
        # UnavailableTool will be skipped by Toolset due to skip_unavailable=True
        parent_toolset = Toolset(agent_context, tools=[GrepTool, UnavailableTool])

        config = SubagentConfig(
            name="test_subagent",
            description="Test subagent",
            system_prompt="You are a test agent",
            tools=["grep_tool", "unavailable_tool"],
        )

        tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
        tool_instance = tool_cls(agent_context)

        # unavailable_tool is not in parent_toolset because it was skipped
        assert tool_instance.is_available() is False

    def test_subagent_available_when_tools_none(self, agent_context) -> None:
        """Subagent should be available when tools=None (inherit all)."""
        parent_toolset = Toolset(agent_context, tools=[GrepTool, ViewTool])

        config = SubagentConfig(
            name="test_subagent",
            description="Test subagent",
            system_prompt="You are a test agent",
            tools=None,  # inherit all
        )

        tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
        tool_instance = tool_cls(agent_context)

        assert tool_instance.is_available() is True

    def test_subagent_dynamic_availability(self, agent_context) -> None:
        """Subagent availability should be checked dynamically."""
        # Start with dynamic tool available
        DynamicTool._available = True
        parent_toolset = Toolset(agent_context, tools=[GrepTool, DynamicTool])

        config = SubagentConfig(
            name="test_subagent",
            description="Test subagent",
            system_prompt="You are a test agent",
            tools=["grep_tool", "dynamic_tool"],
        )

        tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
        tool_instance = tool_cls(agent_context)

        # Initially available
        assert tool_instance.is_available() is True

        # Make dynamic tool unavailable
        DynamicTool._available = False

        # Now subagent should be unavailable (dynamic check)
        assert tool_instance.is_available() is False

        # Restore
        DynamicTool._available = True
        assert tool_instance.is_available() is True


class TestToolsetIsToolAvailable:
    """Tests for Toolset.is_tool_available method."""

    def test_is_tool_available_for_existing_tool(self, agent_context) -> None:
        """Should return True for existing and available tool."""
        toolset = Toolset(agent_context, tools=[GrepTool, ViewTool])

        assert toolset.is_tool_available("grep_tool") is True
        assert toolset.is_tool_available("view") is True

    def test_is_tool_available_for_missing_tool(self, agent_context) -> None:
        """Should return False for non-existent tool."""
        toolset = Toolset(agent_context, tools=[GrepTool])

        assert toolset.is_tool_available("view") is False
        assert toolset.is_tool_available("nonexistent") is False

    def test_is_tool_available_for_unavailable_tool(self, agent_context) -> None:
        """Should return False for tool that was skipped due to is_available=False."""
        # UnavailableTool will be skipped
        toolset = Toolset(agent_context, tools=[GrepTool, UnavailableTool])

        assert toolset.is_tool_available("grep_tool") is True
        assert toolset.is_tool_available("unavailable_tool") is False

    def test_is_tool_available_dynamic(self, agent_context) -> None:
        """Should dynamically check tool availability."""
        DynamicTool._available = True
        toolset = Toolset(agent_context, tools=[DynamicTool])

        assert toolset.is_tool_available("dynamic_tool") is True

        # Change availability
        DynamicTool._available = False
        assert toolset.is_tool_available("dynamic_tool") is False

        # Restore
        DynamicTool._available = True


class TestOptionalTools:
    """Tests for optional_tools functionality."""

    def test_subagent_available_with_optional_tools_missing(self, agent_context) -> None:
        """Subagent should be available even if optional tools are missing."""
        parent_toolset = Toolset(agent_context, tools=[GrepTool, ViewTool])

        config = SubagentConfig(
            name="test_subagent",
            description="Test subagent",
            system_prompt="You are a test agent",
            tools=["grep_tool"],  # required
            optional_tools=["nonexistent_tool"],  # optional, missing
        )

        tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
        tool_instance = tool_cls(agent_context)

        # Should still be available because required tools exist
        assert tool_instance.is_available() is True

    def test_subagent_unavailable_when_required_missing_but_optional_present(self, agent_context) -> None:
        """Subagent should be unavailable if required tools are missing, even with optional present."""
        parent_toolset = Toolset(agent_context, tools=[GrepTool, ViewTool])

        config = SubagentConfig(
            name="test_subagent",
            description="Test subagent",
            system_prompt="You are a test agent",
            tools=["nonexistent_tool"],  # required, missing
            optional_tools=["grep_tool"],  # optional, present
        )

        tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
        tool_instance = tool_cls(agent_context)

        # Should be unavailable because required tool is missing
        assert tool_instance.is_available() is False

    def test_subagent_with_both_required_and_optional_tools(self, agent_context) -> None:
        """Subagent should include both required and optional tools in subset."""
        parent_toolset = Toolset(agent_context, tools=[GrepTool, ViewTool])

        config = SubagentConfig(
            name="test_subagent",
            description="Test subagent",
            system_prompt="You are a test agent",
            tools=["grep_tool"],  # required
            optional_tools=["view"],  # optional
        )

        tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
        tool_instance = tool_cls(agent_context)

        assert tool_instance.is_available() is True

    def test_subagent_only_optional_tools_always_available(self, agent_context) -> None:
        """Subagent with only optional_tools (no required) should always be available."""
        parent_toolset = Toolset(agent_context, tools=[GrepTool, ViewTool])

        config = SubagentConfig(
            name="test_subagent",
            description="Test subagent",
            system_prompt="You are a test agent",
            tools=None,  # no required tools
            optional_tools=["grep_tool", "nonexistent"],  # optional only
        )

        tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
        tool_instance = tool_cls(agent_context)

        # Should be available because tools=None means inherit all (no required check)
        assert tool_instance.is_available() is True
