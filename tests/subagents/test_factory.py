"""Tests for subagent factory with availability checking."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai import Agent, RunContext

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.environment.local import LocalEnvironment
from pai_agent_sdk.subagents import SubagentConfig, create_subagent_tool_from_config
from pai_agent_sdk.toolsets.core.base import BaseTool, Toolset
from pai_agent_sdk.toolsets.core.subagent.factory import create_subagent_call_func


class GrepTool(BaseTool):
    """Test grep tool."""

    name = "grep"
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

    def is_available(self, ctx) -> bool:
        return False

    async def call(self, ctx) -> str:
        return "should not be called"


class DynamicTool(BaseTool):
    """Test tool with dynamic availability."""

    name = "dynamic_tool"
    description = "This tool has dynamic availability"
    _available = True

    def is_available(self, ctx) -> bool:
        return DynamicTool._available

    async def call(self, ctx) -> str:
        return "dynamic"


class TestSubagentToolAvailability:
    """Tests for subagent tool availability checking."""

    def test_subagent_available_when_all_tools_exist(self, agent_context, mock_run_ctx) -> None:
        """Subagent should be available when all required tools exist and are available."""
        parent_toolset = Toolset(tools=[GrepTool, ViewTool])

        config = SubagentConfig(
            name="test_subagent",
            description="Test subagent",
            system_prompt="You are a test agent",
            tools=["grep", "view"],
        )

        tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
        tool_instance = tool_cls()

        assert tool_instance.is_available(mock_run_ctx) is True

    def test_subagent_unavailable_when_tool_missing(self, agent_context, mock_run_ctx) -> None:
        """Subagent should be unavailable when a required tool is missing."""
        parent_toolset = Toolset(tools=[GrepTool])  # ViewTool missing

        config = SubagentConfig(
            name="test_subagent",
            description="Test subagent",
            system_prompt="You are a test agent",
            tools=["grep", "view"],  # requires view which is missing
        )

        tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
        tool_instance = tool_cls()

        assert tool_instance.is_available(mock_run_ctx) is False

    def test_subagent_unavailable_when_tool_not_available(self, agent_context, mock_run_ctx) -> None:
        """Subagent should be unavailable when a required tool exists but is_available=False."""
        # UnavailableTool will be skipped by Toolset due to skip_unavailable=True
        parent_toolset = Toolset(tools=[GrepTool, UnavailableTool])

        config = SubagentConfig(
            name="test_subagent",
            description="Test subagent",
            system_prompt="You are a test agent",
            tools=["grep", "unavailable_tool"],
        )

        tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
        tool_instance = tool_cls()

        # unavailable_tool is not in parent_toolset because it was skipped
        assert tool_instance.is_available(mock_run_ctx) is False

    def test_subagent_available_when_tools_none(self, agent_context, mock_run_ctx) -> None:
        """Subagent should be available when tools=None (inherit all)."""
        parent_toolset = Toolset(tools=[GrepTool, ViewTool])

        config = SubagentConfig(
            name="test_subagent",
            description="Test subagent",
            system_prompt="You are a test agent",
            tools=None,  # inherit all
        )

        tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
        tool_instance = tool_cls()

        assert tool_instance.is_available(mock_run_ctx) is True

    def test_subagent_dynamic_availability(self, agent_context, mock_run_ctx) -> None:
        """Subagent availability should be checked dynamically."""
        # Start with dynamic tool available
        DynamicTool._available = True
        parent_toolset = Toolset(tools=[GrepTool, DynamicTool])

        config = SubagentConfig(
            name="test_subagent",
            description="Test subagent",
            system_prompt="You are a test agent",
            tools=["grep", "dynamic_tool"],
        )

        tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
        tool_instance = tool_cls()

        # Initially available
        assert tool_instance.is_available(mock_run_ctx) is True

        # Make dynamic tool unavailable
        DynamicTool._available = False

        # Now subagent should be unavailable (dynamic check)
        assert tool_instance.is_available(mock_run_ctx) is False

        # Restore
        DynamicTool._available = True
        assert tool_instance.is_available(mock_run_ctx) is True


class TestToolsetIsToolAvailable:
    """Tests for Toolset.is_tool_available method."""

    def test_is_tool_available_for_existing_tool(self, agent_context, mock_run_ctx) -> None:
        """Should return True for existing and available tool."""
        toolset = Toolset(tools=[GrepTool, ViewTool])

        assert toolset.is_tool_available("grep", mock_run_ctx) is True
        assert toolset.is_tool_available("view", mock_run_ctx) is True

    def test_is_tool_available_for_missing_tool(self, agent_context, mock_run_ctx) -> None:
        """Should return False for non-existent tool."""
        toolset = Toolset(tools=[GrepTool])

        assert toolset.is_tool_available("view", mock_run_ctx) is False
        assert toolset.is_tool_available("nonexistent", mock_run_ctx) is False

    def test_is_tool_available_for_unavailable_tool(self, agent_context, mock_run_ctx) -> None:
        """Should return False for tool that was skipped due to is_available=False."""
        # UnavailableTool is registered but is_available returns False
        toolset = Toolset(tools=[GrepTool, UnavailableTool])

        assert toolset.is_tool_available("grep", mock_run_ctx) is True
        assert toolset.is_tool_available("unavailable_tool", mock_run_ctx) is False

    def test_is_tool_available_dynamic(self, agent_context, mock_run_ctx) -> None:
        """Should dynamically check tool availability."""
        DynamicTool._available = True
        toolset = Toolset(tools=[DynamicTool])

        assert toolset.is_tool_available("dynamic_tool", mock_run_ctx) is True

        # Change availability
        DynamicTool._available = False
        assert toolset.is_tool_available("dynamic_tool", mock_run_ctx) is False

        # Restore
        DynamicTool._available = True


class TestOptionalTools:
    """Tests for optional_tools functionality."""

    def test_subagent_available_with_optional_tools_missing(self, agent_context, mock_run_ctx) -> None:
        """Subagent should be available even if optional tools are missing."""
        parent_toolset = Toolset(tools=[GrepTool, ViewTool])

        config = SubagentConfig(
            name="test_subagent",
            description="Test subagent",
            system_prompt="You are a test agent",
            tools=["grep"],  # required
            optional_tools=["nonexistent_tool"],  # optional, missing
        )

        tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
        tool_instance = tool_cls()

        # Should still be available because required tools exist
        assert tool_instance.is_available(mock_run_ctx) is True

    def test_subagent_unavailable_when_required_missing_but_optional_present(self, agent_context, mock_run_ctx) -> None:
        """Subagent should be unavailable if required tools are missing, even with optional present."""
        parent_toolset = Toolset(tools=[GrepTool, ViewTool])

        config = SubagentConfig(
            name="test_subagent",
            description="Test subagent",
            system_prompt="You are a test agent",
            tools=["nonexistent_tool"],  # required, missing
            optional_tools=["grep"],  # optional, present
        )

        tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
        tool_instance = tool_cls()

        # Should be unavailable because required tool is missing
        assert tool_instance.is_available(mock_run_ctx) is False

    def test_subagent_with_both_required_and_optional_tools(self, agent_context, mock_run_ctx) -> None:
        """Subagent should include both required and optional tools in subset."""
        parent_toolset = Toolset(tools=[GrepTool, ViewTool])

        config = SubagentConfig(
            name="test_subagent",
            description="Test subagent",
            system_prompt="You are a test agent",
            tools=["grep"],  # required
            optional_tools=["view"],  # optional
        )

        tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
        tool_instance = tool_cls()

        assert tool_instance.is_available(mock_run_ctx) is True

    def test_subagent_only_optional_tools_always_available(self, agent_context, mock_run_ctx) -> None:
        """Subagent with only optional_tools (no required) should always be available."""
        parent_toolset = Toolset(tools=[GrepTool, ViewTool])

        config = SubagentConfig(
            name="test_subagent",
            description="Test subagent",
            system_prompt="You are a test agent",
            tools=None,  # no required tools
            optional_tools=["grep", "nonexistent"],  # optional only
        )

        tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
        tool_instance = tool_cls()

        # Should be available because tools=None means inherit all (no required check)
        assert tool_instance.is_available(mock_run_ctx) is True


class TestModelCfgResolution:
    """Tests for model_cfg resolution in subagent creation."""

    def test_model_cfg_from_preset_string(self, agent_context) -> None:
        """Subagent should resolve model_cfg from preset string."""
        parent_toolset = Toolset(tools=[GrepTool])

        config = SubagentConfig(
            name="test_subagent",
            description="Test subagent",
            system_prompt="You are a test agent",
            model_cfg="claude_200k",  # preset string
        )

        # Just verify it doesn't raise - actual ModelConfig creation happens internally
        tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
        assert tool_cls is not None

    def test_model_cfg_from_dict(self, agent_context) -> None:
        """Subagent should accept model_cfg as dict."""
        parent_toolset = Toolset(tools=[GrepTool])

        config = SubagentConfig(
            name="test_subagent",
            description="Test subagent",
            system_prompt="You are a test agent",
            model_cfg={"context_window": 100000, "max_images": 5},
        )

        tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
        assert tool_cls is not None

    def test_model_cfg_inherit(self, agent_context) -> None:
        """Subagent should inherit model_cfg when set to 'inherit'."""
        parent_toolset = Toolset(tools=[GrepTool])

        config = SubagentConfig(
            name="test_subagent",
            description="Test subagent",
            system_prompt="You are a test agent",
            model_cfg="inherit",
        )

        tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
        assert tool_cls is not None

    def test_model_cfg_none_inherits(self, agent_context) -> None:
        """Subagent should inherit model_cfg when None (default)."""
        parent_toolset = Toolset(tools=[GrepTool])

        config = SubagentConfig(
            name="test_subagent",
            description="Test subagent",
            system_prompt="You are a test agent",
            model_cfg=None,  # default, inherit
        )

        tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
        assert tool_cls is not None


# =============================================================================
# Agent registry cleanup on failure tests
# =============================================================================


@pytest.fixture
async def async_agent_context(tmp_path):
    """Create an async AgentContext for tests that need create_subagent_context."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            yield ctx


async def test_agent_registry_cleaned_up_on_new_agent_failure(async_agent_context: AgentContext) -> None:
    """Agent registry should not have ghost entries when a new subagent fails."""
    agent: Agent[AgentContext, str] = Agent(
        model="test",
        system_prompt="You are a test agent",
        name="failing_agent",
    )
    call_func = create_subagent_call_func(agent)

    mock_ctx = MagicMock(spec=RunContext)
    mock_ctx.deps = async_agent_context
    mock_ctx.tool_call_id = "test-tool-call"

    mock_self = MagicMock(spec=BaseTool)

    with patch(
        "pai_agent_sdk.toolsets.core.subagent.factory._run_subagent_iter",
        new_callable=AsyncMock,
        side_effect=RuntimeError("Model API failed"),
    ):
        with pytest.raises(RuntimeError, match="Model API failed"):
            await call_func(mock_self, mock_ctx, "test prompt")

    # agent_registry should NOT contain a ghost entry
    assert len(async_agent_context.agent_registry) == 0
    # subagent_history should also be empty
    assert len(async_agent_context.subagent_history) == 0


async def test_agent_registry_preserved_on_resume_agent_failure(async_agent_context: AgentContext) -> None:
    """Agent registry entry should be preserved when a resumed subagent fails.

    If the agent was already registered (e.g., from a previous successful call),
    the registry entry should not be removed even on failure, since the agent
    has valid history from before.
    """
    agent: Agent[AgentContext, str] = Agent(
        model="test",
        system_prompt="You are a test agent",
        name="resume_agent",
    )
    call_func = create_subagent_call_func(agent)

    mock_ctx = MagicMock(spec=RunContext)
    mock_ctx.deps = async_agent_context
    mock_ctx.tool_call_id = "test-tool-call"

    mock_self = MagicMock(spec=BaseTool)

    # Pre-populate agent_registry to simulate a previously successful call
    from pai_agent_sdk.context import AgentInfo

    agent_id = "resume_agent-abcd"
    async_agent_context.agent_registry[agent_id] = AgentInfo(
        agent_id=agent_id,
        agent_name="resume_agent",
        parent_agent_id=None,
    )
    # Also populate subagent_history to simulate prior success
    async_agent_context.subagent_history[agent_id] = []

    with patch(
        "pai_agent_sdk.toolsets.core.subagent.factory._run_subagent_iter",
        new_callable=AsyncMock,
        side_effect=RuntimeError("Model API failed on resume"),
    ):
        with pytest.raises(RuntimeError, match="Model API failed on resume"):
            await call_func(mock_self, mock_ctx, "continue work", agent_id)

    # agent_registry should still contain the entry (not cleaned up for resume)
    assert agent_id in async_agent_context.agent_registry
    assert async_agent_context.agent_registry[agent_id].agent_name == "resume_agent"
