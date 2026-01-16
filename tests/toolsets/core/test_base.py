"""Tests for pai_agent_sdk.toolsets.base module."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic_ai import RunContext
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.tools import ToolApproved, ToolDenied

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.toolsets.base import (
    BaseTool,
    BaseToolset,
    InstructableToolset,
    UserInputPreprocessResult,
)
from pai_agent_sdk.toolsets.core.base import (
    GlobalHooks,
    HookableToolsetTool,
    Toolset,
    UserInteraction,
)


# --- UserInteraction tests ---
def test_user_interaction_approved() -> None:
    """Should create approved interaction."""
    interaction = UserInteraction(
        tool_call_id="test-id",
        approved=True,
        user_input={"key": "value"},
    )
    assert interaction.tool_call_id == "test-id"
    assert interaction.approved is True
    assert interaction.reason is None


def test_user_interaction_rejected() -> None:
    """Should create rejected interaction with reason."""
    interaction = UserInteraction(
        tool_call_id="test-id",
        approved=False,
        reason="Not allowed",
    )
    assert interaction.approved is False
    assert interaction.reason == "Not allowed"


# --- UserInputPreprocessResult tests ---
def test_user_input_preprocess_result_with_override_args() -> None:
    """Should store override args."""
    result = UserInputPreprocessResult(
        override_args={"path": "/new/path"},
        metadata={"source": "user"},
    )
    assert result.override_args == {"path": "/new/path"}
    assert result.metadata == {"source": "user"}


def test_user_input_preprocess_result_empty() -> None:
    """Should handle empty result."""
    result = UserInputPreprocessResult()
    assert result.override_args is None
    assert result.metadata is None


# --- GlobalHooks tests ---
def test_global_hooks_empty() -> None:
    """Should create with no hooks."""
    hooks = GlobalHooks()
    assert hooks.pre is None
    assert hooks.post is None


def test_global_hooks_with_hooks() -> None:
    """Should accept hook functions."""

    async def pre_hook(ctx: Any, name: str, args: dict, metadata: dict) -> dict:
        return args

    async def post_hook(ctx: Any, name: str, result: Any, metadata: dict) -> Any:
        return result

    hooks = GlobalHooks(pre=pre_hook, post=post_hook)
    assert hooks.pre is pre_hook
    assert hooks.post is post_hook


# --- Test tool classes ---
class DummyTool(BaseTool):
    """A simple test tool."""

    name = "dummy_tool"
    description = "A dummy tool for testing"

    def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        return "Use this dummy tool for testing purposes."

    async def call(self, ctx: RunContext[AgentContext], message: str = "hello") -> str:
        return f"Dummy: {message}"


class UnavailableTool(BaseTool):
    """A tool that is not available."""

    name = "unavailable_tool"
    description = "An unavailable tool"

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return False

    async def call(self, ctx: RunContext[AgentContext]) -> str:
        return "Should not be called"


# --- BaseTool tests ---
def test_base_tool_default_availability(agent_context: AgentContext) -> None:
    """Should be available by default."""
    from unittest.mock import MagicMock

    from pydantic_ai import RunContext

    tool = DummyTool()
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context
    assert tool.is_available(mock_run_ctx) is True


def test_base_tool_unavailable(agent_context: AgentContext) -> None:
    """Should report unavailability correctly."""
    from unittest.mock import MagicMock

    from pydantic_ai import RunContext

    tool = UnavailableTool()
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context
    assert tool.is_available(mock_run_ctx) is False


def test_base_tool_initialization(agent_context: AgentContext) -> None:
    """Should initialize without context."""
    tool = DummyTool()
    assert tool.name == "dummy_tool"
    assert tool.description == "A dummy tool for testing"


async def test_base_tool_process_user_input_returns_none(agent_context: AgentContext) -> None:
    """Should return None by default."""
    tool = DummyTool()
    result = await tool.process_user_input(agent_context, {"input": "data"})
    assert result is None


# --- BaseToolset tests ---
def test_base_toolset_get_instructions_returns_none() -> None:
    """Should return None by default."""

    class SimpleToolset(BaseToolset):
        @property
        def id(self) -> str | None:
            return None

        async def get_tools(self, ctx: RunContext) -> dict:
            return {}

        async def call_tool(self, name: str, tool_args: dict, ctx: RunContext, tool: Any) -> Any:
            pass

    toolset = SimpleToolset()
    mock_ctx = MagicMock(spec=RunContext)
    assert toolset.get_instructions(mock_ctx) is None


# --- Toolset tests ---
def test_toolset_initialization(agent_context: AgentContext) -> None:
    """Should initialize with tools."""
    toolset = Toolset(tools=[DummyTool])
    assert len(toolset._tool_classes) == 1
    assert "dummy_tool" in toolset._tool_classes


async def test_toolset_skip_unavailable_tools(agent_context: AgentContext) -> None:
    """Should skip unavailable tools when skip_unavailable=True in get_tools()."""
    from unittest.mock import MagicMock

    from pydantic_ai import RunContext

    toolset = Toolset(tools=[DummyTool, UnavailableTool], skip_unavailable=True)
    # All tools are registered in _tool_classes
    assert "dummy_tool" in toolset._tool_classes
    assert "unavailable_tool" in toolset._tool_classes
    # But unavailable tools are filtered out in get_tools()
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context
    tools = await toolset.get_tools(mock_run_ctx)
    assert "dummy_tool" in tools
    assert "unavailable_tool" not in tools


def test_toolset_duplicate_tool_name_raises(agent_context: AgentContext) -> None:
    """Should raise on duplicate tool names."""
    from pydantic_ai import UserError

    with pytest.raises(UserError, match="Duplicate tool name"):
        Toolset(tools=[DummyTool, DummyTool])


def test_toolset_id(agent_context: AgentContext) -> None:
    """Should store and return toolset ID."""
    toolset = Toolset(tools=[DummyTool], toolset_id="my-toolset")
    assert toolset.id == "my-toolset"


async def test_toolset_get_instructions(agent_context: AgentContext) -> None:
    """Should collect instructions from tools."""
    toolset = Toolset(tools=[DummyTool])
    mock_run_ctx = MagicMock(spec=RunContext)
    instructions = await toolset.get_instructions(mock_run_ctx)
    assert instructions is not None
    assert "Use this dummy tool for testing purposes." in instructions


async def test_toolset_get_tools(agent_context: AgentContext) -> None:
    """Should return tool definitions."""
    toolset = Toolset(tools=[DummyTool])
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context
    tools = await toolset.get_tools(mock_run_ctx)
    assert "dummy_tool" in tools
    assert isinstance(tools["dummy_tool"], HookableToolsetTool)


async def test_toolset_call_tool_with_hooks(agent_context: AgentContext) -> None:
    """Should execute hooks in order."""
    call_order: list[str] = []

    async def global_pre(ctx: Any, name: str, args: dict, metadata: dict) -> dict:
        call_order.append("global_pre")
        return args

    async def global_post(ctx: Any, name: str, result: Any, metadata: dict) -> Any:
        call_order.append("global_post")
        return result

    async def tool_pre(ctx: Any, args: dict, metadata: dict) -> dict:
        call_order.append("tool_pre")
        return args

    async def tool_post(ctx: Any, result: Any, metadata: dict) -> Any:
        call_order.append("tool_post")
        return result

    toolset = Toolset(
        tools=[DummyTool],
        pre_hooks={"dummy_tool": tool_pre},
        post_hooks={"dummy_tool": tool_post},
        global_hooks=GlobalHooks(pre=global_pre, post=global_post),
    )
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context
    tools = await toolset.get_tools(mock_run_ctx)
    tool = tools["dummy_tool"]
    await toolset.call_tool("dummy_tool", {"message": "test"}, mock_run_ctx, tool)
    assert call_order == ["global_pre", "tool_pre", "tool_post", "global_post"]


async def test_toolset_call_tool_metadata_shared_across_hooks(agent_context: AgentContext) -> None:
    """Should share metadata dict across all hooks in a single call_tool invocation."""
    captured_metadata: list[dict] = []

    async def global_pre(ctx: Any, name: str, args: dict, metadata: dict) -> dict:
        metadata["global_pre_time"] = "t0"
        captured_metadata.append(dict(metadata))
        return args

    async def tool_pre(ctx: Any, args: dict, metadata: dict) -> dict:
        metadata["tool_pre_time"] = "t1"
        captured_metadata.append(dict(metadata))
        return args

    async def tool_post(ctx: Any, result: Any, metadata: dict) -> Any:
        metadata["tool_post_time"] = "t2"
        captured_metadata.append(dict(metadata))
        return result

    async def global_post(ctx: Any, name: str, result: Any, metadata: dict) -> Any:
        metadata["global_post_time"] = "t3"
        captured_metadata.append(dict(metadata))
        return result

    toolset = Toolset(
        tools=[DummyTool],
        pre_hooks={"dummy_tool": tool_pre},
        post_hooks={"dummy_tool": tool_post},
        global_hooks=GlobalHooks(pre=global_pre, post=global_post),
    )
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context
    tools = await toolset.get_tools(mock_run_ctx)
    tool = tools["dummy_tool"]
    await toolset.call_tool("dummy_tool", {"message": "test"}, mock_run_ctx, tool)
    # Verify metadata accumulates across hooks
    assert len(captured_metadata) == 4
    assert captured_metadata[0] == {"global_pre_time": "t0"}
    assert captured_metadata[1] == {"global_pre_time": "t0", "tool_pre_time": "t1"}
    assert captured_metadata[2] == {"global_pre_time": "t0", "tool_pre_time": "t1", "tool_post_time": "t2"}
    assert captured_metadata[3] == {
        "global_pre_time": "t0",
        "tool_pre_time": "t1",
        "tool_post_time": "t2",
        "global_post_time": "t3",
    }


async def test_toolset_process_hitl_call_approved(agent_context: AgentContext) -> None:
    """Should process approved HITL interactions."""
    toolset = Toolset(tools=[DummyTool])
    interactions = [
        UserInteraction(tool_call_id="call-1", approved=True),
    ]
    result = await toolset.process_hitl_call(agent_context, interactions, [])
    assert result is not None
    assert "call-1" in result.approvals
    assert isinstance(result.approvals["call-1"], ToolApproved)


async def test_toolset_process_hitl_call_rejected(agent_context: AgentContext) -> None:
    """Should process rejected HITL interactions."""
    toolset = Toolset(tools=[DummyTool])
    interactions = [
        UserInteraction(tool_call_id="call-1", approved=False, reason="Not safe"),
    ]
    result = await toolset.process_hitl_call(agent_context, interactions, [])
    assert result is not None
    assert "call-1" in result.approvals
    denied = result.approvals["call-1"]
    assert isinstance(denied, ToolDenied)
    assert denied.message == "Not safe"


async def test_toolset_process_hitl_call_none(agent_context: AgentContext) -> None:
    """Should return None when no interactions."""
    toolset = Toolset(tools=[DummyTool])
    result = await toolset.process_hitl_call(agent_context, None, [])
    assert result is None


async def test_toolset_process_hitl_with_user_input(agent_context: AgentContext) -> None:
    """Should process user input for approved interactions."""
    toolset = Toolset(tools=[DummyTool])
    tool_call = ToolCallPart(
        tool_name="dummy_tool",
        tool_call_id="call-1",
        args={},
    )
    message_history = [ModelResponse(parts=[tool_call])]
    interactions = [
        UserInteraction(
            tool_call_id="call-1",
            approved=True,
            user_input={"custom": "data"},
        ),
    ]
    result = await toolset.process_hitl_call(agent_context, interactions, message_history)
    assert result is not None
    assert isinstance(result.approvals["call-1"], ToolApproved)


# --- InstructableToolset protocol tests ---
def test_instructable_toolset_protocol_check(agent_context: AgentContext) -> None:
    """Should recognize conforming toolsets."""
    toolset = Toolset(tools=[DummyTool])
    assert isinstance(toolset, InstructableToolset)


# --- Toolset.subset tests ---
class AnotherTool(BaseTool):
    """Another test tool for subset tests."""

    name = "another_tool"
    description = "Another tool"

    async def call(self, ctx: RunContext[AgentContext]) -> str:
        return "another"


def test_toolset_tool_names(agent_context: AgentContext) -> None:
    """Should return list of tool names."""
    toolset = Toolset(tools=[DummyTool, AnotherTool])
    names = toolset.tool_names
    assert set(names) == {"dummy_tool", "another_tool"}


def test_toolset_subset_all_tools(agent_context: AgentContext) -> None:
    """Should return all tools when tool_names is None."""
    toolset = Toolset(tools=[DummyTool, AnotherTool])
    subset = toolset.subset(None)
    assert set(subset.tool_names) == {"dummy_tool", "another_tool"}


def test_toolset_subset_specific_tools(agent_context: AgentContext) -> None:
    """Should return only specified tools."""
    toolset = Toolset(tools=[DummyTool, AnotherTool])
    subset = toolset.subset(["dummy_tool"])
    assert subset.tool_names == ["dummy_tool"]
    assert "another_tool" not in subset.tool_names


def test_toolset_subset_inherit_hooks(agent_context: AgentContext) -> None:
    """Should inherit hooks when inherit_hooks=True."""

    async def pre_hook(ctx: Any, args: dict, metadata: dict) -> dict:
        return args

    async def post_hook(ctx: Any, result: Any, metadata: dict) -> Any:
        return result

    async def global_pre(ctx: Any, name: str, args: dict, metadata: dict) -> dict:
        return args

    toolset = Toolset(
        tools=[DummyTool, AnotherTool],
        pre_hooks={"dummy_tool": pre_hook},
        post_hooks={"dummy_tool": post_hook},
        global_hooks=GlobalHooks(pre=global_pre),
    )
    subset = toolset.subset(["dummy_tool"], inherit_hooks=True)
    assert "dummy_tool" in subset.pre_hooks
    assert "dummy_tool" in subset.post_hooks
    assert subset.global_hooks.pre is global_pre


def test_toolset_subset_no_inherit_hooks(agent_context: AgentContext) -> None:
    """Should not inherit hooks by default."""

    async def pre_hook(ctx: Any, args: dict, metadata: dict) -> dict:
        return args

    toolset = Toolset(
        tools=[DummyTool],
        pre_hooks={"dummy_tool": pre_hook},
    )
    subset = toolset.subset(["dummy_tool"], inherit_hooks=False)
    assert subset.pre_hooks == {}
    assert subset.post_hooks == {}
    assert subset.global_hooks.pre is None


def test_toolset_subset_nonexistent_tool_skipped(agent_context: AgentContext) -> None:
    """Should skip non-existent tools with warning."""
    toolset = Toolset(tools=[DummyTool])
    subset = toolset.subset(["dummy_tool", "nonexistent_tool"])
    assert subset.tool_names == ["dummy_tool"]


# --- Toolset.with_subagents tests ---
def test_toolset_with_subagents_empty_configs(agent_context: AgentContext) -> None:
    """Should return self when configs is empty."""
    toolset = Toolset(tools=[DummyTool])
    result = toolset.with_subagents([])
    assert result is toolset


def test_toolset_with_subagents_creates_new_toolset(agent_context: AgentContext) -> None:
    """Should create new toolset with subagent tools added."""
    from pai_agent_sdk.subagents import SubagentConfig

    toolset = Toolset(tools=[DummyTool])
    config = SubagentConfig(
        name="test_subagent",
        description="A test subagent",
        system_prompt="You are a test subagent.",
        tools=["dummy_tool"],
    )
    result = toolset.with_subagents([config])
    assert result is not toolset
    assert "dummy_tool" in result.tool_names
    assert "test_subagent" in result.tool_names


def test_toolset_with_subagents_preserves_hooks(agent_context: AgentContext) -> None:
    """Should preserve hooks in new toolset."""
    from pai_agent_sdk.subagents import SubagentConfig

    async def pre_hook(ctx: Any, args: dict, metadata: dict) -> dict:
        return args

    toolset = Toolset(
        tools=[DummyTool],
        pre_hooks={"dummy_tool": pre_hook},
        max_retries=5,
        timeout=30.0,
    )
    config = SubagentConfig(
        name="test_subagent",
        description="A test subagent",
        system_prompt="You are a test subagent.",
    )
    result = toolset.with_subagents([config])
    assert result.max_retries == 5
    assert result.timeout == 30.0
    assert "dummy_tool" in result.pre_hooks


def test_toolset_with_subagents_multiple_configs(agent_context: AgentContext) -> None:
    """Should handle multiple subagent configs."""
    from pai_agent_sdk.subagents import SubagentConfig

    toolset = Toolset(tools=[DummyTool])
    configs = [
        SubagentConfig(
            name="subagent_a",
            description="Subagent A",
            system_prompt="You are subagent A.",
        ),
        SubagentConfig(
            name="subagent_b",
            description="Subagent B",
            system_prompt="You are subagent B.",
        ),
    ]
    result = toolset.with_subagents(configs)
    assert "dummy_tool" in result.tool_names
    assert "subagent_a" in result.tool_names
    assert "subagent_b" in result.tool_names
