"""Tests for pai_agent_sdk.toolsets.base module."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic_ai import RunContext
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.tools import ToolApproved, ToolDenied

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.toolsets.core.base import (
    BaseTool,
    BaseToolset,
    GlobalHooks,
    HookableToolsetTool,
    InstructableToolset,
    Toolset,
    UserInputPreprocessResult,
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

    async def pre_hook(ctx: Any, name: str, args: dict) -> dict:
        return args

    async def post_hook(ctx: Any, name: str, result: Any) -> Any:
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

    def is_available(self) -> bool:
        return False

    async def call(self, ctx: RunContext[AgentContext]) -> str:
        return "Should not be called"


# --- BaseTool tests ---


def test_base_tool_default_availability(agent_context: AgentContext) -> None:
    """Should be available by default."""
    tool = DummyTool(agent_context)
    assert tool.is_available() is True


def test_base_tool_unavailable(agent_context: AgentContext) -> None:
    """Should report unavailability correctly."""
    tool = UnavailableTool(agent_context)
    assert tool.is_available() is False


def test_base_tool_initialization(agent_context: AgentContext) -> None:
    """Should initialize with context."""
    ctx = agent_context
    tool = DummyTool(ctx)
    assert tool.ctx is ctx
    assert tool.name == "dummy_tool"
    assert tool.description == "A dummy tool for testing"


async def test_base_tool_process_user_input_returns_none(agent_context: AgentContext) -> None:
    """Should return None by default."""
    ctx = agent_context
    tool = DummyTool(ctx)
    result = await tool.process_user_input(ctx, {"input": "data"})
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
    ctx = agent_context
    toolset = Toolset(ctx, tools=[DummyTool])
    assert len(toolset._tool_instances) == 1
    assert "dummy_tool" in toolset._tool_instances


def test_toolset_skip_unavailable_tools(agent_context: AgentContext) -> None:
    """Should skip unavailable tools when skip_unavailable=True."""
    ctx = agent_context
    toolset = Toolset(ctx, tools=[DummyTool, UnavailableTool], skip_unavailable=True)
    assert "dummy_tool" in toolset._tool_instances
    assert "unavailable_tool" not in toolset._tool_instances


def test_toolset_duplicate_tool_name_raises(agent_context: AgentContext) -> None:
    """Should raise on duplicate tool names."""
    from pydantic_ai import UserError

    ctx = agent_context
    with pytest.raises(UserError, match="Duplicate tool name"):
        Toolset(ctx, tools=[DummyTool, DummyTool])


def test_toolset_id(agent_context: AgentContext) -> None:
    """Should store and return toolset ID."""
    ctx = agent_context
    toolset = Toolset(ctx, tools=[DummyTool], toolset_id="my-toolset")
    assert toolset.id == "my-toolset"


def test_toolset_get_instructions(agent_context: AgentContext) -> None:
    """Should collect instructions from tools."""
    ctx = agent_context
    toolset = Toolset(ctx, tools=[DummyTool])
    mock_run_ctx = MagicMock(spec=RunContext)
    instructions = toolset.get_instructions(mock_run_ctx)
    assert instructions is not None
    assert "Use this dummy tool for testing purposes." in instructions


async def test_toolset_get_tools(agent_context: AgentContext) -> None:
    """Should return tool definitions."""
    ctx = agent_context
    toolset = Toolset(ctx, tools=[DummyTool])
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = ctx
    tools = await toolset.get_tools(mock_run_ctx)
    assert "dummy_tool" in tools
    assert isinstance(tools["dummy_tool"], HookableToolsetTool)


async def test_toolset_call_tool_with_hooks(agent_context: AgentContext) -> None:
    """Should execute hooks in order."""
    ctx = agent_context
    call_order: list[str] = []

    async def global_pre(ctx: Any, name: str, args: dict) -> dict:
        call_order.append("global_pre")
        return args

    async def global_post(ctx: Any, name: str, result: Any) -> Any:
        call_order.append("global_post")
        return result

    async def tool_pre(ctx: Any, args: dict) -> dict:
        call_order.append("tool_pre")
        return args

    async def tool_post(ctx: Any, result: Any) -> Any:
        call_order.append("tool_post")
        return result

    toolset = Toolset(
        ctx,
        tools=[DummyTool],
        pre_hooks={"dummy_tool": tool_pre},
        post_hooks={"dummy_tool": tool_post},
        global_hooks=GlobalHooks(pre=global_pre, post=global_post),
    )

    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = ctx
    tools = await toolset.get_tools(mock_run_ctx)
    tool = tools["dummy_tool"]

    await toolset.call_tool("dummy_tool", {"message": "test"}, mock_run_ctx, tool)
    assert call_order == ["global_pre", "tool_pre", "tool_post", "global_post"]


async def test_toolset_process_hitl_call_approved(agent_context: AgentContext) -> None:
    """Should process approved HITL interactions."""
    ctx = agent_context
    toolset = Toolset(ctx, tools=[DummyTool])

    interactions = [
        UserInteraction(tool_call_id="call-1", approved=True),
    ]

    result = await toolset.process_hitl_call(interactions, [])
    assert result is not None
    assert "call-1" in result.approvals
    assert isinstance(result.approvals["call-1"], ToolApproved)


async def test_toolset_process_hitl_call_rejected(agent_context: AgentContext) -> None:
    """Should process rejected HITL interactions."""
    ctx = agent_context
    toolset = Toolset(ctx, tools=[DummyTool])

    interactions = [
        UserInteraction(tool_call_id="call-1", approved=False, reason="Not safe"),
    ]

    result = await toolset.process_hitl_call(interactions, [])
    assert result is not None
    assert "call-1" in result.approvals
    denied = result.approvals["call-1"]
    assert isinstance(denied, ToolDenied)
    assert denied.message == "Not safe"


async def test_toolset_process_hitl_call_none(agent_context: AgentContext) -> None:
    """Should return None when no interactions."""
    ctx = agent_context
    toolset = Toolset(ctx, tools=[DummyTool])

    result = await toolset.process_hitl_call(None, [])
    assert result is None


async def test_toolset_process_hitl_with_user_input(agent_context: AgentContext) -> None:
    """Should process user input for approved interactions."""
    ctx = agent_context
    toolset = Toolset(ctx, tools=[DummyTool])

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

    result = await toolset.process_hitl_call(interactions, message_history)
    assert result is not None
    assert isinstance(result.approvals["call-1"], ToolApproved)


# --- InstructableToolset protocol tests ---


def test_instructable_toolset_protocol_check(agent_context: AgentContext) -> None:
    """Should recognize conforming toolsets."""
    ctx = agent_context
    toolset = Toolset(ctx, tools=[DummyTool])
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
    ctx = agent_context
    toolset = Toolset(ctx, tools=[DummyTool, AnotherTool])
    names = toolset.tool_names
    assert set(names) == {"dummy_tool", "another_tool"}


def test_toolset_subset_all_tools(agent_context: AgentContext) -> None:
    """Should return all tools when tool_names is None."""
    ctx = agent_context
    toolset = Toolset(ctx, tools=[DummyTool, AnotherTool])
    subset = toolset.subset(None)

    assert set(subset.tool_names) == {"dummy_tool", "another_tool"}


def test_toolset_subset_specific_tools(agent_context: AgentContext) -> None:
    """Should return only specified tools."""
    ctx = agent_context
    toolset = Toolset(ctx, tools=[DummyTool, AnotherTool])
    subset = toolset.subset(["dummy_tool"])

    assert subset.tool_names == ["dummy_tool"]
    assert "another_tool" not in subset.tool_names


def test_toolset_subset_with_new_context(agent_context: AgentContext) -> None:
    """Should use new context when provided."""
    ctx1 = agent_context

    # Create another context
    from pai_agent_sdk.environment.local import LocalFileOperator, LocalShell

    ctx2 = AgentContext(
        file_operator=LocalFileOperator(),
        shell=LocalShell(),
    )

    toolset = Toolset(ctx1, tools=[DummyTool])
    subset = toolset.subset(None, ctx=ctx2)

    assert subset.ctx is ctx2
    assert subset.ctx is not ctx1


def test_toolset_subset_inherit_hooks(agent_context: AgentContext) -> None:
    """Should inherit hooks when inherit_hooks=True."""

    async def pre_hook(ctx: Any, args: dict) -> dict:
        return args

    async def post_hook(ctx: Any, result: Any) -> Any:
        return result

    async def global_pre(ctx: Any, name: str, args: dict) -> dict:
        return args

    ctx = agent_context
    toolset = Toolset(
        ctx,
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

    async def pre_hook(ctx: Any, args: dict) -> dict:
        return args

    ctx = agent_context
    toolset = Toolset(
        ctx,
        tools=[DummyTool],
        pre_hooks={"dummy_tool": pre_hook},
    )

    subset = toolset.subset(["dummy_tool"], inherit_hooks=False)

    assert subset.pre_hooks == {}
    assert subset.post_hooks == {}
    assert subset.global_hooks.pre is None


def test_toolset_subset_nonexistent_tool_skipped(agent_context: AgentContext) -> None:
    """Should skip non-existent tools with warning."""
    ctx = agent_context
    toolset = Toolset(ctx, tools=[DummyTool])
    subset = toolset.subset(["dummy_tool", "nonexistent_tool"])

    assert subset.tool_names == ["dummy_tool"]
