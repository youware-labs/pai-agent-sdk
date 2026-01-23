"""Tests for Instruction class and group-based deduplication."""

import pytest
from pydantic_ai import RunContext

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.toolsets import Instruction, Toolset
from pai_agent_sdk.toolsets.base import BaseTool, resolve_instruction


class ToolWithStringInstruction(BaseTool):
    """Tool returning plain string instruction."""

    name = "string_tool"
    description = "Test tool with string instruction"

    def get_instruction(self, ctx: RunContext[AgentContext]) -> str:
        return "String instruction content"

    async def call(self, ctx: RunContext[AgentContext]) -> str:
        return "ok"


class ToolWithGroupedInstruction(BaseTool):
    """Tool returning Instruction with group."""

    name = "grouped_tool"
    description = "Test tool with grouped instruction"

    def get_instruction(self, ctx: RunContext[AgentContext]) -> Instruction:
        return Instruction(group="my-group", content="Grouped instruction content")

    async def call(self, ctx: RunContext[AgentContext]) -> str:
        return "ok"


class ToolWithSameGroup(BaseTool):
    """Another tool with the same group (should be deduplicated)."""

    name = "same_group_tool"
    description = "Test tool with same group"

    def get_instruction(self, ctx: RunContext[AgentContext]) -> Instruction:
        return Instruction(group="my-group", content="This should be skipped")

    async def call(self, ctx: RunContext[AgentContext]) -> str:
        return "ok"


class ToolWithNoInstruction(BaseTool):
    """Tool with no instruction."""

    name = "no_instruction_tool"
    description = "Test tool without instruction"

    async def call(self, ctx: RunContext[AgentContext]) -> str:
        return "ok"


# Tests for Instruction model


def test_instruction_model():
    """Test Instruction model creation."""
    instr = Instruction(group="test-group", content="test content")
    assert instr.group == "test-group"
    assert instr.content == "test content"


# Tests for resolve_instruction


@pytest.mark.asyncio
async def test_resolve_instruction_none():
    """Test resolve_instruction with None."""
    result = await resolve_instruction(None)
    assert result is None


@pytest.mark.asyncio
async def test_resolve_instruction_string():
    """Test resolve_instruction with string."""
    result = await resolve_instruction("test string")
    assert result == "test string"


@pytest.mark.asyncio
async def test_resolve_instruction_instruction():
    """Test resolve_instruction with Instruction."""
    instr = Instruction(group="test", content="content")
    result = await resolve_instruction(instr)
    assert result == instr


@pytest.mark.asyncio
async def test_resolve_instruction_awaitable():
    """Test resolve_instruction with awaitable."""

    async def get_instruction():
        return "async result"

    result = await resolve_instruction(get_instruction())
    assert result == "async result"


# Tests for Toolset.get_instructions deduplication


@pytest.mark.asyncio
async def test_get_instructions_string_tool(mock_run_context: RunContext[AgentContext]):
    """Test get_instructions with tool returning string."""
    toolset = Toolset(tools=[ToolWithStringInstruction])
    instructions = await toolset.get_instructions(mock_run_context)

    assert instructions is not None
    assert 'name="string_tool"' in instructions
    assert "String instruction content" in instructions


@pytest.mark.asyncio
async def test_get_instructions_grouped_tool(mock_run_context: RunContext[AgentContext]):
    """Test get_instructions with tool returning Instruction."""
    toolset = Toolset(tools=[ToolWithGroupedInstruction])
    instructions = await toolset.get_instructions(mock_run_context)

    assert instructions is not None
    assert 'name="my-group"' in instructions
    assert "Grouped instruction content" in instructions


@pytest.mark.asyncio
async def test_get_instructions_deduplication(mock_run_context: RunContext[AgentContext]):
    """Test that tools with same group are deduplicated."""
    toolset = Toolset(tools=[ToolWithGroupedInstruction, ToolWithSameGroup])
    instructions = await toolset.get_instructions(mock_run_context)

    assert instructions is not None
    # First tool's content should be present
    assert "Grouped instruction content" in instructions
    # Second tool's content should be skipped
    assert "This should be skipped" not in instructions
    # Only one instruction block for the group
    assert instructions.count('name="my-group"') == 1


@pytest.mark.asyncio
async def test_get_instructions_mixed_tools(mock_run_context: RunContext[AgentContext]):
    """Test get_instructions with mixed tool types."""
    toolset = Toolset(
        tools=[
            ToolWithStringInstruction,
            ToolWithGroupedInstruction,
            ToolWithSameGroup,
            ToolWithNoInstruction,
        ]
    )
    instructions = await toolset.get_instructions(mock_run_context)

    assert instructions is not None
    # String tool uses tool name as group
    assert 'name="string_tool"' in instructions
    # Grouped tools share one instruction
    assert 'name="my-group"' in instructions
    assert instructions.count('name="my-group"') == 1
    # No instruction tool contributes nothing
    assert 'name="no_instruction_tool"' not in instructions


@pytest.mark.asyncio
async def test_get_instructions_empty(mock_run_context: RunContext[AgentContext]):
    """Test get_instructions with only no-instruction tools."""
    toolset = Toolset(tools=[ToolWithNoInstruction])
    instructions = await toolset.get_instructions(mock_run_context)

    assert instructions is None
