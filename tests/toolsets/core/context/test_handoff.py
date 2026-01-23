"""Tests for pai_agent_sdk.toolsets.core.context.handoff module."""

from unittest.mock import MagicMock

from pydantic_ai import RunContext

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.toolsets.core.context.handoff import HandoffMessage, HandoffTool


def test_handoff_message_render() -> None:
    """Should render handoff message with content."""
    msg = HandoffMessage(
        content="## User Intent\nBuild a REST API\n\n## Current State\nInitial setup complete",
    )
    result = msg.render()
    assert "# Context Handoff" in result
    assert "Build a REST API" in result
    assert "Initial setup complete" in result


def test_handoff_message_render_with_auto_load_files() -> None:
    """Should render handoff message, auto_load_files is not included in render."""
    msg = HandoffMessage(
        content="Working on API implementation",
        auto_load_files=["main.py", "config.py"],
    )
    result = msg.render()
    assert "# Context Handoff" in result
    assert "Working on API implementation" in result
    # auto_load_files should not be in the rendered output
    # They are stored separately for the filter to process
    assert msg.auto_load_files == ["main.py", "config.py"]


def test_handoff_tool_attributes(agent_context: AgentContext) -> None:
    """Should have correct name and description."""
    assert HandoffTool.name == "handoff"
    assert "Summarize current work" in HandoffTool.description
    assert "clear context" in HandoffTool.description


def test_handoff_tool_initialization(agent_context: AgentContext) -> None:
    """Should initialize with context."""
    tool = HandoffTool()
    assert tool.name == "handoff"


def test_handoff_tool_is_available(agent_context: AgentContext, mock_run_ctx) -> None:
    """Should be available by default."""
    tool = HandoffTool()
    assert tool.is_available(mock_run_ctx) is True


def test_handoff_tool_get_instruction(agent_context: AgentContext) -> None:
    """Should load instruction from prompts/handoff.md."""
    tool = HandoffTool()
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context
    instruction = tool.get_instruction(mock_run_ctx)
    # Verify instruction is loaded and contains expected content
    assert instruction is not None
    assert len(instruction) > 0
    assert "handoff" in instruction.lower()


async def test_handoff_tool_call(agent_context: AgentContext) -> None:
    """Should store handoff message and return summary."""
    tool = HandoffTool()

    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context

    msg = HandoffMessage(
        content="## User Intent\nBuild API\n\n## Key Decisions\n- Use REST",
    )

    result = await tool.call(mock_run_ctx, message=msg)

    # Verify handoff message is stored in context
    assert agent_context.handoff_message is not None
    assert "# Context Handoff" in agent_context.handoff_message
    assert "Build API" in agent_context.handoff_message

    # Verify return value contains summary
    assert "Handoff complete" in result
    assert "# Context Handoff" in result


async def test_handoff_tool_call_sets_auto_load_files(agent_context: AgentContext) -> None:
    """Should set auto_load_files on context."""
    tool = HandoffTool()

    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context

    msg = HandoffMessage(
        content="Working on implementation",
        auto_load_files=["main.py", "utils.py"],
    )

    await tool.call(mock_run_ctx, message=msg)

    # Verify auto_load_files is set on context
    assert agent_context.auto_load_files == ["main.py", "utils.py"]


async def test_handoff_tool_call_overwrites_previous(agent_context: AgentContext) -> None:
    """Should overwrite previous handoff message."""
    tool = HandoffTool()

    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context

    # First handoff
    msg1 = HandoffMessage(
        content="First request content",
    )
    await tool.call(mock_run_ctx, message=msg1)
    assert "First request content" in agent_context.handoff_message

    # Second handoff overwrites
    msg2 = HandoffMessage(
        content="Second request content",
    )
    await tool.call(mock_run_ctx, message=msg2)
    assert "Second request content" in agent_context.handoff_message
    assert "First request content" not in agent_context.handoff_message
