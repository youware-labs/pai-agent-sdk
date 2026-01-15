"""Tests for pai_agent_sdk.toolsets.core.context.handoff module."""

from unittest.mock import MagicMock

from pydantic_ai import RunContext

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.toolsets.core.context.handoff import HandoffMessage, HandoffTool


def test_handoff_message_render_minimal() -> None:
    """Should render minimal handoff message with only required fields."""
    msg = HandoffMessage(
        primary_request="Build a REST API",
        current_state="Initial setup complete",
    )
    result = msg.render()
    assert "<context-handoff>" in result
    assert "<primary-request>Build a REST API</primary-request>" in result
    assert "<current-state>Initial setup complete</current-state>" in result
    assert "<key-decisions>" not in result
    assert "<files-modified>" not in result
    assert "<pending-tasks>" not in result
    assert "<next-step>" not in result


def test_handoff_message_render_full() -> None:
    """Should render full handoff message with all fields."""
    msg = HandoffMessage(
        primary_request="Build a REST API",
        current_state="Database models created",
        key_decisions=["Use PostgreSQL", "Use FastAPI"],
        files_modified=["models.py", "api.py"],
        pending_tasks=["Add authentication", "Write tests"],
        next_step="Implement user endpoint",
    )
    result = msg.render()
    assert "<context-handoff>" in result
    assert "<primary-request>Build a REST API</primary-request>" in result
    assert "<current-state>Database models created</current-state>" in result
    assert "<key-decisions>" in result
    assert "<decision>Use PostgreSQL</decision>" in result
    assert "<decision>Use FastAPI</decision>" in result
    assert "<files-modified>" in result
    assert "<file>models.py</file>" in result
    assert "<file>api.py</file>" in result
    assert "<pending-tasks>" in result
    assert "<task>Add authentication</task>" in result
    assert "<task>Write tests</task>" in result
    assert "<next-step>Implement user endpoint</next-step>" in result


def test_handoff_message_render_partial() -> None:
    """Should render partial handoff message with some optional fields."""
    msg = HandoffMessage(
        primary_request="Fix bug",
        current_state="Issue identified",
        key_decisions=["Root cause found"],
    )
    result = msg.render()
    assert "<key-decisions>" in result
    assert "<decision>Root cause found</decision>" in result
    assert "<files-modified>" not in result
    assert "<pending-tasks>" not in result
    assert "<next-step>" not in result


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
        primary_request="Build API",
        current_state="Done",
        key_decisions=["Use REST"],
        files_modified=["main.py"],
    )

    result = await tool.call(mock_run_ctx, message=msg)

    # Verify handoff message is stored in context
    assert agent_context.handoff_message is not None
    assert "<context-handoff>" in agent_context.handoff_message
    assert "<primary-request>Build API</primary-request>" in agent_context.handoff_message

    # Verify return value contains summary
    assert "Handoff complete" in result
    assert "<context-handoff>" in result


async def test_handoff_tool_call_overwrites_previous(agent_context: AgentContext) -> None:
    """Should overwrite previous handoff message."""
    tool = HandoffTool()

    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context

    # First handoff
    msg1 = HandoffMessage(
        primary_request="First request",
        current_state="First state",
    )
    await tool.call(mock_run_ctx, message=msg1)
    assert "First request" in agent_context.handoff_message

    # Second handoff overwrites
    msg2 = HandoffMessage(
        primary_request="Second request",
        current_state="Second state",
    )
    await tool.call(mock_run_ctx, message=msg2)
    assert "Second request" in agent_context.handoff_message
    assert "First request" not in agent_context.handoff_message
