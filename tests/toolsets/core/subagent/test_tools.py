"""Tests for subagent management tools."""

from unittest.mock import MagicMock

import pytest
from pydantic_ai import RunContext
from pydantic_ai.messages import (
    ImageUrl,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from pai_agent_sdk.context import AgentContext, AgentInfo
from pai_agent_sdk.toolsets.core.subagent.tools import (
    SubagentInfoTool,
    _extract_first_user_prompt,
    _format_agent_info,
    _has_subagent_info,
)


@pytest.fixture
def mock_ctx() -> AgentContext:
    """Create a mock AgentContext for testing."""
    return AgentContext(run_id="main-123")


@pytest.fixture
def mock_run_context(mock_ctx: AgentContext) -> RunContext[AgentContext]:
    """Create a mock RunContext."""
    ctx = MagicMock(spec=RunContext)
    ctx.deps = mock_ctx
    return ctx


@pytest.fixture
def ctx_with_subagents(mock_ctx: AgentContext) -> AgentContext:
    """Create a context with subagent data."""
    mock_ctx.agent_registry = {
        "main-123": AgentInfo(agent_id="main-123", agent_name="main", parent_agent_id=None),
        "search-001": AgentInfo(agent_id="search-001", agent_name="search", parent_agent_id="main-123"),
        "reason-002": AgentInfo(agent_id="reason-002", agent_name="reasoning", parent_agent_id="main-123"),
    }
    mock_ctx.subagent_history = {
        "search-001": [
            ModelRequest(parts=[UserPromptPart(content="Find information about Python")]),
            ModelResponse(parts=[TextPart(content="Python is a programming language...")]),
        ],
        "reason-002": [
            ModelRequest(parts=[UserPromptPart(content="Analyze this code")]),
            ModelResponse(parts=[TextPart(content="The code looks good")]),
        ],
    }
    return mock_ctx


# Tests for _has_subagent_info


def test_has_subagent_info_returns_false_for_empty_ctx(mock_ctx: AgentContext):
    """Should return False when no subagent info exists."""
    assert _has_subagent_info(mock_ctx) is False


def test_has_subagent_info_returns_true_with_registry(mock_ctx: AgentContext):
    """Should return True when agent_registry has entries."""
    mock_ctx.agent_registry = {"sub-1": AgentInfo("sub-1", "search", "main")}
    assert _has_subagent_info(mock_ctx) is True


def test_has_subagent_info_returns_true_with_history(mock_ctx: AgentContext):
    """Should return True when subagent_history has entries."""
    mock_ctx.subagent_history = {"sub-1": []}
    assert _has_subagent_info(mock_ctx) is True


# Tests for _format_agent_info


def test_format_agent_info_formats_registered_agent(ctx_with_subagents: AgentContext):
    """Should format agent from registry correctly."""
    info = _format_agent_info("search-001", ctx_with_subagents)
    assert info["agent_id"] == "search-001"
    assert info["agent_name"] == "search"
    assert info["parent_agent_id"] == "main-123"
    assert info["has_history"] is True


def test_format_agent_info_formats_unregistered_agent_with_history(mock_ctx: AgentContext):
    """Should handle agent with history but not in registry."""
    mock_ctx.subagent_history = {"orphan-001": []}
    info = _format_agent_info("orphan-001", mock_ctx)
    assert info["agent_id"] == "orphan-001"
    assert info["agent_name"] == "unknown"
    assert info["parent_agent_id"] is None
    assert info["has_history"] is True


# Tests for _extract_first_user_prompt


def test_extract_first_user_prompt_extracts_from_first_request():
    """Should extract user prompt from first ModelRequest."""
    messages = [
        ModelRequest(parts=[UserPromptPart(content="Hello world")]),
        ModelResponse(parts=[TextPart(content="Hi there")]),
    ]
    assert _extract_first_user_prompt(messages) == "Hello world"


def test_extract_first_user_prompt_returns_none_for_empty_messages():
    """Should return None for empty message list."""
    assert _extract_first_user_prompt([]) is None


def test_extract_first_user_prompt_returns_none_for_no_user_prompt():
    """Should return None when no UserPromptPart exists."""
    messages = [
        ModelResponse(parts=[TextPart(content="Response only")]),
    ]
    assert _extract_first_user_prompt(messages) is None


def test_extract_first_user_prompt_skips_response_to_find_request():
    """Should skip ModelResponse to find first ModelRequest."""
    messages = [
        ModelResponse(parts=[TextPart(content="First response")]),
        ModelRequest(parts=[UserPromptPart(content="User question")]),
    ]
    assert _extract_first_user_prompt(messages) == "User question"


def test_extract_first_user_prompt_extracts_first_str_from_multipart():
    """Should extract first str part from multipart content."""
    messages = [
        ModelRequest(parts=[UserPromptPart(content=[ImageUrl(url="http://example.com/img.png"), "Text after image"])]),
    ]
    assert _extract_first_user_prompt(messages) == "Text after image"


def test_extract_first_user_prompt_returns_none_for_multipart_without_str():
    """Should return None when multipart has no str."""
    messages = [
        ModelRequest(parts=[UserPromptPart(content=[ImageUrl(url="http://example.com/img.png")])]),
    ]
    assert _extract_first_user_prompt(messages) is None


# Tests for SubagentInfoTool


def test_subagent_info_tool_is_not_available_without_subagents(
    mock_ctx: AgentContext, mock_run_context: RunContext[AgentContext]
):
    """Should not be available when no subagent info exists."""
    tool = SubagentInfoTool()
    assert tool.is_available(mock_run_context) is False


def test_subagent_info_tool_is_available_with_subagents(ctx_with_subagents: AgentContext):
    """Should be available when subagent info exists."""
    run_ctx = MagicMock(spec=RunContext)
    run_ctx.deps = ctx_with_subagents
    tool = SubagentInfoTool()
    assert tool.is_available(run_ctx) is True


async def test_subagent_info_tool_lists_subagents(ctx_with_subagents: AgentContext):
    """Should list all subagents."""
    run_ctx = MagicMock(spec=RunContext)
    run_ctx.deps = ctx_with_subagents
    tool = SubagentInfoTool()
    result = await tool.call(run_ctx)

    assert result["total_count"] == 2  # search-001 and reason-002 (excludes main)
    agent_ids = [s["agent_id"] for s in result["subagents"]]
    assert "search-001" in agent_ids
    assert "reason-002" in agent_ids
    assert "main-123" not in agent_ids  # Main agent excluded


async def test_subagent_info_tool_returns_empty_list_when_only_main(mock_ctx: AgentContext):
    """Should return empty list when only main agent exists."""
    mock_ctx.agent_registry = {"main-123": AgentInfo("main-123", "main", None)}
    run_ctx = MagicMock(spec=RunContext)
    run_ctx.deps = mock_ctx
    tool = SubagentInfoTool()
    result = await tool.call(run_ctx)

    assert result["subagents"] == []
    assert "No subagents found" in result["message"]


async def test_subagent_info_tool_includes_history_length(ctx_with_subagents: AgentContext):
    """Should include history length for subagents with history."""
    run_ctx = MagicMock(spec=RunContext)
    run_ctx.deps = ctx_with_subagents
    tool = SubagentInfoTool()
    result = await tool.call(run_ctx)

    # Find search subagent
    search_info = next(s for s in result["subagents"] if s["agent_id"] == "search-001")
    assert search_info["history_length"] == 2


async def test_subagent_info_tool_includes_hint(ctx_with_subagents: AgentContext):
    """Should include hint from first user prompt."""
    run_ctx = MagicMock(spec=RunContext)
    run_ctx.deps = ctx_with_subagents
    tool = SubagentInfoTool()
    result = await tool.call(run_ctx)

    # Find search subagent and check hint
    search_info = next(s for s in result["subagents"] if s["agent_id"] == "search-001")
    assert search_info["hint"] == "Find information about Python"

    # Find reasoning subagent and check hint
    reason_info = next(s for s in result["subagents"] if s["agent_id"] == "reason-002")
    assert reason_info["hint"] == "Analyze this code"


# Tests for tool metadata


def test_subagent_info_tool_has_correct_name(mock_ctx: AgentContext):
    """SubagentInfoTool should have correct name."""
    tool = SubagentInfoTool()
    assert tool.name == "subagent_info"


def test_subagent_info_tool_has_description(mock_ctx: AgentContext):
    """Tool should have meaningful description."""
    tool = SubagentInfoTool()
    assert len(tool.description) > 20
