"""Tests for pai_agent_sdk.mcp module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from pai_agent_sdk.mcp import MCPServerSpec, create_mcp_approval_hook

# =============================================================================
# MCPServerSpec Tests
# =============================================================================


def test_mcp_server_spec_defaults() -> None:
    """Test MCPServerSpec with default values."""
    spec = MCPServerSpec()
    assert spec.transport == "stdio"
    assert spec.command is None
    assert spec.args == []
    assert spec.env == {}
    assert spec.url is None
    assert spec.headers == {}


def test_mcp_server_spec_stdio() -> None:
    """Test MCPServerSpec for stdio transport."""
    spec = MCPServerSpec(
        transport="stdio",
        command="uvx",
        args=["mcp-server-filesystem"],
        env={"HOME": "/home/user"},
    )
    assert spec.transport == "stdio"
    assert spec.command == "uvx"
    assert spec.args == ["mcp-server-filesystem"]
    assert spec.env == {"HOME": "/home/user"}


def test_mcp_server_spec_streamable_http() -> None:
    """Test MCPServerSpec for streamable_http transport."""
    spec = MCPServerSpec(
        transport="streamable_http",
        url="http://localhost:8000/mcp",
        headers={"Authorization": "Bearer token"},
    )
    assert spec.transport == "streamable_http"
    assert spec.url == "http://localhost:8000/mcp"
    assert spec.headers == {"Authorization": "Bearer token"}


# =============================================================================
# create_mcp_approval_hook Tests
# =============================================================================


@pytest.fixture
def mock_context() -> MagicMock:
    """Create a mock RunContext."""
    ctx = MagicMock()
    ctx.deps = MagicMock()
    ctx.deps.need_user_approve_mcps = []
    ctx.tool_call_approved = False
    return ctx


@pytest.fixture
def mock_call_tool() -> AsyncMock:
    """Create a mock call_tool function."""
    return AsyncMock(return_value="tool result")


@pytest.mark.asyncio
async def test_hook_no_approval_needed(mock_context: MagicMock, mock_call_tool: AsyncMock) -> None:
    """Test hook when server is not in approval list."""
    hook = create_mcp_approval_hook("filesystem")
    mock_context.deps.need_user_approve_mcps = []

    result = await hook(mock_context, mock_call_tool, "read_file", {"path": "/home/user/test.txt"})

    assert result == "tool result"
    mock_call_tool.assert_called_once_with("read_file", {"path": "/home/user/test.txt"}, None)


@pytest.mark.asyncio
async def test_hook_approval_required_raises(mock_context: MagicMock, mock_call_tool: AsyncMock) -> None:
    """Test hook raises ApprovalRequired when server needs approval."""
    from pydantic_ai import ApprovalRequired

    hook = create_mcp_approval_hook("filesystem")
    mock_context.deps.need_user_approve_mcps = ["filesystem"]
    mock_context.tool_call_approved = False

    with pytest.raises(ApprovalRequired) as exc_info:
        await hook(mock_context, mock_call_tool, "write_file", {"path": "/home/user/test.txt"})

    assert exc_info.value.metadata["mcp_server"] == "filesystem"
    assert exc_info.value.metadata["mcp_tool"] == "write_file"
    assert exc_info.value.metadata["full_name"] == "filesystem_write_file"
    mock_call_tool.assert_not_called()


@pytest.mark.asyncio
async def test_hook_already_approved(mock_context: MagicMock, mock_call_tool: AsyncMock) -> None:
    """Test hook proceeds when tool_call_approved is True."""
    hook = create_mcp_approval_hook("filesystem")
    mock_context.deps.need_user_approve_mcps = ["filesystem"]
    mock_context.tool_call_approved = True

    result = await hook(mock_context, mock_call_tool, "write_file", {"path": "/home/user/test.txt"})

    assert result == "tool result"
    mock_call_tool.assert_called_once_with("write_file", {"path": "/home/user/test.txt"}, None)


@pytest.mark.asyncio
async def test_hook_different_server_not_affected(mock_context: MagicMock, mock_call_tool: AsyncMock) -> None:
    """Test hook for server not in approval list proceeds normally."""
    hook = create_mcp_approval_hook("github")
    mock_context.deps.need_user_approve_mcps = ["filesystem"]  # Only filesystem needs approval
    mock_context.tool_call_approved = False

    result = await hook(mock_context, mock_call_tool, "create_issue", {"title": "Test"})

    assert result == "tool result"
    mock_call_tool.assert_called_once()
