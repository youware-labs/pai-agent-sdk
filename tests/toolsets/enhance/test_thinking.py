"""Tests for pai_agent_sdk.toolsets.enhance.thinking module."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic_ai import RunContext

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.toolsets.enhance.thinking import ThinkingTool

# --- ThinkingTool tests ---


def test_thinking_tool_attributes() -> None:
    """Should have correct name and description."""
    assert ThinkingTool.name == "thinking"
    assert "think" in ThinkingTool.description.lower()
    assert ThinkingTool.instruction is None


def test_thinking_tool_initialization() -> None:
    """Should initialize with context."""
    ctx = AgentContext()
    tool = ThinkingTool(ctx)
    assert tool.ctx is ctx
    assert tool.name == "thinking"


def test_thinking_tool_is_available() -> None:
    """Should be available by default."""
    assert ThinkingTool.is_available() is True
    assert ThinkingTool.unavailable_reason() is None


@pytest.mark.asyncio
async def test_thinking_tool_call_returns_thought() -> None:
    """Should return the thought in a dictionary."""
    ctx = AgentContext(tmp_dir=Path("/tmp"))  # noqa: S108
    tool = ThinkingTool(ctx)

    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = ctx

    result = await tool.call(mock_run_ctx, thought="This is a test thought")
    assert result == {"thought": "This is a test thought"}


@pytest.mark.asyncio
async def test_thinking_tool_call_with_markdown() -> None:
    """Should handle markdown formatted thoughts."""
    ctx = AgentContext(tmp_dir=Path("/tmp"))  # noqa: S108
    tool = ThinkingTool(ctx)

    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = ctx

    markdown_thought = """
## Analysis
- Point 1
- Point 2
```python
code_example()
```
"""
    result = await tool.call(mock_run_ctx, thought=markdown_thought)
    assert result == {"thought": markdown_thought}


@pytest.mark.asyncio
async def test_thinking_tool_call_with_empty_thought() -> None:
    """Should handle empty thought string."""
    ctx = AgentContext(tmp_dir=Path("/tmp"))  # noqa: S108
    tool = ThinkingTool(ctx)

    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = ctx

    result = await tool.call(mock_run_ctx, thought="")
    assert result == {"thought": ""}


@pytest.mark.asyncio
async def test_thinking_tool_call_with_unicode() -> None:
    """Should handle unicode characters."""
    ctx = AgentContext(tmp_dir=Path("/tmp"))  # noqa: S108
    tool = ThinkingTool(ctx)

    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = ctx

    unicode_thought = "Thinking about: Hello World"
    result = await tool.call(mock_run_ctx, thought=unicode_thought)
    assert result == {"thought": unicode_thought}
