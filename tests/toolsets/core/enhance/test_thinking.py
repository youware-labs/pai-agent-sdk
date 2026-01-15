"""Tests for pai_agent_sdk.toolsets.core.enhance.thinking module."""

from unittest.mock import MagicMock

from inline_snapshot import snapshot
from pydantic_ai import RunContext

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.toolsets.core.enhance.thinking import ThinkingTool


def test_thinking_tool_attributes(agent_context: AgentContext) -> None:
    """Should have correct name, description and instruction."""
    assert ThinkingTool.name == "thinking"
    assert ThinkingTool.description == snapshot(
        "Think about something without obtaining new information or making changes."
    )
    # Test get_instruction with a mock context
    tool = ThinkingTool()
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context
    assert tool.get_instruction(mock_run_ctx) == snapshot(
        """\
<thinking-guidelines>

<when-to-use>
Use `thinking` for complex reasoning or to cache intermediate thoughts. The tool appends thoughts to the log without obtaining new information or making changes.
</when-to-use>

<appropriate-scenarios>
- Complex multi-step reasoning that benefits from explicit thinking
- Caching intermediate analysis or observations for later reference
- Breaking down problems before taking action
</appropriate-scenarios>

<inappropriate-scenarios>
- Task planning and management (use `to_do` tools instead)
- Simple straightforward operations
</inappropriate-scenarios>

<language>
Use user's language when writing thoughts.
</language>

</thinking-guidelines>
"""
    )


def test_thinking_tool_initialization(agent_context: AgentContext) -> None:
    """Should initialize with context."""
    tool = ThinkingTool()
    assert tool.name == "thinking"


def test_thinking_tool_is_available(agent_context: AgentContext, mock_run_ctx) -> None:
    """Should be available by default."""
    tool = ThinkingTool()
    assert tool.is_available(mock_run_ctx) is True


async def test_thinking_tool_call_returns_thought(agent_context: AgentContext) -> None:
    """Should return the thought in a dictionary."""
    tool = ThinkingTool()

    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context

    result = await tool.call(mock_run_ctx, thought="This is a test thought")
    assert result == {"thought": "This is a test thought"}


async def test_thinking_tool_call_with_markdown(agent_context: AgentContext) -> None:
    """Should handle markdown formatted thoughts."""
    tool = ThinkingTool()

    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context

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


async def test_thinking_tool_call_with_empty_thought(agent_context: AgentContext) -> None:
    """Should handle empty thought string."""
    tool = ThinkingTool()

    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context

    result = await tool.call(mock_run_ctx, thought="")
    assert result == {"thought": ""}


async def test_thinking_tool_call_with_unicode(agent_context: AgentContext) -> None:
    """Should handle unicode characters."""
    tool = ThinkingTool()

    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context

    unicode_thought = "Thinking about: Hello World"
    result = await tool.call(mock_run_ctx, thought=unicode_thought)
    assert result == {"thought": unicode_thought}
