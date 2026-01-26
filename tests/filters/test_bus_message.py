"""Tests for bus message injection filter."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai import RunContext
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from pai_agent_sdk.context import MessageBus
from pai_agent_sdk.filters.bus_message import inject_bus_messages


def create_mock_ctx(agent_id: str = "main", message_bus: MessageBus | None = None) -> RunContext:
    """Create a mock RunContext with mocked deps."""
    bus = message_bus or MessageBus()

    mock_deps = MagicMock()
    mock_deps.agent_id = agent_id
    mock_deps.message_bus = bus
    mock_deps.emit_event = AsyncMock()

    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = mock_deps
    return mock_run_ctx


@pytest.mark.asyncio
async def test_inject_bus_messages_no_pending() -> None:
    """Test filter with no pending messages."""
    bus = MessageBus()
    bus.subscribe("main")
    ctx = create_mock_ctx(message_bus=bus)
    messages = [ModelRequest(parts=[UserPromptPart(content="Hello")])]

    result = await inject_bus_messages(ctx, messages)

    assert result == messages  # Unchanged
    assert len(result[0].parts) == 1  # No parts added


@pytest.mark.asyncio
async def test_inject_bus_messages_with_pending() -> None:
    """Test filter injects pending messages into last request parts."""
    bus = MessageBus()
    bus.subscribe("main")
    bus.send("Please stop", source="user", target="main")
    ctx = create_mock_ctx(message_bus=bus)
    messages = [ModelRequest(parts=[UserPromptPart(content="Hello")])]

    result = await inject_bus_messages(ctx, messages)

    assert len(result) == 1  # Same message count
    assert len(result[0].parts) == 2  # Part added
    assert isinstance(result[0].parts[1], UserPromptPart)
    assert '<bus-message source="user">' in result[0].parts[1].content
    # No template, so raw content is used
    assert "Please stop" in result[0].parts[1].content


@pytest.mark.asyncio
async def test_inject_bus_messages_custom_template() -> None:
    """Test filter uses custom template from BusMessage."""
    bus = MessageBus()
    bus.subscribe("main")
    bus.send("Hello", source="system", target="main", template="[SYSTEM] {{ content }}")
    ctx = create_mock_ctx(message_bus=bus)
    messages = [ModelRequest(parts=[UserPromptPart(content="Hi")])]

    result = await inject_bus_messages(ctx, messages)

    assert '<bus-message source="system">' in result[0].parts[1].content
    assert "[SYSTEM] Hello" in result[0].parts[1].content


@pytest.mark.asyncio
async def test_inject_bus_messages_multiple() -> None:
    """Test filter injects multiple pending messages as separate parts."""
    bus = MessageBus()
    bus.subscribe("main")
    bus.send("First message", source="user", target="main")
    bus.send("Second message", source="system", target="main", template="From system: {{ content }}")
    ctx = create_mock_ctx(message_bus=bus)
    messages = [ModelRequest(parts=[UserPromptPart(content="Hello")])]

    result = await inject_bus_messages(ctx, messages)

    assert len(result) == 1
    assert len(result[0].parts) == 3  # Original + 2 injected
    assert '<bus-message source="user">' in result[0].parts[1].content
    assert "First message" in result[0].parts[1].content
    assert '<bus-message source="system">' in result[0].parts[2].content
    assert "From system: Second message" in result[0].parts[2].content


@pytest.mark.asyncio
async def test_inject_bus_messages_consumes() -> None:
    """Test filter advances cursor after injection (messages remain in queue)."""
    bus = MessageBus()
    bus.subscribe("main")
    bus.send("Hello", source="user", target="main")
    ctx = create_mock_ctx(message_bus=bus)
    messages = [ModelRequest(parts=[UserPromptPart(content="Hello")])]

    assert len(bus) == 1
    await inject_bus_messages(ctx, messages)
    # Message still in queue (not deleted), but cursor advanced
    assert len(bus) == 1
    # No more pending for main (cursor advanced)
    assert not bus.has_pending("main")


@pytest.mark.asyncio
async def test_inject_bus_messages_emits_event() -> None:
    """Test filter emits single event with all messages."""
    bus = MessageBus()
    bus.subscribe("main")
    bus.send("First", source="user", target="main")
    bus.send("Second", source="system", target="main", template="[SYSTEM] {{ content }}")
    ctx = create_mock_ctx(message_bus=bus)
    messages = [ModelRequest(parts=[UserPromptPart(content="Hello")])]

    await inject_bus_messages(ctx, messages)

    ctx.deps.emit_event.assert_called_once()
    event = ctx.deps.emit_event.call_args[0][0]
    assert len(event.messages) == 2
    # First message - no template, content equals rendered_content
    assert event.messages[0].source == "user"
    assert event.messages[0].content == "First"
    assert event.messages[0].rendered_content == "First"
    # Second message - with template, rendered_content has template applied
    assert event.messages[1].source == "system"
    assert event.messages[1].content == "Second"
    assert event.messages[1].rendered_content == "[SYSTEM] Second"


@pytest.mark.asyncio
async def test_inject_bus_messages_respects_target() -> None:
    """Test filter only consumes messages for current agent."""
    bus = MessageBus()
    bus.subscribe("main")
    bus.subscribe("other")
    bus.send("For main", source="user", target="main")
    bus.send("For other", source="user", target="other")
    ctx = create_mock_ctx(agent_id="main", message_bus=bus)
    messages = [ModelRequest(parts=[UserPromptPart(content="Hello")])]

    result = await inject_bus_messages(ctx, messages)

    # Only "For main" should be injected
    assert len(result[0].parts) == 2
    assert "For main" in result[0].parts[1].content
    # "For other" still pending for other agent
    assert bus.has_pending("other")
    assert not bus.has_pending("main")


@pytest.mark.asyncio
async def test_inject_bus_messages_skips_non_request() -> None:
    """Test filter skips when last message is not ModelRequest."""
    bus = MessageBus()
    bus.subscribe("main")
    bus.send("Hello", source="user", target="main")
    ctx = create_mock_ctx(message_bus=bus)
    messages = [
        ModelRequest(parts=[UserPromptPart(content="Hello")]),
        ModelResponse(parts=[TextPart(content="Hi there")]),
    ]

    result = await inject_bus_messages(ctx, messages)

    # Not injected, messages unchanged
    assert result == messages
    # Message still pending (not consumed)
    assert bus.has_pending("main")


@pytest.mark.asyncio
async def test_inject_bus_messages_empty_history() -> None:
    """Test filter handles empty message history."""
    bus = MessageBus()
    bus.subscribe("main")
    bus.send("Hello", source="user", target="main")
    ctx = create_mock_ctx(message_bus=bus)

    result = await inject_bus_messages(ctx, [])

    assert result == []
    assert bus.has_pending("main")  # Not consumed
