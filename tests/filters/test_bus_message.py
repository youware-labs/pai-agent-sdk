"""Tests for bus message injection filter."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai import RunContext
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from pai_agent_sdk.context import BusMessage, MessageBus
from pai_agent_sdk.filters.bus_message import inject_bus_messages


def create_mock_ctx(agent_id: str = "main", message_bus: MessageBus | None = None) -> RunContext:
    """Create a mock RunContext with mocked deps."""
    bus = message_bus or MessageBus()

    def consume_messages() -> list[BusMessage]:
        """Delegate to message_bus.consume() which is now idempotent."""
        return bus.consume(agent_id)

    mock_deps = MagicMock()
    mock_deps.agent_id = agent_id
    mock_deps.message_bus = bus
    mock_deps.emit_event = AsyncMock()
    mock_deps.consume_messages = consume_messages
    mock_deps.steering_messages = []  # Add steering_messages list

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
    bus.send(BusMessage(content="Please stop", source="user", target="main"))
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
    bus.send(BusMessage(content="Hello", source="system", target="main", template="[SYSTEM] {{ content }}"))
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
    bus.send(BusMessage(content="First message", source="user", target="main"))
    bus.send(
        BusMessage(content="Second message", source="system", target="main", template="From system: {{ content }}")
    )
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
    bus.send(BusMessage(content="Hello", source="user", target="main"))
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
    bus.send(BusMessage(content="First", source="user", target="main"))
    bus.send(BusMessage(content="Second", source="system", target="main", template="[SYSTEM] {{ content }}"))
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
    bus.send(BusMessage(content="For main", source="user", target="main"))
    bus.send(BusMessage(content="For other", source="user", target="other"))
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
    bus.send(BusMessage(content="Hello", source="user", target="main"))
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
    bus.send(BusMessage(content="Hello", source="user", target="main"))
    ctx = create_mock_ctx(message_bus=bus)

    result = await inject_bus_messages(ctx, [])

    assert result == []
    assert bus.has_pending("main")  # Not consumed


@pytest.mark.asyncio
async def test_inject_bus_messages_idempotent() -> None:
    """Test filter is idempotent - same message not injected twice."""
    bus = MessageBus()
    bus.subscribe("main")
    bus.send(BusMessage(id="msg-001", content="Hello", source="user", target="main"))
    ctx = create_mock_ctx(message_bus=bus)
    messages = [ModelRequest(parts=[UserPromptPart(content="Hello")])]

    # First call
    result1 = await inject_bus_messages(ctx, messages)
    assert len(result1[0].parts) == 2

    # Second call (simulating retry) - should not inject again
    messages2 = [ModelRequest(parts=[UserPromptPart(content="Hello")])]
    result2 = await inject_bus_messages(ctx, messages2)
    assert len(result2[0].parts) == 1  # No new parts added


@pytest.mark.asyncio
async def test_inject_bus_messages_accumulates_user_steering() -> None:
    """Test filter accumulates user steering messages for compact."""
    bus = MessageBus()
    bus.subscribe("main")
    bus.send(BusMessage(content="User steering 1", source="user", target="main"))
    bus.send(BusMessage(content="System message", source="system", target="main"))
    bus.send(BusMessage(content="User steering 2", source="user", target="main"))
    ctx = create_mock_ctx(message_bus=bus)
    messages = [ModelRequest(parts=[UserPromptPart(content="Hello")])]

    await inject_bus_messages(ctx, messages)

    # Only user messages should be accumulated
    assert len(ctx.deps.steering_messages) == 2
    assert ctx.deps.steering_messages[0] == "User steering 1"
    assert ctx.deps.steering_messages[1] == "User steering 2"


@pytest.mark.asyncio
async def test_inject_bus_messages_accumulates_rendered_content() -> None:
    """Test filter accumulates rendered (with template) content for steering."""
    bus = MessageBus()
    bus.subscribe("main")
    bus.send(BusMessage(content="Stop task", source="user", target="main", template="[URGENT] {{ content }}"))
    ctx = create_mock_ctx(message_bus=bus)
    messages = [ModelRequest(parts=[UserPromptPart(content="Hello")])]

    await inject_bus_messages(ctx, messages)

    # Rendered content (with template) should be accumulated
    assert len(ctx.deps.steering_messages) == 1
    assert ctx.deps.steering_messages[0] == "[URGENT] Stop task"
