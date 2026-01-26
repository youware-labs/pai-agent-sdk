"""Tests for message bus functionality."""

from datetime import datetime

from pai_agent_sdk.bus import BusMessage, MessageBus

# =============================================================================
# BusMessage Tests
# =============================================================================


def test_bus_message_creation() -> None:
    """Test BusMessage creation with required fields."""
    msg = BusMessage(content="Hello", source="user")
    assert msg.content == "Hello"
    assert msg.source == "user"
    assert msg.target is None
    assert isinstance(msg.timestamp, datetime)


def test_bus_message_with_target() -> None:
    """Test BusMessage creation with target."""
    msg = BusMessage(content="Hello", source="user", target="main")
    assert msg.target == "main"


# =============================================================================
# MessageBus Tests
# =============================================================================


def test_message_bus_send() -> None:
    """Test sending messages to the bus."""
    bus = MessageBus()
    msg = bus.send("Hello", source="user")

    assert msg.content == "Hello"
    assert msg.source == "user"
    assert len(bus) == 1


def test_message_bus_send_with_target() -> None:
    """Test sending messages with specific target."""
    bus = MessageBus()
    bus.send("Hello", source="user", target="main")

    assert len(bus) == 1
    assert bus.has_pending("main")
    assert not bus.has_pending("other")


def test_message_bus_consume_targeted() -> None:
    """Test consuming messages targeted at specific agent."""
    bus = MessageBus()
    bus.send("For main", source="user", target="main")
    bus.send("For other", source="user", target="other")

    messages = bus.consume("main")

    assert len(messages) == 1
    assert messages[0].content == "For main"
    assert len(bus) == 1  # "For other" remains


def test_message_bus_consume_broadcast() -> None:
    """Test consuming broadcast messages (target=None)."""
    bus = MessageBus()
    bus.send("Broadcast", source="user")  # No target = broadcast
    bus.send("For main", source="user", target="main")

    messages = bus.consume("main")

    assert len(messages) == 2
    assert messages[0].content == "Broadcast"
    assert messages[1].content == "For main"
    assert len(bus) == 0


def test_message_bus_consume_fifo_order() -> None:
    """Test that messages are consumed in FIFO order."""
    bus = MessageBus()
    bus.send("First", source="user", target="main")
    bus.send("Second", source="user", target="main")
    bus.send("Third", source="user", target="main")

    messages = bus.consume("main")

    assert [m.content for m in messages] == ["First", "Second", "Third"]


def test_message_bus_consume_removes_messages() -> None:
    """Test that consumed messages are removed from bus."""
    bus = MessageBus()
    bus.send("Hello", source="user", target="main")

    assert len(bus) == 1
    messages = bus.consume("main")
    assert len(messages) == 1
    assert len(bus) == 0

    # Second consume returns empty
    messages = bus.consume("main")
    assert len(messages) == 0


def test_message_bus_has_pending() -> None:
    """Test has_pending check."""
    bus = MessageBus()
    assert not bus.has_pending("main")

    bus.send("Hello", source="user", target="main")
    assert bus.has_pending("main")
    assert not bus.has_pending("other")


def test_message_bus_has_pending_broadcast() -> None:
    """Test has_pending with broadcast messages."""
    bus = MessageBus()
    bus.send("Broadcast", source="user")  # No target

    assert bus.has_pending("main")
    assert bus.has_pending("other")
    assert bus.has_pending("any_agent")


def test_message_bus_peek() -> None:
    """Test peeking at messages without consuming."""
    bus = MessageBus()
    bus.send("Hello", source="user", target="main")

    messages = bus.peek("main")
    assert len(messages) == 1
    assert len(bus) == 1  # Not removed

    # Can peek again
    messages = bus.peek("main")
    assert len(messages) == 1


def test_message_bus_clear() -> None:
    """Test clearing all messages."""
    bus = MessageBus()
    bus.send("One", source="user")
    bus.send("Two", source="user")
    bus.send("Three", source="user")

    assert len(bus) == 3
    bus.clear()
    assert len(bus) == 0


def test_message_bus_bool() -> None:
    """Test boolean evaluation of bus."""
    bus = MessageBus()
    assert not bus

    bus.send("Hello", source="user")
    assert bus


def test_message_bus_multiple_agents() -> None:
    """Test message routing between multiple agents."""
    bus = MessageBus()

    # User sends to main
    bus.send("User to main", source="user", target="main")
    # Main sends to subagent
    bus.send("Main to sub", source="main", target="subagent-123")
    # Subagent sends to main
    bus.send("Sub to main", source="subagent-123", target="main")

    # Main agent consumes
    main_messages = bus.consume("main")
    assert len(main_messages) == 2
    assert main_messages[0].content == "User to main"
    assert main_messages[1].content == "Sub to main"

    # Subagent consumes
    sub_messages = bus.consume("subagent-123")
    assert len(sub_messages) == 1
    assert sub_messages[0].content == "Main to sub"
