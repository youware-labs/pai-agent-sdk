"""Tests for message bus functionality."""

from datetime import datetime

from pai_agent_sdk.context import BusMessage, MessageBus

# =============================================================================
# BusMessage Tests
# =============================================================================


def test_bus_message_creation() -> None:
    """Test BusMessage creation with required fields."""
    msg = BusMessage(id="msg-001", content="Hello", source="user")
    assert msg.id == "msg-001"
    assert msg.content == "Hello"
    assert msg.source == "user"
    assert msg.target is None
    assert isinstance(msg.timestamp, datetime)


def test_bus_message_auto_generates_id() -> None:
    """Test BusMessage auto-generates UUID id if not provided."""
    msg = BusMessage(content="Hello", source="user")
    assert len(msg.id) == 32  # UUID hex format


def test_bus_message_with_target() -> None:
    """Test BusMessage creation with target."""
    msg = BusMessage(id="msg-001", content="Hello", source="user", target="main")
    assert msg.target == "main"


def test_bus_message_render_without_template() -> None:
    """Test message rendering without template."""
    msg = BusMessage(id="msg-001", content="Hello", source="user")
    assert msg.render() == "Hello"


def test_bus_message_render_with_template() -> None:
    """Test message rendering with Jinja2 template."""
    msg = BusMessage(id="msg-001", content="Stop", source="user", template="[URGENT] {{ content }}")
    assert msg.render() == "[URGENT] Stop"


# =============================================================================
# MessageBus Subscribe/Unsubscribe Tests
# =============================================================================


def test_message_bus_subscribe() -> None:
    """Test subscribing to the bus."""
    bus = MessageBus()
    assert bus.subscriber_count == 0

    bus.subscribe("main")
    assert bus.subscriber_count == 1

    # Idempotent
    bus.subscribe("main")
    assert bus.subscriber_count == 1


def test_message_bus_unsubscribe() -> None:
    """Test unsubscribing from the bus."""
    bus = MessageBus()
    bus.subscribe("main")
    bus.subscribe("subagent")
    assert bus.subscriber_count == 2

    bus.unsubscribe("subagent")
    assert bus.subscriber_count == 1

    # Unsubscribe non-existent is no-op
    bus.unsubscribe("nonexistent")
    assert bus.subscriber_count == 1


def test_message_bus_auto_subscribe_on_consume() -> None:
    """Test that consume auto-subscribes if not already subscribed."""
    bus = MessageBus()
    bus.send(BusMessage(content="Hello", source="user", target="main"))

    # Not subscribed yet
    assert bus.subscriber_count == 0

    # Consume auto-subscribes
    messages = bus.consume("main")
    assert bus.subscriber_count == 1
    # But since we subscribed at latest, we won't see the old message
    assert len(messages) == 0


# =============================================================================
# MessageBus Send/Consume Tests
# =============================================================================


def test_message_bus_send() -> None:
    """Test sending messages to the bus."""
    bus = MessageBus()
    msg = bus.send(BusMessage(content="Hello", source="user"))

    assert len(msg.id) == 32  # UUID hex
    assert msg.content == "Hello"
    assert msg.source == "user"
    assert len(bus) == 1


def test_message_bus_send_with_explicit_id() -> None:
    """Test sending messages with explicit id."""
    bus = MessageBus()
    msg = bus.send(BusMessage(id="custom-id-123", content="Hello", source="user"))

    assert msg.id == "custom-id-123"
    assert msg.content == "Hello"


def test_message_bus_send_generates_unique_ids() -> None:
    """Test that auto-generated message IDs are unique."""
    bus = MessageBus()
    msg1 = bus.send(BusMessage(content="First", source="user"))
    msg2 = bus.send(BusMessage(content="Second", source="user"))
    msg3 = bus.send(BusMessage(content="Third", source="user"))

    assert msg1.id != msg2.id != msg3.id


def test_message_bus_send_idempotent() -> None:
    """Test that send is idempotent with same message id."""
    bus = MessageBus()
    msg1 = bus.send(BusMessage(id="same-id", content="Hello", source="user"))
    msg2 = bus.send(BusMessage(id="same-id", content="Different", source="other"))

    # Should return the same message, not create duplicate
    assert msg1.id == msg2.id
    assert msg1.content == "Hello"  # Original content preserved
    assert len(bus) == 1


def test_message_bus_consume_targeted() -> None:
    """Test consuming messages targeted at specific agent."""
    bus = MessageBus()
    bus.subscribe("main")
    bus.subscribe("other")

    bus.send(BusMessage(content="For main", source="user", target="main"))
    bus.send(BusMessage(content="For other", source="user", target="other"))

    messages = bus.consume("main")
    assert len(messages) == 1
    assert messages[0].content == "For main"

    messages = bus.consume("other")
    assert len(messages) == 1
    assert messages[0].content == "For other"


def test_message_bus_consume_broadcast() -> None:
    """Test consuming broadcast messages (target=None)."""
    bus = MessageBus()
    bus.subscribe("main")
    bus.subscribe("other")

    bus.send(BusMessage(content="Broadcast", source="user"))  # No target = broadcast

    # Both subscribers receive the broadcast
    main_messages = bus.consume("main")
    assert len(main_messages) == 1
    assert main_messages[0].content == "Broadcast"

    other_messages = bus.consume("other")
    assert len(other_messages) == 1
    assert other_messages[0].content == "Broadcast"


def test_message_bus_consume_fifo_order() -> None:
    """Test that messages are consumed in FIFO order."""
    bus = MessageBus()
    bus.subscribe("main")

    bus.send(BusMessage(content="First", source="user", target="main"))
    bus.send(BusMessage(content="Second", source="user", target="main"))
    bus.send(BusMessage(content="Third", source="user", target="main"))

    messages = bus.consume("main")
    assert [m.content for m in messages] == ["First", "Second", "Third"]


def test_message_bus_consume_advances_cursor() -> None:
    """Test that cursor advances after consume."""
    bus = MessageBus()
    bus.subscribe("main")

    bus.send(BusMessage(content="First", source="user", target="main"))
    messages = bus.consume("main")
    assert len(messages) == 1

    # Second consume returns empty (already read)
    messages = bus.consume("main")
    assert len(messages) == 0

    # New message is visible
    bus.send(BusMessage(content="Second", source="user", target="main"))
    messages = bus.consume("main")
    assert len(messages) == 1
    assert messages[0].content == "Second"


def test_message_bus_has_pending() -> None:
    """Test has_pending check."""
    bus = MessageBus()
    bus.subscribe("main")

    assert not bus.has_pending("main")

    bus.send(BusMessage(content="Hello", source="user", target="main"))
    assert bus.has_pending("main")

    bus.consume("main")
    assert not bus.has_pending("main")


def test_message_bus_has_pending_broadcast() -> None:
    """Test has_pending with broadcast messages."""
    bus = MessageBus()
    bus.subscribe("main")
    bus.subscribe("other")

    bus.send(BusMessage(content="Broadcast", source="user"))  # No target

    assert bus.has_pending("main")
    assert bus.has_pending("other")


def test_message_bus_has_pending_unsubscribed() -> None:
    """Test has_pending for unsubscribed agent returns False."""
    bus = MessageBus()
    bus.send(BusMessage(content="Hello", source="user", target="main"))

    # Not subscribed, so has_pending is False
    assert not bus.has_pending("main")


def test_message_bus_peek() -> None:
    """Test peeking at messages without advancing cursor."""
    bus = MessageBus()
    bus.subscribe("main")
    bus.send(BusMessage(content="Hello", source="user", target="main"))

    messages = bus.peek("main")
    assert len(messages) == 1

    # Peek again still returns same messages
    messages = bus.peek("main")
    assert len(messages) == 1

    # Consume advances cursor
    bus.consume("main")
    messages = bus.peek("main")
    assert len(messages) == 0


def test_message_bus_peek_unsubscribed() -> None:
    """Test peek returns empty for unsubscribed agent."""
    bus = MessageBus()
    bus.send(BusMessage(content="Hello", source="user", target="main"))

    # Not subscribed, should return empty
    assert bus.peek("main") == []


def test_message_bus_clear() -> None:
    """Test clearing all messages and cursors."""
    bus = MessageBus()
    bus.subscribe("main")
    bus.send(BusMessage(content="One", source="user"))
    bus.send(BusMessage(content="Two", source="user"))

    assert len(bus) == 2
    assert bus.subscriber_count == 1

    bus.clear()
    assert len(bus) == 0
    assert bus.subscriber_count == 0


def test_message_bus_bool() -> None:
    """Test boolean evaluation of bus."""
    bus = MessageBus()
    assert not bus

    bus.send(BusMessage(content="Hello", source="user"))
    assert bus


# =============================================================================
# MessageBus Maxlen Tests
# =============================================================================


def test_message_bus_maxlen() -> None:
    """Test that old messages are trimmed when maxlen exceeded."""
    bus = MessageBus(maxlen=3)
    bus.subscribe("main")

    bus.send(BusMessage(content="One", source="user", target="main"))
    bus.send(BusMessage(content="Two", source="user", target="main"))
    bus.send(BusMessage(content="Three", source="user", target="main"))
    assert len(bus) == 3

    bus.send(BusMessage(content="Four", source="user", target="main"))
    assert len(bus) == 3

    messages = bus.consume("main")
    # "One" was trimmed
    assert [m.content for m in messages] == ["Two", "Three", "Four"]


def test_message_bus_cursor_correction_on_trim() -> None:
    """Test that cursor is corrected when it points to trimmed messages."""
    bus = MessageBus(maxlen=3)
    bus.subscribe("main")

    # Send and don't consume
    bus.send(BusMessage(content="One", source="user", target="main"))
    bus.send(BusMessage(content="Two", source="user", target="main"))
    bus.send(BusMessage(content="Three", source="user", target="main"))

    # Add more messages, trimming old ones
    bus.send(BusMessage(content="Four", source="user", target="main"))
    bus.send(BusMessage(content="Five", source="user", target="main"))

    # Cursor still at 0, but messages 1-2 are trimmed
    # Should get messages from earliest available
    messages = bus.consume("main")
    assert [m.content for m in messages] == ["Three", "Four", "Five"]


def test_message_bus_idempotency_does_not_survive_trim() -> None:
    """Test that idempotency does NOT survive trim.

    After a message is trimmed, its ID is removed from tracking.
    Resending creates a new message. This is expected behavior to bound memory.
    """
    bus = MessageBus(maxlen=2)

    # Send message that will be trimmed
    bus.send(BusMessage(id="id-001", content="First", source="user"))
    bus.send(BusMessage(id="id-002", content="Second", source="user"))
    bus.send(BusMessage(id="id-003", content="Third", source="user"))

    # id-001 is trimmed AND removed from _message_ids
    assert len(bus) == 2

    # Resending id-001 is NOT idempotent - treated as new message
    msg_retry = bus.send(BusMessage(id="id-001", content="First retry", source="user"))
    assert msg_retry.id == "id-001"
    assert msg_retry.content == "First retry"  # New content accepted
    assert len(bus) == 2  # Still 2 because id-002 was trimmed


# =============================================================================
# MessageBus Multi-Agent Tests
# =============================================================================


def test_message_bus_subagent_lifecycle() -> None:
    """Test typical subagent subscribe/unsubscribe lifecycle."""
    bus = MessageBus()
    bus.subscribe("main")

    # Send message to main
    bus.send(BusMessage(content="For main", source="user", target="main"))

    # Subagent starts, subscribes
    bus.subscribe("subagent-123")

    # Broadcast goes to both
    bus.send(BusMessage(content="Broadcast", source="user"))

    # Main gets both messages
    main_msgs = bus.consume("main")
    assert len(main_msgs) == 2

    # Subagent only gets broadcast (subscribed after first message)
    sub_msgs = bus.consume("subagent-123")
    assert len(sub_msgs) == 1
    assert sub_msgs[0].content == "Broadcast"

    # Subagent exits, unsubscribes
    bus.unsubscribe("subagent-123")
    assert bus.subscriber_count == 1


def test_message_bus_targeted_inter_agent() -> None:
    """Test targeted messages between agents."""
    bus = MessageBus()
    bus.subscribe("main")
    bus.subscribe("subagent")

    # Main sends to subagent
    bus.send(BusMessage(content="Task for you", source="main", target="subagent"))
    # Subagent sends to main
    bus.send(BusMessage(content="Result", source="subagent", target="main"))

    main_msgs = bus.consume("main")
    assert len(main_msgs) == 1
    assert main_msgs[0].content == "Result"

    sub_msgs = bus.consume("subagent")
    assert len(sub_msgs) == 1
    assert sub_msgs[0].content == "Task for you"


def test_message_bus_new_subscriber_only_sees_new_messages() -> None:
    """Test that new subscribers only see messages sent after subscribing."""
    bus = MessageBus()
    bus.subscribe("main")

    # Send before subagent subscribes
    bus.send(BusMessage(content="Old message", source="user", target="main"))

    # Subagent subscribes (should not see old broadcast)
    bus.subscribe("subagent")

    # New broadcast
    bus.send(BusMessage(content="New broadcast", source="user"))

    # Subagent only sees new message
    sub_msgs = bus.consume("subagent")
    assert len(sub_msgs) == 1
    assert sub_msgs[0].content == "New broadcast"
