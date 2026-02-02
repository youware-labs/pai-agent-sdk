"""Message bus for inter-agent communication.

This module provides a message bus for communication between agents,
inspired by Redis Streams design with subscriber cursors.

Key features:
- Subscriber-based consumption: Each subscriber maintains its own cursor
- Broadcast support: target=None delivers to all subscribers
- Bounded queue: maxlen limits memory usage, old messages auto-trimmed
- Idempotent send: Duplicate message.id is silently ignored
- Idempotent consume: Each message consumed only once per subscriber
- No message loss: Multiple subscribers can read the same broadcast message

Example:
    Basic usage with AgentContext::

        from pai_agent_sdk.context import AgentContext, BusMessage

        # Messages are sent via AgentContext.send_message()
        msg = BusMessage(content="Please prioritize task A", source="user", target="main")
        ctx.send_message(msg)

    Direct MessageBus usage (advanced)::

        from pai_agent_sdk.context import MessageBus, BusMessage

        bus = MessageBus(maxlen=100)
        bus.subscribe("main")
        bus.send(BusMessage(content="Hello", source="user", target="main"))
        messages = bus.consume("main")  # Returns unread messages (idempotent)
"""

from __future__ import annotations

import uuid
from datetime import datetime

from jinja2 import Template
from pydantic import BaseModel, Field


def render_template(content: str, template: str | None) -> str:
    """Render content using a Jinja2 template, or return raw content if no template.

    Args:
        content: The message content.
        template: Jinja2 template string with {{ content }} placeholder. None for raw content.

    Returns:
        Rendered string.
    """
    if template is None:
        return content
    return Template(template).render(content=content)


class BusMessage(BaseModel):
    """A message in the message bus.

    Attributes:
        id: Unique message ID (UUID string, auto-generated if not provided).
        content: The message content (typically markdown text).
        source: Who sent the message (e.g., "user", agent_id).
        target: Who should receive the message (agent_id, or None for broadcast).
        template: Jinja2 template string for rendering. Use {{ content }} placeholder.
        timestamp: When the message was created.

    Example::

        # Minimal message (id auto-generated)
        msg = BusMessage(content="Hello", source="user")

        # Targeted message with explicit id
        msg = BusMessage(
            id="unique-id-123",
            content="Focus on security",
            source="user",
            target="main",
        )

        # With template
        msg = BusMessage(
            content="Stop",
            source="user",
            template="[URGENT] {{ content }}",
        )
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    """Unique message ID (UUID string). Used for idempotent send and consume."""

    content: str
    """The message content (typically markdown text)."""

    source: str
    """Who sent the message (e.g., "user", agent_id)."""

    target: str | None = None
    """Who should receive the message (agent_id, or None for broadcast)."""

    template: str | None = None
    """Jinja2 template string for rendering. Use {{ content }} placeholder."""

    timestamp: datetime = Field(default_factory=datetime.now)
    """When the message was created."""

    def render(self) -> str:
        """Render the message using its Jinja2 template, or return raw content if no template."""
        return render_template(self.content, self.template)


class MessageBus:
    """Message bus for inter-agent communication (Redis Streams style).

    The message bus supports:
    - Targeted messages: target=agent_id for specific recipient
    - Broadcast messages: target=None for all subscribers
    - Subscriber cursors: Each subscriber tracks its read position
    - Bounded queue: Old messages auto-trimmed when maxlen exceeded
    - Idempotent send: Duplicate message.id returns existing message
    - Idempotent consume: Each message consumed only once per subscriber

    Subscriber Lifecycle:
    - subscribe(): Register subscriber, cursor starts at current position
    - consume(): Get unread messages (idempotent), advance cursor
    - unsubscribe(): Remove subscriber, free cursor (for subagents on exit)

    Thread Safety:
        This class is NOT thread-safe. It's designed to be used within
        a single asyncio event loop. For concurrent access from multiple
        coroutines, operations are atomic (no await points).

    Example:
        >>> bus = MessageBus(maxlen=100)
        >>> bus.subscribe("main")
        >>> bus.send(BusMessage(content="Hello", source="user", target="main"))
        >>> bus.send(BusMessage(content="World", source="user"))  # broadcast
        >>> messages = bus.consume("main")
        >>> [m.content for m in messages]
        ['Hello', 'World']
        >>> bus.consume("main")  # Already consumed (idempotent)
        []
    """

    def __init__(self, maxlen: int = 500) -> None:
        """Initialize an empty message bus.

        Args:
            maxlen: Maximum number of messages to retain. Oldest messages
                    are trimmed when exceeded. Default: 500.
        """
        self._messages: list[BusMessage] = []
        self._message_ids: set[str] = set()  # For idempotent send
        self._cursors: dict[str, int] = {}  # agent_id -> position index in _messages
        self._consumed_ids: dict[str, set[str]] = {}  # agent_id -> consumed message IDs
        self._maxlen = maxlen

    def subscribe(self, agent_id: str) -> None:
        """Register a subscriber.

        New subscribers start with cursor at the current position,
        meaning they will only receive messages sent after subscribing.

        Args:
            agent_id: Unique identifier for the subscriber.
        """
        if agent_id not in self._cursors:
            # Start from current position (only see new messages)
            self._cursors[agent_id] = len(self._messages)
            self._consumed_ids[agent_id] = set()

    def unsubscribe(self, agent_id: str) -> None:
        """Remove a subscriber.

        Should be called when a subagent exits to free cursor memory.

        Args:
            agent_id: The subscriber to remove.
        """
        self._cursors.pop(agent_id, None)
        self._consumed_ids.pop(agent_id, None)

    def send(self, message: BusMessage) -> BusMessage:
        """Send a message to the bus.

        This operation is idempotent: if a message with the same id
        already exists, the existing message is returned without creating a duplicate.

        Args:
            message: The message to send.

        Returns:
            The sent message, or existing message if id already exists.
        """
        # Idempotent send: return existing message if ID already exists
        if message.id in self._message_ids:
            for msg in self._messages:
                if msg.id == message.id:
                    return msg
            # ID exists but message was trimmed - return the provided message
            # without adding it (maintains idempotency)
            return message

        self._messages.append(message)
        self._message_ids.add(message.id)

        # Trim old messages if exceeding maxlen
        self._trim()

        return message

    def _trim(self) -> None:
        """Trim old messages if exceeding maxlen."""
        if len(self._messages) > self._maxlen:
            trim_count = len(self._messages) - self._maxlen
            trimmed_ids = {msg.id for msg in self._messages[:trim_count]}

            # Remove trimmed message IDs from the set
            self._message_ids -= trimmed_ids

            # Remove trimmed IDs from all subscribers' consumed_ids (memory cleanup)
            for consumed in self._consumed_ids.values():
                consumed -= trimmed_ids

            # Trim the messages list
            self._messages = self._messages[trim_count:]

            # Adjust all cursors (shift down by trim_count, min 0)
            for agent_id in self._cursors:
                self._cursors[agent_id] = max(0, self._cursors[agent_id] - trim_count)

    def consume(self, agent_id: str) -> list[BusMessage]:
        """Consume unread messages for a subscriber (idempotent).

        Returns messages that:
        1. Are after subscriber's cursor position (unread)
        2. Are targeted to this subscriber OR are broadcasts (target=None)
        3. Have not been consumed before by this subscriber

        This operation is idempotent: each message is returned only once
        per subscriber, even if consume() is called multiple times.

        Automatically subscribes if not already subscribed.

        Args:
            agent_id: The subscriber ID to consume messages for.

        Returns:
            List of new messages, in FIFO order (oldest first).
        """
        # Auto-subscribe if not registered
        if agent_id not in self._cursors:
            self.subscribe(agent_id)

        cursor = self._cursors[agent_id]
        consumed = self._consumed_ids[agent_id]

        # Find matching unread messages (from cursor position to end)
        result: list[BusMessage] = []
        for msg in self._messages[cursor:]:
            # Check: (broadcast OR targeted) AND not already consumed
            if (msg.target is None or msg.target == agent_id) and msg.id not in consumed:
                result.append(msg)
                consumed.add(msg.id)

        # Update cursor to end of list
        self._cursors[agent_id] = len(self._messages)

        return result

    def has_pending(self, agent_id: str) -> bool:
        """Check if there are unread messages for a subscriber.

        Args:
            agent_id: The subscriber ID to check.

        Returns:
            True if there are unread messages (broadcast or targeted) that
            have not been consumed yet.
        """
        if agent_id not in self._cursors:
            return False

        cursor = self._cursors[agent_id]
        consumed = self._consumed_ids.get(agent_id, set())

        return any(
            (msg.target is None or msg.target == agent_id) and msg.id not in consumed for msg in self._messages[cursor:]
        )

    def peek(self, agent_id: str) -> list[BusMessage]:
        """Peek at unread messages without consuming.

        Args:
            agent_id: The subscriber ID to check.

        Returns:
            List of unread messages that have not been consumed
            (cursor not updated, messages not marked as consumed).
        """
        if agent_id not in self._cursors:
            return []

        cursor = self._cursors[agent_id]
        consumed = self._consumed_ids.get(agent_id, set())

        return [
            msg
            for msg in self._messages[cursor:]
            if (msg.target is None or msg.target == agent_id) and msg.id not in consumed
        ]

    def clear(self) -> None:
        """Clear all messages and reset subscribers."""
        self._messages.clear()
        self._message_ids.clear()
        self._cursors.clear()
        self._consumed_ids.clear()

    @property
    def subscriber_count(self) -> int:
        """Return the number of registered subscribers."""
        return len(self._cursors)

    def __len__(self) -> int:
        """Return the total number of messages in the queue."""
        return len(self._messages)

    def __bool__(self) -> bool:
        """Return True if there are any messages in the queue."""
        return len(self._messages) > 0
